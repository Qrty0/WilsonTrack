"""Entraîne l'« IA de référence » v2 de WilsonTrack sur un grand nombre de traces SIMULÉES.

Principe (voir aussi wilsontrack/detection/reference_model.py) :
  1. génération — chaque « tâche » est un enregistrement complet simulé EN MÉMOIRE (aucun
     fichier vidéo) par le simulateur physique v2 (wilsontrack/synth.py), avec des conditions
     tirées au hasard (randomisation de domaine : résolution 640×360 à 1920×1080, 15 à 30 im/s,
     largeur de traitement, calibration, zone d'intérêt, bruit, compression, éclairage, épaisseur
     de la couche, taux de chaque source, artefacts…). Les images passent dans le VRAI détecteur
     de l'application ; chaque détection reçoit une étiquette de la vérité terrain
     (evaluate.label_detections : particule, morceau d'une particule, ou « artefact ») ;
     un fichier JSONL par tâche (reprise possible : les tâches déjà faites sont sautées) ;
  2. entraînement — 5 classes (alpha, beta, muon, proton, artefact), HistGradientBoosting,
     séparation PAR ENREGISTREMENT (60 % entraînement / 20 % validation / 20 % test) ;
     calibration des probabilités sur la validation (température ou isotonique, choisie par
     validation croisée), poids w de l'hybride règles + IA et température de l'hybride choisis
     sur la validation ; seuil d'abstention 0,85 (« incertain ») ; mesures sur le test et sur un
     jeu « conditions extrêmes » (graines disjointes, conditions plus dures).

Les protons sont volontairement sur-représentés dans les simulations (pour les apprendre) ; les
probabilités sont ramenées à une fréquence RÉALISTE (1 % des particules, PDG) par une pondération
documentée (« prior_ref »), et toutes les mesures publiées utilisent cette même pondération.

Exemples (PowerShell, depuis la racine du projet) :
  .\\.venv\\Scripts\\python.exe tools\\train_reference_model.py --jobs 300 --workers 5
  .\\.venv\\Scripts\\python.exe tools\\train_reference_model.py --train-only
  .\\.venv\\Scripts\\python.exe tools\\train_reference_model.py --examples-only

Grande génération finale (≥ 60 000 traces étiquetées ; ≈ 42 traces par enregistrement, ≈ 3 h avec
10 processus ; à lancer quand le code du détecteur est figé ; reprise possible : relancer la même
commande saute les enregistrements déjà faits) :
  .\\.venv\\Scripts\\python.exe tools\\train_reference_model.py --jobs 1450 --stress-jobs 180 --workers 10 --subdir final --no-train
  .\\.venv\\Scripts\\python.exe tools\\train_reference_model.py --subdir final --train-only --threads 10 --examples
"""
from __future__ import annotations

import os

# Un fil par processus pour les bibliothèques numériques (les tâches tournent en parallèle).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("WILSONTRACK_CV_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import wilsontrack  # noqa: E402,F401  (applique cv2.setNumThreads(1))
from wilsontrack import synth as SY  # noqa: E402
from wilsontrack.config import AppConfig, ClassificationParams  # noqa: E402
from wilsontrack.detection import TrackDetector  # noqa: E402
from wilsontrack.detection import reference_model as RM  # noqa: E402
from wilsontrack.detection.classifier import ML_FEATURES  # noqa: E402
from wilsontrack.evaluate import evaluate, label_detections  # noqa: E402
from wilsontrack.synth import SynthConfig, SyntheticChamber  # noqa: E402

CLASSES = list(RM.ALL_CLASSES)                 # alpha, beta, muon, proton, artefact
PARTICLES = list(RM.CLASSES)
FEATURES = list(RM.V2_FEATURES)
DEFAULT_DATA = ROOT / "_training" / "data"
DEFAULT_OUT = ROOT / "wilsontrack" / "assets" / "modele_reference.joblib"
DEFAULT_PILOT = ROOT / "_training" / "modele_pilote_v1.joblib"
EXAMPLES_DIR = ROOT / "wilsontrack" / "assets" / "ia_exemples"
STRESS_SEED0 = 1_000_000
W_GRID = [round(0.1 * i, 1) for i in range(11)]
PROTON_PRIOR_FRAC = 0.01        # fréquence réaliste des protons parmi les particules (PDG : ≈ 1 % des muons)

RESOLUTIONS = [(640, 360), (960, 540), (1280, 720), (1920, 1080)]
GENERATOR = {
    "simulateur": f"wilsontrack.synth v{SY.SYNTH_VERSION} (transport 3 D, tables NIST, Highland, rendu commun)",
    "resolution": ["640x360 (15 %)", "960x540 (30 %)", "1280x720 (40 %)", "1920x1080 (15 %)"],
    "fps": ["15 (10 %)", "20 (15 %)", "25 (30 %)", "30 (45 %)"],
    "duration_s": [30.0, 45.0],
    "process_width": ["640 (30 %)", "960 (50 %)", "1280 (20 %)"],
    "calibrated_fraction": 0.6,
    "roi_fraction": 0.35,
    "fov_mm": [140.0, 340.0],
    "noise_sigma": [1.2, 6.0],
    "shot_noise": "0 (50 %) ou U(0,02 ; 0,15)",
    "jpeg_quality": "aucune (55 %) ou U(55 ; 95)",
    "fog_amp": [2.0, 14.0],
    "rain_per_frame": [80, 500],
    "base_level": [8.0, 30.0],
    "light_gradient": [0.0, 0.8],
    "brightness": [0.5, 1.6],
    "width_scale": [0.75, 1.35],
    "droplet_scale": [0.6, 1.6],
    "sat_density_mm2": [12.0, 45.0],
    "diffusion_mm": [0.25, 0.6],
    "supersat_sigma": [0.15, 0.45],
    "life_scale": [0.75, 1.3],
    "chamber_temp_c": [-35.0, -10.0],
    "pressure_kpa": [95.0, 103.0],
    "sensitive_depth_mm (alphas, protons)": [5.0, 25.0],
    "deff_mm (traces fines)": [5.0, 25.0],
    "alpha_per_min": "pauvre en radon 0,5–3 (25 %), normal 3–12 (50 %), riche 12–40 (25 %)",
    "alpha_mix": "Dirichlet autour de Rn-222 0,38 / Po-218 0,24 / Po-214 0,18 / Po-210 0,08 / thoron 0,08 / "
                 "Po-212 0,025 / Bi-212 0,015",
    "beta_compton_per_min": [5.0, 40.0],
    "source_beta_per_min": "0 (65 %) ou U(15, 80) (source de type KCl)",
    "muon_per_min": [15.0, 60.0],
    "proton_per_min": "U(2, 6) — sur-représentés, ramenés à 1 % des particules par pondération",
    "cosmic_e_frac": [0.05, 0.25],
    "shower_per_min": [0.0, 1.0],
    "muon_decay_frac": [0.002, 0.02],
    "artefacts": "chaque type U(0, 2) × défaut × facteur U(0,3 ; 3)",
    "stress": {"seeds_from": STRESS_SEED0, "noise_sigma": [6.0, 9.0], "brightness": "U(0,35 ; 0,5) ou U(1,6 ; 2,0)",
               "fog_amp": [12.0, 20.0], "rain_per_frame": [500, 900], "artefact_factor": [3.0, 6.0],
               "jpeg_quality": [35, 60]},
}


# --------------------------------------------------------------------------- #
# Génération (une tâche = un enregistrement simulé)
# --------------------------------------------------------------------------- #
def draw_params(seed: int, stress: bool = False, pilot: bool = False) -> dict:
    """Conditions de l'enregistrement simulé, tirées avec un générateur propre à la graine."""
    rng = np.random.default_rng([int(seed), 0x5EED, 2])
    u = lambda a, b: float(rng.uniform(a, b))      # noqa: E731

    def pick(options, p):
        return options[int(rng.choice(len(options), p=p))]
    w, h = pick(RESOLUTIONS, [0.15, 0.30, 0.40, 0.15])
    fps = pick([15.0, 20.0, 25.0, 30.0], [0.10, 0.15, 0.30, 0.45])
    dur = u(20.0, 30.0) if pilot else u(30.0, 45.0)
    pw = pick([640, 960, 1280], [0.3, 0.5, 0.2])
    calibrated = bool(rng.uniform() < 0.6)
    roi = []
    if rng.uniform() < 0.35:
        if rng.uniform() < 0.5:
            m = u(0.03, 0.12)
            roi = [[m, m], [1 - m, m], [1 - m, 1 - m], [m, 1 - m]]
        else:                                     # polygone « ellipse » (chambre ronde)
            n = 16
            ax, ay = u(0.38, 0.49), u(0.38, 0.49)
            roi = [[0.5 + ax * math.cos(2 * math.pi * k / n), 0.5 + ay * math.sin(2 * math.pi * k / n)]
                   for k in range(n)]
    fov = u(140.0, 340.0)
    noise = u(1.2, 6.0)
    shot = 0.0 if rng.uniform() < 0.5 else u(0.02, 0.15)
    jpeg = 0 if rng.uniform() < 0.55 else int(round(u(55, 95)))
    fog = u(2.0, 14.0)
    rain = int(round(u(80, 500)))
    base = u(8.0, 30.0)
    grad = u(0.0, 0.8)
    grad_ang = u(0.0, 360.0)
    bright = u(0.5, 1.6)
    width_scale = u(0.75, 1.35)
    droplet_scale = u(0.6, 1.6)
    sat = u(12.0, 45.0)
    diff = u(0.25, 0.6)
    ssig = u(0.15, 0.45)
    life_scale = u(0.75, 1.3)
    temp = u(-35.0, -10.0)
    press = u(95.0, 103.0)
    depth = u(5.0, 25.0)
    deff = u(5.0, 25.0)
    room = rng.uniform()
    alpha = u(0.5, 3.0) if room < 0.25 else (u(3.0, 12.0) if room < 0.75 else u(12.0, 40.0))
    base_mix = SY._default_alpha_mix()
    names = list(base_mix)
    mix_v = rng.dirichlet([8.0 * base_mix[k] + 0.2 for k in names])
    mix = {k: float(v) for k, v in zip(names, mix_v)}
    rates = {"alpha": alpha, "beta": u(5.0, 40.0), "muon": u(15.0, 60.0), "proton": u(2.0, 6.0)}
    src = u(15.0, 80.0) if rng.uniform() < 0.35 else 0.0
    w_mm, h_mm = fov, fov * h / w
    src_xy = (u(0.2, 0.8) * w_mm, u(0.2, 0.8) * h_mm)
    src_r = u(8.0, 30.0)
    self_abs = u(0.2, 0.8)
    art_factor = u(0.3, 3.0)
    arts = {k: float(v) * u(0.0, 2.0) * art_factor for k, v in SY._default_artefacts().items()}
    cosmic_e = u(0.05, 0.25)
    shower = u(0.0, 1.0)
    mdecay = u(0.002, 0.02)
    if stress:                            # conditions plus dures, hors de la plage d'entraînement
        noise = u(6.0, 9.0)
        bright = u(0.35, 0.5) if rng.uniform() < 0.5 else u(1.6, 2.0)
        fog = u(12.0, 20.0)
        rain = int(round(u(500, 900)))
        f = u(3.0, 6.0)
        arts = {k: float(v) * f for k, v in SY._default_artefacts().items()}
        jpeg = int(round(u(35, 60)))
    return {"seed": int(seed), "kind": "stress" if stress else "main", "width": w, "height": h, "fps": fps,
            "duration_s": dur, "process_width": pw, "calibrated": calibrated, "roi": roi, "fov_mm": fov,
            "noise_sigma": noise, "shot_noise": shot, "jpeg_quality": jpeg, "fog_amp": fog, "rain_per_frame": rain,
            "base_level": base, "light_gradient": grad, "light_angle_deg": grad_ang, "brightness": bright,
            "width_scale": width_scale, "droplet_scale": droplet_scale, "sat_density_mm2": sat,
            "diffusion_mm": diff, "supersat_sigma": ssig, "life_scale": life_scale, "chamber_temp_c": temp,
            "pressure_kpa": press, "sensitive_depth_mm": depth, "deff_mm": deff, "rates_per_min": rates,
            "alpha_mix": mix, "source_beta_per_min": src, "source_xy_mm": list(src_xy), "source_radius_mm": src_r,
            "source_self_absorption": self_abs, "artefact_rates_per_min": arts, "cosmic_e_frac": cosmic_e,
            "shower_per_min": shower, "muon_decay_frac": mdecay}


_SYNTH_KEYS = ("width", "height", "fps", "duration_s", "fov_mm", "noise_sigma", "shot_noise", "jpeg_quality",
               "fog_amp", "rain_per_frame", "base_level", "light_gradient", "light_angle_deg", "brightness",
               "width_scale", "droplet_scale", "sat_density_mm2", "diffusion_mm", "supersat_sigma", "life_scale",
               "chamber_temp_c", "pressure_kpa", "sensitive_depth_mm", "deff_mm", "source_beta_per_min",
               "source_radius_mm", "source_self_absorption", "cosmic_e_frac", "shower_per_min", "muon_decay_frac")


def build_configs(p: dict) -> tuple[SynthConfig, AppConfig]:
    kw = {k: p[k] for k in _SYNTH_KEYS if k in p}
    sc = SynthConfig(seed=int(p["seed"]), rates_per_min=dict(p["rates_per_min"]), alpha_mix=dict(p["alpha_mix"]),
                     artefact_rates_per_min=dict(p["artefact_rates_per_min"]),
                     source_xy_mm=tuple(p["source_xy_mm"]), **kw)
    cfg = AppConfig()
    cfg.classification.mode = "regles"          # décision enregistrée = règles physiques (recalculées ensuite)
    cfg.detection.process_width = int(p["process_width"])
    cfg.detection.roi = [list(map(float, q)) for q in p.get("roi") or []]
    cfg.physics.px_per_mm = sc.px_per_mm if p["calibrated"] else 0.0
    cfg.physics.chamber_temp_c = p["chamber_temp_c"]
    cfg.physics.sensitive_depth_mm = p["sensitive_depth_mm"]
    return sc, cfg


def _code_fingerprint() -> str:
    """Empreinte du code qui produit les mesures (détection, mesures, simulateur)."""
    import hashlib
    h = hashlib.sha1()
    for rel in ("wilsontrack/synth.py", "wilsontrack/detection/pipeline.py", "wilsontrack/detection/features.py",
                "wilsontrack/detection/preprocess.py", "wilsontrack/detection/segment.py", "wilsontrack/evaluate.py"):
        try:
            h.update((ROOT / rel).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


def _clean(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _record(e, label: str, how: str, g: dict | None, calibrated: bool, idx: int) -> dict:
    f = {k: _clean(v) for k, v in e.features.items()
         if not k.startswith("_") and isinstance(v, (int, float, np.integer, np.floating))}
    g = g or {}
    return {"type": "track", "label": label, "how": how, "idx": idx, "f": f,
            "duration_s": round(float(e.duration_s), 4), "touches_border": bool(e.touches_border),
            "n_seen": int(e.n_seen), "calibrated": calibrated, "rules_cls": e.cls, "rules_conf": _clean(e.confidence),
            "rules_scores": {k: _clean(v) for k, v in (e.scores or {}).items()},
            "gt_cls": g.get("cls"), "gt_kind": g.get("kind"), "gt_len_mm": g.get("length_mm"),
            "gt_energy_mev": g.get("energy_mev"), "gt_q": g.get("q_mean"), "gt_source": bool(g.get("source", False))}


def simulate_and_detect(p: dict, keep_thumbs: bool = False):
    sc, cfg = build_configs(p)
    cv2.setRNGSeed(int(p["seed"]) % (2 ** 31))
    sim = SyntheticChamber(sc)
    det = TrackDetector(cfg, sc.fps, (sc.height, sc.width))
    for _ in range(sim.n_frames):
        det.process(sim.render_next(), None)
    det.finalize_run()
    gt = {"px_per_mm": sc.px_per_mm, "config": {"fps": sc.fps, "width": sc.width, "height": sc.height},
          "tracks": sim.gt}
    return sc, cfg, sim, det, gt


def run_job(seed: int, kind: str, pilot: bool, out_path: str) -> dict:
    """Simule un enregistrement, le fait analyser par le détecteur et écrit les traces étiquetées."""
    t0 = time.perf_counter()
    p = draw_params(seed, kind == "stress", pilot)
    sc, cfg, sim, det, gt = simulate_and_detect(p)
    t_sim = time.perf_counter() - t0
    res = evaluate(det.events, gt, min_len_mm=0.0)
    labs = label_detections(det.events, gt, res=res)
    order = {id(e): i for i, e in enumerate(det.valid_events)}
    recs = [_record(d["event"], d["label"], d["how"], d["gt"], p["calibrated"], order[id(d["event"])])
            for d in labs]
    # Efficacité de détection par type de vérité terrain (traces ≥ 5 mm)
    eff: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g, e in res["_matches_all"]:
        if g["length_mm"] >= 5.0:
            k = f"{g['cls']}/{g.get('kind', g['cls'])}"
            eff[k][1] += 1
            eff[k][0] += int(e is not None)
    n_by = Counter(r["label"] for r in recs)
    how_by = Counter(r["how"] for r in recs)
    dt = time.perf_counter() - t0
    header = {"type": "job", **p, "n_frames": sim.n_frames, "n_gt": len(sim.gt), "n_labelled": len(recs),
              "n_by_label": dict(n_by), "n_by_how": dict(how_by), "n_valid": len(det.valid_events),
              "detection_by_kind": {k: v for k, v in eff.items()}, "seconds": round(dt, 2),
              "seconds_sim_detect": round(t_sim, 2), "px_per_mm_proc": det.px_per_mm_proc,
              "code": _code_fingerprint(), "synth_version": SY.SYNTH_VERSION}
    out = Path(out_path)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return {"seed": seed, "kind": kind, "n_labelled": len(recs), "n_artefact": int(n_by.get("artefact", 0)),
            "n_frames": sim.n_frames, "seconds": dt, "pixels": sc.width * sc.height}


def _fmt(x: float, nd: int = 1) -> str:
    """Nombre au format québécois (virgule décimale, espace pour les milliers)."""
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _fmt_dur(s: float) -> str:
    s = max(0, int(round(s)))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s:02d} s" if m else f"{s} s")


def generate(data_dir: Path, seeds: list[tuple[int, str]], workers: int, pilot: bool) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for seed, kind in seeds:
        out = data_dir / f"{kind}_{seed:07d}.jsonl"
        if not out.exists():
            todo.append((seed, kind, str(out)))
    print(f"Génération : {len(seeds)} enregistrements simulés demandés, {len(seeds) - len(todo)} déjà faits, "
          f"{len(todo)} à faire ({workers} processus).", flush=True)
    stats = {"jobs": 0, "labelled": 0, "artefact": 0, "seconds_jobs": 0.0, "frames": 0, "failed": 0}
    if not todo:
        return stats
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_job, s, k, pilot, o): (s, k) for s, k, o in todo}
        for fut in as_completed(futs):
            s, k = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:          # une tâche qui échoue n'arrête pas les autres
                stats["failed"] += 1
                print(f"  ! graine {s} ({k}) : échec — {exc!r}", flush=True)
                continue
            stats["jobs"] += 1
            stats["labelled"] += r["n_labelled"]
            stats["artefact"] += r["n_artefact"]
            stats["seconds_jobs"] += r["seconds"]
            stats["frames"] += r["n_frames"]
            el = time.perf_counter() - t0
            rate = stats["labelled"] / max(el, 1e-9) * 60.0
            eta = el / stats["jobs"] * (len(todo) - stats["jobs"])
            print(f"  [{stats['jobs']:>5d}/{len(todo)}] graine {s:>7d} ({'extrême' if k == 'stress' else 'principal'}) : "
                  f"{r['n_labelled']:>3d} traces étiquetées dont {r['n_artefact']:>2d} artefacts, "
                  f"{_fmt(r['seconds'])} s | total {stats['labelled']} traces, {_fmt(rate, 0)} traces/min, "
                  f"reste ≈ {_fmt_dur(eta)}", flush=True)
    stats["wall_s"] = time.perf_counter() - t0
    stats["tracks_per_min"] = stats["labelled"] / max(stats["wall_s"], 1e-9) * 60.0
    stats["s_per_job"] = stats["seconds_jobs"] / max(stats["jobs"], 1)
    stats["workers"] = workers
    return stats


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_data(data_dir: Path, kind: str) -> tuple[list[dict], dict]:
    recs, jobs = [], {}
    for fp in sorted(data_dir.glob(f"{kind}_*.jsonl")):
        seed = None
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("type") == "job":
                    seed = d["seed"]
                    jobs[seed] = d
                else:
                    d["job"] = seed
                    recs.append(d)
    return recs, jobs


def _ppm_proc(h: dict) -> float:
    if h.get("px_per_mm_proc"):
        return float(h["px_per_mm_proc"])
    dp = AppConfig().detection
    full = h["width"] / h["fov_mm"] if h.get("calibrated") else h["width"] / AppConfig().physics.default_fov_mm
    pw = h.get("process_width", dp.process_width)
    scale = pw / h["width"] if pw and pw < h["width"] else 1.0
    return full * scale


def recompute_rules(recs: list[dict], jobs: dict, cp: ClassificationParams, store: bool = True) -> list[tuple]:
    """Refait la décision des règles avec le code ACTUEL (classify_rules) à partir des mesures
    enregistrées : références recalculées sur toutes les traces valides de l'enregistrement
    (comme reclassify_all). Renvoie [(classe, confiance, scores, court_et_droit)] dans l'ordre de recs."""
    from wilsontrack.detection.classifier import classify_rules, compute_references
    by_job = defaultdict(list)
    for i, r in enumerate(recs):
        by_job[r["job"]].append(i)
    out = [None] * len(recs)
    for seed, idxs in by_job.items():
        h = jobs.get(seed, {})
        idxs = sorted(idxs, key=lambda i: recs[i].get("idx", 0))
        feats = []
        for i in idxs:
            f = {k: (v if v is not None else (float("inf") if k == "radius_mm" else float("nan")))
                 for k, v in recs[i]["f"].items()}
            feats.append(f)
        refs = compute_references(feats, cp.ref_percentile)
        ppm = _ppm_proc(h) if h else 1.0
        for i, f in zip(idxs, feats):
            cls, conf, scores, _reason = classify_rules(f, refs, cp, bool(recs[i]["calibrated"]), ppm)
            ss = bool(f.get("ev_short_straight", 0.0))
            out[i] = (cls, float(conf), {k: float(v) for k, v in scores.items()}, ss)
            if store:
                recs[i]["rules_cls"], recs[i]["rules_conf"] = cls, _clean(conf)
                recs[i]["rules_scores"] = {k: _clean(v) for k, v in scores.items()}
                recs[i]["rules_ss"] = ss
                recs[i]["f"] = {k: _clean(v) for k, v in f.items() if isinstance(v, (int, float, np.integer, np.floating))}
    return out


def vectorize(recs: list[dict], names: list[str] | None = None) -> np.ndarray:
    names = names or FEATURES
    return np.array([RM.feature_vector(r["f"], names, r["duration_s"], r["touches_border"], r["calibrated"])
                     for r in recs], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Mesures
# --------------------------------------------------------------------------- #
def class_weights_for_prior(y: list[str], prior: dict) -> dict:
    """Poids par classe w_c = π_réf(c) / π_données(c) (pondération d'importance)."""
    n = len(y)
    cnt = Counter(y)
    return {c: (prior.get(c, 0.0) / (cnt[c] / n)) if cnt.get(c) else 0.0 for c in CLASSES}


def compute_metrics(y_true: list[str], y_pred: list[str], weights=None, y_argmax: list[str] | None = None) -> dict:
    """Mesures avec abstention (« incertain ») ; ``weights`` : pondération par trace (priors réalistes)."""
    labels_true = [c for c in CLASSES if c in set(y_true)] or list(CLASSES)
    labels_pred = list(CLASSES) + ["incertain"]
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, float)
    M = np.zeros((len(labels_true), len(labels_pred)))
    Mi = np.zeros((len(labels_true), len(labels_pred)), int)
    for t, p, wi in zip(y_true, y_pred, w):
        if t not in labels_true:
            continue
        j = labels_pred.index(p if p in labels_pred else "incertain")
        M[labels_true.index(t), j] += wi
        Mi[labels_true.index(t), j] += 1
    tot = M.sum()
    conf_mask = np.array([lp != "incertain" for lp in labels_pred])
    n_conf = M[:, conf_mask].sum()
    correct = sum(M[i, labels_pred.index(c)] for i, c in enumerate(labels_true))
    per = {}
    rec_list, recc_list = [], []
    for i, c in enumerate(labels_true):
        nt = M[i].sum()
        nconf_c = M[i, conf_mask].sum()
        j = labels_pred.index(c)
        npred = M[:, j].sum()
        prec = M[i, j] / npred if npred > 0 else None
        rec = M[i, j] / nt if nt > 0 else None
        recc = M[i, j] / nconf_c if nconf_c > 0 else None
        per[c] = {"precision": prec, "recall": rec, "recall_confident": recc,
                  "coverage": nconf_c / nt if nt > 0 else None, "n": int(Mi[i].sum()),
                  "n_predicted": int(Mi[:, j].sum())}
        if rec is not None:
            rec_list.append(rec)
        if recc is not None:
            recc_list.append(recc)
    out = {"n": int(Mi.sum()), "coverage": float(n_conf / tot) if tot else None,
           "accuracy_confident": float(correct / n_conf) if n_conf else None,
           "accuracy": float(correct / tot) if tot else None,
           "balanced_accuracy": float(np.mean(rec_list)) if rec_list else None,
           "balanced_accuracy_confident": float(np.mean(recc_list)) if recc_list else None,
           "per_class": per,
           "confusion": {"labels_true": labels_true, "labels_pred": labels_pred, "matrix": Mi.tolist()}}
    if y_argmax is not None:
        ok = [(t == p) for t, p in zip(y_true, y_argmax) if t in labels_true]
        wa = [wi for t, wi in zip(y_true, w) if t in labels_true]
        rec_a = []
        for c in labels_true:
            idx = [k for k, t in enumerate(y_true) if t == c]
            if idx:
                rec_a.append(sum(w[k] * (y_argmax[k] == c) for k in idx) / sum(w[k] for k in idx))
        out["balanced_accuracy_argmax"] = float(np.mean(rec_a)) if rec_a else None
        out["accuracy_argmax"] = float(np.dot(ok, wa) / max(sum(wa), 1e-12)) if wa else None
    # Famille : fine (β, μ) / lourde (α, p), sur les particules
    fam = {"alpha": "lourde", "proton": "lourde", "beta": "fine", "muon": "fine"}
    fc = [(fam[t] == fam[p], wi) for t, p, wi in zip(y_true, y_pred, w) if t in fam and p in fam]
    out["famille_accuracy_confident"] = (float(sum(a * b for a, b in fc) / max(sum(b for _a, b in fc), 1e-12))
                                         if fc else None)
    if y_argmax is not None:
        fa = [(fam[t] == fam[p], wi) for t, p, wi in zip(y_true, y_argmax, w) if t in fam and p in fam]
        out["famille_accuracy_argmax"] = (float(sum(a * b for a, b in fa) / max(sum(b for _a, b in fa), 1e-12))
                                          if fa else None)
    return out


def reliability(p_top: np.ndarray, correct: np.ndarray, weights=None, n_bins: int = 10) -> dict:
    """Fiabilité : probabilité moyenne prédite vs exactitude observée par tranche ; ECE."""
    w = np.ones(len(p_top)) if weights is None else np.asarray(weights, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, tot = [], 0.0, float(w.sum())
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p_top >= a) & ((p_top < b) if b < 1.0 else (p_top <= b))
        if not m.any():
            continue
        ww = w[m]
        mp = float(np.average(p_top[m], weights=ww))
        acc = float(np.average(correct[m].astype(float), weights=ww))
        bins.append({"from": round(float(a), 2), "to": round(float(b), 2), "n": int(m.sum()),
                     "mean_predicted": round(mp, 4), "observed_accuracy": round(acc, 4)})
        ece += float(ww.sum()) / tot * abs(mp - acc)
    return {"ece": round(ece, 4), "bins": bins}


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
def make_model(max_iter: int, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    # Variante retenue sur la VALIDATION (log-loss calibrée la plus basse parmi : 31 feuilles, 63 feuilles,
    # 15 feuilles + L2 = 3, min_samples_leaf 60, ensachage de 5 modèles ; écarts de couverture < 1 point).
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=25,
        l2_regularization=3.0, early_stopping=True, validation_fraction=0.12,
        n_iter_no_change=40, random_state=seed)


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    cnt = Counter(y.tolist())
    k = len(cnt)
    return np.array([len(y) / (k * cnt[c]) for c in y])


def _proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    cls = [str(c) for c in model.classes_]
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(CLASSES):
        if c in cls:
            out[:, j] = p[:, cls.index(c)]
    return out


def _wlogloss(P: np.ndarray, y_idx: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(P[np.arange(len(P)), y_idx], 1e-6, 1.0)
    return float(-(w * np.log(p)).sum() / w.sum())


def rule_matrix(recs: list[dict]) -> np.ndarray:
    return np.array([[RM.rules_probabilities(r["rules_scores"])[c] for c in PARTICLES] for r in recs])


def _hybrid_raw(Pr: np.ndarray, Pm: np.ndarray, w: float) -> np.ndarray:
    """Vectorisé : même combinaison que reference_model.blend (5 classes)."""
    out = np.zeros_like(Pm)
    art = Pm[:, CLASSES.index("artefact")]
    for j, c in enumerate(PARTICLES):
        out[:, CLASSES.index(c)] = (1 - w) * (1 - art) * Pr[:, j] + w * Pm[:, CLASSES.index(c)]
    out[:, CLASSES.index("artefact")] = art
    return out / out.sum(1, keepdims=True)


def decisions(P: np.ndarray, threshold: float, ss: list[bool] | None = None,
              class_thr: dict | None = None) -> tuple[list[str], list[str], np.ndarray]:
    """Classe (ou « incertain ») selon le seuil ; ss = ambiguïté « courte et droite » des règles."""
    top = P.argmax(1)
    ptop = P[np.arange(len(P)), top]
    preds, argm = [], []
    for i, (k, pt) in enumerate(zip(top, ptop)):
        c = CLASSES[int(k)]
        argm.append(c)
        thr = threshold
        if class_thr and c in class_thr:
            thr = max(thr, class_thr[c])
        if ss is not None and ss[i] and c in ("muon", "beta"):
            thr = max(thr, RM.ML_OVERRIDE)
        preds.append(c if pt >= thr else "incertain")
    return preds, argm, ptop


def wilson_lower(p: float, n: float, z: float = 1.2816) -> float:
    """Borne inférieure de Wilson (unilatérale à 90 % pour z = 1,2816) d'une proportion p sur n essais."""
    if n <= 0:
        return 0.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return c - h


def choose_class_thresholds(H: np.ndarray, y: list[str], w: np.ndarray, ss: list[bool], base: float,
                            target: float = 0.85) -> dict:
    """Seuil par classe (≥ base) : le plus petit tel que la borne inférieure de Wilson de la précision
    pondérée (taille effective Kish) atteigne ``target`` ; 1,01 = classe jamais affirmée."""
    y = np.asarray(y)
    w = np.asarray(w, float)
    top = H.argmax(1)
    ptop = H[np.arange(len(H)), top]
    ssa = np.asarray(ss, bool)
    grid = [base] + [round(base + 0.01 * k, 3) for k in range(1, 15)] + [0.995]
    out = {}
    for j, c in enumerate(CLASSES):
        sel = top == j
        floor = np.where(ssa & (c in ("muon", "beta")), RM.ML_OVERRIDE, 0.0)
        chosen = None
        for t in grid:
            conf = sel & (ptop >= np.maximum(t, floor))
            if not conf.any():
                break                         # aucune preuve de précision sur la validation : jamais affirmée
            wc = w[conf]
            p_hat = float((wc * (y[conf] == c)).sum() / wc.sum())
            n_eff = float(wc.sum() ** 2 / max((wc ** 2).sum(), 1e-12))
            if wilson_lower(p_hat, n_eff) >= target:
                chosen = t
                break
        if chosen is None:
            out[c] = 1.01
        elif chosen > base:
            out[c] = chosen
    return out


def train(data_dir: Path, out_path: Path, max_iter: int, gen_stats: dict | None, threads: int,
          pilot_path: Path | None = None, threshold: float = 0.85, write: bool = True) -> dict:
    import joblib
    import sklearn
    from sklearn.model_selection import GroupShuffleSplit
    from threadpoolctl import threadpool_limits

    recs_all, jobs = load_data(data_dir, "main")
    stress_all, stress_jobs = load_data(data_dir, "stress")
    cp = ClassificationParams()
    codes = Counter(h.get("code", "?") for h in list(jobs.values()) + list(stress_jobs.values()))
    if len(codes) > 1 or (codes and _code_fingerprint() not in codes):
        print(f"  Attention : données produites par {len(codes)} version(s) du code de détection/simulation "
              f"{dict(codes)} (version actuelle : {_code_fingerprint()}).", flush=True)
    # Règles physiques recalculées avec le code et les paramètres ACTUELS ; règles « v1 » (paramètres d'avant)
    v1 = ClassificationParams(alpha_max_length_mm=75.0, muon_min_length_mm=45.0)
    for name in ("muon_max_rms_mm",):
        if hasattr(v1, name):
            setattr(v1, name, 0.0)
    rules_v1 = recompute_rules(recs_all, jobs, v1, store=False) + recompute_rules(stress_all, stress_jobs, v1, store=False)
    recompute_rules(recs_all, jobs, cp)
    recompute_rules(stress_all, stress_jobs, cp)
    for r, rv in zip(recs_all + stress_all, rules_v1):
        r["rules_v1_cls"] = rv[0]
    recs = [r for r in recs_all if r["label"] in CLASSES]
    stress = [r for r in stress_all if r["label"] in CLASSES]
    if len(recs) < 200 or len({r["job"] for r in recs}) < 10:
        raise SystemExit(f"Trop peu de données dans {data_dir} ({len(recs)} traces) : lancez d'abord la génération.")
    X = vectorize(recs)
    y = np.array([r["label"] for r in recs])
    groups = np.array([r["job"] for r in recs])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=0)
    tr, rest = next(gss.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=1)
    va_r, te_r = next(gss2.split(X[rest], y[rest], groups[rest]))
    va, te = rest[va_r], rest[te_r]
    counts = {c: int((y == c).sum()) for c in CLASSES}
    print(f"Entraînement : {len(recs)} traces étiquetées ({len(jobs)} enregistrements) → {len(tr)} entraînement / "
          f"{len(va)} validation / {len(te)} test ; {len(stress)} traces « conditions extrêmes » "
          f"({len(stress_jobs)} enregistrements).", flush=True)
    print("  Répartition : " + ", ".join(f"{c} {n}" for c, n in counts.items()), flush=True)
    # Priors de référence : fréquences simulées, protons ramenés à 1 % des particules (PDG)
    freq = {c: counts[c] / len(y) for c in CLASSES}
    part_tot = sum(freq[c] for c in PARTICLES)
    prior = dict(freq)
    others = part_tot - freq["proton"]
    prior["proton"] = PROTON_PRIOR_FRAC * part_tot
    for c in ("alpha", "beta", "muon"):
        prior[c] = freq[c] / others * (part_tot - prior["proton"]) if others > 0 else freq[c]
    s = sum(prior.values())
    prior = {c: v / s for c, v in prior.items()}

    def wts(ys) -> np.ndarray:
        cw = class_weights_for_prior(list(ys), prior)
        return np.array([cw[c] for c in ys])

    t0 = time.perf_counter()
    with threadpool_limits(limits=threads):
        model = make_model(max_iter).fit(X[tr], y[tr], sample_weight=_balanced_weights(y[tr]))
        P_va_bal = _proba(model, X[va])
    # Calibration (validation) : décalage de priors (équilibré → π_réf), puis température ou isotonique
    pri = np.array([prior[c] for c in CLASSES])
    y_va = y[va]
    yi_va = np.array([CLASSES.index(c) for c in y_va])
    w_va = wts(y_va)
    P_va_shift = RM.apply_prior(P_va_bal, pri)
    cal_ml = RM.fit_calibrator(P_va_shift, yi_va, w_va, groups[va])
    P_va = RM.apply_calibrator(P_va_shift, cal_ml)
    recs_va = [recs[i] for i in va]
    Pr_va = rule_matrix(recs_va)
    ss_va = [bool(r.get("rules_ss")) for r in recs_va]
    # Poids hybride + température de l'hybride + seuils par classe, choisis ENSEMBLE sur la validation :
    # pour chaque poids w, seuils par classe relevés jusqu'à ce que la borne inférieure (Wilson, 80 %)
    # de la précision de chaque classe atteigne 0,85 ; on garde le w de plus grande couverture (à
    # 0,5 point près, le plus FAIBLE : règles physiques plus présentes, plus robuste hors simulation).
    grid = []
    for w in W_GRID:
        H_raw = _hybrid_raw(Pr_va, P_va, w)
        T_h_w = RM.fit_temperature(H_raw, yi_va, w_va)
        H = RM.apply_temperature(H_raw, T_h_w)
        thr_w = choose_class_thresholds(H, list(y_va), w_va, ss_va, threshold)
        preds, argm, ptop = decisions(H, threshold, ss_va, thr_w)
        m = compute_metrics(list(y_va), preds, w_va, argm)
        precs = [m["per_class"][c]["precision"] for c in CLASSES if m["per_class"].get(c, {}).get("precision") is not None]
        grid.append({"w_ml": w, "T_h": round(T_h_w, 4), "coverage": m["coverage"],
                     "min_precision": min(precs) if precs else None, "class_thresholds": thr_w,
                     "accuracy_confident": m["accuracy_confident"], "balanced_accuracy": m["balanced_accuracy"],
                     "logloss": _wlogloss(H, yi_va, w_va)})
    top_cov = max(g["coverage"] or 0.0 for g in grid)
    best = min([g for g in grid if (g["coverage"] or 0.0) >= top_cov - 0.005], key=lambda g: g["w_ml"])
    w_best, T_h, class_thr = best["w_ml"], best["T_h"], dict(best["class_thresholds"])
    # Température des règles seules (repli sans IA)
    Pr5_va = np.column_stack([Pr_va, np.zeros(len(Pr_va))])
    T_r = RM.fit_temperature(np.clip(Pr5_va, 1e-6, 1), yi_va, w_va)
    calib = {"prior_ref": {c: round(float(prior[c]), 6) for c in CLASSES}, "ml": cal_ml,
             "hybrid": {"w": w_best, "T": T_h}, "rules": {"T": T_r}, "class_thresholds": class_thr,
             "trained_balanced": True}

    def predict_all(rs, Xm):
        P = RM.apply_calibrator(RM.apply_prior(_proba(model, Xm), pri), cal_ml)
        Pr = rule_matrix(rs)
        H = RM.apply_temperature(_hybrid_raw(Pr, P, w_best), T_h)
        return P, Pr, H

    def evaluate_split(rs, Xm, pilot) -> dict:
        ys = [r["label"] for r in rs]
        ww = wts(ys)
        P, Pr, H = predict_all(rs, Xm)
        ss = [bool(r.get("rules_ss")) for r in rs]
        pv2, av2, ptop = decisions(H, threshold, ss, class_thr)
        pml, aml, _ = decisions(P, threshold, None, class_thr)
        out = {"v2": compute_metrics(ys, pv2, ww, av2), "ml": compute_metrics(ys, pml, ww, aml),
               "rules": compute_metrics(ys, [r["rules_cls"] for r in rs], ww),
               "rules_v1": compute_metrics(ys, [r["rules_v1_cls"] for r in rs], ww),
               "v2_raw": compute_metrics(ys, pv2, None, av2)}
        Pr5 = RM.apply_temperature(np.column_stack([Pr, np.zeros(len(Pr))]).clip(1e-6, 1), T_r)
        prs, ars, _ = decisions(Pr5, threshold, ss)
        out["rules_seuil"] = compute_metrics(ys, prs, ww, ars)
        corr = np.array([a == t for a, t in zip(av2, ys)])
        out["reliability_v2"] = reliability(ptop, corr, ww)
        # Famille (fine β/μ, lourde α/p) d'après les probabilités hybrides, sur les particules
        ci = {c: CLASSES.index(c) for c in CLASSES}
        p_thin = H[:, ci["beta"]] + H[:, ci["muon"]]
        p_heavy = H[:, ci["alpha"]] + H[:, ci["proton"]]
        fam_true = {"alpha": "lourde", "proton": "lourde", "beta": "fine", "muon": "fine"}
        mk = np.array([t in fam_true for t in ys])
        if mk.any():
            fp = np.where(p_heavy > p_thin, "lourde", "fine")[mk]
            fc = np.maximum(p_thin, p_heavy)[mk]
            ft = np.array([fam_true[t] for t in np.asarray(ys)[mk]])
            wf = ww[mk]
            okc = fc >= threshold
            out["famille"] = {"n": int(mk.sum()),
                              "accuracy_all": float(np.average(fp == ft, weights=wf)),
                              "coverage": float(np.average(okc, weights=wf)),
                              "accuracy_confident": float(np.average((fp == ft)[okc], weights=wf[okc])) if okc.any() else None}
        top_ml = P.max(1)
        out["reliability_ml"] = reliability(top_ml, np.array([a == t for a, t in zip(aml, ys)]), ww)
        cov_curve = []
        for thr in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            pp, _a, _p = decisions(H, thr, ss, class_thr if thr >= threshold else None)
            mm = compute_metrics(ys, pp, ww)
            cov_curve.append({"threshold": thr, "coverage": mm["coverage"], "accuracy_confident": mm["accuracy_confident"]})
        out["coverage_curve_v2"] = cov_curve
        # Précision / couverture de chaque classe selon le seuil (sans seuils par classe) : montre où
        # se situe la limite physique (ex. : muon contre électron cosmique)
        curve = {}
        for thr in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            pp, _a, _p = decisions(H, thr, ss, None)
            mm = compute_metrics(ys, pp, ww)
            curve[str(thr)] = {c: {"precision": d["precision"], "coverage": d["coverage"],
                                   "n_predicted": d["n_predicted"]} for c, d in mm["per_class"].items()}
        out["per_class_curve_v2"] = curve
        if pilot is not None:
            out["pilot"] = pilot_metrics(pilot, rs, ys, ww, cp)
        return out

    pilot = None
    if pilot_path is not None and Path(pilot_path).exists():
        pilot = RM.ReferenceModel.load(pilot_path)
    with threadpool_limits(limits=threads):
        recs_te = [recs[i] for i in te]
        test_metrics = evaluate_split(recs_te, X[te], pilot)
        stress_metrics = evaluate_split(stress, vectorize(stress), pilot) if stress else {}
        val_metrics = evaluate_split(recs_va, X[va], None)
        sub = {}
        for flag, name in ((True, "test_calibre"), (False, "test_non_calibre")):
            idx = [i for i in te if bool(recs[i]["calibrated"]) == flag]
            if idx:
                sub[name] = evaluate_split([recs[i] for i in idx], X[idx], None)["v2"]
        by_res = {}
        for i in te:
            h = jobs.get(recs[i]["job"], {})
            by_res.setdefault(f"{h.get('width')}x{h.get('height')}", []).append(i)
        for k, idx in sorted(by_res.items()):
            sub[f"test_{k}"] = evaluate_split([recs[i] for i in idx], X[idx], None)["v2"]
        # Famille (fine / lourde) par la règle physique (ev_heavy) sur le test
        fam = {"alpha": "lourde", "proton": "lourde", "beta": "fine", "muon": "fine"}
        fr = [(RM.family(r["f"]) == fam[r["label"]]) for r in recs_te if r["label"] in fam]
        famille_rules = float(np.mean(fr)) if fr else None
    t_train = time.perf_counter() - t0
    imp = _importances(model, X[te], y[te]) if len(te) >= 100 else {}
    heads = list(jobs.values()) + list(stress_jobs.values())
    cpu_s = sum(float(h.get("seconds", 0.0)) for h in heads)
    n_lab = sum(int(h.get("n_labelled", 0)) for h in heads)
    det_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for h in jobs.values():
        for k, v in (h.get("detection_by_kind") or {}).items():
            det_kind[k][0] += v[0]
            det_kind[k][1] += v[1]
    gen_summary = {
        "n_jobs": len(heads), "labelled_per_job": round(n_lab / max(1, len(heads)), 2),
        "seconds_per_job_per_process": round(cpu_s / max(1, len(heads)), 2),
        "labelled_per_process_minute": round(n_lab / max(cpu_s, 1e-9) * 60.0, 2),
        "simulated_minutes": round(sum(float(h.get("duration_s", 0.0)) for h in heads) / 60.0, 1),
        "labels_how": dict(Counter(r["how"] for r in recs_all + stress_all)),
        "detection_efficiency_by_kind": {k: {"detected": v[0], "total": v[1],
                                             "fraction": round(v[0] / v[1], 4) if v[1] else None}
                                         for k, v in sorted(det_kind.items())},
    }
    v2t = test_metrics["v2"]
    v2s = stress_metrics.get("v2", {}) if stress_metrics else {}

    def contract_m(m: dict) -> dict:
        if not m:
            return {}
        return {"n": m["n"], "coverage": m["coverage"], "accuracy_confident": m["accuracy_confident"],
                "balanced_accuracy": m["balanced_accuracy"],
                "balanced_accuracy_argmax": m.get("balanced_accuracy_argmax"),
                "famille_accuracy": m.get("famille_accuracy_confident"),
                "per_class": {c: {"precision": d["precision"], "recall": d["recall"], "n": d["n"],
                                  "coverage": d["coverage"]} for c, d in m["per_class"].items()}}
    test_c, stress_c = contract_m(v2t), contract_m(v2s)
    if test_c:
        test_c["famille"] = test_metrics.get("famille")
        test_c["ece"] = test_metrics["reliability_v2"]["ece"]
    if stress_c:
        stress_c["famille"] = stress_metrics.get("famille")
        stress_c["ece"] = stress_metrics["reliability_v2"]["ece"]
    metrics = {
        "test": test_c, "stress": stress_c, "validation": contract_m(val_metrics["v2"]),
        "details": {"test": test_metrics, "stress": stress_metrics, **sub},
        "w_ml_grid_validation": grid, "w_ml": w_best, "threshold": threshold, "class_thresholds": class_thr,
        "class_counts": counts, "prior_ref": calib["prior_ref"], "famille_rules_test": famille_rules,
        "n_iter": int(getattr(model, "n_iter_", 0)), "train_seconds": round(t_train, 1),
        "feature_importance_test": imp, "code_versions": dict(codes),
    }
    # Clés « à plat » (compatibilité : interface et rapport 1.x). Même sens qu'en 1.x : exactitude de la
    # décision SANS abstention (classe la plus probable) ; les mesures avec abstention sont dans test / stress.
    for split, mm in (("test", test_metrics), ("stress", stress_metrics)):
        for k_old, k_new in (("rules", "rules"), ("ml", "ml"), ("hybrid", "v2")):
            d = (mm or {}).get(k_new) or {}
            ba = d.get("balanced_accuracy_argmax")
            metrics[f"{split}_{k_old}_balanced_accuracy"] = ba if ba is not None else d.get("balanced_accuracy")
            ac = d.get("accuracy_argmax")
            metrics[f"{split}_{k_old}_accuracy"] = ac if ac is not None else d.get("accuracy")
    conf_test = v2t["confusion"]
    payload = {"format": 2, "model": model, "features": FEATURES, "classes": CLASSES, "w_ml": w_best,
               "threshold": threshold, "calibration": calib, "metrics": metrics, "confusion_test": conf_test,
               "n_train": int(len(tr)), "n_tracks_total": int(len(recs) + len(stress)),
               "n_jobs": int(len(jobs) + len(stress_jobs)), "sklearn_version": sklearn.__version__,
               "created": datetime.now().isoformat(timespec="seconds"), "generator": GENERATOR,
               "model_type": type(model).__name__,
               "generation": {**gen_summary, **{k: v for k, v in (gen_stats or {}).items()
                                                if k in ("wall_s", "tracks_per_min", "workers")}}}
    # Temps d'inférence par trace (prédicteur rapide sans OpenMP, comme dans l'application)
    try:
        mo = RM.ReferenceModel(payload)
        mo._enable_fast_path()
        sample = recs_te[:200] or recs[:200]
        t1 = time.perf_counter()
        for r in sample:
            mo.predict_proba(r["f"], r["duration_s"], r["touches_border"], r["calibrated"])
        metrics["inference_ms_per_track"] = round(1000 * (time.perf_counter() - t1) / max(1, len(sample)), 3)
        metrics["fast_predictor"] = mo._fast is not None
    except Exception:
        metrics["inference_ms_per_track"] = None
    info = {k: v for k, v in payload.items() if k != "model"}
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, out_path, compress=3)
        size_mb = out_path.stat().st_size / 1e6
        info["model_size_mb"] = round(size_mb, 3)
        info["description"] = ("IA de référence v2 de WilsonTrack : gradient boosting (5 classes dont « artefact ») "
                               "entraîné sur des traces simulées par un simulateur physique (tables NIST, Highland, "
                               "cos²θ) passées dans le vrai détecteur ; probabilités calibrées ; décision hybride "
                               "règles + IA avec abstention (« incertain ») sous le seuil.")
        out_path.with_suffix(".json").write_text(json.dumps(_jsonable(info), indent=1, ensure_ascii=False),
                                                 encoding="utf-8")
    print_summary(metrics, w_best, T_h, class_thr)
    if write:
        print(f"Modèle enregistré : {out_path} ({_fmt(out_path.stat().st_size / 1e6, 2)} Mo)")
    info["_model"] = model
    return info


def pilot_metrics(pilot, rs, ys, ww, cp) -> dict:
    """Pilote v1 (4 classes, hybride w = 0,7, seuil 0,30) évalué sur les mêmes traces."""
    Xp = np.array([RM.feature_vector(r["f"], pilot.features, r["duration_s"], r["touches_border"], r["calibrated"])
                   for r in rs], dtype=np.float64)
    P = pilot.predict_proba_many(Xp)
    preds = []
    for r, p in zip(rs, P):
        cls, _c, _b, _a, _cap = RM.hybrid_classify(r["rules_scores"], dict(zip(RM.CLASSES, p)), pilot.w_ml,
                                                   cp.min_confidence, r["f"].get("length_mm") or 0.0,
                                                   45.0)
        preds.append(cls)
    return compute_metrics(ys, preds, ww)


def print_summary(metrics: dict, w: float, T_h: float, class_thr: dict) -> None:
    def pct(x):
        return "—" if x is None else _fmt(100 * x) + " %"
    print(f"\nPoids de l'IA dans l'hybride (validation) : w = {_fmt(w)} ; température de l'hybride {_fmt(T_h, 3)} ; "
          f"seuils par classe relevés : {class_thr or 'aucun'}")
    for split in ("test", "stress"):
        d = metrics["details"].get(split) or {}
        if not d:
            continue
        print(f"  {split.upper()} (pondération réaliste des protons) :")
        for k in ("rules_v1", "rules", "rules_seuil", "pilot", "ml", "v2"):
            m = d.get(k)
            if not m:
                continue
            pcs = ", ".join(f"{c} {pct(v['precision'])}/{pct(v['recall'])}" for c, v in m["per_class"].items())
            print(f"    {k:<11s} couverture {pct(m['coverage'])}, exactitude (sûres) {pct(m['accuracy_confident'])}, "
                  f"exactitude équilibrée {pct(m['balanced_accuracy'])} | précision/rappel : {pcs}")
        print(f"    ECE v2 {d['reliability_v2']['ece']}, ECE IA {d['reliability_ml']['ece']}")


def _importances(model, X, y) -> dict:
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, n_repeats=3, random_state=0, scoring="balanced_accuracy", n_jobs=1)
        order = np.argsort(-r.importances_mean)
        return {FEATURES[i]: round(float(r.importances_mean[i]), 4) for i in order}
    except Exception:
        return {}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return round(f, 6) if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


# --------------------------------------------------------------------------- #
# Vignettes d'exemple (centre d'apprentissage de l'interface)
# --------------------------------------------------------------------------- #
def export_examples(out_dir: Path, n_per_class: int = 8, seed0: int = 5_000_000, max_jobs: int = 14) -> dict:
    """Vignettes (≤ 160 px, PNG) de traces SIMULÉES (vérité connue) que le modèle livré classe en tête dans
    la bonne classe ; on garde les plus sûres. index.json indique la confiance et si elle atteint le seuil
    (les muons et les protons n'atteignent presque jamais 0,85 : ambiguïté physique, affichée honnêtement)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    RM.clear_cache()
    model = RM.ReferenceModel.load_default()
    cand = {c: [] for c in CLASSES}
    cp = ClassificationParams()
    for k in range(max_jobs):
        if all(len(v) >= 3 * n_per_class for v in cand.values()):
            break
        p = draw_params(seed0 + k)
        p.update({"width": 1280, "height": 720, "fps": 30.0, "duration_s": 20.0, "calibrated": True, "roi": [],
                  "jpeg_quality": 0, "noise_sigma": min(p["noise_sigma"], 3.5), "brightness": max(p["brightness"], 0.9),
                  "process_width": 960, "deff_mm": max(p["deff_mm"], 12.0),
                  "sensitive_depth_mm": max(p["sensitive_depth_mm"], 12.0)})
        p["rates_per_min"].update({"proton": 15.0, "alpha": max(p["rates_per_min"]["alpha"], 15.0),
                                   "muon": 60.0, "beta": 40.0})
        p["artefact_rates_per_min"] = {kk: 3.0 * v for kk, v in SY._default_artefacts().items()}
        sc, cfg, sim, det, gt = simulate_and_detect(p)
        for d in label_detections(det.events, gt):
            e, lab = d["event"], d["label"]
            if lab not in cand or e.thumb is None or d["how"] == "dup":
                continue
            if model is not None:
                dec = RM.decide(e.scores, model.predict_proba(e.features, e.duration_s, e.touches_border, True),
                                threshold=cp.ml_threshold, model=model)
                top = dec.competitors[0][0]
                if top != lab:
                    continue
                conf, sure = dec.confidence, not dec.abstained
            else:
                conf, sure = None, False
            th = e.thumb
            f = min(1.0, 160.0 / max(th.shape[:2]))
            if f < 1.0:
                th = cv2.resize(th, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            if min(th.shape[:2]) < 12:
                continue
            cand[lab].append((th, e, d, conf, sure))
    for old in out_dir.glob("*.png"):
        old.unlink()
    index = []
    got = {}
    for c, items in cand.items():
        items = sorted(items, key=lambda t: -(t[3] or 0.0))[:n_per_class]
        got[c] = len(items)
        for k, (th, e, d, conf, sure) in enumerate(items):
            name = f"{c}_{k}.png"
            cv2.imwrite(str(out_dir / name), th)
            g = d["gt"] or {}
            index.append({"fichier": name, "classe": c, "type": g.get("kind") or ("artefact" if c == "artefact" else c),
                          "longueur_mm": round(float(e.features.get("length_mm", 0.0)), 1),
                          "duree_s": round(float(e.duration_s), 2),
                          "confiance": round(float(conf), 3) if conf is not None else None,
                          "au_dessus_du_seuil": bool(sure), "simulee": True})
    (out_dir / "index.json").write_text(json.dumps(
        {"description": "Exemples SIMULÉS (vérité connue), classés en tête dans la bonne classe par l'IA de "
                        "référence ; « confiance » = probabilité calibrée ; « au_dessus_du_seuil » = classe affirmée "
                        "(sinon la trace serait affichée « incertaine »). Vignettes ≤ 160 px.",
         "exemples": index}, indent=1, ensure_ascii=False), encoding="utf-8")
    return got


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Entraîne l'IA de référence v2 de WilsonTrack sur des traces simulées.")
    ap.add_argument("--jobs", type=int, default=None, help="nombre d'enregistrements simulés (jeu principal)")
    ap.add_argument("--start-seed", type=int, default=1, help="première graine du jeu principal")
    ap.add_argument("--stress-jobs", type=int, default=None,
                    help="nombre d'enregistrements « conditions extrêmes » (défaut : max(12, jobs/8))")
    ap.add_argument("--workers", type=int, default=5, help="processus en parallèle (défaut 5)")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dossier des données générées")
    ap.add_argument("--subdir", default="v2", help="sous-dossier de --data (défaut « v2 »)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="fichier du modèle (.joblib)")
    ap.add_argument("--train-only", action="store_true", help="entraîner sur les données existantes seulement")
    ap.add_argument("--no-train", action="store_true", help="générer les données sans entraîner")
    ap.add_argument("--pilot", action="store_true", help="essai rapide : 36 enregistrements de 20–30 s (+12 extrêmes)")
    ap.add_argument("--max-iter", type=int, default=600, help="itérations maximales du gradient boosting")
    ap.add_argument("--threads", type=int, default=5, help="fils de calcul pour l'entraînement")
    ap.add_argument("--threshold", type=float, default=0.85, help="seuil de confiance (abstention en dessous)")
    ap.add_argument("--compare", type=Path, default=DEFAULT_PILOT, help="modèle pilote v1 à comparer (si présent)")
    ap.add_argument("--examples", action="store_true", help="exporter aussi les vignettes d'exemple après l'entraînement")
    ap.add_argument("--examples-only", action="store_true", help="exporter seulement les vignettes d'exemple")
    a = ap.parse_args(argv)
    if a.examples_only:
        print(export_examples(EXAMPLES_DIR))
        return
    jobs = a.jobs if a.jobs is not None else (36 if a.pilot else 1500)
    n_stress = a.stress_jobs if a.stress_jobs is not None else max(12, jobs // 8)
    data_dir = Path(a.data) / ("pilote_v2" if a.pilot else a.subdir)
    workers = max(1, min(a.workers, os.cpu_count() or 1))
    gen_stats = None
    if not a.train_only:
        seeds = [(a.start_seed + i, "main") for i in range(jobs)]
        seeds += [(STRESS_SEED0 + i, "stress") for i in range(n_stress)]
        gen_stats = generate(data_dir, seeds, workers, a.pilot)
        if gen_stats.get("jobs"):
            print(f"Débit : {_fmt(gen_stats['tracks_per_min'], 0)} traces étiquetées/min avec {workers} processus, "
                  f"{_fmt(gen_stats['s_per_job'])} s par enregistrement (par processus).", flush=True)
    if not a.no_train:
        train(data_dir, Path(a.out), a.max_iter, gen_stats, a.threads, a.compare, a.threshold)
        if a.examples:
            print(export_examples(EXAMPLES_DIR))


if __name__ == "__main__":
    mp.freeze_support()
    main()
