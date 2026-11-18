"""Pipeline complet sur des vidéos synthétiques avec vérité terrain.

Vidéo 960×540 (même échelle de traitement que 1280×720 → 960 px), 15 s, taux élevés.
La vidéo est générée une seule fois par module (fixture « module »).

Deux mesures de la qualité de détection :
- ``evaluate()`` de wilsontrack (fenêtre temporelle stricte : apparition détectée entre
  −0.2 s et +0.8 s de l'apparition vraie) — exigence du cahier des charges ;
- un appariement tolérant sur l'instant d'apparition (voir ``match_tracks``), qui isole la
  qualité de DÉTECTION de la datation de l'apparition.

Les tests de non-régression en fin de fichier couvrent les défauts corrigés : trace
comptée plusieurs fois, héritage de l'instant d'une gouttelette voisine, mesures au
« meilleur instant », saut global de luminosité, reproductibilité du générateur.
"""
from __future__ import annotations

import csv
import math

import cv2
import numpy as np
import pytest

from wilsontrack import physics
from wilsontrack.analysis.run import RunResult, analyze_video
from wilsontrack.config import AppConfig
from wilsontrack.detection import TrackDetector
from wilsontrack.detection.events import EXPORT_COLUMNS
from wilsontrack.evaluate import evaluate, format_report, load_gt
from wilsontrack.synth import SynthConfig, SyntheticChamber, generate_video

MAIN_SEED = 3
RATES = {"alpha": 30.0, "beta": 30.0, "muon": 40.0, "proton": 0.0}
THIN = ("beta", "muon")
# Simulateur v2 : scénario « conditions claires » pour tester la détection et le comptage, proche des
# conditions du simulateur 1.x (traces fines visibles ≈ 0,8 s comme les muons 1.x, plus brillantes, couche
# plus épaisse, muons d'au moins 6 mm, pas d'artefacts). Le régime réaliste (durées de Sussman, muons
# surtout courts, artefacts) est mesuré dans test_synth_realism.py et par l'IA de référence.
CLEAR = dict(thin_life_median_s=0.8, thin_life_sigma=0.3, deff_mm=15.0, crossing_min_mm=6.0,
             sensitive_depth_mm=15.0, sat_density_mm2=14.0, brightness=1.2,
             artefact_rates_per_min={}, shower_per_min=0.0, cosmic_e_frac=0.0)


def _generate(path, cfg: SynthConfig, cv_seed: int) -> dict:
    cv2.setRNGSeed(cv_seed)
    return generate_video(path, cfg)


# --------------------------------------------------------------------------- #
# Appariement tolérant (vérité terrain ↔ traces détectées)
# --------------------------------------------------------------------------- #
def _track_mask(g: dict, gt: dict, tol_mm: float = 3.5) -> np.ndarray:
    """Masque de la polyligne vraie dilatée de ``tol_mm``."""
    W, H = gt["config"]["width"], gt["config"]["height"]
    tol = max(2, int(round(tol_mm * gt["px_per_mm"])))
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [np.round(np.array(g["polyline"], np.float32)).astype(np.int32)], False, 255, 1)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1)))


def _frac_on(e, mask: np.ndarray) -> float:
    """Fraction du chemin détecté située dans le masque."""
    H, W = mask.shape
    p = np.round(e.path_full).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    return float((mask[p[ok, 1], p[ok, 0]] > 0).mean()) if ok.any() else 0.0


def match_tracks(events, gt: dict, min_len_mm: float = 5.0, t_before: float = 0.6, t_after: float = 0.8,
                 tol_mm: float = 3.5, min_frac: float = 0.5, warmup_frames: int = 12) -> dict:
    """Associe chaque trace vraie à au plus une trace détectée (et réciproquement).

    - géométrie : ≥ ``min_frac`` des points du chemin détecté à moins de ``tol_mm`` de la
      polyligne vraie (comme evaluate()) ; parmi les candidates, on garde celle qui couvre
      le mieux la trace vraie (un petit fragment ne l'emporte pas sur la trace entière) ;
    - temps : apparition détectée entre ``t_birth − t_before`` et ``t_birth + t_after`` ;
    - traces « comptées » (rappel) : longueur visible ≥ ``min_len_mm`` et visibles au moins
      ~0.2 s pendant l'analyse (hors apprentissage du fond / fin de vidéo) ;
    - les autres traces vraies (très courtes, cachées par l'apprentissage du fond…) sont
      « indifférentes » : les détecter n'est pas un faux positif, les rater n'est pas un raté.
    """
    fps = gt["config"]["fps"]
    n_frames = gt["n_frames"]
    evs = [e for e in events if e.valid and len(e.path_full)]

    def visible_frames(g):
        start = max(g["frame_birth"], warmup_frames)
        end = min(g["frame_birth"] + g["life_frames"], n_frames)
        return end - start

    counted = [g for g in gt["tracks"] if g["length_mm"] >= min_len_mm and visible_frames(g) >= 0.2 * fps]
    counted_ids = {g["id"] for g in counted}
    dont_care = [g for g in gt["tracks"] if g["id"] not in counted_ids]
    used: set[int] = set()

    def best_event(g):
        mask = _track_mask(g, gt, tol_mm)
        best, best_key = None, None
        for k, e in enumerate(evs):
            if k in used or not (g["t_birth"] - t_before <= e.t_first <= g["t_birth"] + t_after):
                continue
            frac = _frac_on(e, mask)
            L_e, L_g = e.features.get("length_mm", 0.0), max(g["length_mm"], 1e-6)
            coverage = frac * min(1.0, L_e / L_g)
            key = (round(coverage, 2), -abs(e.t_first - g["t_birth"]))
            if frac >= min_frac and (best_key is None or key > best_key):
                best, best_key = k, key
        if best is not None:
            used.add(best)
        return evs[best] if best is not None else None

    matches = [(g, best_event(g)) for g in sorted(counted, key=lambda g: -g["length_mm"])]
    for g in dont_care:
        best_event(g)
    n_found = sum(1 for _g, e in matches if e is not None)
    return {"matches": matches, "n_counted": len(counted), "n_det": len(evs),
            "recall": n_found / max(1, len(counted)), "precision": len(used) / max(1, len(evs)),
            "unmatched": [e for k, e in enumerate(evs) if k not in used]}


def _describe(m: dict) -> str:
    lines = [f"rappel {m['recall']:.3f} ({m['n_counted']} traces comptées)  précision {m['precision']:.3f} "
             f"({m['n_det']} détectées)"]
    for g, e in sorted(m["matches"], key=lambda x: x[0]["t_birth"]):
        d = (f"→ {e.final_cls:9s} L={e.features['length_mm']:6.1f} dt={e.t_first - g['t_birth']:+.2f}s"
             if e is not None else "→ RATÉE")
        lines.append(f"  t={g['t_birth']:5.2f} {g['cls']:6s} L={g['length_mm']:6.1f} {d}")
    for e in m["unmatched"]:
        lines.append(f"  faux positif t={e.t_first:5.2f} {e.final_cls} L={e.features['length_mm']:.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Fixtures : vidéo synthétique principale
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synth")
    sc = SynthConfig(width=960, height=540, duration_s=15.0, seed=MAIN_SEED, rates_per_min=dict(RATES), **CLEAR)
    vid = folder / "synth_15s.mp4"
    gt = _generate(vid, sc, 1234)
    return vid, gt, sc


@pytest.fixture(scope="module")
def analysed(synth_video):
    vid, gt, sc = synth_video
    cfg = AppConfig()
    cfg.physics.px_per_mm = gt["px_per_mm"]      # calibration exacte
    cfg.run.condition = "Fond (rayons cosmiques)"
    run, det = analyze_video(str(vid), cfg)
    res = evaluate(run.events, gt)
    tol = match_tracks(run.events, gt, warmup_frames=cfg.detection.bg_warmup_frames)
    print("\n" + format_report(res) + "\n" + _describe(tol))
    return run, det, res, gt, cfg, tol


# --------------------------------------------------------------------------- #
def test_ground_truth_written(synth_video):
    vid, gt, sc = synth_video
    assert vid.exists() and vid.stat().st_size > 10_000
    assert load_gt(vid) == gt
    assert gt["px_per_mm"] == pytest.approx(960 / 250.0)
    assert gt["n_frames"] == int(15.0 * sc.fps)
    assert len(gt["tracks"]) >= 15
    classes = {t["cls"] for t in gt["tracks"]}
    assert {"alpha", "beta", "muon"} <= classes
    for t in gt["tracks"]:
        assert 0 <= t["frame_birth"] < gt["n_frames"]
        assert t["t_birth"] == pytest.approx(t["frame_birth"] / sc.fps)
        assert len(t["polyline"]) >= 2 and t["length_mm"] >= 0.0




def test_detection_recall_precision(analysed):
    """Exigence : rappel ≥ 75 % et précision ≥ 75 % (evaluate() de wilsontrack)."""
    run, det, res, gt, cfg, tol = analysed
    assert res["n_gt"] >= 15
    assert res["recall_total"] >= 0.75, format_report(res)
    assert res["precision"] >= 0.75, format_report(res)


def test_detection_recall_time_tolerant(analysed):
    """Rappel ≥ 80 % avec un appariement tolérant sur l'instant d'apparition (−0.6 s / +0.8 s)."""
    run, det, res, gt, cfg, tol = analysed
    assert tol["n_counted"] >= 15
    assert tol["recall"] >= 0.8, _describe(tol)


def test_detection_precision_time_tolerant(analysed):
    """Précision ≥ 75 % (les détections de traces vraies très courtes ne sont pas des faux positifs)."""
    run, det, res, gt, cfg, tol = analysed
    assert tol["precision"] >= 0.75, _describe(tol)


def test_each_track_counted_once(analysed):
    """Aucune trace vraie (≥ 10 mm) ne doit donner plusieurs traces détectées valides (≥ 5 mm)
    posées sur elle pendant sa durée de vie."""
    run, det, res, gt, cfg, tol = analysed
    fps = gt["config"]["fps"]
    evs = [e for e in run.events if e.valid and len(e.path_full) and e.features["length_mm"] >= 5.0]
    multiple = []
    tracks = [g for g in gt["tracks"] if g["length_mm"] >= 10.0]
    for g in tracks:
        mask = _track_mask(g, gt)
        t0, t1 = g["t_birth"] - 0.6, g["t_birth"] + g["life_frames"] / fps
        on = [e for e in evs if e.t_first <= t1 and e.t_last >= t0 and _frac_on(e, mask) >= 0.8]
        if len(on) >= 2:
            multiple.append((g["cls"], g["length_mm"], [(round(e.t_first, 2), round(e.features["length_mm"], 1))
                                                        for e in on]))
    assert len(multiple) <= 0.1 * len(tracks), multiple


def test_alpha_tracks_found(analysed):
    run, det, res, gt, cfg, tol = analysed
    n_alpha = sum(1 for g, _e in tol["matches"] if g["cls"] == "alpha")
    found = sum(1 for g, e in tol["matches"] if g["cls"] == "alpha" and e is not None)
    assert n_alpha >= 3, _describe(tol)
    assert found / n_alpha >= 0.66, _describe(tol)


def test_thin_tracks_rarely_called_alpha(analysed):
    """Une trace fine (β, μ) prise pour une alpha fausserait le comptage des alphas (radon)."""
    run, det, res, gt, cfg, tol = analysed
    thin = [(g, e) for g, e in tol["matches"] if e is not None and g["cls"] in THIN]
    assert len(thin) >= 8, _describe(tol)
    assert sum(1 for _g, e in thin if e.final_cls == "alpha") <= 0.1 * len(thin), _describe(tol)


def test_classification_accuracy(analysed):
    run, det, res, gt, cfg, tol = analysed
    found = [(g, e) for g, e in tol["matches"] if e is not None]
    alphas = [(g, e) for g, e in found if g["cls"] == "alpha"]
    thin = [(g, e) for g, e in found if g["cls"] in THIN]
    # Alphas majoritairement reconnues (trace épaisse et dense)
    assert sum(1 for _g, e in alphas if e.final_cls == "alpha") >= 0.5 * len(alphas), _describe(tol)
    # Traces fines jamais (ou presque) classées lourdes ; reconnues comme fines
    # (β et μ courts sont physiquement difficiles à distinguer vus de dessus : pas exigé)
    assert sum(1 for _g, e in thin if e.final_cls in ("alpha", "proton")) <= 0.1 * len(thin), _describe(tol)
    assert sum(1 for _g, e in thin if e.final_cls in THIN + ("fine", "incertain")) >= 0.8 * len(thin), _describe(tol)
    # Séparation « lourde / fine » (le critère physique principal)
    heavy_ok = sum(1 for g, e in found if (g["cls"] == "alpha") == (e.final_cls == "alpha"))
    assert heavy_ok >= 0.85 * len(found), _describe(tol)


def test_measured_lengths_match_truth(analysed):
    run, det, res, gt, cfg, tol = analysed
    ratios = {"alpha": [], "beta": [], "muon": []}
    for g, e in tol["matches"]:
        if e is not None and g["cls"] in ratios and g["length_mm"] >= 5.0:
            ratios[g["cls"]].append(e.features["length_mm"] / g["length_mm"])
    for c, r in ratios.items():
        if r:
            # alphas : leurs extrémités épaisses ajoutent ≈ 1 mm à une trace courte
            lo, hi = (0.85, 1.2) if c == "alpha" else (0.85, 1.15)
            assert lo <= float(np.median(r)) <= hi, (c, r)


def test_alpha_energy_from_range(analysed):
    """Énergie des alphas estimée par la longueur (air à −25 °C).

    Vue de dessus, la trace est la projection de la partie contenue dans la couche sensible :
    l'énergie déduite est une énergie MINIMALE (≤ énergie vraie), cohérente avec la longueur
    visible réelle de la trace (relation portée-énergie de Geiger).
    """
    run, det, res, gt, cfg, tol = analysed
    checked = 0
    for g, e in tol["matches"]:
        if g["cls"] != "alpha" or e is None or e.final_cls != "alpha":
            continue
        est = e.features.get("energy_mev")
        if not est:
            continue       # trace qui touche le bord : pas d'estimation
        assert est <= 1.1 * g["energy_mev"], (est, g)
        # relation de Geiger appliquée à la longueur mesurée…
        assert est == pytest.approx(physics.alpha_energy_from_range(
            e.features["length_mm"], cfg.physics.chamber_temp_c, cfg.physics.pressure_kpa), rel=1e-6)
        # … cohérente avec la longueur visible vraie de la trace
        expected = physics.alpha_energy_from_range(g["length_mm"], cfg.physics.chamber_temp_c)
        assert est == pytest.approx(expected, rel=0.2), (est, expected, g)
        checked += 1
    assert checked >= 1, _describe(tol)


def test_run_result_consistency(analysed, synth_video):
    run, det, res, gt, cfg, tol = analysed
    vid, gt, sc = synth_video
    assert run.n_frames == gt["n_frames"]
    assert run.calibrated and run.px_per_mm == pytest.approx(gt["px_per_mm"])
    # Durée analysée = images traitées (hors apprentissage du fond) / fps
    assert run.duration_s == pytest.approx(det.n_frames_processed / sc.fps, rel=1e-6)
    assert det.n_frames_processed + det.n_frames_disturbed + cfg.detection.bg_warmup_frames == run.n_frames
    assert run.roi_area_mm2 == pytest.approx(250.0 * 250.0 * 540 / 960, rel=0.01)
    rows = run.table()
    assert len(rows) == run.total == sum(run.counts().values()) > 0
    ids = [r["id"] for r in rows]       # uniques et croissants (trous = détections rejetées comme artefacts)
    assert ids == sorted(set(ids)) and ids[0] >= 1
    for r in rows:
        assert set(EXPORT_COLUMNS) <= set(r)
        assert r["classe"] in physics.COUNTED_CLASSES
        assert 0.0 <= r["t_debut_s"] <= sc.duration_s
        assert 0.0 <= r["t_analyse_s"] <= run.duration_s + 1e-6
        assert r["longueur_mm"] >= cfg.detection.min_length_mm
        assert 0.0 <= r["confiance"] <= 1.0
        assert r["raison"]


def test_save_analysed_run(analysed, tmp_path):
    run, det, res, gt, cfg, tol = analysed
    folder = run.save(tmp_path)
    thumbs = list((folder / "vignettes").glob("trace_*.png"))
    assert len(thumbs) == len(run.table())
    with open(folder / "traces.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(run.table())
    loaded = RunResult.load(folder)
    assert loaded.counts() == run.counts()


def test_analyze_video_max_frames_and_progress(synth_video):
    vid, gt, sc = synth_video
    seen = []
    frames = []
    run, det = analyze_video(str(vid), AppConfig(), progress=lambda f: seen.append(f),
                             on_frame=lambda fr, r, d: frames.append(r.frame_idx), max_frames=40)
    assert run.n_frames == 40
    assert frames == list(range(40))
    assert seen and all(0 < f <= 1.0 for f in seen) and seen == sorted(seen)
    assert not run.calibrated        # pas de calibration → champ de vision supposé (250 mm)
    assert run.px_per_mm == pytest.approx(960 / 250.0)


def test_analyze_video_can_be_cancelled(synth_video):
    vid, gt, sc = synth_video
    run, det = analyze_video(str(vid), AppConfig(), progress=lambda f: True)
    assert run.n_frames == 10        # interrompu au premier appel de progress (image 10)


def test_analyze_video_missing_file(tmp_path):
    with pytest.raises(IOError):
        analyze_video(str(tmp_path / "absent.mp4"), AppConfig())


# --------------------------------------------------------------------------- #
# Instant d'apparition d'une trace (scène minimale, sans vidéo)
# --------------------------------------------------------------------------- #
_H, _W, _FPS = 180, 320, 30.0
_TRACK_START = 24


def _scene(blob: bool) -> list[np.ndarray]:
    """Fond bruité ; éventuellement une gouttelette isolée (images 20–21) ; puis une trace
    droite de ≈ 40 mm qui apparaît à l'image 24, juste à côté de la gouttelette."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(60):
        img = 20.0 + rng.normal(0, 2.0, (_H, _W)).astype(np.float32)
        if blob and 20 <= i < 22:
            cv2.circle(img, (100, 90), 3, 70.0, -1)
        if _TRACK_START <= i < _TRACK_START + 25:
            cv2.line(img, (106, 90), (260, 60), 80.0, 3, cv2.LINE_AA)
        out.append(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return out


def _run_scene(blob: bool):
    cfg = AppConfig()
    cfg.physics.px_per_mm = 4.0
    det = TrackDetector(cfg, _FPS, (_H, _W))
    for i, fr in enumerate(_scene(blob)):
        det.process(fr, i / _FPS)
    return det.finalize_run()


def test_event_start_time_isolated_track():
    """Témoin : sans gouttelette voisine, l'apparition est datée à l'image près."""
    (ev,) = _run_scene(blob=False)
    assert ev.t_first == pytest.approx(_TRACK_START / _FPS, abs=1.01 / _FPS)
    assert ev.features["length_mm"] == pytest.approx(math.hypot(154, 30) / 4.0, rel=0.1)


def test_event_start_not_inherited_from_noise_blob():
    evs = _run_scene(blob=True)
    assert len(evs) == 1
    (ev,) = evs
    assert ev.t_first == pytest.approx(_TRACK_START / _FPS, abs=1.01 / _FPS)


# --------------------------------------------------------------------------- #
# Perturbation (lumière allumée pendant ~0.7 s)
# --------------------------------------------------------------------------- #
DIST_AT_S = 5.0


@pytest.fixture(scope="module")
def disturbed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synth_dist")
    sc = SynthConfig(width=640, height=360, duration_s=7.0, seed=5, disturbance_at_s=DIST_AT_S)
    vid = folder / "dist.mp4"
    gt = _generate(vid, sc, 99)
    cfg = AppConfig()
    cfg.physics.px_per_mm = gt["px_per_mm"]
    run, det = analyze_video(str(vid), cfg)
    return run, det, gt, sc, cfg


def test_disturbance_does_not_crash(disturbed):
    run, det, gt, sc, cfg = disturbed
    assert run.n_frames == gt["n_frames"]
    assert det.n_frames_processed + det.n_frames_disturbed + cfg.detection.bg_warmup_frames == run.n_frames
    assert run.duration_s == pytest.approx(det.n_frames_processed / sc.fps, rel=1e-6)
    rows = run.table()
    assert all(r["classe"] in physics.COUNTED_CLASSES for r in rows)
    assert all(math.isfinite(r["longueur_mm"]) for r in rows)


def test_disturbance_is_flagged_and_ignored(disturbed):
    run, det, gt, sc, cfg = disturbed
    assert det.n_frames_disturbed > 0
    # Pas de rafale de fausses traces à l'allumage / l'extinction de la lumière
    burst = [r for r in run.table() if DIST_AT_S - 0.5 <= r["t_debut_s"] <= DIST_AT_S + 0.6]
    assert len(burst) <= 1, burst


# --------------------------------------------------------------------------- #
# Reproductibilité du générateur
# --------------------------------------------------------------------------- #
def test_synth_seed_reproducible():
    def frames():
        sim = SyntheticChamber(SynthConfig(width=160, height=90, duration_s=1.0, seed=42))
        return [sim.render_next() for _ in range(3)]
    cv2.setRNGSeed(1)
    a = frames()
    b = frames()                     # le générateur d'OpenCV a avancé entre les deux
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_synth_reproducible_with_cv2_seed():
    """Contournement documenté : avec cv2.setRNGSeed, la même graine donne la même vidéo."""
    def frames():
        cv2.setRNGSeed(7)
        sim = SyntheticChamber(SynthConfig(width=160, height=90, duration_s=1.0, seed=42))
        return [sim.render_next() for _ in range(3)]
    a, b = frames(), frames()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
