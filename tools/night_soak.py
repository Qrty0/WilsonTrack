"""Essai d'endurance ACCÉLÉRÉ d'une session de nuit (mémoire, points de sauvegarde).

Une chambre simulée (SynthSource realtime=False) est analysée aussi vite que possible par
``longrun.run_session`` avec tous les mécanismes d'une vraie nuit (vidéo segmentée, photos,
points de sauvegarde, délestage des vignettes). La mémoire du processus (RSS / working set)
est relevée toutes les 5 minutes SIMULÉES, puis on calcule la pente (Mo par heure simulée).

Exemple :
    .venv\\Scripts\\python.exe tools\\night_soak.py --minutes 30 --largeur 960 --hauteur 540

Interprétation : chaque trace valide garde en mémoire ses mesures (quelques Ko, nécessaires au
résultat final) ; tout le reste (vignettes, détections rejetées) doit rester borné. La pente
attendue est donc ≈ (traces par heure) × (Ko par trace), sans croissance liée aux images.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rss_bytes() -> int:
    """Mémoire résidente du processus (psutil si disponible, sinon GetProcessMemoryInfo sous Windows)."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize)
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # noqa: BLE001
        return -1


def deep_size(o, _seen=None) -> int:
    """Taille mémoire approximative d'un objet Python (dict, liste, tableau numpy, dataclass…)."""
    import numpy as np
    if _seen is None:
        _seen = set()
    if id(o) in _seen:
        return 0
    _seen.add(id(o))
    if isinstance(o, np.ndarray):
        return int(o.nbytes) + 112
    n = sys.getsizeof(o)
    if isinstance(o, dict):
        n += sum(deep_size(k, _seen) + deep_size(v, _seen) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        n += sum(deep_size(v, _seen) for v in o)
    elif hasattr(o, "__dict__") and not isinstance(o, type):
        n += deep_size(vars(o), _seen)
    return n


def slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else float("nan")


def main(argv=None) -> int:
    from wilsontrack.config import AppConfig
    from wilsontrack.gui.worker import SynthSource
    from wilsontrack.longrun import NightSessionConfig, fmt_num, run_session
    from wilsontrack.synth import SynthConfig

    ap = argparse.ArgumentParser(description="Essai d'endurance accéléré d'une session de nuit.")
    ap.add_argument("--minutes", type=float, default=30.0, help="durée SIMULÉE (min), défaut 30")
    ap.add_argument("--largeur", type=int, default=960)
    ap.add_argument("--hauteur", type=int, default=540)
    ap.add_argument("--ips", type=float, default=30.0, help="images par seconde simulées (défaut 30)")
    ap.add_argument("--sauvegarde-min", dest="autosave", type=float, default=5.0)
    ap.add_argument("--photo-min", dest="photo", type=float, default=10.0)
    ap.add_argument("--video", default="reduite", choices=["aucune", "reduite", "complete"])
    ap.add_argument("--releve-min", dest="sample_min", type=float, default=5.0,
                    help="relevé de la mémoire toutes les N minutes simulées (défaut 5)")
    ap.add_argument("--sortie", default="", help="dossier de sortie (défaut : dossier temporaire)")
    ap.add_argument("--graine", type=int, default=11)
    a = ap.parse_args(argv)

    out = Path(a.sortie) if a.sortie else Path(tempfile.mkdtemp(prefix="wilsontrack_soak_"))
    sc = SynthConfig(width=a.largeur, height=a.hauteur, fps=a.ips, duration_s=a.minutes * 60.0 + 30.0,
                     seed=a.graine, rates_per_min={"alpha": 8, "beta": 14, "muon": 16, "proton": 0.3})
    cfg = NightSessionConfig.from_dict({
        "duration_h": a.minutes / 60.0, "autosave_every_min": a.autosave, "snapshot_every_min": a.photo,
        "segment_min": 10, "record_video": a.video, "condition": "Essai d'endurance (simulation)",
        "output_dir": str(out), "keep_awake": False})
    samples: list[dict] = []
    period = a.sample_min * 60.0
    state = {"next": 0.0, "t0": time.perf_counter()}
    holder: dict = {}

    def factory():
        holder["src"] = SynthSource(sc, realtime=False)
        return holder["src"]

    def on_frame(frame, res, det):
        # La vérité terrain du SIMULATEUR (une entrée par trace simulée) n'appartient pas à
        # l'application : on la vide pour ne mesurer que la mémoire de WilsonTrack.
        src = holder.get("src")
        if src is not None and len(src.sim.gt) > 50:
            src.sim.gt.clear()
        if res.t + 1e-9 >= state["next"]:
            state["next"] += period
            gc.collect()
            thumbs = sum(1 for e in det.events if getattr(e, "thumb", None) is not None)
            invalid = sum(1 for e in det.events if not e.valid)
            valid_ev = [e for e in det.events if e.valid]
            spilled = [e for e in valid_ev if e.thumb is None][-50:]
            per_ev = (sum(deep_size(e) for e in spilled) / len(spilled)) if spilled else float("nan")
            s = {"sim_min": res.t / 60.0, "wall_s": time.perf_counter() - state["t0"], "rss_mb": rss_bytes() / 2 ** 20,
                 "events": len(det.events), "valid": len(valid_ev), "invalid_kept": invalid,
                 "thumbs_in_memory": thumbs, "rejected": det.rejected,
                 "event_kb_sans_vignette": None if math.isnan(per_ev) else round(per_ev / 1024, 2)}
            samples.append(s)
            print(f"  t = {fmt_num(s['sim_min'], 1)} min simulées ({fmt_num(s['wall_s'], 0)} s réelles) : "
                  f"RSS {fmt_num(s['rss_mb'], 1)} Mo, {s['valid']} traces valides, {s['invalid_kept']} rejets gardés, "
                  f"{thumbs} vignettes en mémoire, {fmt_num(per_ev / 1024, 1)} Ko par trace délestée", flush=True)

    print(f"Essai d'endurance : {fmt_num(a.minutes, 0)} min simulées, {a.largeur}×{a.hauteur} à {fmt_num(a.ips, 0)} img/s, "
          f"vidéo {a.video}, sortie {out}", flush=True)
    t0 = time.perf_counter()
    run = run_session(cfg, AppConfig(), factory, source_name="simulation (endurance)",
                      on_frame=on_frame, make_report=True)
    wall = time.perf_counter() - t0
    gc.collect()
    end_rss = rss_bytes() / 2 ** 20
    # Pente : on ignore le premier relevé (t = 0, avant l'apprentissage du fond et les premiers caches)
    pts = [s for s in samples if s["sim_min"] > 0.5]
    xs = [s["sim_min"] / 60.0 for s in pts]
    ys = [s["rss_mb"] for s in pts]
    sl = slope(xs, ys)
    per_track_kb = float("nan")
    if len(pts) >= 2 and pts[-1]["valid"] > pts[0]["valid"]:
        per_track_kb = (pts[-1]["rss_mb"] - pts[0]["rss_mb"]) * 1024 / (pts[-1]["valid"] - pts[0]["valid"])
    tracks_per_h = (pts[-1]["valid"] - pts[0]["valid"]) / max(1e-9, xs[-1] - xs[0]) if len(pts) >= 2 else float("nan")
    res = {
        "minutes_simulees": a.minutes, "taille": [a.largeur, a.hauteur], "ips": a.ips, "video": a.video,
        "duree_reelle_s": round(wall, 1), "acceleration": round(a.minutes * 60.0 / wall, 2),
        "releves": samples, "rss_fin_mb": round(end_rss, 1),
        "pente_mo_par_heure_simulee": None if math.isnan(sl) else round(sl, 2),
        "memoire_par_trace_ko": None if math.isnan(per_track_kb) else round(per_track_kb, 2),
        "traces_par_heure": None if math.isnan(tracks_per_h) else round(tracks_per_h, 1),
        "points_de_sauvegarde": run.session.get("checkpoints"), "photos": run.session.get("photos"),
        "segments_video": len(run.session.get("video_segments", [])), "traces_valides": run.total,
        "comptes": run.counts(), "duree_analysee_s": round(run.duration_s, 1), "dossier": run.folder,
        "fin": run.session.get("stop_reason"),
    }
    Path(run.folder, "soak_resultats.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print(f"Terminé en {fmt_num(wall / 60, 1)} min réelles (×{fmt_num(res['acceleration'], 2)}) : {run.total} traces "
          f"valides, {res['points_de_sauvegarde']} points de sauvegarde, {res['photos']} photos, "
          f"{res['segments_video']} segment(s) vidéo.")
    print(f"Mémoire : pente {fmt_num(sl, 1)} Mo par heure simulée ; ≈ {fmt_num(per_track_kb, 1)} Ko par trace valide "
          f"({fmt_num(tracks_per_h, 0)} traces/h) ; RSS final {fmt_num(end_rss, 1)} Mo.")
    late = [s for s in pts if s["thumbs_in_memory"] is not None and s["valid"] > 300]
    if len(late) >= 2:
        xl = [s["sim_min"] / 60.0 for s in late]
        sl2 = slope(xl, [s["rss_mb"] for s in late])
        res["pente_apres_300_traces_mo_par_heure"] = None if math.isnan(sl2) else round(sl2, 2)
        Path(run.folder, "soak_resultats.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
        print(f"  Pente une fois les 300 vignettes récentes en mémoire (régime établi) : {fmt_num(sl2, 1)} Mo/h.")
    print(f"Résultats : {Path(run.folder, 'soak_resultats.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
