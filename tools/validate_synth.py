"""Valide l'algorithme sur des vidéos synthétiques (rappel / précision / classification).

Usage : python tools/validate_synth.py [--dur 60] [--seed 1] [--out DOSSIER] [--brightness 1.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wilsontrack.analysis.run import analyze_video  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.evaluate import evaluate, format_report, load_gt  # noqa: E402
from wilsontrack.synth import SynthConfig, generate_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="_validation")
    ap.add_argument("--brightness", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=2.5)
    ap.add_argument("--rate", type=float, default=1.0, help="multiplicateur des taux")
    ap.add_argument("--source", type=float, default=0.0, help="β/min supplémentaires (source KCl)")
    ap.add_argument("--calib", action="store_true", help="donner la vraie calibration au détecteur")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="param=valeur (section.param)")
    args = ap.parse_args()
    out = Path(args.out)
    sc = SynthConfig(duration_s=args.dur, seed=args.seed, brightness=args.brightness, noise_sigma=args.noise,
                     source_beta_per_min=args.source)
    sc.rates_per_min = {k: v * args.rate for k, v in sc.rates_per_min.items()}
    vid = out / f"synth_s{args.seed}_d{int(args.dur)}_b{args.brightness}_n{args.noise}_r{args.rate}_src{args.source}.mp4"
    if args.regen or not vid.exists():
        t = time.time()
        generate_video(vid, sc)
        print(f"Vidéo générée en {time.time() - t:.1f} s : {vid}")
    gt = load_gt(vid)
    cfg = AppConfig()
    if args.calib:
        cfg.physics.px_per_mm = gt["px_per_mm"]
    for kv in args.set:
        k, v = kv.split("=")
        sec, name = k.split(".")
        obj = getattr(cfg, sec)
        cur = getattr(obj, name)
        setattr(obj, name, type(cur)(eval(v)) if not isinstance(cur, str) else v)
    run, det = analyze_video(str(vid), cfg)
    fps_proc = run.n_frames / run.processing_s
    print(f"Analyse : {run.n_frames} images en {run.processing_s:.1f} s ({fps_proc:.1f} img/s), "
          f"rejetées : {det.rejected}, réf : {det.refs}")
    res = evaluate(run.events, gt)
    print(format_report(res))
    return res, run, det, gt


if __name__ == "__main__":
    main()
