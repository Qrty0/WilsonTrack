"""Génère les vidéos d'exemple réalistes (simulateur 2.0) dans exemples/simulateur 2.0/.

Chaque vidéo est accompagnée de <vidéo>.gt.json (vraie nature de chaque trace) : l'échelle
px/mm est alors remplie automatiquement par l'application.

    python tools/make_example_videos.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "exemples" / "simulateur 2.0"

# (nom de fichier, durée s, graine, β supplémentaires/min (source), multiplicateur alphas, résolution)
VIDEOS = [
    ("01 fond (rayons cosmiques) 3 min.mp4", 180, 101, 0.0, 1.0, (1280, 720)),
    ("02 KCl 100 g 3 min.mp4", 180, 102, 40.0, 1.0, (1280, 720)),
    ("03 banane 3 min.mp4", 180, 103, 6.0, 1.0, (1280, 720)),
    ("04 piece riche en radon (alphas) 2 min.mp4", 120, 104, 0.0, 4.0, (1280, 720)),
    ("05 fond longue duree 10 min.mp4", 600, 105, 0.0, 1.0, (960, 540)),
    ("06 KCl longue duree 10 min.mp4", 600, 106, 40.0, 1.0, (960, 540)),
]


def make(spec):
    name, dur, seed, src, alpha_mult, (w, h) = spec
    from wilsontrack.synth import SynthConfig, generate_video
    sc = SynthConfig(width=w, height=h, duration_s=float(dur), seed=seed, source_beta_per_min=src)
    sc.rates_per_min = dict(sc.rates_per_min)
    sc.rates_per_min["alpha"] = sc.rates_per_min.get("alpha", 5.0) * alpha_mult
    path = OUT / name
    gt = generate_video(path, sc)
    n = {}
    for t in gt["tracks"]:
        n[t["cls"]] = n.get(t["cls"], 0) + 1
    return name, round(gt["px_per_mm"], 3), n


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        for name, ppm, n in ex.map(make, VIDEOS):
            print(f"{name} : échelle {ppm} px/mm, traces vraies {n}", flush=True)
