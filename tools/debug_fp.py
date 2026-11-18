"""Outil de diagnostic : caractéristiques des faux positifs sur une vidéo synthétique."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from wilsontrack.analysis.run import analyze_video  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.evaluate import evaluate, format_report, load_gt  # noqa: E402

vid = sys.argv[1]
outdir = Path(sys.argv[2])
outdir.mkdir(exist_ok=True, parents=True)
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 600
gt = load_gt(vid)
cfg = AppConfig()
cfg.physics.px_per_mm = gt["px_per_mm"]
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    sec, name = k.split(".")
    obj = getattr(cfg, sec)
    cur = getattr(obj, name)
    setattr(obj, name, type(cur)(eval(v)))

dbg = {}


def on_frame(frame, res, det):
    if res.frame_idx == 200:
        dbg["frame"] = frame.copy()
        d = det
        g = d.pre(frame) if False else None
    if res.frame_idx in (200,) and res.debug:
        dbg.update(res.debug)


# analyse avec debug à l'image 200
cap_cfg = cfg
run, det = analyze_video(vid, cfg, max_frames=nmax)
res = evaluate(run.events, {**gt, "tracks": [t for t in gt["tracks"] if t["frame_birth"] < nmax - 30]})
print(format_report(res))
matched = {id(m[1]) for m in res["_matches"] if m[1] is not None}
fps = [e for e in run.events if e.valid and id(e) not in matched]
tps = [e for e in run.events if e.valid and id(e) in matched]
keys = ["length_mm", "width_mm", "straightness", "elongation", "density", "peak_snr", "area_px"]
print("\nFaux positifs (médianes / p10 / p90) :")
for k in keys:
    v = np.array([e.features[k] for e in fps])
    w = np.array([e.features[k] for e in tps])
    if v.size and w.size:
        print(f"  {k:14s} FP {np.median(v):8.2f} [{np.percentile(v, 10):7.2f},{np.percentile(v, 90):7.2f}]"
              f"   TP {np.median(w):8.2f} [{np.percentile(w, 10):7.2f},{np.percentile(w, 90):7.2f}]")
v = np.array([e.n_seen for e in fps]); w = np.array([e.n_seen for e in tps])
print(f"  n_seen FP {np.median(v)} TP {np.median(w)}")
# mosaïque de vignettes des faux positifs
th = [cv2.resize(e.thumb, (120, 120)) for e in fps[:48] if e.thumb is not None]
if th:
    rows = [np.hstack(th[i:i + 8] + [np.zeros((120, 120, 3), np.uint8)] * (8 - len(th[i:i + 8])))
            for i in range(0, len(th), 8)]
    mos = np.vstack(rows)
    mos = cv2.convertScaleAbs(mos, alpha=3.0)
    cv2.imwrite(str(outdir / "fp_mosaic.png"), mos)
th = [cv2.resize(e.thumb, (120, 120)) for e in tps[:32] if e.thumb is not None]
if th:
    rows = [np.hstack(th[i:i + 8] + [np.zeros((120, 120, 3), np.uint8)] * (8 - len(th[i:i + 8])))
            for i in range(0, len(th), 8)]
    cv2.imwrite(str(outdir / "tp_mosaic.png"), cv2.convertScaleAbs(np.vstack(rows), alpha=3.0))

# Image de diagnostic : évidence d'une image
cap = cv2.VideoCapture(vid)
from wilsontrack.detection import TrackDetector  # noqa: E402
ok, fr = cap.read()
d2 = TrackDetector(cfg, 30, fr.shape[:2])
for i in range(200):
    r = d2.process(fr, want_debug=(i == 199))
    ok, fr = cap.read()
E = r.debug["evidence"]
print("E percentiles 50/90/99/99.9:", np.percentile(E, [50, 90, 99, 99.9]), "sigma", r.noise_sigma)
vis = np.clip(E * 20, 0, 255).astype(np.uint8)
cv2.imwrite(str(outdir / "evidence.png"), vis)
cv2.imwrite(str(outdir / "labels.png"), ((r.debug["labels"] > 0) * 255).astype(np.uint8))
cv2.imwrite(str(outdir / "diff.png"), np.clip(r.debug["diff"] * 8 + 30, 0, 255).astype(np.uint8))
