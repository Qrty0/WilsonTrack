"""Analyse une vidéo et produit : vidéo annotée, planche d'images annotées, mosaïque des traces.

Usage : python tools/annotate_video.py VIDEO SORTIE [--max N] [--roi x0,y0,x1,y1] [--set sec.param=val ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from wilsontrack.analysis.run import analyze_video  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.render import class_color, contact_sheet, draw_overlay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--max", type=int, default=0)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--roi", default="")
ap.add_argument("--snap", type=int, default=6)
ap.add_argument("--novideo", action="store_true")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cfg = AppConfig()
if a.roi:
    x0, y0, x1, y1 = map(float, a.roi.split(","))
    cfg.detection.roi = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
for kv in a.set:
    k, v = kv.split("=")
    sec, name = k.split(".")
    obj = getattr(cfg, sec)
    cur = getattr(obj, name)
    setattr(obj, name, v if isinstance(cur, str) else type(cur)(eval(v)))

cap = cv2.VideoCapture(a.video)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
n = a.max or n_total
snap_at = set(int(n * (i + 1) / (a.snap + 1)) for i in range(a.snap))
writer = None
snaps = []
stem = Path(a.video).stem


def on_frame(frame, res, det):
    global writer
    recent = det.events[-40:]
    img = draw_overlay(frame, res, det, [e for e in recent if e.valid])
    if not a.novideo:
        if writer is None:
            h, w = img.shape[:2]
            f = min(1.0, 1280 / w)
            writer = (cv2.VideoWriter(str(out / f"{stem}_annot.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), det.fps,
                                      (int(w * f), int(h * f))), f)
        wr, f = writer
        wr.write(cv2.resize(img, None, fx=f, fy=f) if f < 1 else img)
    if res.frame_idx in snap_at:
        snaps.append(cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1]))))


run, det = analyze_video(a.video, cfg, on_frame=on_frame, max_frames=a.max, start_s=a.start)
if writer:
    writer[0].release()
for i, s in enumerate(snaps):
    cv2.imwrite(str(out / f"{stem}_snap{i}.jpg"), s)
ev = [e for e in run.events if e.valid]
sheet = contact_sheet([e.thumb for e in ev[:64]], cols=8, size=110,
                      labels=[f"#{e.id} {e.final_cls[:5]} {e.features['length_mm']:.0f}mm" for e in ev[:64]],
                      colors=[class_color(e.final_cls) for e in ev[:64]])
cv2.imwrite(str(out / f"{stem}_traces.jpg"), sheet)
print(f"{run.n_frames} images, {run.processing_s:.1f} s ({run.n_frames / run.processing_s:.1f} img/s); "
      f"traces : {run.counts()} ; rejetées : {det.rejected} ; perturbées : {det.n_frames_disturbed}; refs {det.refs}")
