"""Liste les mesures des traces détectées sur une vidéo réelle (diagnostic)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilsontrack.analysis.run import analyze_video  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402

vid = sys.argv[1]
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 900
cfg = AppConfig()
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    if k == "roi":
        x0, y0, x1, y1 = map(float, v.split(","))
        cfg.detection.roi = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        continue
    sec, name = k.split(".")
    obj = getattr(cfg, sec)
    setattr(obj, name, type(getattr(obj, name))(eval(v)))
t = time.time()
run, det = analyze_video(vid, cfg, max_frames=nmax)
print(f"{run.n_frames / (time.time() - t):.1f} img/s ; {run.counts()} ; rejetées {det.rejected} ; refs {det.refs}")
for e in [e for e in det.events if e.valid][:80]:
    f = e.features
    print(f"#{e.id:3d} {e.final_cls:9s} L={f['length_mm']:5.1f} W={f['width_mm']:4.2f} el={f['elongation']:5.1f} "
          f"S={f['straightness']:.2f} snr={f['peak_snr']:5.1f} dens={f['density']:6.1f} n={e.n_seen:3d} "
          f"Wm={f['width_mask_mm']:4.2f} cv={f['profile_cv']:.2f} Drel={f.get('density_rel', 0):.2f}")
