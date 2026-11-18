"""Affiche les mesures des traces retrouvées, par classe vraie (diagnostic)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilsontrack.analysis.run import analyze_video  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.evaluate import evaluate, format_report, load_gt  # noqa: E402

vid = sys.argv[1]
gt = load_gt(vid)
cfg = AppConfig()
cfg.physics.px_per_mm = gt["px_per_mm"]
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    sec, name = k.split(".")
    obj = getattr(cfg, sec)
    setattr(obj, name, type(getattr(obj, name))(eval(v)))
run, det = analyze_video(vid, cfg)
res = evaluate(run.events, gt)
print("refs", det.refs)
hdr = (f"{'vrai':6s} {'pred':9s} {'Lvrai':>6s} {'L':>6s} {'W':>5s} {'rms':>5s} {'lat':>5s} {'Wrel':>5s} {'Drel':>5s} "
       f"{'S':>6s} {'T':>6s} {'sag':>6s} {'curl':>5s} {'n':>3s} {'dur':>5s} {'E':>6s}")
print(hdr)
for g, e in sorted(res["_matches"], key=lambda m: m[0]["cls"]):
    if e is None:
        print(f"{g['cls']:6s} MANQUEE   L={g['length_mm']:.1f} t={g['t_birth']:.2f}")
        continue
    f = e.features
    print(f"{g['cls']:6s} {e.final_cls:9s} {g['length_mm']:6.1f} {f['length_mm']:6.1f} {f['width_mm']:5.2f} "
          f"{f['line_rms_mm']:5.2f} {f.get('lateral_mm', 0):5.2f} {f.get('width_rel', 0):5.2f} {f.get('density_rel', 0):5.2f} {f['straightness']:6.3f} "
          f"{f['tortuosity']:6.3f} {f['sagitta_ratio']:6.3f} {f['curl_consistency']:5.2f} {e.n_seen:3d} {e.duration_s:5.2f} "
          f"{g.get('energy_mev', 0):6.3f}")
matched = {id(m[1]) for m in res["_matches"] if m[1] is not None}
print("\nFaux positifs :")
for e in run.events:
    if e.valid and id(e) not in matched:
        f = e.features
        print(f"  t={e.t_first:6.2f} {e.final_cls:9s} L={f['length_mm']:.1f} W={f['width_mm']:.2f} S={f['straightness']:.3f} "
              f"snr={f['peak_snr']:.1f} n={e.n_seen} xy={e.centroid_proc}")
print()
print(format_report(res))
