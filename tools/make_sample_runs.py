"""Analyse deux vidéos synthétiques (fond / KCl) et sauvegarde les résultats (données de test)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilsontrack.analysis.conclusions import conclude_comparison, conclude_run, conclusions_text  # noqa: E402
from wilsontrack.analysis.run import RunResult, analyze_video  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.evaluate import load_gt  # noqa: E402

bg_vid, src_vid, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
runs = []
for vid, cond, stype, mass in [(bg_vid, "Fond (rayons cosmiques)", "Aucun", 0.0),
                               (src_vid, "KCl", "KCl (chlorure de potassium)", 100.0)]:
    cfg = AppConfig()
    cfg.physics.px_per_mm = load_gt(vid)["px_per_mm"]
    cfg.run.condition, cfg.run.sample_type, cfg.run.sample_mass_g = cond, stype, mass
    run, det = analyze_video(vid, cfg)
    folder = run.save(out)
    print("sauvegardé :", folder, run.counts())
    runs.append(RunResult.load(folder))
for r in runs:
    s, c = conclude_run(r)
    print("\n=== ", r.name)
    print(conclusions_text(c))
comps, c = conclude_comparison(runs[1], runs[0])
print("\n=== COMPARAISON")
print(conclusions_text(c))
