"""Analyse les vraies vidéos de exemples/vidéos réelles (Wikimedia)/ avec le moteur de WilsonTrack.

Pour chaque vidéo : résultats + rapport HTML dans <dossier>/analyses/, une image annotée, et un
résumé global analyses/RESUME.md. Les vidéos ne sont pas calibrées (échelle inconnue) : les
longueurs en mm sont approximatives (champ supposé de 250 mm) ; les mesures RELATIVES (épaisseur,
densité, forme) qui servent à classer restent valables.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from wilsontrack.analysis.report import build_run_report  # noqa: E402
from wilsontrack.analysis.run import analyze_video  # noqa: E402
from wilsontrack.capture import imwrite  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.detection.explain import explain_event  # noqa: E402
from wilsontrack.render import draw_overlay  # noqa: E402

SRC = ROOT / "exemples" / "vidéos réelles (Wikimedia)"
OUT = SRC / "analyses"


def main():
    OUT.mkdir(exist_ok=True)
    lines = ["# Analyse des vraies vidéos par WilsonTrack 2.0", "",
             "Vidéos non calibrées (échelle inconnue) : les longueurs en mm sont approximatives. Sources et "
             "licences : voir SOURCES.md.", "",
             "| Vidéo | Durée analysée | Traces | α | β | μ | p | fine β/μ | incertaine | artefacts rejetés | img/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    details = []
    for vid in sorted(SRC.glob("*.*")):
        if vid.suffix.lower() not in (".ogv", ".webm", ".mp4", ".avi", ".mov", ".mkv"):
            continue
        cfg = AppConfig()
        cfg.run.condition = "Vidéo réelle (Wikimedia)"
        t0 = time.time()
        best = {"frame": None, "n": -1}

        def on_frame(frame, res, det, _best=best):
            n = len(getattr(res, "active", []) or [])
            if n > _best["n"] and not getattr(res, "warming", False):
                _best.update(n=n, frame=frame.copy(), res=res, det=det)
        try:
            run, det = analyze_video(str(vid), cfg, on_frame=on_frame)
        except TypeError:
            run, det = analyze_video(str(vid), cfg)
        except Exception as e:
            lines.append(f"| {vid.name} | erreur : {e} | | | | | | | | | |")
            continue
        run.name = vid.stem
        folder = run.save(OUT)
        try:
            build_run_report(run)
        except Exception as e:
            print("  rapport :", e)
        c = run.counts()
        art = sum(1 for e in det.events if not e.valid and e.extra.get("ml_artefact"))
        fps = run.n_frames / max(run.processing_s, 1e-6)
        lines.append(f"| {vid.name} | {run.duration_s:.0f} s | {run.total} | {c.get('alpha', 0)} | {c.get('beta', 0)} | "
                     f"{c.get('muon', 0)} | {c.get('proton', 0)} | {c.get('fine', 0)} | {c.get('incertain', 0)} | "
                     f"{art} | {fps:.0f} |")
        if best["frame"] is not None:
            try:
                img = draw_overlay(best["frame"], best["res"], best["det"],
                                   [e for e in det.events[-60:] if e.valid], show_labels=True)
                imwrite(folder / "image_annotee.png", img)
            except Exception as e:
                print("  image :", e)
        ex = [e for e in det.events if e.valid]
        ex.sort(key=lambda e: -e.confidence)
        details.append(f"\n## {vid.name}\n\nRésultats : `{folder.name}` (rapport.html). Exemples de raisonnement :\n")
        shown = set()
        for e in ex:
            if e.final_cls in shown:
                continue
            shown.add(e.final_cls)
            details.append(f"- trace #{e.id} — {explain_event(e).short()}")
        print(f"{vid.name}: {run.total} traces {c} artefacts {art} — {time.time() - t0:.0f} s", flush=True)
    (OUT / "RESUME.md").write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
