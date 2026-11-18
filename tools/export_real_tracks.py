"""Extrait toutes les traces détectées dans les vraies vidéos pour étiquetage humain/visuel.

Sortie : _training/real/
    tracks.jsonl            une ligne par trace (vidéo, id, mesures, scores des règles…)
    planches/planche_NNN.png  planches de 24 vignettes numérotées (gauche : brute, droite : trace surlignée)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from wilsontrack.analysis.run import analyze_video  # noqa: E402
from wilsontrack.capture import FileSource, imwrite  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402

SRC = ROOT / "exemples" / "vidéos réelles (Wikimedia)"
OUT = ROOT / "_training" / "real"
TILE = 150
COLS, ROWS = 6, 4


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def crop_for(src: FileSource, e, margin: int = 30):
    fr = src.grab_frame(e.best_frame)
    if fr is None or len(e.path_full) == 0:
        return None
    p = np.asarray(e.path_full)
    x0, y0 = np.floor(p.min(0)).astype(int) - margin
    x1, y1 = np.ceil(p.max(0)).astype(int) + margin
    h, w = fr.shape[:2]
    side = max(x1 - x0, y1 - y0, 60)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, x1 = max(0, cx - side // 2), min(w, cx + side // 2)
    y0, y1 = max(0, cy - side // 2), min(h, cy + side // 2)
    raw = fr[y0:y1, x0:x1].copy()
    hl = raw.copy()
    pts = np.round(p - [x0, y0]).astype(np.int32)
    cv2.polylines(hl, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
    def fit(img):
        img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_CUBIC if img.shape[0] < TILE else cv2.INTER_AREA)
        # contraste rehaussé pour l'œil
        return cv2.convertScaleAbs(img, alpha=1.6, beta=-20)
    return np.hstack([fit(raw), fit(hl)]), (x1 - x0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "planches").mkdir(exist_ok=True)
    recs, tiles = [], []
    for vid in sorted(SRC.glob("*.*")):
        if vid.suffix.lower() not in (".ogv", ".webm", ".mp4"):
            continue
        cfg = AppConfig()
        cfg.classification.mode = "regles"
        run, det = analyze_video(str(vid), cfg)
        src = FileSource(str(vid))
        for e in det.valid_events:
            r = crop_for(src, e)
            if r is None:
                continue
            img, side_px = r
            k = len(recs)
            f = {kk: (_num(v) if not isinstance(v, (str, bool)) else v) for kk, v in e.features.items()
                 if isinstance(v, (int, float, bool, str))}
            recs.append({"k": k, "video": vid.name, "id": e.id, "t": round(e.t_first, 2),
                         "duration_s": round(e.duration_s, 3), "touches_border": bool(e.touches_border),
                         "calibrated": False, "rules_cls": e.cls, "rules_conf": round(e.confidence, 3),
                         "rules_scores": {c: round(float(s), 4) for c, s in e.scores.items()},
                         "crop_px": int(side_px), "features": f})
            tiles.append(img)
        src.release()
        print(vid.name, len(det.valid_events), flush=True)
    with open(OUT / "tracks.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = COLS * ROWS
    for s in range(0, len(tiles), per):
        sheet = np.zeros((ROWS * (TILE + 18), COLS * (2 * TILE + 6), 3), np.uint8)
        for j, img in enumerate(tiles[s:s + per]):
            r_, c_ = divmod(j, COLS)
            y, x = r_ * (TILE + 18), c_ * (2 * TILE + 6)
            sheet[y + 18:y + 18 + TILE, x:x + 2 * TILE] = img
            cv2.putText(sheet, f"#{s + j}", (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        imwrite(OUT / "planches" / f"planche_{s // per:03d}.png", sheet)
    print("traces :", len(recs), "planches :", math.ceil(len(recs) / per))


if __name__ == "__main__":
    main()
