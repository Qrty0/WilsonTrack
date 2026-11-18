"""Évaluation de la détection par rapport à une vérité terrain (vidéos synthétiques).

Depuis le simulateur v2, la vérité terrain contient aussi des ARTEFACTS (classe « artefact » :
brume, amas de gouttelettes, scintillements, reflets, éblouissement). Une détection posée sur
un artefact n'est PAS une vraie trace : elle compte comme FAUX POSITIF dans la précision et
elle est rapportée à part (``res["artefact"]``). Le rappel reste calculé sur les 4 classes de
particules. Les appariements complets sont exposés (``_matches_all``, ``_unmatched``) et
``label_detections`` donne une étiquette à chaque détection (entraînement de l'IA).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CLS = ["alpha", "beta", "muon", "proton"]
ARTEFACT = "artefact"


def _poly_mask(g: dict, W: int, H: int, tol: int):
    """Masque (limité à la boîte englobante) de la polyligne vraie dilatée de ``tol`` px."""
    pts = np.round(np.array(g["polyline"], np.float32)).astype(np.int32)
    if pts.ndim != 2 or len(pts) == 0:
        return 0, 0, np.zeros((1, 1), bool)
    x0 = int(max(0, pts[:, 0].min() - tol - 1))
    y0 = int(max(0, pts[:, 1].min() - tol - 1))
    x1 = int(min(W, pts[:, 0].max() + tol + 2))
    y1 = int(min(H, pts[:, 1].max() + tol + 2))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.zeros((1, 1), bool)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.polylines(m, [pts - [x0, y0]], False, 255, 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1)))
    return x0, y0, m > 0


def _frac_on(path: np.ndarray, x0: int, y0: int, m: np.ndarray, W: int, H: int) -> float:
    """Fraction des points (dans l'image) du chemin détecté situés dans le masque."""
    if len(path) == 0:
        return 0.0
    pi = np.round(path).astype(int)
    ok = (pi[:, 0] >= 0) & (pi[:, 0] < W) & (pi[:, 1] >= 0) & (pi[:, 1] < H)
    if not ok.any():
        return 0.0
    px, py = pi[ok, 0] - x0, pi[ok, 1] - y0
    h, w = m.shape
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    hit = np.zeros(px.size, bool)
    hit[inside] = m[py[inside], px[inside]]
    return float(hit.mean())


def _bbox_overlap(path: np.ndarray, x0: int, y0: int, m: np.ndarray) -> bool:
    if len(path) == 0:
        return False
    h, w = m.shape
    return not (path[:, 0].max() < x0 or path[:, 0].min() >= x0 + w or path[:, 1].max() < y0
                or path[:, 1].min() >= y0 + h)


def evaluate(events, gt: dict, min_len_mm: float = 5.0, tol_mm: float = 3.5) -> dict:
    ppm = gt["px_per_mm"]
    fps = gt["config"]["fps"]
    W, H = gt["config"]["width"], gt["config"]["height"]
    tol = max(2, int(round(tol_mm * ppm)))
    # On apparie avec TOUTES les traces vraies (même très courtes : une détection qui leur
    # correspond n'est pas un faux positif), mais le rappel n'est calculé que sur celles ≥ min_len_mm.
    # Les particules sont appariées AVANT les artefacts (une détection posée sur une vraie trace
    # et sur un artefact revient à la trace).
    tracks = list(gt["tracks"])
    particles = sorted([g for g in tracks if g.get("cls") != ARTEFACT], key=lambda g: -g["length_mm"])
    artefacts = sorted([g for g in tracks if g.get("cls") == ARTEFACT], key=lambda g: -g["length_mm"])
    gts = [g for g in particles if g["length_mm"] >= min_len_mm]
    evs = [e for e in events if e.valid]
    used: set[int] = set()
    matches = []
    matches_all = []
    art_matches = []
    small_matched = 0
    for g in particles + artefacts:
        x0, y0, mask = _poly_mask(g, W, H, tol)
        best, best_score = None, 0.0
        for k, e in enumerate(evs):
            if k in used:
                continue
            if not (g["t_birth"] - 0.2 <= e.t_first <= g["t_birth"] + 0.8):
                continue
            p = e.path_full
            if len(p) == 0 or not _bbox_overlap(p, x0, y0, mask):
                continue
            frac = _frac_on(p, x0, y0, mask, W, H)
            if frac > best_score:
                best, best_score = k, frac
        e_best = evs[best] if (best is not None and best_score >= 0.5) else None
        if e_best is not None:
            used.add(best)
        matches_all.append((g, e_best))
        if g.get("cls") == ARTEFACT:
            if e_best is not None:
                art_matches.append((g, e_best))
            continue
        is_small = g["length_mm"] < min_len_mm
        if e_best is not None:
            if is_small:
                small_matched += 1
            else:
                matches.append((g, e_best))
        elif not is_small:
            matches.append((g, None))
    art_ids = {id(e) for _g, e in art_matches}
    n_particle_matched = len(used) - len(art_ids)
    res = {"n_gt": len(gts), "n_det": len(evs)}
    rec = {}
    conf = {c: {} for c in CLS}
    len_ratio = {c: [] for c in CLS}
    for c in CLS:
        gc = [m for m in matches if m[0]["cls"] == c]
        found = [m for m in gc if m[1] is not None]
        rec[c] = (len(found), len(gc))
        for g, e in found:
            conf[c][e.final_cls] = conf[c].get(e.final_cls, 0) + 1
            if g["length_mm"] > 0:
                len_ratio[c].append(e.features["length_mm"] / g["length_mm"])
    nm = sum(1 for m in matches if m[1] is not None)
    res["recall"] = rec
    res["recall_total"] = nm / max(1, len(gts))
    # Précision : détections posées sur une VRAIE particule (même très courte) / détections
    res["precision"] = n_particle_matched / max(1, len(evs))
    res["false_positives"] = len(evs) - n_particle_matched
    res["confusion"] = conf
    correct = sum(conf[c].get(c, 0) for c in CLS)
    res["class_accuracy"] = correct / max(1, nm)
    res["length_ratio_median"] = {c: (float(np.median(v)) if v else None) for c, v in len_ratio.items()}
    unmatched_evs = [e for k, e in enumerate(evs) if k not in used]
    res["fp_classes"] = {}
    for e in unmatched_evs + [e for _g, e in art_matches]:
        res["fp_classes"][e.final_cls] = res["fp_classes"].get(e.final_cls, 0) + 1
    by_kind: dict[str, list[int]] = {}
    for g, e in matches_all:
        if g.get("cls") == ARTEFACT:
            k = g.get("kind", "?")
            by_kind.setdefault(k, [0, 0])
            by_kind[k][1] += 1
            by_kind[k][0] += int(e is not None)
    art_cls: dict[str, int] = {}
    for _g, e in art_matches:
        art_cls[e.final_cls] = art_cls.get(e.final_cls, 0) + 1
    res["artefact"] = {"n_gt": len(artefacts), "n_detected": len(art_matches), "by_kind": by_kind,
                       "classes": art_cls}
    res["fp_artefact"] = len(art_matches)
    res["fp_unexplained"] = len(unmatched_evs)
    res["_matches"] = matches
    res["_matches_all"] = matches_all
    res["_artefact_matches"] = art_matches
    res["_unmatched"] = unmatched_evs
    res["_fps"] = fps
    return res


def label_detections(events, gt: dict, tol_mm: float = 3.5, res: dict | None = None) -> list[dict]:
    """Étiquette chaque détection valide à partir de la vérité terrain (entraînement de l'IA).

    Renvoie une liste de dicts {"event", "label", "how", "gt"} :
      - how = "match"      : appariée à une particule (label = sa classe) ;
      - how = "artefact"   : appariée à un artefact de la vérité terrain (label = "artefact") ;
      - how = "dup"        : non appariée mais posée sur une particule visible au même moment
                             (morceau ou double d'une vraie trace : label = classe de la particule) ;
      - how = "far"        : loin de toute particule (brume, pluie, bruit…) → label = "artefact".
    """
    if res is None:
        res = evaluate(events, gt, min_len_mm=0.0, tol_mm=tol_mm)
    out = []
    seen = set()
    for g, e in res["_matches_all"]:
        if e is None:
            continue
        seen.add(id(e))
        if g.get("cls") == ARTEFACT:
            out.append({"event": e, "label": ARTEFACT, "how": "artefact", "gt": g})
        else:
            out.append({"event": e, "label": g["cls"], "how": "match", "gt": g})
    ppm = gt["px_per_mm"]
    fps = gt["config"]["fps"]
    W, H = gt["config"]["width"], gt["config"]["height"]
    tol = max(2, int(round(tol_mm * ppm)))
    parts = [g for g in gt["tracks"] if g.get("cls") != ARTEFACT]
    masks = {}
    for e in res["_unmatched"]:
        if id(e) in seen:
            continue
        best, best_frac = None, 0.0
        for g in parts:
            t0 = g["t_birth"] - 0.3
            t1 = g["t_birth"] + g.get("life_frames", 0) / fps + 0.3
            if not (t0 <= e.t_first <= t1):
                continue
            if g["id"] not in masks:
                masks[g["id"]] = _poly_mask(g, W, H, tol)
            x0, y0, m = masks[g["id"]]
            if not _bbox_overlap(e.path_full, x0, y0, m):
                continue
            frac = _frac_on(e.path_full, x0, y0, m, W, H)
            if frac > best_frac:
                best, best_frac = g, frac
        if best is not None and best_frac >= 0.5:
            out.append({"event": e, "label": best["cls"], "how": "dup", "gt": best})
        else:
            out.append({"event": e, "label": ARTEFACT, "how": "far", "gt": None})
    return out


def format_report(res: dict) -> str:
    lines = [f"Traces vraies (≥5 mm) : {res['n_gt']}   détectées : {res['n_det']}",
             f"Rappel total : {100 * res['recall_total']:.1f} %   Précision : {100 * res['precision']:.1f} %"
             f"   Faux positifs : {res['false_positives']} {res['fp_classes']}",
             f"Exactitude de classification (traces retrouvées) : {100 * res['class_accuracy']:.1f} %"]
    for c in CLS:
        f, n = res["recall"][c]
        if n:
            lr = res["length_ratio_median"][c]
            lines.append(f"  {c:7s} rappel {f:4d}/{n:<4d} ({100 * f / n:5.1f} %)  classées → {res['confusion'][c]}"
                         + (f"  longueur mesurée/vraie ≈ {lr:.2f}" if lr else ""))
    art = res.get("artefact") or {}
    if art.get("n_gt"):
        kinds = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(art.get("by_kind", {}).items()))
        lines.append(f"  artefacts simulés : {art['n_gt']}, détectés (faux positifs) : {art['n_detected']}"
                     + (f" ({kinds})" if kinds else "") + (f" → classés {art['classes']}" if art.get("classes") else ""))
    return "\n".join(lines)


def load_gt(video_path: str | Path) -> dict:
    return json.loads(Path(str(video_path) + ".gt.json").read_text(encoding="utf-8"))
