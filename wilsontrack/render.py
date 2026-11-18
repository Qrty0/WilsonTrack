"""Dessin des annotations (traces, étiquettes, zone d'intérêt, compteurs) sur les images."""
from __future__ import annotations

import cv2
import numpy as np

from .physics import CLASS_INFO

FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(cls: str) -> tuple:
    return CLASS_INFO.get(cls, CLASS_INFO["incertain"])["couleur"]


def _label(img, text, org, color, scale=0.5, thick=1):
    (w, h), b = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - w - 4))
    y = max(h + 4, min(y, img.shape[0] - 2))
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + b - 1), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 2), FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, result, detector, recent_events=(), t_now: float | None = None,
                 show_labels: bool = True, show_roi: bool = True, hud: dict | None = None,
                 keep_s: float = 1.2, out: np.ndarray | None = None, inplace: bool = False) -> np.ndarray:
    """Renvoie une copie annotée de l'image (même taille que l'image source).

    Performance (la copie d'une image 1280×720 coûte plus que tout le dessin) :
    - ``out`` : tampon BGR (même taille, uint8) réutilisé d'une image à l'autre — l'appelant
      doit alors en faire une copie s'il garde l'image annotée au-delà de l'image suivante
      (QImage sans copie, capture d'écran) ;
    - ``inplace=True`` : dessine directement sur ``frame`` (image BGR que l'appelant n'utilise plus).
    Le dessin produit est identique dans tous les cas."""
    if frame.ndim == 2:
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR,
                           dst=out if (out is not None and out.shape == frame.shape + (3,)
                                       and out.dtype == frame.dtype) else None)
    elif inplace:
        img = frame
    elif out is not None and out.shape == frame.shape and out.dtype == frame.dtype and out is not frame:
        np.copyto(out, frame)
        img = out
    else:
        img = frame.copy()
    H, W = img.shape[:2]
    # L'image peut être une version réduite de l'image native (affichage) : facteur k
    k = W / float(detector.frame_hw[1]) if detector.frame_hw[1] else 1.0
    inv = k / detector.scale
    lw = max(1, int(round(W / 900)))
    fs = max(0.45, W / 2400)
    if show_roi and detector.cfg.detection.roi:
        pts = np.array([[p[0] * (W - 1), p[1] * (H - 1)] for p in detector.cfg.detection.roi], np.int32)
        cv2.polylines(img, [pts], True, (120, 120, 120), lw, cv2.LINE_AA)
    t_now = result.t if (t_now is None and result is not None) else (t_now or 0.0)
    # Traces terminées récemment (tracé mesuré)
    for e in recent_events:
        age = t_now - e.t_last
        if age > keep_s or len(e.path_full) < 2:
            continue
        col = class_color(e.final_cls)
        pts = np.round(e.path_full * k).astype(np.int32)
        cv2.polylines(img, [pts], False, col, lw + 1, cv2.LINE_AA)
        if show_labels:
            sym = CLASS_INFO.get(e.final_cls, CLASS_INFO["incertain"])["symbole"]
            p = pts[np.argmin(pts[:, 1])]
            _label(img, f"#{e.id} {sym} {e.features.get('length_mm', 0):.0f}mm", (p[0] + 6, p[1] - 4), col, fs)
    # Traces actives (contour)
    if result is not None:
        for contour, cls, conf, uid in result.active:
            if contour is None:
                continue
            col = class_color(cls)
            c = np.round(contour.astype(np.float32) * inv).astype(np.int32)
            cv2.drawContours(img, [c], -1, col, lw, cv2.LINE_AA)
            if show_labels:
                x, y, w, h = cv2.boundingRect(c)
                sym = CLASS_INFO.get(cls, CLASS_INFO["incertain"])["symbole"]
                _label(img, f"{sym} {int(conf * 100)}%", (x, y - 3), col, fs)
        if result.disturbed:
            _label(img, "PERTURBATION (lumiere / mouvement) - fond reinitialise", (10, H - 12), (0, 0, 255), fs)
        if detector.exp_enabled:
            txt = "FENETRE SENSIBLE" if result.in_window else "attente d'expansion"
            _label(img, txt, (W - 260, 26), (0, 255, 255) if result.in_window else (150, 150, 150), fs)
    if hud:
        y = 24
        for k, (txt, col) in hud.items():
            _label(img, txt, (10, y), col, fs * 1.1, 1)
            y += int(26 * max(1.0, fs / 0.5))
    # Barre d'échelle
    ppm = detector.px_per_mm_full * k
    if ppm > 0:
        L = 10.0 if 10 * ppm < W * 0.3 else 5.0
        n = int(round(L * ppm))
        x1 = W - 20
        x0 = x1 - n
        y0 = H - 20
        cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), lw + 1)
        tag = f"{L:g} mm" + ("" if detector.calibrated else " (non calibre)")
        _label(img, tag, (x0, y0 - 6), (255, 255, 255), fs * 0.9)
    return img


def contact_sheet(images: list[np.ndarray], cols: int = 8, size: int = 120, labels: list[str] | None = None,
                  colors: list | None = None) -> np.ndarray:
    """Mosaïque de vignettes (pour les rapports)."""
    if not images:
        return np.zeros((size, size, 3), np.uint8)
    tiles = []
    for i, im in enumerate(images):
        if im is None:
            im = np.zeros((size, size, 3), np.uint8)
        h, w = im.shape[:2]
        f = size / max(h, w)
        r = cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        t = np.zeros((size, size, 3), np.uint8)
        y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
        t[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r if r.ndim == 3 else cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        if labels:
            col = colors[i] if colors else (255, 255, 255)
            cv2.rectangle(t, (0, 0), (size - 1, size - 1), col, 2)
            cv2.putText(t, labels[i], (4, size - 6), FONT, 0.38, col, 1, cv2.LINE_AA)
        tiles.append(t)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row += [np.zeros((size, size, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def enhance_for_display(img: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Rehaussement du contraste (CLAHE) pour mieux voir les traces faibles à l'écran."""
    if gain <= 1.0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5 * gain, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
