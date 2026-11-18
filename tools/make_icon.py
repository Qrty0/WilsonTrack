"""Génère l'icône de l'application (PNG 512 px + ICO multi-résolutions).

Identité du Cégep Marie-Victorin : le symbole officiel aux 4 « pilules » menthe (#98F29D, géométrie exacte de
assets/cmv/symbole_cmv.svg) centré sur un carré arrondi vert forêt (#154A34).

- grandes tailles (48 px et plus, et le PNG 512 px) : dessin vectoriel à 1024 px avec Qt, puis réduction de
  haute qualité (Lanczos en alpha prémultiplié) ;
- petites tailles (16, 24, 32 px) : dessin direct à la taille finale, pilules calées sur la grille de pixels
  (largeurs entières, interstices d'un pixel) et symbole un peu plus grand, pour rester lisible.

Usage : python tools/make_icon.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath,  # noqa: E402
                           QPen, QTransform)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "wilsontrack" / "assets"
SYMBOL_SVG = ASSETS / "cmv" / "symbole_cmv.svg"

FOREST = "#154A34"
FOREST_TOP = "#1A5A40"
MINT = "#98F29D"

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
SMALL = {          # taille -> (marge du carré px, largeur d'une pilule px, interstice px, hauteur du symbole px)
    16: (0, 2, 1, 8),
    24: (1, 3, 1, 9),
    32: (1, 5, 1, 12),
}

# Tracé officiel (repli si le fichier SVG est absent) — viewBox 48 × 25
_SYMBOL_D = ("M42.6898 24.1797C46.1952 24.1797 48 21.4711 48 17.6997V0.179688C42.6898 0.179688 37.3796 8.09969 "
             "37.3796 13.7225V17.6997C37.3796 21.4711 39.1844 24.1797 42.6898 24.1797ZM30.2299 24.1797C33.7354 "
             "24.1797 35.5401 21.4711 35.5401 17.6997V13.7225C35.5401 8.09969 30.2299 0.179688 24.9197 "
             "0.179688V17.6997C24.9197 21.4711 26.7245 24.1797 30.2299 24.1797ZM17.7701 24.1797C21.2755 24.1797 "
             "23.0803 21.4711 23.0803 17.6997V0.179688H12.4599V17.6997C12.4599 21.4711 14.2646 24.1797 17.7701 "
             "24.1797ZM5.31019 24.1797C8.81562 24.1797 10.6204 21.4711 10.6204 17.6997V0.179688C5.31019 0.179688 "
             "0 8.09969 0 13.7225V17.6997C0 21.4711 1.80477 24.1797 5.31019 24.1797Z")
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _parse(d: str) -> list[QPainterPath]:
    """Commandes absolues M, L, H, V, C, Z : une QPainterPath par sous-tracé."""
    out: list[QPainterPath] = []
    cur = None
    x = y = 0.0
    for cmd, args in re.findall(r"([MLHVCZ])([^MLHVCZ]*)", d):
        v = [float(n) for n in _NUM.findall(args)]
        if cmd == "M":
            cur = QPainterPath()
            out.append(cur)
            x, y = v[0], v[1]
            cur.moveTo(x, y)
            for i in range(2, len(v) - 1, 2):
                x, y = v[i], v[i + 1]
                cur.lineTo(x, y)
        elif cmd == "L":
            for i in range(0, len(v) - 1, 2):
                x, y = v[i], v[i + 1]
                cur.lineTo(x, y)
        elif cmd == "H":
            for x in v:
                cur.lineTo(x, y)
        elif cmd == "V":
            for y in v:
                cur.lineTo(x, y)
        elif cmd == "C":
            for i in range(0, len(v) - 5, 6):
                cur.cubicTo(*v[i:i + 6])
                x, y = v[i + 4], v[i + 5]
        elif cmd == "Z" and cur is not None:
            cur.closeSubpath()
    return out


def symbol_pills() -> list[QPainterPath]:
    """Les 4 pilules, de gauche à droite, dans le repère du SVG."""
    d = _SYMBOL_D
    try:
        m = re.search(r'\sd="([^"]+)"', SYMBOL_SVG.read_text(encoding="utf-8"))
        if m:
            d = m.group(1)
    except OSError:
        pass
    pills = _parse(d)
    pills.sort(key=lambda q: q.boundingRect().left())
    return pills


def _symbol_bounds(pills) -> QRectF:
    r = QRectF(pills[0].boundingRect())
    for q in pills[1:]:
        r = r.united(q.boundingRect())
    return r


def _fit(q: QPainterPath, src: QRectF, dst: QRectF) -> QPainterPath:
    t = QTransform()
    t.translate(dst.left(), dst.top())
    t.scale(dst.width() / src.width(), dst.height() / src.height())
    t.translate(-src.left(), -src.top())
    return t.map(q)


def _background(p: QPainter, size: float, margin: float):
    side = size - 2 * margin
    rect = QRectF(margin, margin, side, side)
    radius = 0.22 * size
    grad = QLinearGradient(QPointF(0, rect.top()), QPointF(0, rect.bottom()))
    grad.setColorAt(0.0, QColor(FOREST_TOP))
    grad.setColorAt(0.75, QColor(FOREST))
    grad.setColorAt(1.0, QColor(FOREST))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(rect, radius, radius)
    return rect, radius


def _new_image(size: int) -> tuple[QImage, QPainter]:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    return img, p


def render_large(size: int = 1024) -> QImage:
    """Icône vectorielle (marge 4 %, rayon 22 %, symbole à 62 % de la largeur)."""
    img, p = _new_image(size)
    rect, radius = _background(p, size, 0.04 * size)
    # reflet intérieur très discret le long du bord supérieur
    hl = QLinearGradient(QPointF(0, rect.top()), QPointF(0, rect.top() + 0.5 * rect.height()))
    hl.setColorAt(0.0, QColor(255, 255, 255, 34))
    hl.setColorAt(1.0, QColor(255, 255, 255, 0))
    w = max(1.0, size / 512)
    p.setPen(QPen(hl, w))
    p.setBrush(Qt.BrushStyle.NoBrush)
    inner = rect.adjusted(w / 2, w / 2, -w / 2, -w / 2)
    p.drawRoundedRect(inner, radius - w / 2, radius - w / 2)

    pills = symbol_pills()
    src = _symbol_bounds(pills)
    sw = 0.62 * size
    sh = sw * src.height() / src.width()
    # centrage optique : les sommets en feuille « allègent » le haut, on remonte d'un poil
    dst = QRectF((size - sw) / 2, (size - sh) / 2 - 0.006 * size, sw, sh)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(MINT))
    for q in pills:
        p.drawPath(_fit(q, src, dst))
    p.end()
    return img


def render_small(size: int) -> QImage:
    """Petite taille : pilules calées sur les pixels (largeur entière, interstice d'un pixel), un peu plus hautes."""
    margin, pw, gap, total_h = SMALL[size]
    img, p = _new_image(size)
    _background(p, size, margin)
    pills = symbol_pills()
    src_all = _symbol_bounds(pills)
    total_w = 4 * pw + 3 * gap
    x0 = (size - total_w) // 2
    y0 = (size - total_h) // 2
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(MINT))
    for k, q in enumerate(pills):
        b = q.boundingRect()
        # chaque pilule garde sa position verticale relative dans le symbole
        top = y0 + (b.top() - src_all.top()) / src_all.height() * total_h
        dst = QRectF(x0 + k * (pw + gap), top, pw, y0 + total_h - top)
        p.drawPath(_fit(q, b, dst))
    p.end()
    return img


def to_pil(img: QImage) -> Image.Image:
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    return Image.frombuffer("RGBA", (img.width(), img.height()), bytes(img.constBits()), "raw", "RGBA",
                            img.bytesPerLine(), 1).copy()


def downscale(im: Image.Image, size: int) -> Image.Image:
    """Réduction Lanczos en alpha prémultiplié (pas de frange sombre sur les bords transparents)."""
    return im.convert("RGBa").resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")


def build() -> dict[int, Image.Image]:
    big = to_pil(render_large(1024))
    frames = {s: (to_pil(render_small(s)) if s in SMALL else downscale(big, s)) for s in ICO_SIZES}
    frames[512] = downscale(big, 512)
    return frames


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
    ASSETS.mkdir(parents=True, exist_ok=True)
    frames = build()
    frames[512].save(ASSETS / "icon.png")
    ico = [frames[s] for s in ICO_SIZES]
    ico[-1].save(ASSETS / "icon.ico", format="ICO", sizes=[(s, s) for s in ICO_SIZES], append_images=ico[:-1])
    print("icône écrite dans", ASSETS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
