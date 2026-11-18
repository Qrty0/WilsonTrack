"""Identité visuelle du Cégep Marie-Victorin appliquée à WilsonTrack.

- jetons de couleur (mesurés sur cegepmv.ca) et couleurs dérivées pour l'interface claire ;
- logos officiels (SVG, rendus avec QtSvg) et géométrie exacte des « pilules » du symbole ;
- petites icônes vectorielles monochromes (barre d'en-tête, menus) ;
- courbe d'animation du site : cubic-bezier(0.215, 0.61, 0.355, 1), et préférence « mouvement réduit ».

Les couleurs des classes de particules (physics.CLASS_INFO) ne sont PAS redéfinies ici : elles codent la
physique. Pour écrire du texte dans ces couleurs sur fond blanc, utiliser class_text_hex() (même teinte,
assombrie pour respecter le contraste WCAG AA).
"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

# --------------------------------------------------------------------------- couleurs de la marque --
FOREST = "#154A34"        # vert forêt : couleur principale (bandeaux, en-tête, boutons principaux)
GREEN = "#247F59"         # vert secondaire (survol, liens, accents)
BRIGHT = "#26B076"        # vert vif (succès, anneau de focus)
MINT = "#98F29D"          # menthe (logo sur fond foncé, mises en évidence)
SOFT = "#F1F5F1"          # fond doux de la fenêtre
INK = "#1C1C1C"           # encre
WHITE = "#FFFFFF"

# --------------------------------------------------------------------------- dérivées (interface) --
MUTED = "#5B6B62"         # texte secondaire (6,1:1 sur blanc)
FAINT = "#66766D"         # texte discret (4,8:1 sur blanc)
BORDER = "#D5E3D8"        # contour doux des cartes
BORDER_STRONG = "#B7CCBD" # contour des champs
HOVER = "#EAF3EC"         # survol léger
ALT_ROW = "#F7FAF7"       # lignes alternées
SELECTION = "#CDEFD0"     # sélection (texte : FOREST)
FOREST_DARK = "#0F3828"   # bouton principal enfoncé
FOREST_DEEP = "#0C2E21"   # ombres du bandeau
MINT_SOFT = "#E3F8E4"     # pastilles menthe sur fond clair
VIDEO_BG = "#0B1712"      # image de la caméra (reste sombre)
LINK = GREEN

WARN_TEXT, WARN_BG, WARN_BORDER = "#8A5300", "#FFF5DF", "#F0C66E"
DANGER_TEXT, DANGER_BG, DANGER_BORDER = "#B4232A", "#FDEDED", "#F2B8BA"
INFO_TEXT, INFO_BG, INFO_BORDER = "#1D5FA8", "#EAF3FD", "#B7D3F2"
SUCCESS_TEXT = "#1A7A50"  # texte « succès » lisible sur blanc (5,2:1)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CMV = ASSETS / "cmv"
LOGO_DARK_BG = CMV / "logo_cmv_fond_fonce.svg"     # symbole menthe + texte blanc (fonds verts)
LOGO_LIGHT_BG = CMV / "logo_cmv_fond_clair.svg"    # symbole vert + texte encre (fonds clairs)
SYMBOL = CMV / "symbole_cmv.svg"                   # les 4 pilules seules (menthe)

ANIM_MS = 1500
EASE_SITE = (0.215, 0.61, 0.355, 1.0)              # courbe de révélation de cegepmv.ca


def ui_font_family() -> str | None:
    """Police système (les polices du site sont commerciales : elles ne sont pas embarquées)."""
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if (fonts / "SegUIVar.ttf").exists():
            return "Segoe UI Variable Text"
        return "Segoe UI"
    return None          # macOS / Linux : police système par défaut de Qt (SF sur macOS)


def reduced_motion() -> bool:
    """Animations désactivées (WILSONTRACK_NO_ANIM=1 : tests, préférence de l'utilisateur)."""
    return os.environ.get("WILSONTRACK_NO_ANIM", "").strip().lower() in ("1", "true", "oui", "yes")


@lru_cache(maxsize=1)
def site_easing() -> QEasingCurve:
    """cubic-bezier(0.215, 0.61, 0.355, 1) — la révélation « douce » du site du Cégep."""
    c = QEasingCurve(QEasingCurve.Type.BezierSpline)
    x1, y1, x2, y2 = EASE_SITE
    c.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return c


def ease(t: float) -> float:
    t = 0.0 if t <= 0 else (1.0 if t >= 1 else t)
    return float(site_easing().valueForProgress(t))


def mix(c1: str, c2: str, t: float) -> QColor:
    a, b = QColor(c1), QColor(c2)
    return QColor(round(a.red() + (b.red() - a.red()) * t), round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


def rgba(c: str, alpha: int) -> str:
    q = QColor(c)
    return f"rgba({q.red()},{q.green()},{q.blue()},{alpha})"


# --------------------------------------------------------------------------- contraste --
def _lum(c: QColor) -> float:
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


def contrast(c1: str, c2: str) -> float:
    l1, l2 = sorted((_lum(QColor(c1)), _lum(QColor(c2))), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


@lru_cache(maxsize=64)
def readable_on(hex_color: str, bg: str = WHITE, target: float = 4.6) -> str:
    """Même teinte, assombrie juste assez pour un texte lisible (WCAG AA) sur `bg`."""
    c = QColor(hex_color)
    h, s, v, _a = c.getHsvF()
    for _ in range(40):
        if contrast(c.name(), bg) >= target:
            break
        v = max(0.0, v - 0.03)
        c = QColor.fromHsvF(h, min(1.0, s * 1.02), v)
    return c.name()


def class_text_hex(cls: str) -> str:
    """Couleur d'une classe de particule pour du TEXTE sur fond clair (la teinte physique est conservée)."""
    from .. import physics
    info = physics.CLASS_INFO.get(cls) or physics.CLASS_INFO.get("incertain") or {"hex": "#8b8d98"}
    return readable_on(info["hex"])


# --------------------------------------------------------------------------- logos (SVG) --
def svg_renderer(path: Path):
    try:
        from PySide6.QtSvg import QSvgRenderer
    except Exception:  # pragma: no cover - QtSvg absent (installation incomplète)
        return None
    r = QSvgRenderer(str(path))
    return r if r.isValid() else None


def svg_size(path: Path) -> QSize:
    r = svg_renderer(path)
    return r.defaultSize() if r is not None else QSize(354, 56)


def svg_pixmap(path: Path, height: int, dpr: float = 1.0, color: str | None = None) -> QPixmap:
    """Rend un SVG à la hauteur demandée (pixmap haute densité). `color` recolore toute l'image."""
    r = svg_renderer(path)
    size = r.defaultSize() if r is not None else QSize(354, 56)
    w = max(1, round(height * size.width() / max(1, size.height())))
    pm = QPixmap(max(1, round(w * dpr)), max(1, round(height * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    if r is None:
        return pm
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    r.render(p, QRectF(0, 0, w, height))
    if color:
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(QRectF(0, 0, w, height), QColor(color))
    p.end()
    return pm


def paint_svg(p: QPainter, path: Path, rect: QRectF):
    r = svg_renderer(path)
    if r is not None:
        r.render(p, rect)


# --------------------------------------------------------------------------- pilules du symbole --
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _parse_svg_path(d: str) -> list[QPainterPath]:
    """Mini-analyseur (commandes absolues M, L, H, V, C, Z) : une QPainterPath par sous-tracé."""
    out: list[QPainterPath] = []
    cur: QPainterPath | None = None
    x = y = 0.0
    for cmd, args in re.findall(r"([MLHVCZmlhvcz])([^MLHVCZmlhvcz]*)", d):
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
            for n in v:
                x = n
                cur.lineTo(x, y)
        elif cmd == "V":
            for n in v:
                y = n
                cur.lineTo(x, y)
        elif cmd == "C":
            for i in range(0, len(v) - 5, 6):
                cur.cubicTo(v[i], v[i + 1], v[i + 2], v[i + 3], v[i + 4], v[i + 5])
                x, y = v[i + 4], v[i + 5]
        elif cmd in "Zz" and cur is not None:
            cur.closeSubpath()
    return out


# Tracé officiel du symbole (symbole_cmv.svg, viewBox 48 × 25) — repli si le fichier est absent
_SYMBOL_D = ("M42.6898 24.1797C46.1952 24.1797 48 21.4711 48 17.6997V0.179688C42.6898 0.179688 37.3796 8.09969 "
             "37.3796 13.7225V17.6997C37.3796 21.4711 39.1844 24.1797 42.6898 24.1797ZM30.2299 24.1797C33.7354 "
             "24.1797 35.5401 21.4711 35.5401 17.6997V13.7225C35.5401 8.09969 30.2299 0.179688 24.9197 "
             "0.179688V17.6997C24.9197 21.4711 26.7245 24.1797 30.2299 24.1797ZM17.7701 24.1797C21.2755 24.1797 "
             "23.0803 21.4711 23.0803 17.6997V0.179688H12.4599V17.6997C12.4599 21.4711 14.2646 24.1797 17.7701 "
             "24.1797ZM5.31019 24.1797C8.81562 24.1797 10.6204 21.4711 10.6204 17.6997V0.179688C5.31019 0.179688 "
             "0 8.09969 0 13.7225V17.6997C0 21.4711 1.80477 24.1797 5.31019 24.1797Z")


@lru_cache(maxsize=1)
def symbol_pills() -> tuple:
    """Les 4 pilules du symbole, de gauche à droite, dans le repère du SVG (48 × 24, origine en haut)."""
    d = _SYMBOL_D
    try:
        m = re.search(r'\sd="([^"]+)"', SYMBOL.read_text(encoding="utf-8"))
        if m:
            d = m.group(1)
    except OSError:
        pass
    pills = _parse_svg_path(d)
    pills.sort(key=lambda q: q.boundingRect().left())
    shifted = []
    for q in pills:
        q = QPainterPath(q)
        q.translate(0, -0.179688)
        shifted.append(q)
    return tuple(shifted)


SYMBOL_W, SYMBOL_H = 48.0, 24.0


def symbol_path(rect: QRectF) -> QPainterPath:
    """Symbole complet ajusté (proportions conservées, centré) dans `rect`."""
    s = min(rect.width() / SYMBOL_W, rect.height() / SYMBOL_H)
    ox = rect.left() + (rect.width() - SYMBOL_W * s) / 2
    oy = rect.top() + (rect.height() - SYMBOL_H * s) / 2
    out = QPainterPath()
    for q in symbol_pills():
        out.addPath(_scaled(q, s, ox, oy))
    return out


def _scaled(q: QPainterPath, s: float, ox: float, oy: float) -> QPainterPath:
    from PySide6.QtGui import QTransform
    return QTransform(s, 0, 0, s, ox, oy).map(q)


def pill_shape(kind: int, rect: QRectF) -> QPainterPath:
    """Une pilule du symbole (0..3 : feuille gauche, droite, feuille droite, feuille gauche) dans `rect`.

    La forme garde ses proportions sur la largeur ; la hauteur est étirée en conservant le bas arrondi.
    """
    q = symbol_pills()[kind % 4]
    b = q.boundingRect()
    from PySide6.QtGui import QTransform
    t = QTransform()
    t.translate(rect.left(), rect.top())
    t.scale(rect.width() / b.width(), rect.height() / b.height())
    t.translate(-b.left(), -b.top())
    return t.map(q)


# --------------------------------------------------------------------------- icônes vectorielles --
def _icon_path(name: str, s: float) -> tuple[QPainterPath, bool]:
    """Tracé d'une icône dans un carré s × s ; renvoie (chemin, rempli ?)."""
    u = s / 24.0
    P = QPainterPath()
    fill = False

    def pt(x, y):
        return QPointF(x * u, y * u)

    if name == "home":
        P.moveTo(pt(4, 11))
        P.lineTo(pt(12, 4.5))
        P.lineTo(pt(20, 11))
        P.moveTo(pt(6.5, 9.5))
        P.lineTo(pt(6.5, 19.5))
        P.lineTo(pt(17.5, 19.5))
        P.lineTo(pt(17.5, 9.5))
        P.moveTo(pt(10.5, 19.5))
        P.lineTo(pt(10.5, 14.5))
        P.lineTo(pt(13.5, 14.5))
        P.lineTo(pt(13.5, 19.5))
    elif name == "camera":
        P.addRoundedRect(QRectF(3 * u, 7 * u, 13 * u, 10 * u), 2.5 * u, 2.5 * u)
        P.moveTo(pt(16, 10.5))
        P.lineTo(pt(21, 7.5))
        P.lineTo(pt(21, 16.5))
        P.lineTo(pt(16, 13.5))
    elif name == "video":
        P.addRoundedRect(QRectF(3.5 * u, 4.5 * u, 17 * u, 15 * u), 2.5 * u, 2.5 * u)
        P.moveTo(pt(10.2, 9))
        P.lineTo(pt(15.2, 12))
        P.lineTo(pt(10.2, 15))
        P.closeSubpath()
    elif name == "pause":
        P.addRoundedRect(QRectF(7 * u, 5.5 * u, 3.4 * u, 13 * u), 1.2 * u, 1.2 * u)
        P.addRoundedRect(QRectF(13.6 * u, 5.5 * u, 3.4 * u, 13 * u), 1.2 * u, 1.2 * u)
        fill = True
    elif name == "stop":
        P.addRoundedRect(QRectF(6.5 * u, 6.5 * u, 11 * u, 11 * u), 2.4 * u, 2.4 * u)
        fill = True
    elif name == "record":
        P.addEllipse(QPointF(12 * u, 12 * u), 6 * u, 6 * u)
        fill = True
    elif name == "photo":
        P.addRoundedRect(QRectF(3 * u, 7 * u, 18 * u, 12.5 * u), 2.6 * u, 2.6 * u)
        P.moveTo(pt(8.5, 7))
        P.lineTo(pt(10, 4.6))
        P.lineTo(pt(14, 4.6))
        P.lineTo(pt(15.5, 7))
        P.addEllipse(QPointF(12 * u, 13.2 * u), 3.4 * u, 3.4 * u)
    elif name == "gear":
        import math
        cx = cy = 12 * u
        r_out, r_in, n = 8.6 * u, 6.6 * u, 8
        for k in range(n * 2):
            a0 = math.pi * k / n
            r = r_out if k % 2 == 0 else r_in
            for da in (-0.17, 0.17):
                p = QPointF(cx + r * math.cos(a0 + da), cy + r * math.sin(a0 + da))
                if k == 0 and da < 0:
                    P.moveTo(p)
                else:
                    P.lineTo(p)
        P.closeSubpath()
        P.addEllipse(QPointF(cx, cy), 2.8 * u, 2.8 * u)
    elif name == "folder":
        P.moveTo(pt(3.5, 7))
        P.lineTo(pt(9.5, 7))
        P.lineTo(pt(11.2, 9))
        P.lineTo(pt(20.5, 9))
        P.lineTo(pt(20.5, 18.5))
        P.lineTo(pt(3.5, 18.5))
        P.closeSubpath()
    elif name == "spark":                  # assistant / IA
        P.moveTo(pt(12, 3.5))
        P.cubicTo(pt(12.8, 9.5), pt(14.5, 11.2), pt(20.5, 12))
        P.cubicTo(pt(14.5, 12.8), pt(12.8, 14.5), pt(12, 20.5))
        P.cubicTo(pt(11.2, 14.5), pt(9.5, 12.8), pt(3.5, 12))
        P.cubicTo(pt(9.5, 11.2), pt(11.2, 9.5), pt(12, 3.5))
        fill = True
    elif name == "help":
        P.addEllipse(QPointF(12 * u, 12 * u), 8.5 * u, 8.5 * u)
        P.moveTo(pt(9.6, 9.8))
        P.cubicTo(pt(9.6, 6.8), pt(14.6, 6.8), pt(14.6, 9.8))
        P.cubicTo(pt(14.6, 11.8), pt(12, 11.6), pt(12, 14))
        P.moveTo(pt(12, 16.6))
        P.lineTo(pt(12, 16.7))
    elif name == "chevron":
        P.moveTo(pt(9.5, 6.5))
        P.lineTo(pt(15, 12))
        P.lineTo(pt(9.5, 17.5))
    elif name == "arrow":
        P.moveTo(pt(5, 12))
        P.lineTo(pt(19, 12))
        P.moveTo(pt(13.5, 6.5))
        P.lineTo(pt(19, 12))
        P.lineTo(pt(13.5, 17.5))
    elif name == "check":
        P.moveTo(pt(5.5, 12.5))
        P.lineTo(pt(10, 17))
        P.lineTo(pt(18.5, 7.5))
    elif name == "moon":
        outer = QPainterPath()
        outer.addEllipse(QPointF(12 * u, 12 * u), 8.2 * u, 8.2 * u)
        bite = QPainterPath()
        bite.addEllipse(QPointF(16.3 * u, 8.6 * u), 6.6 * u, 6.6 * u)
        P = outer.subtracted(bite)
        fill = True
    elif name == "bars":                   # comparer des résultats
        P.addRoundedRect(QRectF(4.5 * u, 11 * u, 3.6 * u, 8.5 * u), 1.2 * u, 1.2 * u)
        P.addRoundedRect(QRectF(10.2 * u, 5 * u, 3.6 * u, 14.5 * u), 1.2 * u, 1.2 * u)
        P.addRoundedRect(QRectF(15.9 * u, 8.5 * u, 3.6 * u, 11 * u), 1.2 * u, 1.2 * u)
        fill = True
    elif name == "images":
        P.addRoundedRect(QRectF(3.5 * u, 5.5 * u, 17 * u, 13 * u), 2.4 * u, 2.4 * u)
        P.moveTo(pt(4.5, 16.5))
        P.lineTo(pt(9.5, 11.5))
        P.lineTo(pt(13, 15))
        P.lineTo(pt(15.2, 12.8))
        P.lineTo(pt(19.5, 17))
        P.addEllipse(QPointF(15.5 * u, 9.3 * u), 1.4 * u, 1.4 * u)
    elif name == "flask":
        P.moveTo(pt(9.5, 3.8))
        P.lineTo(pt(14.5, 3.8))
        P.moveTo(pt(10.3, 4))
        P.lineTo(pt(10.3, 9.6))
        P.lineTo(pt(5.2, 18))
        P.cubicTo(pt(4.6, 19.2), pt(5.3, 20.2), pt(6.6, 20.2))
        P.lineTo(pt(17.4, 20.2))
        P.cubicTo(pt(18.7, 20.2), pt(19.4, 19.2), pt(18.8, 18))
        P.lineTo(pt(13.7, 9.6))
        P.lineTo(pt(13.7, 4))
        P.moveTo(pt(7.4, 14.6))
        P.lineTo(pt(16.6, 14.6))
    elif name == "inbox":
        P.moveTo(pt(3.8, 13))
        P.lineTo(pt(6.5, 5.5))
        P.lineTo(pt(17.5, 5.5))
        P.lineTo(pt(20.2, 13))
        P.lineTo(pt(20.2, 18.5))
        P.lineTo(pt(3.8, 18.5))
        P.closeSubpath()
        P.moveTo(pt(3.8, 13))
        P.lineTo(pt(8.5, 13))
        P.lineTo(pt(9.8, 15.4))
        P.lineTo(pt(14.2, 15.4))
        P.lineTo(pt(15.5, 13))
        P.lineTo(pt(20.2, 13))
    elif name == "book":
        P.moveTo(pt(12, 6.5))
        P.cubicTo(pt(9.5, 4.8), pt(6.5, 4.6), pt(3.8, 5.4))
        P.lineTo(pt(3.8, 18.6))
        P.cubicTo(pt(6.5, 17.8), pt(9.5, 18), pt(12, 19.6))
        P.cubicTo(pt(14.5, 18), pt(17.5, 17.8), pt(20.2, 18.6))
        P.lineTo(pt(20.2, 5.4))
        P.cubicTo(pt(17.5, 4.6), pt(14.5, 4.8), pt(12, 6.5))
        P.lineTo(pt(12, 19.6))
    elif name == "chat":
        P.moveTo(pt(5.5, 5))
        P.lineTo(pt(18.5, 5))
        P.cubicTo(pt(19.6, 5), pt(20.5, 5.9), pt(20.5, 7))
        P.lineTo(pt(20.5, 14.5))
        P.cubicTo(pt(20.5, 15.6), pt(19.6, 16.5), pt(18.5, 16.5))
        P.lineTo(pt(10, 16.5))
        P.lineTo(pt(6, 20))
        P.lineTo(pt(6, 16.5))
        P.lineTo(pt(5.5, 16.5))
        P.cubicTo(pt(4.4, 16.5), pt(3.5, 15.6), pt(3.5, 14.5))
        P.lineTo(pt(3.5, 7))
        P.cubicTo(pt(3.5, 5.9), pt(4.4, 5), pt(5.5, 5))
    elif name == "drop":                   # glisser-déposer
        P.moveTo(pt(12, 4))
        P.lineTo(pt(12, 14.5))
        P.moveTo(pt(7.5, 10.2))
        P.lineTo(pt(12, 14.7))
        P.lineTo(pt(16.5, 10.2))
        P.moveTo(pt(4.5, 15))
        P.lineTo(pt(4.5, 19.5))
        P.lineTo(pt(19.5, 19.5))
        P.lineTo(pt(19.5, 15))
    else:                                  # point
        P.addEllipse(QPointF(12 * u, 12 * u), 3 * u, 3 * u)
        fill = True
    return P, fill


ICONS = ("home", "camera", "video", "pause", "stop", "record", "photo", "gear", "folder", "spark", "help",
         "chevron", "arrow", "check", "moon", "bars", "images", "flask", "inbox", "book", "chat", "drop")


def icon_pixmap(name: str, color: str, size: int = 18, dpr: float = 2.0, stroke: float = 1.8) -> QPixmap:
    pm = QPixmap(round(size * dpr), round(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path, fill = _icon_path(name, float(size))
    c = QColor(color)
    if fill:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawPath(path)
    else:
        pen = QPen(c, stroke * size / 24.0 * 1.15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    p.end()
    return pm


def line_icon(name: str, color: str = FOREST, size: int = 18, disabled: str | None = None) -> QIcon:
    ic = QIcon()
    ic.addPixmap(icon_pixmap(name, color, size), QIcon.Mode.Normal)
    ic.addPixmap(icon_pixmap(name, disabled or rgba_hex(color, 0.4), size), QIcon.Mode.Disabled)
    return ic


def rgba_hex(c: str, a: float) -> str:
    q = QColor(c)
    return f"#{int(round(a * 255)):02x}{q.red():02x}{q.green():02x}{q.blue():02x}"
