"""Écran de chargement aux couleurs du Cégep Marie-Victorin.

Fenêtre sans bordure (≈ 640 × 400) vert forêt aux coins arrondis :
1. les 4 pilules du symbole montent l'une après l'autre (révélation par découpe, 1,5 s, décalage 120 ms,
   courbe du site cubic-bezier(0.215, 0.61, 0.355, 1)) ;
2. le logo (texte « CÉGEP MARIE-VICTORIN » blanc) apparaît en fondu ;
3. « WilsonTrack » en menthe, puis une ligne d'état (« Chargement de l'IA de référence… ») et un fin trait
   de progression menthe.

Utilisé par app.run_gui avant la construction de la fenêtre principale (jamais par l'auto-test ni par
tools/gui_smoke.py). Le temps d'animation est mesuré en temps réel : si l'application est occupée, l'image
saute simplement à l'instant courant. Un clic passe l'animation. Aucune exception ne doit sortir d'ici.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QEventLoop, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPainterPath
from PySide6.QtWidgets import QApplication, QWidget

from .. import APP_NAME, __version__
from . import brand as B

W, H = 640, 400
SCENE_MS = 1300                      # la chambre « s'allume » seule avant que le logo n'arrive
PILL_MS, STAGGER_MS = 1500, 120
PILL_AT = SCENE_MS - 250
LOGO_AT, LOGO_MS = PILL_AT + 650, 650          # fondu du logo complet (texte blanc)
NAME_AT, NAME_MS = PILL_AT + 1050, 600         # « WilsonTrack »
STATUS_AT = PILL_AT + 1300
FLASH_AT = PILL_AT - 60              # le muon qui traverse le centre : éclair derrière le logo
FADE_OUT_MS = 320
MIN_TOTAL_S = 3.6

# Couleurs des traces (mêmes teintes que les classes de l'application, en plus lumineux sur fond sombre)
C_ALPHA, C_BETA, C_MUON, C_DELTA = "#FF6B5E", "#4FB2FF", "#9DF5A3", "#C9A2FF"
TRAIL_MS = 1250                      # durée d'évaporation d'une trace après son passage


def _build_tracks(seed: int = 1907) -> list[dict]:
    """Scénario fixe de traces façon chambre de Wilson (alpha en étoile, bêta sinueux, muons, rayon delta)."""
    import math
    import random
    rng = random.Random(seed)
    tracks: list[dict] = []

    def add(pts, t0, dur, color, width):
        tracks.append({"pts": [QPointF(x, y) for x, y in pts], "t0": float(t0), "dur": float(dur),
                       "color": QColor(color), "w": float(width)})

    # 1. muons : longs traits rectilignes qui traversent toute la chambre
    for t0, (x0, y0, x1, y1) in ((120, (-20, 70, 660, 250)), (900, (520, -20, 230, 420)),
                                 (1850, (-20, 330, 660, 120)), (2700, (90, -20, 360, 420))):
        add([(x0 + (x1 - x0) * k / 40, y0 + (y1 - y0) * k / 40) for k in range(41)], t0, 300, C_MUON, 1.4)
    # 2. le muon « héros » qui passe derrière le logo au moment où il apparaît
    add([(-20 + 680 * k / 50, 236 - 90 * k / 50) for k in range(51)], FLASH_AT, 260, "#FFFFFF", 2.0)

    # 3. étoile alpha : 5 traces épaisses et courtes qui partent d'une même source radioactive
    for cx, cy, t0 in ((508, 96, 380), (122, 318, 1500), (560, 330, 2350)):
        n = rng.choice((4, 5))
        a0 = rng.uniform(0, math.tau)
        for i in range(n):
            ang = a0 + i * math.tau / n + rng.uniform(-0.35, 0.35)
            length = rng.uniform(55, 95)
            add([(cx + math.cos(ang) * length * k / 14, cy + math.sin(ang) * length * k / 14) for k in range(15)],
                t0 + i * 45, 420, C_ALPHA, 4.2)

    # 4. bêta : électrons légers, trajectoires sinueuses (diffusion multiple)
    for x, y, t0 in ((40, 160, 650), (600, 210, 1150), (300, 380, 1700), (200, 40, 2200), (470, 360, 2900)):
        ang = math.atan2(200 - y, 320 - x) + rng.uniform(-0.6, 0.6)
        pts = [(x, y)]
        for _ in range(70):
            ang += rng.gauss(0, 0.17)
            x += math.cos(ang) * 4.2
            y += math.sin(ang) * 4.2
            pts.append((x, y))
        add(pts, t0, 850, C_BETA, 1.7)

    # 5. rayon delta : petite spirale qui se resserre (électron arraché qui perd son énergie)
    for cx, cy, t0 in ((420, 290, 1300), (190, 120, 2550)):
        pts = []
        for k in range(90):
            r = 26 * (1 - k / 95)
            a = k * 0.32
            pts.append((cx + r * math.cos(a) + k * 0.6, cy + r * math.sin(a)))
        add(pts, t0, 700, C_DELTA, 1.5)
    return tracks


def _build_mist(seed: int = 42, n: int = 90) -> list[tuple[float, float, float, float]]:
    """Gouttelettes de brume (x, y, rayon, vitesse de chute)."""
    import random
    rng = random.Random(seed)
    return [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.5, 1.6), rng.uniform(4, 14)) for _ in range(n)]


class SplashScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setFixedSize(W, H)
        self.setWindowTitle(APP_NAME)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Cliquez pour passer")
        self._t0 = time.monotonic()
        self._t_override: float | None = None
        self._status = "Démarrage…"
        self._progress = 0.0
        self._shown_progress = 0.0
        self._opacity = 1.0
        self._skip = False
        self._timer = QTimer(self)
        self._timer.setInterval(16)                 # ≈ 60 images/s au plus
        self._timer.timeout.connect(self._tick)
        self._logo = None
        self._tracks = _build_tracks()
        self._mist = _build_mist()

    # ------------------------------------------------------------------ cycle de vie --
    def start(self):
        scr = QGuiApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            self.move(g.center().x() - W // 2, g.center().y() - H // 2)
        self._t0 = time.monotonic()
        self.show()
        self.raise_()
        if not B.reduced_motion():
            self._timer.start()
        self.pump()

    def elapsed_ms(self) -> float:
        if self._t_override is not None:
            return self._t_override
        if self._skip or B.reduced_motion():
            return 1e6
        return (time.monotonic() - self._t0) * 1000.0

    def set_time(self, ms: float):
        """Fige l'animation à l'instant `ms` (aperçus et tests)."""
        self._t_override = float(ms)
        self._shown_progress = self._progress
        self.update()

    def set_status(self, text: str, progress: float | None = None):
        self._status = str(text)
        if progress is not None:
            self._progress = max(self._progress, min(1.0, float(progress)))
        self.update()

    def pump(self):
        """Laisse Qt redessiner pendant un chargement (sans traiter les clics de l'application)."""
        try:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 30)
        except Exception:
            pass

    def finish(self, window: QWidget, min_total_s: float = MIN_TOTAL_S, timeout_s: float = 8.0):
        """Attend la fin de l'animation (au moins `min_total_s`), affiche `window`, puis s'efface en fondu."""
        self.set_status("Prêt", 1.0)
        start = time.monotonic()
        while not self._skip and (time.monotonic() - self._t0) < min_total_s:
            if time.monotonic() - start > timeout_s:
                break
            self.pump()
            time.sleep(0.008)
        window.show()
        window.raise_()
        window.activateWindow()
        if B.reduced_motion():
            self.close()
            return
        t1 = time.monotonic()
        while (time.monotonic() - t1) * 1000.0 < FADE_OUT_MS:
            self._opacity = max(0.0, 1.0 - (time.monotonic() - t1) * 1000.0 / FADE_OUT_MS)
            self.setWindowOpacity(self._opacity)
            self.pump()
            time.sleep(0.008)
        self._timer.stop()
        self.close()

    def mousePressEvent(self, e):
        self._skip = True
        self.update()
        e.accept()

    def _tick(self):
        # le trait de progression rattrape doucement la valeur demandée
        self._shown_progress += (self._progress - self._shown_progress) * 0.18
        self.update()

    # ------------------------------------------------------------------ dessin --
    def paintEvent(self, _e):
        try:
            self._paint()
        except Exception:
            pass

    def _paint_chamber(self, p: QPainter, t: float):
        """Fond de chambre de Wilson : obscurité, éclairage latéral, brume qui tombe, traces qui s'évaporent."""
        from PySide6.QtGui import QLinearGradient, QPen, QRadialGradient
        # l'obscurité de la chambre s'ouvre doucement sur le vert du Cégep quand le logo arrive
        dark = 1.0 - 0.5 * B.ease((t - LOGO_AT) / 900.0)
        g = QLinearGradient(0, 0, 0, H)
        g.setColorAt(0.0, QColor(5, 16, 11, int(255 * dark)))
        g.setColorAt(1.0, QColor(10, 34, 24, int(255 * dark)))
        p.fillRect(QRectF(0, 0, W, H), g)
        # éclairage rasant venant de la gauche (comme la lampe d'une vraie chambre)
        lamp = QRadialGradient(QPointF(-60, H * 0.62), W * 0.9)
        lamp.setColorAt(0.0, QColor(152, 242, 157, 46))
        lamp.setColorAt(1.0, QColor(152, 242, 157, 0))
        p.fillRect(QRectF(0, 0, W, H), lamp)
        # brume : gouttelettes qui tombent lentement
        p.setPen(Qt.PenStyle.NoPen)
        ts = min(t, 60000.0) / 1000.0
        for x, y, r, v in self._mist:
            yy = (y + v * ts) % H
            a = int(30 + 60 * r / 1.6)
            p.setBrush(QColor(220, 255, 230, a))
            p.drawEllipse(QPointF(x, yy), r, r)
        # éclair derrière le logo au passage du muon central
        fl = (t - FLASH_AT) / 700.0
        if 0.0 < fl < 1.0:
            a = (1.0 - fl) ** 2 * min(1.0, fl * 6)
            glow = QRadialGradient(QPointF(W / 2, H * 0.42), W * 0.55)
            glow.setColorAt(0.0, QColor(200, 255, 210, int(120 * a)))
            glow.setColorAt(1.0, QColor(200, 255, 210, 0))
            p.fillRect(QRectF(0, 0, W, H), glow)
        # traces
        p.setBrush(Qt.BrushStyle.NoBrush)
        for tr in self._tracks:
            local = t - tr["t0"]
            if local <= 0:
                continue
            dur = tr["dur"]
            k = min(1.0, local / dur)
            fade = 1.0 if local < dur else 1.0 - (local - dur) / TRAIL_MS
            if fade <= 0.0:
                continue
            fade = fade * fade
            pts = tr["pts"]
            pos = k * (len(pts) - 1)
            n = int(pos)
            path = QPainterPath(pts[0])
            for q in pts[1:n + 1]:
                path.lineTo(q)
            head = pts[n]
            if n + 1 < len(pts):
                f = pos - n
                a, b = pts[n], pts[n + 1]
                head = QPointF(a.x() + (b.x() - a.x()) * f, a.y() + (b.y() - a.y()) * f)
                path.lineTo(head)
            # la trace s'élargit en s'évaporant (la brume diffuse)
            w = tr["w"] * (1.0 + (1.0 - fade) * 1.6)
            col = QColor(tr["color"])
            for width_k, alpha in ((6.0, 0.08), (3.0, 0.22), (1.0, 0.95)):
                c = QColor(col)
                c.setAlphaF(max(0.0, min(1.0, alpha * fade)))
                pen = QPen(c, w * width_k)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                p.drawPath(path)
            if k < 1.0:
                # cœur blanc brûlant + tête lumineuse tant que la particule avance
                core = QPen(QColor(255, 255, 255, 200), max(0.8, tr["w"] * 0.35))
                core.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(core)
                p.drawPath(path)
                rg = QRadialGradient(head, tr["w"] * 6 + 6)
                hc = QColor(col)
                hc.setAlpha(230)
                rg.setColorAt(0.0, QColor(255, 255, 255, 255))
                rg.setColorAt(0.25, hc)
                hc.setAlpha(0)
                rg.setColorAt(1.0, hc)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(rg)
                p.drawEllipse(head, tr["w"] * 6 + 6, tr["w"] * 6 + 6)
                p.setBrush(Qt.BrushStyle.NoBrush)
        # halo sombre derrière le logo pour qu'il reste lisible par-dessus les traces
        a_logo = B.ease((t - PILL_AT) / 600.0)
        if a_logo > 0:
            v = QRadialGradient(QPointF(W / 2, 175), 250)
            v.setColorAt(0.0, QColor(8, 30, 20, int(170 * a_logo)))
            v.setColorAt(1.0, QColor(8, 30, 20, 0))
            p.fillRect(QRectF(0, 0, W, H), v)
        # vignette
        vg = QRadialGradient(QPointF(W / 2, H / 2), W * 0.72)
        vg.setColorAt(0.6, QColor(0, 0, 0, 0))
        vg.setColorAt(1.0, QColor(0, 0, 0, 120))
        p.fillRect(QRectF(0, 0, W, H), vg)
        p.setPen(Qt.PenStyle.NoPen)

    def _paint(self):
        t = self.elapsed_ms()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(0, 0, W, H)
        card = QPainterPath()
        card.addRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 26, 26)
        p.fillPath(card, QColor(B.FOREST))
        p.setClipPath(card)
        # ---- 0. la chambre de Wilson : fond sombre, brume, traces de particules ----
        self._paint_chamber(p, t)
        # ---- 1. les 4 pilules montent l'une après l'autre (révélation par découpe) ----
        sym_h = 132.0
        sym_w = sym_h * B.SYMBOL_W / B.SYMBOL_H
        sx, sy = (W - sym_w) / 2, 70.0
        s = sym_h / B.SYMBOL_H
        from PySide6.QtGui import QTransform
        tr = QTransform(s, 0, 0, s, sx, sy)
        mint = QColor(B.MINT)
        for i, pill in enumerate(B.symbol_pills()):
            k = B.ease((t - PILL_AT - i * STAGGER_MS) / PILL_MS)
            if k <= 0:
                continue
            shape = tr.map(pill)
            box = shape.boundingRect()
            p.save()
            p.setClipRect(box.adjusted(-2, -2, 2, 2), Qt.ClipOperation.IntersectClip)
            p.translate(0, (1.0 - k) * (box.height() + 4))
            p.fillPath(shape, mint)
            p.restore()
        # ---- 2. logo complet (texte blanc) en fondu, à la place du symbole seul ----
        a_logo = B.ease((t - LOGO_AT) / LOGO_MS)
        y_text = sy + sym_h + 30
        if a_logo > 0:
            if self._logo is None:
                # texte « CÉGEP MARIE-VICTORIN » du logo officiel (le symbole est déjà dessiné au-dessus)
                dpr = max(2.0, self.devicePixelRatioF())
                full = B.svg_pixmap(B.LOGO_DARK_BG, 40, dpr)
                x0 = int(full.width() * 104.0 / 354.0)
                self._logo = full.copy(x0, 0, full.width() - x0, full.height())
                self._logo.setDevicePixelRatio(dpr)
            lw = self._logo.width() / self._logo.devicePixelRatio()
            p.setOpacity(a_logo)
            p.drawPixmap(QPointF((W - lw) / 2, y_text + (1 - a_logo) * 8), self._logo)
            p.setOpacity(1.0)
        # ---- 3. « WilsonTrack » en menthe ----
        a_name = B.ease((t - NAME_AT) / NAME_MS)
        if a_name > 0:
            f = QFont(self.font())
            fam = B.ui_font_family()
            if fam:
                f.setFamily(fam)
            f.setPixelSize(30)
            f.setWeight(QFont.Weight.Bold)
            f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 98.0)
            p.setFont(f)
            c = QColor(B.MINT)
            c.setAlphaF(a_name)
            p.setPen(c)
            p.drawText(QRectF(0, y_text + 48 + (1 - a_name) * 10, W, 40), int(Qt.AlignmentFlag.AlignHCenter), APP_NAME)
        # ---- 4. ligne d'état + trait de progression ----
        a_st = B.ease((t - STATUS_AT) / 500.0) if t < 1e5 else 1.0
        if a_st > 0:
            f = QFont(self.font())
            fam = B.ui_font_family()
            if fam:
                f.setFamily(fam)
            f.setPixelSize(12)
            p.setFont(f)
            c = QColor(255, 255, 255, int(200 * a_st))
            p.setPen(c)
            p.drawText(QRectF(36, H - 58, W - 72, 18), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                       self._status)
            c2 = QColor(255, 255, 255, int(150 * a_st))
            p.setPen(c2)
            p.drawText(QRectF(36, H - 58, W - 72, 18),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       f"v{__version__} · Sciences de la nature")
            track = QRectF(36, H - 32, W - 72, 3)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, int(38 * a_st)))
            p.drawRoundedRect(track, 1.5, 1.5)
            prog = self._shown_progress if self._t_override is None else self._progress
            prog = max(prog, min(1.0, t / 2400.0) * 0.35)
            if prog > 0:
                c3 = QColor(B.MINT)
                c3.setAlphaF(a_st)
                p.setBrush(c3)
                p.drawRoundedRect(QRectF(track.left(), track.top(), track.width() * min(1.0, prog), 3), 1.5, 1.5)
        p.end()
