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
PILL_MS, STAGGER_MS = 1500, 120
LOGO_AT, LOGO_MS = 650, 650          # fondu du logo complet (texte blanc)
NAME_AT, NAME_MS = 1050, 600         # « WilsonTrack »
STATUS_AT = 1300
FADE_OUT_MS = 260


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

    def finish(self, window: QWidget, min_total_s: float = 1.9, timeout_s: float = 6.0):
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
        # ---- 1. les 4 pilules montent l'une après l'autre (révélation par découpe) ----
        sym_h = 132.0
        sym_w = sym_h * B.SYMBOL_W / B.SYMBOL_H
        sx, sy = (W - sym_w) / 2, 70.0
        s = sym_h / B.SYMBOL_H
        from PySide6.QtGui import QTransform
        tr = QTransform(s, 0, 0, s, sx, sy)
        mint = QColor(B.MINT)
        for i, pill in enumerate(B.symbol_pills()):
            k = B.ease((t - i * STAGGER_MS) / PILL_MS)
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
