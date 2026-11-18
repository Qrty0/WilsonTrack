"""Widgets réutilisables : vue vidéo interactive, compteurs, graphiques matplotlib."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout,
                               QWidget)

from ..physics import CLASS_INFO
from . import brand as B


def bgr_to_qimage(img: np.ndarray) -> QImage:
    """Convertit une image OpenCV (BGR ou niveaux de gris) en QImage indépendante."""
    if img is None:
        return QImage()
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8).copy()
    h, w, _ = img.shape
    return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888).copy()


class VideoView(QWidget):
    """Affiche l'image courante (proportions conservées) et permet de dessiner.

    Modes d'interaction :
    - « roi_rect »  : glisser pour tracer une zone rectangulaire ;
    - « roi_poly »  : clics successifs, double-clic pour fermer le polygone ;
    - « calib »     : deux clics sur une règle filmée → distance en pixels ;
    - « source »    : un clic pour indiquer la position de l'échantillon.
    Les coordonnées émises sont normalisées (0..1) par rapport à l'image source.
    """

    roi_defined = Signal(list)          # [[x, y], ...] normalisés
    calib_defined = Signal(float)       # distance en px de l'image native
    point_clicked = Signal(float, float)
    mode_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._img: QImage | None = None
        self._native_wh = (1, 1)
        self.mode = "none"
        self._pts: list[QPointF] = []
        self._drag_start: QPointF | None = None
        self._mouse: QPointF | None = None
        self.placeholder = "Aucune image — choisissez une source (caméra, vidéo, photos ou démonstration)."
        self.setAutoFillBackground(False)

    # ------------------------------------------------------------------------
    def set_image(self, img: QImage, native_wh: tuple[int, int] | None = None):
        self._img = img
        if native_wh:
            self._native_wh = native_wh
        self.update()

    def current_image(self) -> QImage | None:
        return self._img

    def set_mode(self, mode: str):
        self.mode = mode
        self._pts = []
        self._drag_start = None
        self.setCursor(Qt.CursorShape.CrossCursor if mode != "none" else Qt.CursorShape.ArrowCursor)
        self.update()

    def _target_rect(self) -> QRectF:
        if self._img is None or self._img.isNull():
            return QRectF(self.rect())
        iw, ih = self._img.width(), self._img.height()
        W, H = self.width(), self.height()
        s = min(W / iw, H / ih)
        w, h = iw * s, ih * s
        return QRectF((W - w) / 2, (H - h) / 2, w, h)

    def _to_norm(self, p: QPointF) -> tuple[float, float]:
        r = self._target_rect()
        x = (p.x() - r.x()) / max(r.width(), 1)
        y = (p.y() - r.y()) / max(r.height(), 1)
        return float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))

    # ------------------------------------------------------------------ paint --
    def paintEvent(self, _ev):
        # L'image de la caméra reste sombre (fond #0B1712) dans un cadre arrondi, sur le fond clair de l'onglet.
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        frame = QPainterPath()
        frame.addRoundedRect(QRectF(self.rect()), 16, 16)
        p.fillPath(frame, QColor(B.VIDEO_BG))
        p.setClipPath(frame)
        if self._img is None or self._img.isNull():
            p.setPen(QColor(178, 196, 186))
            p.drawText(QRectF(self.rect()).adjusted(30, 0, -30, 0),
                       int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.placeholder)
            return
        r = self._target_rect()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(r, self._img)
        if self.mode == "none":
            return
        pen = QPen(QColor(255, 210, 0), 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        if self.mode == "roi_rect" and self._drag_start is not None and self._mouse is not None:
            p.drawRect(QRectF(self._drag_start, self._mouse).normalized())
        elif self.mode in ("roi_poly", "calib") and self._pts:
            pts = list(self._pts) + ([self._mouse] if self._mouse is not None else [])
            p.drawPolyline(QPolygonF(pts))
            p.setBrush(QBrush(QColor(255, 210, 0)))
            for q in self._pts:
                p.drawEllipse(q, 4, 4)
        hint = {"roi_rect": "Glissez pour tracer la zone à analyser (Échap pour annuler)",
                "roi_poly": "Cliquez les sommets de la zone, double-cliquez pour terminer (Échap pour annuler)",
                "calib": "Cliquez les deux extrémités d'une longueur connue sur la règle (Échap pour annuler)",
                "source": "Cliquez sur la position de l'échantillon (Échap pour annuler)"}.get(self.mode, "")
        p.setPen(QColor(0, 0, 0))
        p.setBrush(QBrush(QColor(255, 210, 0, 220)))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(hint) + 16
        p.drawRect(QRectF(8, 8, w, fm.height() + 10))
        p.drawText(QRectF(8, 8, w, fm.height() + 10), Qt.AlignmentFlag.AlignCenter, hint)

    # ------------------------------------------------------------------ souris --
    def mousePressEvent(self, ev):
        if self.mode == "none" or ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = ev.position()
        if self.mode == "roi_rect":
            self._drag_start = pos
        elif self.mode == "roi_poly":
            self._pts.append(pos)
        elif self.mode == "calib":
            self._pts.append(pos)
            if len(self._pts) == 2:
                (x0, y0), (x1, y1) = self._to_norm(self._pts[0]), self._to_norm(self._pts[1])
                W, H = self._native_wh
                d = float(np.hypot((x1 - x0) * W, (y1 - y0) * H))
                self.set_mode("none")
                self.calib_defined.emit(d)
                self.mode_finished.emit()
        elif self.mode == "source":
            x, y = self._to_norm(pos)
            self.set_mode("none")
            self.point_clicked.emit(x, y)
            self.mode_finished.emit()
        self.update()

    def mouseMoveEvent(self, ev):
        self._mouse = ev.position()
        if self.mode != "none":
            self.update()

    def mouseReleaseEvent(self, ev):
        if self.mode == "roi_rect" and self._drag_start is not None:
            a, b = self._to_norm(self._drag_start), self._to_norm(ev.position())
            self._drag_start = None
            x0, x1 = sorted((a[0], b[0]))
            y0, y1 = sorted((a[1], b[1]))
            self.set_mode("none")
            if (x1 - x0) > 0.05 and (y1 - y0) > 0.05:
                self.roi_defined.emit([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            self.mode_finished.emit()

    def mouseDoubleClickEvent(self, ev):
        if self.mode == "roi_poly" and len(self._pts) >= 3:
            pts = [list(self._to_norm(q)) for q in self._pts]
            self.set_mode("none")
            self.roi_defined.emit(pts)
            self.mode_finished.emit()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape and self.mode != "none":
            self.set_mode("none")
            self.mode_finished.emit()
        else:
            super().keyPressEvent(ev)

    def sizeHint(self):
        return QSize(960, 540)


class CounterTile(QFrame):
    """Compteur d'une classe : carte blanche avec une barre de couleur (couleur physique de la classe) à gauche."""

    def __init__(self, cls: str, parent=None):
        super().__init__(parent)
        info = CLASS_INFO[cls]
        self.setObjectName("tile")
        self.setStyleSheet(f"QFrame#tile {{ border: 1px solid {B.BORDER}; border-left: 5px solid {info['hex']};"
                           f" border-radius: 12px; background: #ffffff; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        self.name = QLabel(f"{info['symbole']}  {info['nom']}")
        self.name.setStyleSheet(f"color: {B.class_text_hex(cls)}; font-weight: 700; font-size: 12px;")
        self.name.setMinimumWidth(40)
        self.name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.name.setToolTip(info["nom"])
        self.count = QLabel("0")
        self.count.setStyleSheet(f"color: {B.INK}; font-size: 26px; font-weight: 700;")
        self.rate = QLabel("— /min")
        self.rate.setStyleSheet(f"color: {B.MUTED}; font-size: 12px;")
        lay.addWidget(self.name)
        lay.addWidget(self.count)
        lay.addWidget(self.rate)
        self.setToolTip(info["description"])

    def set_values(self, n: int, rate: float | None, err: float | None):
        self.count.setText(str(n))
        if rate is None:
            self.rate.setText("— /min")
        else:
            self.rate.setText(f"{rate:.2f} ± {err:.2f} /min")


class CounterPanel(QWidget):
    CLASSES = ["alpha", "beta", "muon", "proton", "fine", "incertain"]

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.tiles = {}
        for i, c in enumerate(self.CLASSES):
            t = CounterTile(c)
            self.tiles[c] = t
            lay.addWidget(t, i // 2, i % 2)
        self.total = QLabel("Total : 0")
        self.total.setStyleSheet(f"color: {B.FOREST}; font-size: 15px; font-weight: 700; padding: 4px 6px;")
        self.total.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(self.total, (len(self.CLASSES) + 1) // 2, 0, 1, 2)

    def update_counts(self, counts: dict, duration_s: float):
        tmin = duration_s / 60.0
        tot = 0
        for c, t in self.tiles.items():
            n = counts.get(c, 0)
            tot += n
            if tmin > 0.05:
                t.set_values(n, n / tmin, (max(n, 1) ** 0.5) / tmin)
            else:
                t.set_values(n, None, None)
        rate = f" — {tot / tmin:.1f} /min" if tmin > 0.05 else ""
        self.total.setText(f"Total : {tot}{rate}")


class MplCanvas(QWidget):
    """Conteneur pour une figure matplotlib (remplaçable)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self.msg = QLabel("")
        self.msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet(f"color: {B.MUTED}; background: #ffffff; border: 1px solid {B.BORDER};"
                               f" border-radius: 14px; padding: 18px;")
        self.lay.addWidget(self.msg)

    def set_figure(self, fig):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        if self.canvas is not None:
            self.lay.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        self.msg.hide()
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setMinimumHeight(260)
        self.lay.addWidget(self.canvas)
        self.canvas.draw_idle()

    def set_message(self, text: str):
        self.msg.setText(text)
        self.msg.show()


class Callout(QLabel):
    COLORS = {"succes": "#30a46c", "attention": "#f76b15", "info": "#2f9bff"}

    def __init__(self, title: str, text: str, level: str = "info"):
        super().__init__()
        col = self.COLORS.get(level, "#2f9bff")
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(f"<b style='color:{B.readable_on(col, B.SOFT)}'>{title}</b><br>{text}")
        self.setStyleSheet(f"QLabel {{ border-left: 4px solid {col}; background: {B.SOFT}; color: {B.INK};"
                           f" padding: 10px; border-radius: 8px; }}")


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {B.BORDER};")
    return f


def hbox(*widgets, stretch_last=False) -> QHBoxLayout:
    lay = QHBoxLayout()
    for w in widgets:
        if isinstance(w, int):
            lay.addStretch(w)
        else:
            lay.addWidget(w)
    if stretch_last:
        lay.addStretch(1)
    return lay
