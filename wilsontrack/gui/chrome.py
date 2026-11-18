"""Habillage de la fenêtre principale : en-tête vert forêt du Cégep et barre d'onglets.

- BrandHeader : bandeau vert forêt #154A34 avec le logo officiel (symbole menthe, texte blanc), le nom
  « WilsonTrack » et le sous-titre menthe, les menus (Fichier / Outils / Aide) puis, à droite, seulement les
  actions essentielles. Les boutons reflètent l'état de leur QAction (activée, cochée) ; quand la fenêtre est
  étroite, le sous-titre puis le texte des boutons sont masqués (icône + infobulle).
- BrandTabs : QTabWidget dont setCurrentWidget() sait (1) réafficher un onglet masqué « jusqu'à utilisation »
  (Lot, Assistant IA) et (2) atteindre un sous-onglet (Graphiques, rangé dans Conclusions).
- TabShell : dessine la bande blanche derrière les onglets (sous l'en-tête).
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QTabWidget, QToolButton, QVBoxLayout,
                               QWidget)

from .. import APP_NAME
from . import brand as B


# ============================================================================ en-tête ====
def _header_icon(name: str, checked_color: str | None = None) -> QIcon:
    ic = QIcon()
    ic.addPixmap(B.icon_pixmap(name, "#ffffff", 18), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(B.icon_pixmap(name, "#ffffff", 18), QIcon.Mode.Active, QIcon.State.Off)
    ic.addPixmap(B.icon_pixmap(name, "#6f9a86", 18), QIcon.Mode.Disabled, QIcon.State.Off)
    on = checked_color or "#ffffff"
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
        ic.addPixmap(B.icon_pixmap(name, on, 18), mode, QIcon.State.On)
    return ic


class HeaderButton(QToolButton):
    """Bouton de l'en-tête lié à une QAction existante (texte, infobulle, état activé / coché)."""

    def __init__(self, action, icon: str, label: str | None = None, parent=None):
        super().__init__(parent)
        self.action_ref = action
        self.label = label
        self.setObjectName("headerBtn")
        self.setProperty("icon_name", icon)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setIconSize(QSize(18, 18))
        role = "toggle" if icon == "gear" else ("record" if icon == "record" else "")
        if role:
            self.setProperty("role", role)
        self.setIcon(_header_icon(icon, B.FOREST if role == "toggle" else None))
        self.setCheckable(action.isCheckable())
        self.clicked.connect(lambda _c=False: action.trigger())
        action.changed.connect(self._sync)
        self._sync()

    def _sync(self):
        a = self.action_ref
        try:
            self.setText(self.label or a.text().replace("&", ""))
            tip = a.toolTip() or a.text()
            sc = a.shortcut().toString()
            self.setToolTip(f"{tip}  ({sc})" if sc else tip)
            self.setAccessibleName(a.text().replace("&", ""))
            self.setEnabled(a.isEnabled())
            if a.isCheckable():
                self.setChecked(a.isChecked())
        except RuntimeError:                       # action détruite à la fermeture
            pass


class BrandHeader(QWidget):
    """Bandeau vert forêt : logo du Cégep, nom de l'application, menus, actions essentielles."""

    def __init__(self, menubar, actions: list, parent=None):
        super().__init__(parent)
        self.setObjectName("BrandHeader")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 10, 12, 10)
        lay.setSpacing(2)
        self.logo = QLabel()
        self.logo.setObjectName("brandLogo")
        self.logo.setPixmap(B.svg_pixmap(B.LOGO_DARK_BG, 29, 2.0))
        self.logo.setAccessibleName("Cégep Marie-Victorin")
        self.logo.setToolTip("Cégep Marie-Victorin")
        lay.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(12)
        sep = QFrame()
        sep.setObjectName("brandSep")
        sep.setFixedSize(1, 30)
        lay.addWidget(sep, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(12)
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        self.title = QLabel(APP_NAME)
        self.title.setObjectName("brandTitle")
        self.subtitle = QLabel("Détection des particules")
        self.subtitle.setObjectName("brandSubtitle")
        tcol.addWidget(self.title)
        tcol.addWidget(self.subtitle)
        self.title_box = QWidget()
        self.title_box.setLayout(tcol)
        tcol.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.title_box, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(10)
        self.menubar = menubar
        menubar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lay.addWidget(menubar, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        self.buttons: list[HeaderButton] = []
        self._seps: list[QFrame] = []
        for item in actions:
            if item is None:
                s = QFrame()
                s.setObjectName("brandSep")
                s.setFixedSize(1, 24)
                lay.addSpacing(4)
                lay.addWidget(s, 0, Qt.AlignmentFlag.AlignVCenter)
                lay.addSpacing(4)
                self._seps.append(s)
                continue
            action, icon, *label = item
            b = HeaderButton(action, icon, label[0] if label else None)
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignVCenter)
            self.buttons.append(b)
        self._mode = -1
        self.setMinimumHeight(60)

    def button_for(self, action) -> HeaderButton | None:
        return next((b for b in self.buttons if b.action_ref is action), None)

    COMPACT_FIRST = ("home", "pause", "stop", "photo")      # icônes assez claires pour se passer de texte

    def _apply_mode(self, mode: int):
        """0 = tout ; 1 = Accueil / Pause / Arrêter / Photo en icônes seules ; 2 = sans le nom de l'application ;
        3 = toutes les actions en icônes seules (sauf Réglages). Le texte reste dans l'infobulle."""
        if mode == self._mode:
            return
        self._mode = mode
        self.title_box.setVisible(mode < 2)
        for b in self.buttons:
            if b.property("role") == "toggle":
                keep_text = True
            elif mode >= 3:
                keep_text = False
            else:
                keep_text = mode < 1 or b.property("icon_name") not in self.COMPACT_FIRST
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon if keep_text
                                 else Qt.ToolButtonStyle.ToolButtonIconOnly)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def showEvent(self, e):
        super().showEvent(e)
        self._fit()

    def _fit(self):
        for mode in (0, 1, 2, 3):
            self._apply_mode(mode)
            self.layout().activate()
            if self.layout().minimumSize().width() <= self.width() or mode == 3:
                break


# ============================================================================ onglets ====
class BrandTabs(QTabWidget):
    """Onglets principaux ; setCurrentWidget() gère les onglets masqués et les sous-onglets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._nested: dict = {}            # widget enfant -> (onglet parent, QTabWidget interne)
        self.tabBar().setDrawBase(False)
        self.tabBar().setExpanding(False)
        self.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)

    def hide_until_used(self, w: QWidget):
        i = self.indexOf(w)
        if i >= 0:
            self.setTabVisible(i, False)

    def register_nested(self, child: QWidget, top: QWidget, inner: QTabWidget):
        self._nested[child] = (top, inner)

    def setCurrentWidget(self, w: QWidget):  # noqa: N802 (API Qt)
        if w in self._nested:
            top, inner = self._nested[w]
            inner.setCurrentWidget(w)
            w = top
        i = self.indexOf(w)
        if i >= 0 and not self.isTabVisible(i):
            self.setTabVisible(i, True)
        super().setCurrentWidget(w)

    def setCurrentIndex(self, i: int):  # noqa: N802
        if 0 <= i < self.count() and not self.isTabVisible(i):
            self.setTabVisible(i, True)
        super().setCurrentIndex(i)


class TabShell(QWidget):
    """Conteneur central : bande blanche derrière la barre d'onglets, fond doux en dessous."""

    def __init__(self, tabs: QTabWidget, parent=None):
        super().__init__(parent)
        self.tabs = tabs
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(B.SOFT))
        h = self.tabs.tabBar().height() if self.tabs.tabBar().isVisible() else 0
        if h:
            p.fillRect(0, 0, self.width(), h, QColor(B.WHITE))
            p.fillRect(0, h - 1, self.width(), 1, QColor(B.BORDER))
        p.end()
