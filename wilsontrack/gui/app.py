"""Lancement de l'interface graphique : thème clair aux couleurs du Cégep Marie-Victorin.

La vue vidéo (image de la caméra) et les vignettes de traces restent sombres : c'est l'image réelle de
la chambre. Tout le reste suit l'identité du Cégep (vert forêt, menthe, fond doux, cartes blanches).
"""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from .. import APP_NAME, __version__
from . import brand as B


# ---------------------------------------------------------------------------------------------
# Petites images du thème (flèches des listes déroulantes et des compteurs, coche des cases).
# Dès qu'une feuille de style redessine la bordure d'un QComboBox / QSpinBox / QCheckBox, Qt
# n'affiche plus les flèches natives : on les génère donc en PNG (Python pur, sans Qt, pour que
# QSS reste une simple chaîne utilisable par n'importe quel script).
# ---------------------------------------------------------------------------------------------
def _png(width: int, height: int, pixels: list[list[tuple[int, int, int, int]]]) -> bytes:
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _seg_dist(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy or 1.0)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _polyline_png(width: int, height: int, pts: list[tuple[float, float]], thick: float, rgb: tuple) -> bytes:
    """Polyligne anticrénelée (champ de distance) sur fond transparent."""
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            d = min(_seg_dist(x + 0.5, y + 0.5, a, b) for a, b in zip(pts, pts[1:]))
            a = max(0.0, min(1.0, thick / 2 + 0.5 - d))
            row.append((rgb[0], rgb[1], rgb[2], int(round(255 * a))))
        rows.append(row)
    return _png(width, height, rows)


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _theme_dir() -> str | None:
    """Écrit (une seule fois) les images du thème ; renvoie le dossier en notation « / », ou None."""
    try:
        d = Path(tempfile.gettempdir()) / "wilsontrack_theme_cmv_v1"
        d.mkdir(parents=True, exist_ok=True)
        specs = {
            "down": (10, 6, [(1.2, 1.2), (5.0, 4.8), (8.8, 1.2)], 1.6),
            "up": (10, 6, [(1.2, 4.8), (5.0, 1.2), (8.8, 4.8)], 1.6),
            "check": (12, 12, [(2.4, 6.3), (4.9, 8.8), (9.6, 3.4)], 1.9),
            "dot": (10, 10, [(5.0, 5.0), (5.0, 5.0)], 5.0),          # bouton radio coché
        }
        colors = {"": _hex_rgb(B.MUTED), "_hover": _hex_rgb(B.FOREST), "_disabled": (178, 192, 184),
                  "_white": (255, 255, 255)}
        for name, (w, h, pts, th) in specs.items():
            for suffix, rgb in colors.items():
                for scale, tag in ((1, ""), (2, "@2x")):
                    f = d / f"{name}{suffix}{tag}.png"
                    if not f.exists() or f.stat().st_size == 0:
                        data = _polyline_png(w * scale, h * scale, [(x * scale, y * scale) for x, y in pts],
                                             th * scale, rgb)
                        f.write_bytes(data)
        return d.as_posix()
    except Exception:
        return None


_INPUTS_WITH_IMAGES = """
QLineEdit, QComboBox, QAbstractSpinBox {
    background: #ffffff; color: @INK@; border: 1px solid @BORDER_STRONG@; border-radius: 9px; padding: 5px 9px;
    min-height: 20px; selection-background-color: @SELECTION@; selection-color: @FOREST@; }
QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover { border-color: @GREEN@; }
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus { border: 2px solid @BRIGHT@; padding: 4px 8px; }
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {
    background: #f4f7f4; color: #8b9a91; border-color: #dfe8e1; }
QLineEdit[readOnly="true"] { background: @SOFT@; color: @MUTED@; }
QComboBox { padding-right: 28px; }
QComboBox:focus { padding-right: 27px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none; }
QComboBox::down-arrow { image: url("@T@/down.png"); width: 10px; height: 6px; }
QComboBox::down-arrow:hover, QComboBox::down-arrow:on { image: url("@T@/down_hover.png"); }
QComboBox::down-arrow:disabled { image: url("@T@/down_disabled.png"); }
QComboBox QAbstractItemView { background: #ffffff; color: @INK@; border: 1px solid @BORDER@; border-radius: 10px;
    padding: 4px; outline: none; selection-background-color: @SELECTION@; selection-color: @FOREST@; }
QComboBox QAbstractItemView::item { min-height: 26px; padding: 2px 10px; border-radius: 6px; }
QAbstractSpinBox { padding-right: 28px; }
QAbstractSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
    width: 24px; border: none; border-left: 1px solid @BORDER@; border-top-right-radius: 9px; background: @SOFT@;
    margin: 1px 1px 0 0; }
QAbstractSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
    width: 24px; border: none; border-left: 1px solid @BORDER@; border-bottom-right-radius: 9px; background: @SOFT@;
    margin: 0 1px 1px 0; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: @HOVER@; }
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed { background: @SELECTION@; }
QAbstractSpinBox::up-arrow { image: url("@T@/up.png"); width: 8px; height: 5px; }
QAbstractSpinBox::down-arrow { image: url("@T@/down.png"); width: 8px; height: 5px; }
QAbstractSpinBox::up-arrow:hover { image: url("@T@/up_hover.png"); }
QAbstractSpinBox::down-arrow:hover { image: url("@T@/down_hover.png"); }
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off { image: url("@T@/up_disabled.png"); }
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off { image: url("@T@/down_disabled.png"); }
QCheckBox, QRadioButton { spacing: 9px; color: @INK@; }
QCheckBox::indicator { width: 18px; height: 18px; border: 1.5px solid @BORDER_STRONG@; border-radius: 5px;
    background: #ffffff; }
QCheckBox::indicator:hover, QCheckBox::indicator:focus { border-color: @BRIGHT@; }
QCheckBox::indicator:checked { background: @FOREST@; border-color: @FOREST@; image: url("@T@/check_white.png"); }
QCheckBox::indicator:checked:hover { background: @GREEN@; border-color: @GREEN@; }
QCheckBox::indicator:disabled { background: #f4f7f4; border-color: #dfe8e1; }
QCheckBox::indicator:checked:disabled { background: #b7ccbd; border-color: #b7ccbd;
    image: url("@T@/check_white.png"); }
QRadioButton::indicator { width: 18px; height: 18px; border: 1.5px solid @BORDER_STRONG@; border-radius: 10px;
    background: #ffffff; }
QRadioButton::indicator:hover, QRadioButton::indicator:focus { border-color: @BRIGHT@; }
QRadioButton::indicator:checked { background: @FOREST@; border-color: @FOREST@; image: url("@T@/dot_white.png"); }
QRadioButton::indicator:disabled { background: #f4f7f4; border-color: #dfe8e1; }
QRadioButton::indicator:checked:disabled { background: #b7ccbd; border-color: #b7ccbd;
    image: url("@T@/dot_white.png"); }
"""

# Repli si les images ne peuvent pas être écrites : on garde les flèches natives (Fusion).
_INPUTS_PLAIN = """
QLineEdit { background: #ffffff; border: 1px solid @BORDER_STRONG@; border-radius: 9px; padding: 5px 9px; }
QLineEdit:focus { border: 2px solid @BRIGHT@; padding: 4px 8px; }
QComboBox, QAbstractSpinBox { padding: 3px 5px; }
QCheckBox, QRadioButton { spacing: 8px; }
"""

_QSS_TEMPLATE = """
QWidget { font-size: 13px; @FONT@ }
QMainWindow, QDialog { background: @SOFT@; }
QToolTip { color: #ffffff; background-color: @FOREST@; border: none; border-radius: 8px; padding: 6px 10px; }
QLabel { color: @INK@; }

/* ---- en-tête vert forêt (barre de marque) ---- */
QWidget#BrandHeader { background: @FOREST@; }
QLabel#brandTitle { color: #ffffff; font-size: 17px; font-weight: 700; }
QLabel#brandSubtitle { color: @MINT@; font-size: 11px; font-weight: 600; }
QFrame#brandSep { background: rgba(152,242,157,70); border: none; }
QToolButton#headerBtn { color: #ffffff; background: transparent; border: 1px solid transparent; border-radius: 10px;
    padding: 6px 9px; font-size: 13px; font-weight: 600; }
QToolButton#headerBtn:hover { background: rgba(255,255,255,26); border-color: rgba(152,242,157,90); }
QToolButton#headerBtn:pressed { background: rgba(0,0,0,40); }
QToolButton#headerBtn:focus { border: 2px solid @MINT@; padding: 5px 8px; }
QToolButton#headerBtn:disabled { color: rgba(255,255,255,105); }
QToolButton#headerBtn[role="record"]:checked { background: #c9353b; border-color: #ff9a9d; color: #ffffff; }
QToolButton#headerBtn[role="toggle"] { border: 1px solid rgba(255,255,255,80); }
QToolButton#headerBtn[role="toggle"]:hover { background: rgba(255,255,255,26); border-color: @MINT@; }
QToolButton#headerBtn[role="toggle"]:checked { background: #ffffff; color: @FOREST@; border-color: #ffffff; }

/* ---- menus ---- */
QMenuBar { background: transparent; color: #ffffff; border: none; padding: 0; }
QMenuBar::item { padding: 6px 8px; border-radius: 8px; background: transparent; color: rgba(255,255,255,225); }
QMenuBar::item:selected, QMenuBar::item:pressed { background: rgba(255,255,255,30); color: #ffffff; }
QMenu { background: #ffffff; color: @INK@; border: 1px solid @BORDER@; border-radius: 12px; padding: 6px; }
QMenu::item { padding: 7px 28px 7px 12px; border-radius: 8px; }
QMenu::item:selected { background: @HOVER@; color: @FOREST@; }
QMenu::item:disabled { color: #9aa79f; }
QMenu::icon { padding-left: 8px; }
QMenu::separator { height: 1px; background: @BORDER@; margin: 5px 10px; }
QToolBar { spacing: 4px; padding: 4px 6px; border: none; background: @SOFT@; }
QToolButton { border-radius: 8px; color: @INK@; }
QToolButton:focus { border: 2px solid @BRIGHT@; }

/* ---- boutons (propriétés dynamiques : primary, success, danger, link) ---- */
QPushButton { padding: 7px 14px; border-radius: 10px; background: #ffffff; border: 1px solid @BORDER_STRONG@;
    color: @FOREST@; font-weight: 600; }
QPushButton:hover { background: @HOVER@; border-color: @GREEN@; }
QPushButton:pressed { background: @SELECTION@; }
QPushButton:focus { border: 2px solid @BRIGHT@; padding: 6px 13px; }
QPushButton:disabled { background: #f4f7f4; color: #9aa79f; border-color: #dfe8e1; }
QPushButton[primary="true"], QPushButton[success="true"] { background: @FOREST@; border: 1px solid @FOREST@;
    color: #ffffff; font-weight: 700; }
QPushButton[primary="true"]:hover, QPushButton[success="true"]:hover { background: @GREEN@; border-color: @GREEN@; }
QPushButton[primary="true"]:pressed, QPushButton[success="true"]:pressed { background: @FOREST_DARK@;
    border-color: @FOREST_DARK@; }
QPushButton[primary="true"]:focus, QPushButton[success="true"]:focus { border: 2px solid @BRIGHT@; }
QPushButton[primary="true"]:disabled, QPushButton[success="true"]:disabled { background: #b7ccbd;
    border-color: #b7ccbd; color: #ffffff; }
QPushButton[danger="true"] { background: #ffffff; border: 1px solid @DANGER_BORDER@; color: @DANGER_TEXT@;
    font-weight: 700; }
QPushButton[danger="true"]:hover { background: @DANGER_BG@; border-color: #e5484d; }
QPushButton[danger="true"]:pressed { background: #f9d9da; }
QPushButton[danger="true"]:focus { border: 2px solid #e5484d; }
QPushButton[danger="true"]:disabled { background: #f7f4f4; border-color: #ece2e2; color: #b9a5a6; }
QPushButton[armed="true"] { background: #c2292e; border: 1px solid #a51f24; color: #ffffff; font-weight: 700; }
QPushButton[link="true"] { background: transparent; border: none; color: @GREEN@; padding: 2px 4px;
    font-weight: 600; }
QPushButton[link="true"]:hover { color: @FOREST@; text-decoration: underline; }
QPushButton[link="true"]:focus { color: @FOREST@; text-decoration: underline; border: none; padding: 2px 4px; }
QPushButton[link="true"]:disabled { color: #9aa79f; }

/* ---- champs de saisie (flèches et coches : images générées, voir _theme_dir) ---- */
@INPUTS@

/* ---- onglets (soulignement vert forêt) ---- */
QTabWidget::pane { border: none; background: @SOFT@; top: 0px; }
QTabWidget::tab-bar { left: 18px; }
QTabBar { background: transparent; qproperty-drawBase: 0; }
QTabBar::tab { padding: 11px 16px 9px 16px; margin: 0 4px 0 0; color: @MUTED@; background: transparent;
    border: none; border-bottom: 3px solid transparent; font-weight: 600; }
QTabBar::tab:hover { color: @FOREST@; border-bottom: 3px solid @BORDER_STRONG@; }
QTabBar::tab:selected { color: @FOREST@; border-bottom: 3px solid @FOREST@; }
QTabBar::tab:focus { color: @FOREST@; }
QTabBar::tab:disabled { color: #a9b6ae; }
QWidget#TabStrip { background: #ffffff; border-bottom: 1px solid @BORDER@; }
QTabWidget#SubTabs::pane { border: none; background: transparent; }
QTabWidget#SubTabs::tab-bar { left: 0px; }
QTabWidget#SubTabs QTabBar::tab { padding: 7px 14px; margin: 0 6px 6px 0; border: 1px solid @BORDER@;
    border-radius: 15px; background: #ffffff; color: @MUTED@; }
QTabWidget#SubTabs QTabBar::tab:hover { color: @FOREST@; border-color: @GREEN@; }
QTabWidget#SubTabs QTabBar::tab:selected { color: #ffffff; background: @FOREST@; border-color: @FOREST@; }

/* ---- cartes génériques ---- */
QFrame#Card { background: #ffffff; border: 1px solid @BORDER@; border-radius: 16px; }
QLabel#cardHeading { color: @INK@; font-size: 15px; font-weight: 700; }
QLabel#muted { color: @MUTED@; }
QLabel#eyebrow { color: @GREEN@; font-size: 11px; font-weight: 700; }

/* ---- listes, tableaux, textes ---- */
QGroupBox { border: 1px solid @BORDER@; border-radius: 14px; margin-top: 2px; padding-top: 6px; background: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: @FOREST@; font-weight: 700; }
QTableWidget, QTableView, QListWidget, QListView, QTreeView { background: #ffffff; color: @INK@;
    border: 1px solid @BORDER@; border-radius: 14px; gridline-color: #e6eee8; selection-background-color: @SELECTION@;
    selection-color: @FOREST@; alternate-background-color: @ALT_ROW@; outline: none; }
QTableWidget:focus, QTableView:focus, QListWidget:focus, QListView:focus, QTreeView:focus { border-color: @BRIGHT@; }
QListWidget::item, QListView::item { padding: 5px 7px; border-radius: 8px; }
QListWidget::item:hover, QListView::item:hover { background: @HOVER@; }
QListWidget::item:selected, QListView::item:selected { background: @SELECTION@; color: @FOREST@; }
QTableView::item { padding: 3px 8px; }
QTableView::item:hover { background: @HOVER@; }
QTableView::item:selected { background: @SELECTION@; color: @FOREST@; }
QHeaderView { background: #ffffff; border: none; border-top-left-radius: 14px; border-top-right-radius: 14px; }
QHeaderView::section { background: #ffffff; color: @MUTED@; padding: 8px 9px; border: none;
    border-bottom: 1px solid @BORDER@; font-weight: 700; font-size: 12px; }
QHeaderView::section:hover { background: @HOVER@; color: @FOREST@; }
QTableCornerButton::section { background: #ffffff; border: none; }
QTextBrowser, QTextEdit, QPlainTextEdit { background: #ffffff; color: @INK@; border: 1px solid @BORDER@;
    border-radius: 14px; padding: 10px; selection-background-color: @SELECTION@; selection-color: @FOREST@; }
QTextBrowser:focus, QTextEdit:focus, QPlainTextEdit:focus { border-color: @BRIGHT@; }

/* ---- barres de défilement fines ---- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 3px 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px 3px; }
QScrollBar::handle:vertical { background: #c3d4c8; border-radius: 3px; min-height: 36px; margin: 0 1px; }
QScrollBar::handle:horizontal { background: #c3d4c8; border-radius: 3px; min-width: 36px; margin: 1px 0; }
QScrollBar::handle:hover { background: @GREEN@; }
QScrollBar::handle:pressed { background: @FOREST@; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QAbstractScrollArea::corner { background: transparent; }
QScrollArea { background: transparent; border: none; }

/* ---- divers ---- */
QSlider::groove:horizontal { height: 4px; background: @BORDER@; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @FOREST@; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; border: 2px solid @FOREST@; width: 12px; height: 12px;
    margin: -7px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { border-color: @BRIGHT@; }
QProgressBar { border: 1px solid @BORDER@; border-radius: 8px; text-align: center; background: #ffffff;
    color: @INK@; min-height: 14px; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @FOREST@, stop:1 @BRIGHT@);
    border-radius: 7px; }
QProgressBar[slim="true"] { border: none; border-radius: 3px; background: @BORDER@; min-height: 6px;
    max-height: 6px; }
QProgressBar[slim="true"]::chunk { border-radius: 3px; }
QProgressBar[success="true"]::chunk { background: @BRIGHT@; }
QProgressBar[warning="true"]::chunk { background: #e8a317; }
QProgressBar[error="true"]::chunk { background: #e5484d; }
QDockWidget { color: @FOREST@; font-weight: 700; }
QDockWidget::title { padding: 10px 12px; background: #ffffff; border-bottom: 1px solid @BORDER@; text-align: left; }
QStatusBar { background: #ffffff; border-top: 1px solid @BORDER@; color: @MUTED@; }
QStatusBar QLabel { color: @MUTED@; }
QStatusBar::item { border: none; }
QSplitter::handle { background: transparent; }
QSplitter::handle:horizontal { width: 8px; }
QSplitter::handle:vertical { height: 8px; }
QSplitter::handle:hover { background: @BORDER@; }
QMessageBox { background: #ffffff; }
QMessageBox QLabel { color: @INK@; }
"""


def _fill(template: str) -> str:
    tokens = {"FOREST": B.FOREST, "GREEN": B.GREEN, "BRIGHT": B.BRIGHT, "MINT": B.MINT, "SOFT": B.SOFT,
              "INK": B.INK, "MUTED": B.MUTED, "BORDER_STRONG": B.BORDER_STRONG, "BORDER": B.BORDER,
              "HOVER": B.HOVER, "ALT_ROW": B.ALT_ROW, "SELECTION": B.SELECTION, "FOREST_DARK": B.FOREST_DARK,
              "DANGER_TEXT": B.DANGER_TEXT, "DANGER_BG": B.DANGER_BG, "DANGER_BORDER": B.DANGER_BORDER}
    for k in sorted(tokens, key=len, reverse=True):          # BORDER_STRONG avant BORDER
        template = template.replace(f"@{k}@", tokens[k])
    return template


def _build_qss() -> str:
    theme = _theme_dir()
    inputs = _INPUTS_WITH_IMAGES.replace("@T@", theme) if theme else _INPUTS_PLAIN
    fam = B.ui_font_family()
    font = f'font-family: "{fam}";' if fam else ""
    return _fill(_QSS_TEMPLATE.replace("@INPUTS@", inputs).replace("@FONT@", font))


QSS = _build_qss()


def light_palette() -> QPalette:
    """Palette claire du Cégep Marie-Victorin (fond doux, cartes blanches, encre, vert forêt)."""
    p = QPalette()
    soft, white, ink = QColor(B.SOFT), QColor(B.WHITE), QColor(B.INK)
    p.setColor(QPalette.ColorRole.Window, soft)
    p.setColor(QPalette.ColorRole.WindowText, ink)
    p.setColor(QPalette.ColorRole.Base, white)
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(B.ALT_ROW))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(B.FOREST))
    p.setColor(QPalette.ColorRole.ToolTipText, white)
    p.setColor(QPalette.ColorRole.Text, ink)
    p.setColor(QPalette.ColorRole.Button, white)
    p.setColor(QPalette.ColorRole.ButtonText, QColor(B.FOREST))
    p.setColor(QPalette.ColorRole.BrightText, QColor("#c2292e"))
    p.setColor(QPalette.ColorRole.Highlight, QColor(B.SELECTION))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(B.FOREST))
    p.setColor(QPalette.ColorRole.Link, QColor(B.GREEN))
    p.setColor(QPalette.ColorRole.LinkVisited, QColor(B.FOREST))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor("#7d8c84"))
    p.setColor(QPalette.ColorRole.Light, white)
    p.setColor(QPalette.ColorRole.Midlight, QColor(B.BORDER))
    p.setColor(QPalette.ColorRole.Mid, QColor(B.BORDER_STRONG))
    p.setColor(QPalette.ColorRole.Dark, QColor("#8fa597"))
    p.setColor(QPalette.ColorRole.Shadow, QColor("#5b6b62"))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#9aa79f"))
    return p


# Compatibilité : les scripts et les tests existants appellent dark_palette() (désormais la palette claire).
dark_palette = light_palette


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    fam = B.ui_font_family()
    if fam:
        f = QFont(app.font())
        f.setFamily(fam)
        app.setFont(f)
    app.setStyleSheet(QSS)


def _want_splash() -> bool:
    if os.environ.get("WILSONTRACK_NO_SPLASH", "").strip() in ("1", "true", "oui"):
        return False
    return os.environ.get("QT_QPA_PLATFORM", "").lower() not in ("offscreen", "minimal")


def run_gui(argv=None) -> int:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"Marie-Victorin.{APP_NAME}.{__version__}")
        except Exception:
            pass
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Cégep Marie-Victorin")
    apply_theme(app)
    try:
        from PySide6.QtGui import QIcon
        icon = B.ASSETS / "icon.png"
        if icon.exists():
            app.setWindowIcon(QIcon(str(icon)))
    except Exception:
        pass

    splash = None
    if _want_splash():
        try:
            from .splash import SplashScreen
            splash = SplashScreen()
            splash.start()
        except Exception:
            splash = None

    def step(msg: str, progress: float):
        if splash is not None:
            try:
                splash.set_status(msg, progress)
                splash.pump()
            except Exception:
                pass

    step("Démarrage…", 0.05)
    try:
        import cv2  # noqa: F401  (chargement long : fait pendant l'animation)
        import numpy  # noqa: F401
    except Exception:
        pass
    step("Chargement de l'IA de référence…", 0.30)
    try:
        from ..detection.reference_model import ReferenceModel
        ReferenceModel.load_default()
    except Exception:
        pass
    step("Préparation des graphiques…", 0.55)
    try:
        import pyqtgraph  # noqa: F401
    except Exception:
        pass
    step("Préparation de la caméra…", 0.75)
    from .main_window import MainWindow
    step("Préparation de l'interface…", 0.88)
    w = MainWindow()
    step("Prêt", 1.0)
    if splash is not None:
        try:
            splash.finish(w)
        except Exception:
            w.show()
    else:
        w.show()
    rc = app.exec()
    # Aucun fil ne doit survivre à la fermeture : sinon Qt arrête brutalement l'application (SIGABRT)
    # pendant le nettoyage de Python. On demande l'arrêt, on attend un peu, puis on sort directement.
    from .worker import running_threads
    busy = running_threads()
    for t in busy:
        if hasattr(t, "stop"):
            try:
                t.stop()
            except Exception:
                pass
    for t in busy:
        t.wait(5000)
    if running_threads() or getattr(w, "unclean_exit", False):
        try:
            from ..analysis.run import _cleanup_spill_dirs
            _cleanup_spill_dirs()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc
