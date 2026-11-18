"""Fenêtre principale de WilsonTrack."""
from __future__ import annotations

import html
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFrame,
                               QGridLayout,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenuBar, QMessageBox, QProgressBar, QPushButton,
                               QSlider, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, physics
from ..analysis.conclusions import Conclusion, conclude_comparison, conclude_run
from ..analysis.run import RunResult, default_run_name, safe_name
from ..capture import CameraSource, FileSource, ImageFolderSource, SerialTrigger, imwrite, list_cameras
from ..config import AppConfig, default_output_dir
from ..detection.classifier import MLClassifier
from ..detection.explain import explain_event
from ..settings import Settings, settings_dir
from . import brand as B
from .chrome import BrandHeader, BrandTabs, TabShell
from .home import HomePage, VideoSetupDialog
from .learning import LearningCenter
from .session_dialogs import BatchQueueWidget, SessionStatusBar
from .settings_panel import SettingsPanel
from .widgets import Callout, CounterPanel, MplCanvas, VideoView, bgr_to_qimage
from .worker import VIEW_MODES, CallWorker, ProcessingWorker, SynthSource

CLASS_ORDER = ["alpha", "beta", "muon", "proton", "fine", "incertain"]
MANUAL_CHOICES = ["(automatique)", "alpha", "beta", "muon", "proton", "fine", "incertain", "artefact"]


def qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_BGR888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.sizeInBytes()).reshape(h, img.bytesPerLine())
    return arr[:, :w * 3].reshape(h, w, 3).copy()


def conclusions_html(items: list[Conclusion]) -> str:
    """Encadrés des conclusions (fond clair, barre de couleur à gauche, titre lisible WCAG AA)."""
    bars = {"succes": B.BRIGHT, "attention": "#f76b15", "info": "#2f9bff"}
    texts = {"succes": B.SUCCESS_TEXT, "attention": "#a3480b", "info": B.INFO_TEXT}
    parts = []
    for c in items:
        bar = bars.get(c.level, "#2f9bff")
        col = texts.get(c.level, B.INFO_TEXT)
        parts.append(f"<table width='100%' cellspacing='0' cellpadding='10' style='margin:8px 0;'>"
                     f"<tr><td width='4' bgcolor='{bar}'></td><td bgcolor='{B.SOFT}'>"
                     f"<b style='color:{col}'>{html.escape(c.title)}</b><br>"
                     f"<span style='color:{B.INK}'>{html.escape(c.text)}</span></td></tr></table>")
    return "".join(parts)


BROWSER_CSS = (f"h1, h2, h3 {{ color: {B.FOREST}; }} h4 {{ color: {B.GREEN}; }} "
               f"a {{ color: {B.GREEN}; }} th {{ color: {B.MUTED}; }} body {{ color: {B.INK}; }}")


def _browser() -> QTextBrowser:
    """QTextBrowser blanc aux titres vert forêt (charte du Cégep)."""
    tb = QTextBrowser()
    tb.setOpenExternalLinks(True)
    tb.document().setDefaultStyleSheet(BROWSER_CSS)
    return tb


def _card(widget=None, margins=(14, 10, 14, 10)):
    """Carte blanche arrondie (1 px de contour vert doux) contenant une rangée horizontale."""
    f = QFrame()
    f.setObjectName("Card")
    lay = QHBoxLayout(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(10)
    if widget is not None:
        lay.addWidget(widget)
    return f, lay


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__} — Analyse des traces de la chambre de Wilson")
        self.resize(1500, 920)
        icon_path = Path(__file__).resolve().parents[1] / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.settings = Settings()
        self.cfg: AppConfig = self.settings.load_config()
        if int(self.settings.get("config_version", 1) or 1) < 2:
            # Passage à la 2.0 : méthode hybride (règles + IA de référence) et nouveaux seuils par défaut
            fresh = AppConfig().classification
            c = self.cfg.classification
            if c.mode != "ia":
                c.mode = fresh.mode
            for k in ("alpha_max_length_mm", "ml_threshold", "muon_max_rms_mm"):
                if hasattr(fresh, k):
                    setattr(c, k, getattr(fresh, k))
            self.settings.set("config_version", 2)
        if not self.cfg.output_dir:
            self.cfg.output_dir = str(default_output_dir())
        self.worker: ProcessingWorker | None = None
        self.serial: SerialTrigger | None = None
        self.run: RunResult | None = None
        self.events: list = []
        self.source_kind = ""
        self.worker2 = None                 # 2e caméra simultanée
        self.cam2_label = ""
        self.source_arg = None
        self.last_stats: dict = {}
        self.plot_hist: list[tuple[float, dict]] = []
        self.assistant = None
        self._calls: list = []
        self._restart_hint = False

        self._build_ui()
        self.panel.load(self.cfg)
        self.panel.cam_fps.setValue(int(self.settings.get("cam_fps", 30)))
        res = self.settings.get("cam_res", "1920×1080")
        i = self.panel.resolution.findText(res)
        if i >= 0:
            self.panel.resolution.setCurrentIndex(i)
        self._refresh_cameras(quick=True)
        self._refresh_ports()
        geo = self.settings.get("geometry")
        if geo:
            try:
                self.restoreGeometry(bytes.fromhex(geo))
            except Exception:
                pass
        self._cfg_timer = QTimer(self, singleShot=True, interval=350)
        self._cfg_timer.timeout.connect(self._apply_settings)
        self.panel.changed.connect(self._cfg_timer.start)
        self._update_actions()
        self.tabs.setCurrentWidget(self.home)
        self.statusBar().showMessage("Prêt. Choisissez une source : caméra, vidéo, dossier de photos ou démonstration.")

    # ================================================================ UI ====
    def _build_ui(self):
        def act(text, slot, shortcut=None, tip="", checkable=False, icon=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip(tip or text)
            a.setCheckable(checkable)
            if icon:
                a.setIcon(B.line_icon(icon, B.FOREST, 16))
            self.addAction(a)                      # raccourcis actifs même hors des menus
            return a

        self.a_home = act("Accueil", self.go_home, "Ctrl+H", "Revenir au menu d'accueil", icon="home")
        self.a_camera = act("Caméra", lambda: self.start_source("camera"), "Ctrl+K",
                            "Démarrer l'acquisition en direct depuis la caméra", icon="camera")
        self.a_switch_cam = act("Changer de caméra", self.switch_camera, "Ctrl+Shift+K",
                                "Passer à la caméra suivante (Mac ↔ iPhone…) sans arrêter la détection",
                                icon="swap")
        self.a_video = act("Vidéo…", self.open_video, "Ctrl+O", "Analyser un fichier vidéo", icon="video")
        self.a_photos = act("Série de photos…", self.open_folder, "Ctrl+Shift+O",
                            "Analyser une série de photos (rafale / time-lapse)", icon="images")
        self.a_demo = act("Démonstration", lambda: self.start_source("demo"), "Ctrl+D",
                          "Chambre de Wilson simulée en temps réel (sans matériel)", icon="flask")
        self.a_pause = act("Pause", self.toggle_pause, "Space", "Pause / reprise", icon="pause")
        self.a_cam2 = act("2e caméra", self.toggle_cam2, "Ctrl+Shift+2",
                          "Filmer avec une 2e caméra en même temps (ex. : C930e + iPhone), chacune avec sa détection",
                          checkable=True, icon="images")
        self.a_stop = act("Arrêter", self.stop_all, "Ctrl+.", "Terminer l'enregistrement et calculer les résultats",
                          icon="stop")
        self.a_record = act("Enregistrer la vidéo", self.toggle_record, "R",
                            "Enregistrer la vidéo brute (MP4) pendant l'analyse", checkable=True, icon="record")
        self.a_snap = act("Photo", self.snapshot, "P", "Enregistrer l'image actuelle (brute + annotée)", icon="photo")
        self.a_trigger = act("Déclencher une expansion", self.trigger_expansion, "E",
                             "Déclencher une expansion (mode chambre à expansion)")
        self.a_claude = act("Demander à Claude (image actuelle)", self.ai_analyze_frame, "Ctrl+I",
                            "Faire analyser l'image actuelle par Claude (clé API nécessaire)", icon="spark")

        # ---- réglages (dock, masqué par défaut) -------------------------------------------
        self.panel = SettingsPanel()
        dock = QDockWidget("Réglages", self)
        dock.setObjectName("SettingsDock")
        dock.setWidget(self.panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        dock.hide()
        self.settings_dock = dock
        self.a_settings = dock.toggleViewAction()
        self.a_settings.setText("Réglages")
        self.a_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.a_settings.setToolTip("Afficher ou masquer les réglages (expérience, caméra, échelle…)")
        self.a_settings.setIcon(B.line_icon("gear", B.FOREST, 16))
        self.addAction(self.a_settings)
        p = self.panel
        p.roi_rect_requested.connect(lambda: self._start_draw("roi_rect"))
        p.roi_poly_requested.connect(lambda: self._start_draw("roi_poly"))
        p.roi_clear_requested.connect(self.clear_roi)
        p.calib_requested.connect(lambda: self._start_draw("calib"))
        p.camera_refresh_requested.connect(lambda: self._refresh_cameras(quick=False))
        p.camera.currentIndexChanged.connect(self._on_camera_changed)
        p.camera_start_requested.connect(lambda: self.start_source("camera"))
        p.camera_dialog_requested.connect(lambda: self.worker and self.worker.send("camera_dialog"))
        p.trigger_requested.connect(self.trigger_expansion)
        p.save_preset_requested.connect(self.save_preset)
        p.load_preset_requested.connect(self.load_preset)
        p.browse_output_requested.connect(self.browse_output)
        p.reset_requested.connect(self.reset_defaults)

        # ---- onglets --------------------------------------------------------------
        self.tabs = BrandTabs()
        self.tabs.setObjectName("MainTabs")
        self.tabs.setDocumentMode(True)
        shell = TabShell(self.tabs)
        self.setCentralWidget(shell)
        self.home = HomePage(self.settings)
        self.tabs.addTab(self.home, "Accueil")
        self._connect_home()
        self._build_live_tab()
        self._build_tracks_tab()
        self._build_charts_tab()
        self._build_conclusions_tab()
        self._build_compare_tab()
        self._build_ai_tab()
        self._build_batch_tab()
        self.learning = LearningCenter()
        self.learning.validate_requested.connect(self.validate_on_my_tracks)
        self.tab_learning = self.learning
        self.tabs.addTab(self.learning, "IA")
        self.tabs.setTabToolTip(self.tabs.indexOf(self.learning), "Centre d'apprentissage de l'IA")
        self._build_help_tab()
        # onglets affichés seulement quand on s'en sert (analyse par lot, assistant Claude)
        self.tabs.hide_until_used(self.tab_batch)
        self.tabs.hide_until_used(self.tab_ai)
        self.setStatusBar(QStatusBar())

        # ---- menus (dans l'en-tête vert) ----------------------------------------------
        mb = QMenuBar()
        mb.setNativeMenuBar(sys.platform == "darwin")
        self.menu_bar = mb
        h = self.home
        m = mb.addMenu("&Fichier")
        m.addAction(self.a_video)
        m.addAction(self.a_photos)
        m.addAction(act("Session de nuit…", h.open_night_setup, icon="moon"))
        m.addAction(act("Vidéos reçues / analyse par lot…", lambda: h.open_batch_setup(), icon="inbox"))
        m.addSeparator()
        m.addAction(act("Ouvrir des résultats enregistrés…", self.open_results, icon="folder"))
        m.addAction(act("Ouvrir le dossier des résultats", self.open_output_dir))
        m.addAction(act("Comparer des résultats", self._go_compare, icon="bars"))
        m.addSeparator()
        m.addAction(act("Quitter", self.close, "Ctrl+Q"))
        m = mb.addMenu("&Outils")
        m.addAction(self.a_settings)
        m.addSeparator()
        m.addAction(self.a_demo)
        m.addAction(act("Démonstration avec source KCl simulée", lambda: self.start_source("demo_kcl")))
        m.addAction(self.a_trigger)
        m.addSeparator()
        m.addAction(act("Assistant Claude…", lambda: self.tabs.setCurrentWidget(self.tab_ai), icon="chat"))
        m.addAction(self.a_claude)
        m.addAction(act("Centre d'apprentissage de l'IA", lambda: self.tabs.setCurrentWidget(self.tab_learning),
                        icon="book"))
        m.addSeparator()
        m.addAction(act("Générer une vidéo de démonstration…", self.tool_generate_video))
        m.addAction(act("Valider l'algorithme (vidéo simulée avec vérité terrain)…", self.tool_validate))
        m.addAction(act("Réinitialiser le fond", lambda: self.worker and self.worker.send("reset_background")))
        m = mb.addMenu("&Aide")
        m.addAction(act("Guide rapide", lambda: self.tabs.setCurrentWidget(self.tab_help), "F1", icon="help"))
        m.addAction(act("À propos", self.about))

        # ---- en-tête vert forêt : logo, nom, menus et actions essentielles ----------------
        self.header = BrandHeader(mb, [
            (self.a_home, "home"), (self.a_camera, "camera"), (self.a_switch_cam, "swap", "Changer"),
            (self.a_cam2, "images", "2 caméras"),
            (self.a_video, "video"), None,
            (self.a_pause, "pause"), (self.a_stop, "stop"), None,
            (self.a_record, "record", "Enregistrer"), (self.a_snap, "photo"), None,
            (self.a_settings, "gear")])
        self.setMenuWidget(self.header)
        self._compact_combos()

    def _compact_combos(self):
        """Listes déroulantes compressibles (la fenêtre tient sur un écran 1366 × 768, réglages ouverts) ;
        la liste ouverte reste assez large pour lire chaque choix."""
        for cb in self.findChildren(QComboBox):
            if cb.window() is not self or cb.parent() is None:
                continue
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(10)
            try:
                cb.view().setMinimumWidth(max(cb.view().sizeHintForColumn(0) + 36, 160))
            except Exception:
                pass

    def _build_live_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(18, 14, 18, 12)
        split = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(split)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(10)
        self.view_mode = QComboBox()
        for k, v in VIEW_MODES.items():
            self.view_mode.addItem(v, k)
        self.view_mode.currentIndexChanged.connect(self._on_view_mode)
        self.enhance = QSlider(Qt.Orientation.Horizontal)
        self.enhance.setRange(10, 40)
        self.enhance.setValue(10)
        self.enhance.setFixedWidth(96)
        self.enhance.setToolTip("Rehausser le contraste de l'affichage (n'influence pas la détection)")
        self.enhance.valueChanged.connect(self._on_view_mode)
        self.labels_chk = QCheckBox("Étiquettes")
        self.labels_chk.setChecked(True)
        self.labels_chk.toggled.connect(self._on_view_mode)
        self.realtime_chk = QCheckBox("Vitesse réelle")
        self.realtime_chk.setToolTip("Vidéo à vitesse réelle. Décoché : les fichiers vidéo sont analysés aussi vite "
                                     "que possible")
        self.rec_label = QLabel("")
        self.rec_label.setStyleSheet(f"color:{B.DANGER_TEXT}; font-weight:700")
        bar_card, bar = _card(margins=(14, 8, 14, 8))
        lab = QLabel("Affichage")
        lab.setObjectName("muted")
        bar.addWidget(lab)
        bar.addWidget(self.view_mode)
        bar.addSpacing(8)
        lab = QLabel("Contraste")
        lab.setObjectName("muted")
        bar.addWidget(lab)
        bar.addWidget(self.enhance)
        bar.addSpacing(8)
        bar.addWidget(self.labels_chk)
        bar.addWidget(self.realtime_chk)
        bar.addStretch(1)
        bar.addWidget(self.rec_label)
        self.session_bar = SessionStatusBar()
        self.session_bar.stop_requested.connect(self.stop_source)
        self.session_bar.hide()
        ll.addWidget(self.session_bar)
        ll.addWidget(bar_card)
        self.video = VideoView()
        self.video.roi_defined.connect(self.set_roi)
        self.video.calib_defined.connect(self.set_calibration)
        # 2e caméra simultanée : seconde image côte à côte, avec sa propre détection
        self.video_split = QSplitter(Qt.Orientation.Horizontal)
        self.video_split.setChildrenCollapsible(False)
        cam1_box = QWidget()
        c1 = QVBoxLayout(cam1_box)
        c1.setContentsMargins(0, 0, 0, 0)
        c1.setSpacing(4)
        self.cam1_title = QLabel("Caméra 1")
        self.cam1_title.setStyleSheet(f"color:{B.FOREST}; font-weight:700; padding: 0 4px;")
        self.cam1_title.hide()               # visible seulement en mode 2 caméras
        c1.addWidget(self.cam1_title)
        c1.addWidget(self.video, 1)
        self.video_split.addWidget(cam1_box)
        cam2_box = QWidget()
        c2 = QVBoxLayout(cam2_box)
        c2.setContentsMargins(0, 0, 0, 0)
        c2.setSpacing(4)
        self.cam2_title = QLabel("Caméra 2")
        self.cam2_title.setStyleSheet(f"color:{B.FOREST}; font-weight:700; padding: 0 4px;")
        self.video2 = VideoView()
        self.cam2_info = QLabel("")
        self.cam2_info.setStyleSheet(f"color:{B.MUTED}; font-size:12px; padding: 0 4px;")
        self.cam2_info.setWordWrap(True)
        c2.addWidget(self.cam2_title)
        c2.addWidget(self.video2, 1)
        c2.addWidget(self.cam2_info)
        self.video_split.addWidget(cam2_box)
        self.cam2_box = cam2_box
        cam2_box.hide()
        ll.addWidget(self.video_split, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setMaximumHeight(16)
        self.progress.hide()
        ll.addWidget(self.progress)
        self.live_info = QLabel("")
        self.live_info.setStyleSheet(f"color:{B.MUTED}; font-size:12px; padding: 0 4px;")
        ll.addWidget(self.live_info)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        rl.setSpacing(10)
        self.counters = CounterPanel()
        rl.addWidget(self.counters)
        try:
            import pyqtgraph as pg
            pg.setConfigOptions(antialias=True)
            self.plot = pg.PlotWidget(background=B.WHITE)
            self.plot.setStyleSheet(f"border: 1px solid {B.BORDER}; border-radius: 14px;")
            for side in ("left", "bottom"):
                ax = self.plot.getAxis(side)
                ax.setPen(pg.mkPen(B.BORDER_STRONG))
                ax.setTextPen(pg.mkPen(B.MUTED))
            self.plot.setLabel("bottom", "temps (s)", color=B.MUTED)
            self.plot.setLabel("left", "traces cumulées", color=B.MUTED)
            self.plot.addLegend(offset=(8, 8), labelTextColor=B.INK, brush=pg.mkBrush(255, 255, 255, 220),
                                pen=pg.mkPen(B.BORDER))
            self.plot.showGrid(x=True, y=True, alpha=0.18)
            self.curves = {c: self.plot.plot([], [], pen=pg.mkPen(physics.CLASS_INFO[c]["hex"], width=2.2),
                                             name=physics.CLASS_INFO[c]["symbole"]) for c in CLASS_ORDER[:4]}
            self.plot.setMinimumHeight(180)
            self.plot.getViewBox().setLimits(xMin=-0.5, yMin=-0.5)
            self.plot.setYRange(0, 5, padding=0.05)
            rl.addWidget(self.plot, 1)
        except Exception:
            self.plot = None
        lab = QLabel("DERNIÈRES TRACES")
        lab.setObjectName("eyebrow")
        rl.addWidget(lab)
        self.recent = QListWidget()
        self.recent.setIconSize(QSize(64, 64))
        self.recent.setMaximumHeight(260)
        self.recent.itemDoubleClicked.connect(self._recent_clicked)
        rl.addWidget(self.recent, 1)
        split.addWidget(right)
        split.setSizes([1000, 420])
        self.tab_live = w
        self.tabs.addTab(w, "Direct")

    def _build_tracks_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 14, 18, 12)
        lay.setSpacing(10)
        top_card, top = _card(margins=(14, 8, 14, 8))
        self.filter_cls = QComboBox()
        self.filter_cls.addItem("Toutes les classes", "")
        for c in CLASS_ORDER:
            self.filter_cls.addItem(physics.CLASS_INFO[c]["nom"], c)
        self.filter_cls.currentIndexChanged.connect(self._fill_tracks)
        train = QPushButton("Entraîner l'IA locale")
        train.setToolTip("Entraîner l'IA locale avec mes corrections : apprentissage automatique (forêt aléatoire) "
                         "à partir des classes que vous avez corrigées")
        train.clicked.connect(self.train_local_ai)
        self.use_confident = QCheckBox("+ traces sûres (≥ 85 %)")
        self.use_confident.setToolTip("Inclure aussi les traces classées avec une confiance d'au moins 85 %")
        self.use_confident.setChecked(True)
        exp = QPushButton("Exporter…")
        exp.setToolTip("Exporter le tableau des traces (CSV et Excel)")
        exp.clicked.connect(self.export_tables)
        lab = QLabel("Filtre")
        lab.setObjectName("muted")
        top.addWidget(lab)
        top.addWidget(self.filter_cls)
        top.addStretch(1)
        top.addWidget(train)
        top.addWidget(self.use_confident)
        top.addWidget(exp)
        lay.addWidget(top_card)
        split = QSplitter(Qt.Orientation.Horizontal)
        inner = QTabWidget()
        inner.setObjectName("SubTabs")
        inner.setDocumentMode(True)
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(["#", "Classe", "Conf.", "t (s)", "Longueur (mm)", "Épaisseur (mm)",
                                              "Rectitude", "Densité rel.", "Visible (s)", "E min. α (MeV)",
                                              "Pourquoi (résumé)"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(10, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_track_selected)
        inner.addTab(self.table, "Tableau")
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(128, 128))
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setSpacing(6)
        self.gallery.itemClicked.connect(self._gallery_clicked)
        inner.addTab(self.gallery, "Galerie")
        split.addWidget(inner)
        det = QWidget()
        dl = QVBoxLayout(det)
        dl.setContentsMargins(6, 0, 0, 0)
        dl.setSpacing(8)
        self.detail_img = QLabel("Sélectionnez une trace")
        self.detail_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_img.setMinimumSize(300, 240)
        self.detail_img.setStyleSheet(f"background:{B.VIDEO_BG}; border-radius:14px; color:#9fb5a9;")
        dl.addWidget(self.detail_img)
        row = QHBoxLayout()
        lab = QLabel("Corriger la classe")
        lab.setObjectName("muted")
        row.addWidget(lab)
        self.manual = QComboBox()
        self.manual.addItems(MANUAL_CHOICES)
        self.manual.activated.connect(self._on_manual_class)
        row.addWidget(self.manual, 1)
        dl.addLayout(row)
        ask = QPushButton("Pourquoi ? — 2e avis de Claude Opus 5.5")
        ask.setIcon(B.line_icon("spark", B.FOREST, 16))
        ask.setToolTip("Claude examine la vignette et les mesures, donne sa classification et la justifie "
                       "(clé API nécessaire, onglet « Assistant IA »)")
        ask.clicked.connect(self.ai_analyze_track)
        dl.addWidget(ask)
        self.detail_txt = _browser()
        dl.addWidget(self.detail_txt, 3)
        self.detail_claude = _browser()
        self.detail_claude.hide()
        dl.addWidget(self.detail_claude, 2)
        split.addWidget(det)
        split.setSizes([820, 540])
        lay.addWidget(split, 1)
        self.tab_tracks = w
        self.tabs.addTab(w, "Traces")

    def _build_charts_tab(self):
        """Graphiques : sous-onglet de « Conclusions » (tabs.setCurrentWidget(tab_charts) y mène)."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(10)
        top = QHBoxLayout()
        b = QPushButton("↻ Actualiser les graphiques")
        b.clicked.connect(self.refresh_charts)
        top.addWidget(b)
        top.addStretch(1)
        lay.addLayout(top)
        grid = QGridLayout()
        grid.setSpacing(12)
        self.charts = {k: MplCanvas() for k in ("lw", "time", "hist", "dens", "energy", "orient")}
        for i, k in enumerate(self.charts):
            self.charts[k].set_message("Pas encore de données")
            grid.addWidget(self.charts[k], i // 3, i % 3)
        lay.addLayout(grid, 1)
        self.tab_charts = w

    def _build_conclusions_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 14, 18, 12)
        lay.setSpacing(10)
        row_card, row = _card(margins=(14, 8, 14, 8))
        self.b_save = QPushButton("Sauvegarder les résultats")
        self.b_save.setProperty("primary", True)
        self.b_save.clicked.connect(self.save_results)
        self.b_report = QPushButton("Rapport HTML")
        self.b_report.clicked.connect(self.make_report)
        self.b_folder = QPushButton("Ouvrir le dossier")
        self.b_folder.setIcon(B.line_icon("folder", B.FOREST, 16))
        self.b_folder.clicked.connect(self.open_run_folder)
        self.b_ai_interp = QPushButton("Interprétation par Claude")
        self.b_ai_interp.setIcon(B.line_icon("spark", B.FOREST, 16))
        self.b_ai_interp.clicked.connect(self.ai_interpret_run)
        for b in (self.b_save, self.b_report, self.b_folder, self.b_ai_interp):
            row.addWidget(b)
        row.addStretch(1)
        lay.addWidget(row_card)
        self.concl = _browser()
        self.concl.setHtml(f"<p style='color:{B.MUTED}'>Les conclusions apparaîtront ici à la fin d'un "
                           "enregistrement (bouton « Arrêter », ou fin de la vidéo).</p>")
        self.concl_tabs = QTabWidget()
        self.concl_tabs.setObjectName("SubTabs")
        self.concl_tabs.setDocumentMode(True)
        self.concl_tabs.addTab(self.concl, "Synthèse")
        self.concl_tabs.addTab(self.tab_charts, "Graphiques")
        lay.addWidget(self.concl_tabs, 1)
        self.tab_concl = w
        self.tabs.addTab(w, "Conclusions")
        self.tabs.register_nested(self.tab_charts, w, self.concl_tabs)

    def _build_compare_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 14, 18, 12)
        lay.setSpacing(10)
        row = QHBoxLayout()
        rb = QPushButton("↻ Actualiser la liste")
        rb.clicked.connect(self.refresh_runs)
        ab = QPushButton("Ajouter un dossier de résultats…")
        ab.clicked.connect(self.add_run_folder)
        row.addWidget(rb)
        row.addWidget(ab)
        row.addStretch(1)
        lay.addLayout(row)
        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setHorizontalHeaderLabels(["Nom", "Condition", "Échantillon", "Durée (min)", "Traces",
                                                   "α / β / μ", "Dossier"])
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.runs_table.setMaximumHeight(230)
        lay.addWidget(self.runs_table)
        row_card, row = _card(margins=(14, 8, 14, 8))
        self.cmp_sample = QComboBox()
        self.cmp_bg = QComboBox()
        cb = QPushButton("Comparer")
        cb.setProperty("primary", True)
        cb.setIcon(B.line_icon("bars", "#ffffff", 16))
        cb.clicked.connect(self.compare_runs)
        rp = QPushButton("Rapport")
        rp.setToolTip("Rapport HTML de la comparaison")
        rp.clicked.connect(self.compare_report)
        ai = QPushButton("Claude")
        ai.setToolTip("Interprétation de la comparaison par Claude (clé API nécessaire)")
        ai.setIcon(B.line_icon("spark", B.FOREST, 16))
        ai.clicked.connect(self.ai_interpret_comparison)
        for text, combo in (("Échantillon", self.cmp_sample), ("Référence (fond)", self.cmp_bg)):
            lab = QLabel(text)
            lab.setObjectName("muted")
            row.addWidget(lab)
            row.addWidget(combo, 1)
        row.addWidget(cb)
        row.addWidget(rp)
        row.addWidget(ai)
        lay.addWidget(row_card)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.cmp_text = _browser()
        self.cmp_chart = MplCanvas()
        self.cmp_chart.set_message("Choisissez deux enregistrements puis « Comparer ».")
        split.addWidget(self.cmp_text)
        split.addWidget(self.cmp_chart)
        split.setSizes([700, 600])
        lay.addWidget(split, 1)
        self.extra_run_dirs: list[str] = list(self.settings.get("extra_run_dirs", []))
        self.runs_index: list[RunResult] = []
        self.tab_compare = w
        self.tabs.addTab(w, "Comparer")

    def _build_ai_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 14, 18, 12)
        lay.setSpacing(10)
        intro = QLabel("L'assistant IA est <b>facultatif</b> : toute la détection, la classification et les "
                       "statistiques fonctionnent hors ligne. Avec une clé API Anthropic, Claude peut donner un "
                       "second avis sur les images, rédiger une interprétation scientifique et répondre à vos questions. "
                       "Seules les images et mesures que vous envoyez quittent l'ordinateur.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        g = QGridLayout()
        self.api_key = QLineEdit(self.settings.get("anthropic_api_key", ""))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("sk-ant-…  (ou variable d'environnement ANTHROPIC_API_KEY)")
        self.remember_key = QCheckBox("Mémoriser sur cet ordinateur")
        self.remember_key.setChecked(bool(self.settings.get("anthropic_api_key")))
        from ..ai_assistant import EFFORT_CHOICES, MODEL_CHOICES
        self.ai_model = QComboBox()
        for mid, label in MODEL_CHOICES:
            self.ai_model.addItem(label, mid)
        saved = self.settings.get("ai_model", MODEL_CHOICES[0][0])
        j = self.ai_model.findData(saved)
        self.ai_model.setCurrentIndex(j if j >= 0 else 0)
        self.ai_effort = QComboBox()
        for item in EFFORT_CHOICES:
            val, label = (item if isinstance(item, (tuple, list)) else (item, item))[:2]
            self.ai_effort.addItem(label, val)
        j = self.ai_effort.findData(self.settings.get("ai_effort", "medium"))
        self.ai_effort.setCurrentIndex(j if j >= 0 else max(0, self.ai_effort.findData("medium")))
        tb = QPushButton("Tester la connexion")
        tb.clicked.connect(self.ai_test)
        g.addWidget(QLabel("Clé API :"), 0, 0)
        g.addWidget(self.api_key, 0, 1)
        g.addWidget(self.remember_key, 0, 2)
        g.addWidget(QLabel("Modèle :"), 1, 0)
        g.addWidget(self.ai_model, 1, 1)
        g.addWidget(tb, 1, 2)
        g.addWidget(QLabel("Effort :"), 2, 0)
        g.addWidget(self.ai_effort, 2, 1)
        lay.addLayout(g)
        row = QHBoxLayout()
        for text, slot in [("Analyser l'image actuelle", self.ai_analyze_frame),
                           ("Interpréter l'enregistrement", self.ai_interpret_run),
                           ("Interpréter la comparaison", self.ai_interpret_comparison),
                           ("Nouvelle conversation", self.ai_reset)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        self.ai_out = _browser()
        lay.addWidget(self.ai_out, 1)
        row = QHBoxLayout()
        self.ai_q = QLineEdit()
        self.ai_q.setPlaceholderText("Posez une question sur vos résultats (Entrée pour envoyer)…")
        self.ai_q.returnPressed.connect(self.ai_chat)
        sb = QPushButton("Envoyer")
        sb.setProperty("primary", True)
        sb.clicked.connect(self.ai_chat)
        row.addWidget(self.ai_q, 1)
        row.addWidget(sb)
        lay.addLayout(row)
        self.ai_status = QLabel("")
        self.ai_status.setStyleSheet(f"color:{B.MUTED}")
        lay.addWidget(self.ai_status)
        self.ai_log: list[str] = []
        self.tab_ai = w
        self.tabs.addTab(w, "Assistant IA")

    def _build_help_tab(self):
        tb = _browser()
        tb.setStyleSheet("QTextBrowser { border: none; border-radius: 0px; padding: 18px 28px; }")
        tb.setHtml(_readable_class_colors(HELP_HTML + _reference_model_help()))
        self.tab_help = tb
        self.tabs.addTab(tb, "Aide")

    # ============================================================ sources ====
    def _refresh_cameras(self, quick: bool):
        cams = self.settings.get("cameras_cache", []) if quick else []
        if not quick or not cams:
            self.statusBar().showMessage("Recherche des caméras…")
            QApplicationProcess()
            try:
                cams = list_cameras(4 if quick else 8, self.panel.backend.currentText())
            except Exception:
                cams = []
            self.settings.set("cameras_cache", cams)
        self.panel.camera.clear()
        for c in cams:
            self.panel.camera.addItem(c["label"], c["index"])
        if not cams:
            self.panel.camera.addItem("Caméra 0 (par défaut)", 0)
        idx = self.settings.get("cam_index", 0)
        j = self.panel.camera.findData(idx)
        if j >= 0:
            self.panel.camera.setCurrentIndex(j)
        self._on_camera_changed()
        try:
            self.home.set_cameras(cams)
        except Exception:
            pass
        self.statusBar().showMessage(f"{len(cams)} caméra(s) détectée(s).", 4000)

    # ============================================================ accueil ====
    def _connect_home(self):
        h = self.home
        h.camera_requested.connect(self._home_camera)
        h.camera_refresh_requested.connect(lambda: self._refresh_cameras(quick=False))
        h.video_requested.connect(self._start_video_from_dict)
        h.photos_requested.connect(self.open_folder)
        h.demo_requested.connect(lambda kind: self.start_source(kind))
        h.compare_requested.connect(self._go_compare)
        h.results_requested.connect(self.open_results)
        h.assistant_requested.connect(lambda: self.tabs.setCurrentWidget(self.tab_ai))
        h.guide_requested.connect(lambda: self.tabs.setCurrentWidget(self.tab_help))
        h.night_requested.connect(self.start_night_session)
        h.batch_requested.connect(self.start_batch)
        h.learning_requested.connect(lambda: self.tabs.setCurrentWidget(self.tab_learning))
        if hasattr(h, "run_open_requested"):
            h.run_open_requested.connect(self._open_run_folder_path)

    # ===================================================== session de nuit ==
    def start_night_session(self, d: dict):
        """Session longue durée (toute une nuit) avec la caméra : sauvegardes automatiques, vidéo segmentée…"""
        p = self.panel
        if d.get("condition"):
            p.condition.setCurrentText(str(d["condition"]))
        if d.get("sample"):
            k = p.sample.findText(str(d["sample"]))
            if k >= 0:
                p.sample.setCurrentIndex(k)
        p.mass.setValue(float(d.get("mass_g") or 0.0))
        if float(d.get("px_per_mm") or 0) > 0:
            p.px_per_mm.setValue(float(d["px_per_mm"]))
        idx = int(d.get("camera_index", self.settings.get("cam_index", 0)) or 0)
        j = p.camera.findData(idx)
        if j < 0:
            p.camera.addItem(f"Caméra {idx}", idx)
            j = p.camera.count() - 1
        p.camera.setCurrentIndex(j)
        self._apply_settings()
        self.start_source("camera", session=d)
        self.statusBar().showMessage("🌙 Session de nuit démarrée : sauvegarde automatique régulière, "
                                     "vous pouvez laisser l'ordinateur tourner.", 15000)

    def _open_run_folder_path(self, folder: str):
        try:
            run = RunResult.load(folder)
        except Exception:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
            return
        self._show_loaded_run(run)

    # ================================================ vidéos reçues / lot ==
    def _build_batch_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 14, 18, 12)
        lay.setSpacing(10)
        intro = QLabel("<b>Vidéos reçues et analyse par lot</b> — depuis l'accueil, choisissez des vidéos ou un "
                       "dossier à surveiller (par exemple un dossier OneDrive où l'on vous envoie des vidéos) : "
                       "chaque vidéo est analysée automatiquement, avec ses résultats et son rapport, et un "
                       "tableau récapitulatif <i>resume_lot.csv</i> est créé.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        b = QPushButton("Nouvelles vidéos / dossier à surveiller…")
        b.setIcon(B.line_icon("inbox", "#ffffff", 16))
        b.setProperty("primary", True)
        b.clicked.connect(lambda: self.home.open_batch_setup())
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.batch_queue = BatchQueueWidget()
        self.batch_queue.open_folder_requested.connect(self._open_run_folder_path)
        self.batch_queue.cancel_requested.connect(self.stop_batch)
        lay.addWidget(self.batch_queue, 1)
        self.batch_ctl = None
        self.tab_batch = w
        self.tabs.addTab(w, "Lot")

    def start_batch(self, d: dict):
        from ..batch import QtBatchController
        d = dict(d)
        if not d.get("output_dir"):
            d["output_dir"] = str(self._out_dir())
        if self.batch_ctl is None:
            self.batch_ctl = QtBatchController(self)
            self.batch_ctl.jobs_changed.connect(self.batch_queue.update_jobs)
            self.batch_ctl.job_finished.connect(lambda _j: self.refresh_runs())
            self.batch_ctl.all_finished.connect(
                lambda: self.statusBar().showMessage("📥 Toutes les vidéos ont été analysées.", 15000))
            self.batch_ctl.error.connect(lambda m: self.statusBar().showMessage(f"Lot : {m}", 15000))
            self.batch_ctl.info.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.cfg = self.panel.apply_to(self.cfg)
        started = self.batch_ctl.start(d, app_cfg=self.cfg.copy())
        self.tabs.setCurrentWidget(self.tab_batch)
        n = len(d.get("files") or [])
        msg = (f"{n} vidéo(s) en file d'attente" if n else "Aucune vidéo pour l'instant")
        if d.get("watch_folder"):
            msg += f" — dossier surveillé : {d['watch_folder']}"
        if not started:
            msg = "Vidéos ajoutées au lot déjà en cours. " + msg
        self.statusBar().showMessage(msg, 15000)

    def stop_batch(self):
        if self.batch_ctl is not None:
            self.batch_ctl.stop()
            self.statusBar().showMessage("Analyse par lot arrêtée.", 8000)

    # ============================================== centre d'apprentissage ==
    def validate_on_my_tracks(self):
        """Précision de l'IA de référence sur les traces RÉELLES corrigées à la main par l'équipe."""
        from ..detection import reference_model as rm
        lab = [e for e in self.events if e.manual_cls in ("alpha", "beta", "muon", "proton", "artefact")]
        if len(lab) < 5:
            QMessageBox.information(
                self, "Mesurer la précision",
                "Corrigez d'abord au moins 5 traces dans l'onglet « Traces » (choisissez leur vraie classe), "
                "idéalement 30 ou plus, de chaque type. L'application comparera ensuite ce que l'IA aurait "
                "répondu à vos réponses.")
            return
        feats = [e.features for e in lab]
        labels = [e.manual_cls for e in lab]
        extras = [{"duration_s": e.duration_s, "touches_border": e.touches_border,
                   "calibrated": e.extra.get("calibrated"),
                   "rule_scores": e.extra.get("rules_scores") or e.scores} for e in lab]
        try:
            res = rm.evaluate_on_labeled(feats, labels, extras)
        except Exception as ex:
            QMessageBox.warning(self, "Mesurer la précision", f"Calcul impossible : {ex}")
            return
        self.learning.show_validation(res)

    def go_home(self):
        self.tabs.setCurrentWidget(self.home)

    def _go_compare(self):
        self.refresh_runs()
        self.tabs.setCurrentWidget(self.tab_compare)

    def _home_camera(self, index: int):
        j = self.panel.camera.findData(index)
        if j < 0:
            self.panel.camera.addItem(f"Caméra {index}", index)
            j = self.panel.camera.count() - 1
        self.panel.camera.setCurrentIndex(j)
        self.start_source("camera")

    # ==================================================== 2e caméra ==
    def toggle_cam2(self, checked: bool):
        if not checked:
            self.stop_cam2()
            return
        cams = [c for c in self.settings.get("cameras_cache", []) if isinstance(c, dict) and "index" in c]
        if len(cams) < 2:
            self._refresh_cameras(quick=False)
            cams = [c for c in self.settings.get("cameras_cache", []) if isinstance(c, dict) and "index" in c]
        main_idx = self.panel.camera_settings()["index"]
        if self.worker is not None and self.source_kind == "camera" and self.source_arg is not None:
            main_idx = int(self.source_arg)
        others = [c for c in cams if int(c["index"]) != int(main_idx)]
        if not others:
            self.a_cam2.setChecked(False)
            QMessageBox.information(self, "2e caméra", "Il faut au moins deux caméras branchées (ex. : la Logitech "
                                    "C930e et l'iPhone en Caméra de continuité). Branchez-la, puis réessayez.")
            return
        # Choix proposé : l'iPhone si la caméra principale n'en est pas un, sinon la première autre caméra
        others.sort(key=lambda c: (not c.get("phone"), int(c["index"])))
        names = [str(c.get("label") or f"Caméra {c['index']}") for c in others]
        name, ok = QInputDialog.getItem(self, "2e caméra", "Caméra à utiliser en même temps :", names, 0, False)
        if not ok:
            self.a_cam2.setChecked(False)
            return
        cam = others[names.index(name)]
        # La caméra principale démarre aussi si elle ne tourne pas encore
        if self.worker is None:
            self.start_source("camera")
        self.start_cam2(cam)

    def start_cam2(self, cam: dict):
        if self.worker2 is not None:
            old = self.worker2
            self.stop_cam2(save=False)
            old.wait(4000)
            self._zombies = getattr(self, "_zombies", []) + ([old] if old.isRunning() else [])
            self.worker2 = None
        idx = int(cam["index"])
        w = int(cam.get("width") or 1280)
        h = int(cam.get("height") or 720)
        set_ = self.panel.camera_settings()
        fps, backend = set_["fps"], set_["backend"]
        self.cam2_label = str(cam.get("name") or cam.get("label") or f"Caméra {idx}")
        cfg = self.panel.apply_to(self.cfg).copy()
        factory = lambda: CameraSource(idx, w, h, fps, backend)  # noqa: E731
        self.worker2 = ProcessingWorker(factory, cfg, f"camera:{idx} ({self.cam2_label})")
        self.worker2.view_mode = self.view_mode.currentData()
        wk2 = self.worker2
        self.worker2.frame_ready.connect(lambda img, st, wk2=wk2: wk2 is self.worker2 and self._on_frame2(img, st))
        self.worker2.run_finished.connect(self._on_cam2_finished)
        self.worker2.error.connect(lambda m: QMessageBox.warning(self, "2e caméra", m))
        self.worker2.info.connect(lambda m: self.cam2_info.setText(m))
        self.worker2.finished.connect(lambda wk2=wk2: self._on_cam2_done(wk2))
        self.cam2_title.setText(f"Caméra 2 — {self.cam2_label}")
        self.cam1_title.setText(f"Caméra 1 — {self._camera_name(self.panel.camera_settings()['index'])}")
        self.cam1_title.show()
        self.cam2_info.setText("Ouverture de la caméra…")
        self.cam2_box.show()
        self.video_split.setSizes([1, 1])
        self.worker2.start()
        self.a_cam2.setChecked(True)
        self._update_actions()
        self.tabs.setCurrentWidget(self.tab_live)
        self.statusBar().showMessage(f"2e caméra : {self.cam2_label} — apprentissage du fond…", 6000)

    def _camera_name(self, index) -> str:
        for c in self.settings.get("cameras_cache", []):
            if isinstance(c, dict) and int(c.get("index", -1)) == int(index):
                return str(c.get("name") or c.get("label") or f"Caméra {index}")
        return f"Caméra {index}"

    def stop_cam2(self, save: bool = True):
        w = self.worker2
        if w is None:
            return
        self._cam2_save = save
        w.stop()
        self.cam2_info.setText("Arrêt… calcul des résultats de la caméra 2.")

    def _on_frame2(self, qimg: QImage, st: dict):
        try:
            self.video2.set_image(qimg, st.get("native_wh"))
            counts = ", ".join(f"{physics.CLASS_INFO.get(c, {}).get('symbole', c)} {n}"
                               for c, n in (st.get("counts") or {}).items() if n) or "aucune trace pour l'instant"
            state = "apprentissage du fond…" if st.get("warming") else "détection active"
            self.cam2_info.setText(f"{counts} | {st.get('proc_fps', 0):.0f} img/s | {state}")
        finally:
            w = self.worker2
            if w is not None:
                w.frame_consumed()

    def _on_cam2_finished(self, run):
        if getattr(self, "_cam2_save", True) and run is not None and run.total > 0:
            try:
                run.name = f"{run.name} — caméra 2 ({self.cam2_label})"
            except Exception:
                pass
            self._save_run(run, quiet=True)

    def _on_cam2_done(self, wk2=None):
        self._zombies = [z for z in getattr(self, "_zombies", []) if z is not wk2]
        if wk2 is not None and wk2 is not self.worker2:
            return
        self.worker2 = None
        self.a_cam2.setChecked(False)
        self.cam2_box.hide()
        self.cam1_title.hide()
        self._update_actions()

    # ================================================== changer de caméra ==
    def switch_camera(self):
        """Passe à la caméra suivante, comme le bouton de FaceTime (Mac ↔ iPhone ↔ webcam USB).

        Pendant une acquisition en direct, la détection continue : seule la source change.
        Sinon, on choisit simplement la caméra suivante pour le prochain démarrage."""
        combo = self.panel.camera
        cams = [c for c in self.settings.get("cameras_cache", []) if isinstance(c, dict) and "index" in c]
        if len(cams) < 2:
            self._refresh_cameras(quick=False)
            cams = [c for c in self.settings.get("cameras_cache", []) if isinstance(c, dict) and "index" in c]
        if len(cams) < 2:
            QMessageBox.information(
                self, "Changer de caméra",
                "Une seule caméra est disponible pour le moment.\n\n"
                "Pour utiliser l'iPhone comme caméra (Caméra de continuité, comme dans FaceTime) :\n"
                "• même identifiant Apple sur le Mac et l'iPhone ;\n"
                "• Wi-Fi et Bluetooth activés sur les deux ;\n"
                "• iPhone à proximité, verrouillé et immobile (sur un support, au-dessus de la chambre) ;\n"
                "• iPhone : Réglages › Général › AirPlay et Continuité › Caméra de continuité activée.\n\n"
                "Un câble USB fonctionne aussi. Cliquez ensuite à nouveau sur « Changer de caméra ».")
            return
        cur = self.panel.camera_settings()["index"]
        w2 = self.worker2
        if w2 is not None:
            used = str(w2.source_name).split(":")[1].split(" ")[0] if ":" in str(w2.source_name) else ""
            cams = [c for c in cams if str(c["index"]) != used or int(c["index"]) == int(cur)]
        if not cams:
            return
        order = [int(c["index"]) for c in cams]
        nxt = cams[(order.index(cur) + 1) % len(cams)] if cur in order else cams[0]
        label = nxt.get("name") or nxt.get("label") or f"Caméra {nxt['index']}"
        j = combo.findData(int(nxt["index"]))
        if j >= 0:
            combo.setCurrentIndex(j)
        self.settings.set("cam_index", int(nxt["index"]))
        try:
            self.home.set_cameras(cams)
        except Exception:
            pass
        if self.worker is not None and self.source_kind == "camera":
            req = self.panel.camera_settings()
            req["index"], req["label"] = int(nxt["index"]), str(label)
            self.worker.send("switch_camera", req)
            self.statusBar().showMessage(f"Passage à {label}…", 6000)
        else:
            self.statusBar().showMessage(f"Caméra choisie : {label}. Cliquez sur « Caméra » pour démarrer.", 6000)

    def _on_camera_switched(self, index: int):
        self.source_arg = index
        self.settings.set("cam_index", int(index))

    def _video_defaults(self) -> dict:
        p = self.panel
        return {"condition": p.condition.currentText(), "sample": p.sample.currentText(),
                "mass_g": float(p.mass.value()), "px_per_mm": float(p.px_per_mm.value()),
                "realtime": self.realtime_chk.isChecked()}

    def _start_video_from_dict(self, d: dict):
        """Lance l'analyse d'une vidéo avec les informations saisies dans la fenêtre « Analyser une vidéo »."""
        path = str(d.get("path") or "")
        if not path:
            return
        p = self.panel
        if d.get("condition"):
            p.condition.setCurrentText(str(d["condition"]))
        if d.get("sample"):
            k = p.sample.findText(str(d["sample"]))
            if k >= 0:
                p.sample.setCurrentIndex(k)
        p.mass.setValue(float(d.get("mass_g") or 0.0))
        p.px_per_mm.setValue(float(d.get("px_per_mm") or 0.0))
        self.realtime_chk.setChecked(bool(d.get("realtime", False)))
        self._apply_settings()
        try:
            self.home.add_recent_video(path)
        except Exception:
            pass
        self.settings.set("last_dir", str(Path(path).parent))
        self.start_source("file", path)

    def _on_camera_changed(self, *_):
        """Cale la résolution sur celle réellement détectée pour la caméra choisie
        (évite de demander du 1920×1080 à une webcam 1280×720, ce qui empêchait le démarrage)."""
        label = self.panel.camera.currentText()
        m = re.search(r"(\d{3,5})\s*[×x]\s*(\d{3,5})", label)
        if not m:
            return
        res = f"{int(m.group(1))}×{int(m.group(2))}"
        combo = self.panel.resolution
        k = combo.findText(res)
        if k < 0:
            combo.insertItem(0, res)
            k = 0
        combo.setCurrentIndex(k)

    def _refresh_ports(self):
        self.panel.exp_port.clear()
        for p in SerialTrigger.ports():
            self.panel.exp_port.addItem(p)
        if self.cfg.expansion.serial_port:
            self.panel.exp_port.setEditText(self.cfg.expansion.serial_port)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une vidéo", self.settings.get("last_dir", ""),
                                              "Vidéos (*.mp4 *.avi *.mov *.mkv *.webm *.ogv *.wmv *.m4v);;Tous (*.*)")
        if path:
            self.settings.set("last_dir", str(Path(path).parent))
            dlg = VideoSetupDialog(path, self._video_defaults(), self)
            if dlg.exec():
                d = dlg.result_dict()
                d["path"] = d.get("path") or path
                self._start_video_from_dict(d)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Dossier de photos (rafale / time-lapse)",
                                                self.settings.get("last_dir", ""))
        if not path:
            return
        fps, ok = QInputDialog.getDouble(self, "Cadence des photos", "Nombre de photos par seconde :", 2.0, 0.05,
                                         120, 2)
        if ok:
            self.settings.set("last_dir", path)
            self.start_source("folder", (path, fps))

    def start_source(self, kind: str, arg=None, session: dict | None = None):
        if self.worker is not None:
            old = self.worker
            self.stop_source(wait=True)
            if old.isRunning():
                # Il finira tout seul ; on le garde en vie (détruire un QThread actif fait planter l'app)
                self._zombies = getattr(self, "_zombies", []) + [old]
        self.cfg = self.panel.apply_to(self.cfg)
        cfg = self.cfg.copy()
        cam = self.panel.camera_settings()
        self.settings.set("cam_index", cam["index"])
        self.settings.set("cam_fps", cam["fps"])
        self.settings.set("cam_res", self.panel.resolution.currentText())
        if kind == "camera":
            factory = lambda: CameraSource(cam["index"], cam["width"], cam["height"], cam["fps"], cam["backend"])  # noqa: E731
            name = f"camera:{cam['index']}"
        elif kind == "file":
            factory = lambda: FileSource(arg)  # noqa: E731
            name = str(arg)
        elif kind == "folder":
            factory = lambda: ImageFolderSource(arg[0], arg[1])  # noqa: E731
            name = str(arg[0])
        elif kind in ("demo", "demo_kcl"):
            from ..synth import SynthConfig
            sc = SynthConfig(duration_s=3600, seed=int(time.time()) % 10000, width=960, height=540,
                             rates_per_min={"alpha": 8, "beta": 12, "muon": 16, "proton": 0.3},
                             source_beta_per_min=45.0 if kind == "demo_kcl" else 0.0)
            factory = lambda: SynthSource(sc, realtime=True)  # noqa: E731
            name = "démonstration (simulation)" + (" avec source KCl" if kind == "demo_kcl" else "")
            if cfg.physics.px_per_mm <= 0:
                cfg.physics.px_per_mm = sc.px_per_mm
        else:
            return
        self.source_kind, self.source_arg = kind, arg
        self._reset_live()
        if session:
            sess = dict(session)
            if not sess.get("output_dir"):
                sess["output_dir"] = str(self._out_dir())
            self.worker = ProcessingWorker(factory, cfg, name, realtime_file=self.realtime_chk.isChecked(),
                                           session=sess)
            self.worker.session_status.connect(self.session_bar.update_status)
            self.worker.session_warning.connect(lambda s: self.statusBar().showMessage(f"⚠ {s}", 15000))
            self.session_bar.reset()
            self.session_bar.show()
        else:
            self.worker = ProcessingWorker(factory, cfg, name, realtime_file=self.realtime_chk.isChecked())
            self.session_bar.hide()
        self._on_view_mode()
        wk = self.worker
        self.worker.frame_ready.connect(lambda img, st, wk=wk: wk is self.worker and self._on_frame(img, st))
        self.worker.track_finished.connect(self._on_track)
        self.worker.run_finished.connect(lambda run, wk=wk: wk is self.worker and self._on_run_finished(run))
        self.worker.error.connect(self._on_worker_error)
        self.worker.info.connect(lambda s: self.statusBar().showMessage(s, 8000))
        self.worker.camera_switched.connect(self._on_camera_switched)
        self.worker.snapshot_saved.connect(lambda s: self.statusBar().showMessage(f"Photo enregistrée : {s}", 8000))
        self.worker.finished.connect(lambda wk=wk: self._on_worker_done(wk))
        self.worker.start()
        self._setup_serial(cfg)
        self.tabs.setCurrentWidget(self.tab_live)
        self._update_actions()
        self.statusBar().showMessage(f"Source : {name} — apprentissage du fond (quelques images)…")

    def _setup_serial(self, cfg: AppConfig):
        if self.serial is not None:
            self.serial.close()
            self.serial = None
        ex = cfg.expansion
        if ex.enabled and ex.trigger == "serie" and ex.serial_port:
            try:
                w = self.worker
                self.serial = SerialTrigger(ex.serial_port, ex.serial_baud,
                                            on_expansion=lambda: w is not None and w.send("trigger"))
                self.statusBar().showMessage(f"Arduino connecté sur {ex.serial_port}", 6000)
            except Exception as e:
                QMessageBox.warning(self, "Port série", f"Impossible d'ouvrir {ex.serial_port} : {e}")

    def stop_all(self, *_):
        """Bouton « Arrêter » : arrête la caméra principale et la 2e caméra."""
        self.stop_cam2()
        self.stop_source()

    def stop_source(self, wait: bool = False):
        if self.worker is None:
            return
        self.worker.stop()
        if wait:
            self.worker.wait(8000)
        self.statusBar().showMessage("Arrêt… calcul des résultats finaux.")

    def toggle_pause(self):
        if self.worker is None:
            return
        self.worker.set_paused(not self.worker.paused)
        self.statusBar().showMessage("En pause — utilisez « Demander à Claude » pour analyser l'image figée."
                                     if self.worker.paused else "Reprise.", 5000)

    def toggle_record(self):
        if self.worker is None:
            self.a_record.setChecked(False)
            return
        if self.a_record.isChecked():
            folder = Path(self.cfg.output_dir) / "Vidéos"
            name = safe_name(default_run_name(self.cfg.run.condition, self.cfg.run.trial))
            path = folder / f"{name}.mp4"
            k = 2
            while path.exists():
                path = folder / f"{name} ({k}).mp4"
                k += 1
            self.worker.send("record_start", str(path))
            self.rec_label.setText("⏺ ENREGISTREMENT")
        else:
            self.worker.send("record_stop")
            self.rec_label.setText("")

    def snapshot(self):
        if self.worker is None:
            return
        folder = Path(self.cfg.output_dir) / "Photos"
        folder.mkdir(parents=True, exist_ok=True)
        self.worker.send("snapshot", str(folder))

    def trigger_expansion(self):
        if self.serial is not None:
            self.serial.fire()
        elif self.worker is not None:
            self.worker.send("trigger")
        self.statusBar().showMessage("Expansion déclenchée.", 3000)

    def _update_actions(self):
        running = self.worker is not None
        for a in (self.a_pause, self.a_record, self.a_snap):
            a.setEnabled(running)
        self.a_stop.setEnabled(running or self.worker2 is not None)
        self.a_trigger.setEnabled(running and self.cfg.expansion.enabled)

    # ========================================================= callbacks ====
    def _reset_live(self):
        self.events = []
        self.run = None
        self.plot_hist = []
        self.table.setRowCount(0)
        self.gallery.clear()
        self.recent.clear()
        self.counters.update_counts({}, 0)
        if self.plot is not None:
            for c in self.curves.values():
                c.setData([], [])
        self.progress.setValue(0)
        self.a_record.setChecked(False)
        self.rec_label.setText("")

    def _on_view_mode(self, *_):
        if self.worker is not None:
            self.worker.view_mode = self.view_mode.currentData()
            self.worker.enhance = self.enhance.value() / 10.0
            self.worker.show_labels = self.labels_chk.isChecked()

    def _on_frame(self, qimg: QImage, st: dict):
        try:
            self._show_frame(qimg, st)
        finally:
            w = self.worker
            if w is not None:
                w.frame_consumed()

    def _show_frame(self, qimg: QImage, st: dict):
        self.last_stats = st
        self.video.set_image(qimg, st.get("native_wh"))
        self.counters.update_counts(st["counts"], st["duration_s"])
        if st.get("progress") is not None:
            self.progress.show()
            self.progress.setValue(int(100 * st["progress"]))
        else:
            self.progress.hide()
        state = "apprentissage du fond…" if st["warming"] else ("PERTURBATION" if st["disturbed"] else "détection active")
        cal = f"{st['px_per_mm']:.2f} px/mm" + ("" if st["calibrated"] else " (non calibré)")
        exp = f" | expansions : {st['expansions']}" if self.cfg.expansion.enabled else ""
        self.live_info.setText(
            f"t = {st['t']:.1f} s | image {st['frame'] + 1}"
            + (f"/{st['n_frames']}" if st['n_frames'] else "")
            + f" | {st['proc_fps']:.0f} img/s | bruit σ = {st['noise']:.2f} | traces actives : {st['active']}"
            f" | {state} | échelle {cal}{exp}")
        t = st["t"]
        if not self.plot_hist or t - self.plot_hist[-1][0] >= 1.0:
            self.plot_hist.append((t, dict(st["counts"])))
            if self.plot is not None:
                ts = [p[0] for p in self.plot_hist]
                for c, curve in self.curves.items():
                    curve.setData(ts, [p[1].get(c, 0) for p in self.plot_hist])

    def _on_track(self, e):
        self.events.append(e)
        self._add_table_row(e)
        info = physics.CLASS_INFO.get(e.final_cls, physics.CLASS_INFO["incertain"])
        it = QListWidgetItem(self._thumb_icon(e), f"#{e.id} {info['symbole']} {info['nom']}\n"
                                                  f"{e.features['length_mm']:.0f} mm — {e.confidence:.0%}")
        it.setData(Qt.ItemDataRole.UserRole, e.id)
        it.setForeground(QColor(B.class_text_hex(e.final_cls)))
        it.setToolTip(self._why_short(e))
        self.recent.insertItem(0, it)
        while self.recent.count() > 40:
            self.recent.takeItem(self.recent.count() - 1)

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "Erreur", msg)

    def _on_worker_done(self, wk=None):
        if wk is not None:
            self._zombies = [z for z in getattr(self, "_zombies", []) if z is not wk]
            if wk is not self.worker:
                return                       # ancien traitement qui finit en retard : ne touche à rien
        self.worker = None
        if self.serial is not None:
            self.serial.close()
            self.serial = None
        self.a_record.setChecked(False)
        self.rec_label.setText("")
        self._update_actions()

    def _on_run_finished(self, run: RunResult):
        self.run = run
        self.events = [e for e in run.events if e.valid]
        self._fill_tracks()
        self.counters.update_counts(run.counts(), run.duration_s)
        self.show_conclusions()
        self.refresh_charts()
        if getattr(run, "session", None):
            # Session de nuit : déjà sauvegardée (avec son rapport) dans le dossier de la session
            self.session_bar.set_finished()
            self.refresh_runs()
        elif self.panel.autosave.isChecked() and run.total > 0:
            self.save_results(quiet=True)
        self.statusBar().showMessage(
            f"Terminé : {run.total} traces en {run.duration_s / 60:.1f} min — voir l'onglet Conclusions.", 15000)

    # ============================================================ traces ====
    def _thumb_icon(self, e) -> QIcon:
        if e.thumb is None:
            return QIcon()
        img = e.thumb.copy()
        if e.thumb_mask is not None:
            col = physics.CLASS_INFO.get(e.final_cls, physics.CLASS_INFO["incertain"])["couleur"]
            m = cv2.dilate(e.thumb_mask, np.ones((2, 2), np.uint8)) > 0
            over = img.copy()
            over[m] = col
            img = cv2.addWeighted(img, 0.6, over, 0.4, 0)
        return QIcon(QPixmap.fromImage(bgr_to_qimage(img)))

    def _why(self, e):
        """Explication (« pourquoi cette classe ? ») d'une trace, mise en cache tant que sa classe ne change pas."""
        ph = self.cfg.physics
        key = (e.final_cls, round(float(e.confidence), 4), e.manual_cls, repr(self.cfg.classification),
               ph.chamber_temp_c, ph.pressure_kpa)
        cached = e.extra.get("_why_gui")
        if cached is not None and cached[0] == key:
            return cached[1]
        ex = explain_event(e, self.cfg.classification, self.cfg.physics)
        e.extra["_why_gui"] = (key, ex)
        return ex

    def _why_short(self, e) -> str:
        try:
            return self._why(e).short()
        except Exception:
            return e.reason

    def _add_table_row(self, e):
        flt = self.filter_cls.currentData()
        if flt and e.final_cls != flt:
            return
        self.table.setSortingEnabled(False)
        r = self.table.rowCount()
        self.table.insertRow(r)
        f = e.features
        why = self._why_short(e)
        vals = [e.id, physics.CLASS_INFO.get(e.final_cls, {}).get("nom", e.final_cls), f"{e.confidence:.0%}",
                round(e.t_first, 2), round(f.get("length_mm", 0), 1), round(f.get("width_mm", 0), 2),
                round(f.get("straightness", 0), 3), round(f.get("density_rel", 0), 2), round(e.duration_s, 2),
                round(f["energy_mev"], 2) if f.get("energy_mev") else "", why]
        col = QColor(B.class_text_hex(e.final_cls if e.final_cls in physics.CLASS_INFO else "incertain"))
        for c, v in enumerate(vals):
            it = QTableWidgetItem()
            it.setData(Qt.ItemDataRole.DisplayRole, v)
            if c == 1:
                it.setForeground(col)
            if c == 0:
                it.setData(Qt.ItemDataRole.UserRole, e.id)
            if c == 10:
                it.setToolTip(why)
            self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        gi = QListWidgetItem(self._thumb_icon(e), f"#{e.id} {e.final_cls}")
        gi.setData(Qt.ItemDataRole.UserRole, e.id)
        gi.setForeground(col)
        gi.setToolTip(why)
        self.gallery.addItem(gi)

    def _fill_tracks(self):
        self.table.setRowCount(0)
        self.gallery.clear()
        for e in self.events:
            if e.valid:
                self._add_table_row(e)

    def _event_by_id(self, eid):
        for e in self.events:
            if e.id == eid:
                return e
        return None

    def _selected_event(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows:
            it = self.table.item(rows[0].row(), 0)
            return self._event_by_id(it.data(Qt.ItemDataRole.UserRole)) if it else None
        return getattr(self, "_gallery_sel", None)

    def _on_track_selected(self):
        self._gallery_sel = None
        e = self._selected_event()
        if e is not None:
            self._show_detail(e)

    def _gallery_clicked(self, item):
        e = self._event_by_id(item.data(Qt.ItemDataRole.UserRole))
        self._gallery_sel = e
        if e is not None:
            self._show_detail(e)

    def _recent_clicked(self, item):
        e = self._event_by_id(item.data(Qt.ItemDataRole.UserRole))
        if e is not None:
            self._gallery_sel = e
            self.tabs.setCurrentWidget(self.tab_tracks)
            self._show_detail(e)

    def _show_detail(self, e):
        self._detail_event = e
        if e.thumb is not None:
            img = e.thumb
            h, w = img.shape[:2]
            f = min(3.0, 300 / max(h, w)) if max(h, w) < 300 else 1.0
            big = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC) if f != 1.0 else img
            self.detail_img.setPixmap(QPixmap.fromImage(bgr_to_qimage(big)))
        f = e.features
        info = physics.CLASS_INFO.get(e.final_cls, physics.CLASS_INFO["incertain"])
        try:
            why_html = self._why(e).to_html()
        except Exception as err:
            why_html = (f"<p style='color:{B.WARN_TEXT}'>Explication indisponible : {html.escape(str(err))}</p>"
                        f"<p>{html.escape(e.reason)}</p>")
        rows = [("Classe", f"<b style='color:{B.class_text_hex(e.final_cls)}'>{info['nom']}</b> "
                           f"(confiance {e.confidence:.0%})"
                 + (" — corrigée manuellement" if e.manual_cls else "")),
                ("Scores", ", ".join(f"{k} {v:.2f}" for k, v in sorted(e.scores.items(), key=lambda t: -t[1]))),
                ("Méthode", html.escape(str(e.extra.get("method") or "Règles physiques"))),
                ("Apparition", f"t = {e.t_first:.2f} s, visible {e.duration_s:.2f} s ({e.n_seen} images)"),
                ("Longueur", f"{f['length_mm']:.1f} mm (corde {f['chord_mm']:.1f} mm)"),
                ("Épaisseur (FWHM)", f"{f['width_mm']:.2f} mm (×{f.get('width_rel', 1):.2f} la référence)"),
                ("Densité d'ionisation", f"×{f.get('density_rel', 1):.2f} la référence des traces fines"),
                ("Rectitude", f"{f['straightness']:.3f} — flèche {f['sagitta_ratio']:.3f}, "
                              f"écart RMS {f['line_rms_mm']:.2f} mm, tortuosité {f['tortuosity']:.3f} rad/mm"),
                ("Contraste", f"{f['peak_snr']:.1f} σ"),
                ("Orientation", f"{f['orientation_deg']:.0f}°"),
                ("Touche le bord", "oui (longueur = minimum)" if e.touches_border else "non")]
        if f.get("energy_mev") and e.final_cls == "alpha":
            rows.append(("Énergie minimale", f"{f['energy_mev']:.2f} MeV (portée projetée : la trace peut sortir "
                                             "de la couche sensible, l'énergie réelle est ≥)"))
        elif f.get("energy_mev"):
            rows.append(("Énergie cinétique", f"{f['energy_mev']:.3f} MeV (d'après la courbure)"))
        if f.get("momentum_mev"):
            rows.append(("Impulsion (courbure)", f"{f['momentum_mev']:.3f} MeV/c, R = {f['radius_mm']:.1f} mm"))
        rows.append(("Règle appliquée", html.escape(e.reason)))
        rows.append(("Description", info["description"]))
        measures = (f"<h4 style='color:{B.GREEN}'>Mesures détaillées</h4><table cellpadding='3'>" + "".join(
            f"<tr><td style='color:{B.MUTED}' valign='top'>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>")
        self.detail_txt.setHtml(why_html + "<hr>" + measures)
        md = e.extra.get("claude_md", "")
        if md:
            self.detail_claude.setMarkdown(md)
            self.detail_claude.show()
        else:
            self.detail_claude.clear()
            self.detail_claude.hide()
        self.manual.setCurrentIndex(MANUAL_CHOICES.index(e.manual_cls) if e.manual_cls in MANUAL_CHOICES else 0)

    def _on_manual_class(self, idx: int):
        e = getattr(self, "_detail_event", None)
        if e is None:
            return
        choice = MANUAL_CHOICES[idx]
        if choice == "artefact":
            e.manual_cls = "artefact"
            e.valid = False
            e.reject_reason = "marquée comme artefact par l'utilisateur"
            self.events = [x for x in self.events if x is not e] + [e]
        else:
            e.manual_cls = "" if idx == 0 else choice
            e.valid = True
        self._fill_tracks()
        counts: dict[str, int] = {}
        for x in self.events:
            if x.valid:
                counts[x.final_cls] = counts.get(x.final_cls, 0) + 1
        dur = self.run.duration_s if self.run else self.last_stats.get("duration_s", 0)
        self.counters.update_counts(counts, dur)
        if self.run is not None:
            self.show_conclusions()
        if e.valid:
            self._show_detail(e)
        self.statusBar().showMessage(f"Trace #{e.id} : classe « {choice} » enregistrée.", 5000)

    def train_local_ai(self):
        labeled = [e for e in self.events if e.manual_cls and e.manual_cls != "artefact"]
        feats = [e.features for e in labeled]
        labels = [e.manual_cls for e in labeled]
        if self.use_confident.isChecked():
            for e in self.events:
                if e.valid and not e.manual_cls and e.confidence >= 0.85 and e.cls in CLASS_ORDER[:4]:
                    feats.append(e.features)
                    labels.append(e.cls)
        try:
            ml = MLClassifier()
            msg = ml.train(feats, labels)
        except Exception as ex:
            QMessageBox.warning(self, "Entraînement impossible", str(ex))
            return
        path = settings_dir() / "modele_ia.joblib"
        ml.save(str(path))
        self.cfg.classification.model_path = str(path)
        self.panel.model_label.setText(path.name)
        r = QMessageBox.question(self, "IA locale entraînée",
                                 f"Modèle entraîné sur {len(labels)} traces "
                                 f"({len(labeled)} corrigées à la main).\n{msg}\n\n"
                                 "Utiliser ce modèle pour classer les traces (méthode « IA locale ») ?")
        if r == QMessageBox.StandardButton.Yes:
            self.cfg.classification.mode = "ia"
            self.panel.load(self.cfg)
            if self.run is not None:
                self._reclassify_current()
        self._apply_settings()

    def _reclassify_current(self):
        from ..detection.classifier import classify_rules, compute_references, physics_annotations
        cfg = self.cfg
        ml = None
        if cfg.classification.mode == "ia" and cfg.classification.model_path:
            try:
                ml = MLClassifier.load(cfg.classification.model_path)
            except Exception:
                ml = None
        from types import SimpleNamespace

        from ..detection.pipeline import TrackDetector
        valid = [e for e in self.events if e.valid]
        refs = compute_references([e.features for e in valid], cfg.classification.ref_percentile)
        ppm = self.run.px_per_mm if self.run else 1.0
        # Même logique que le détecteur (règles, hybride v2 avec abstention/« trace fine », IA locale)
        for e in valid:
            sc = e.extra.get("scale", 1.0)
            shim = SimpleNamespace(cfg=cfg, refs=refs, calibrated=bool(e.extra.get("calibrated", False)),
                                   px_per_mm_proc=ppm * sc, ml=ml, ref_model=None, rejected=0)
            TrackDetector._classify_event(shim, e)
        self.events = [e for e in self.events if e.valid] + [e for e in self.events if not e.valid]
        self._fill_tracks()
        if self.run is not None:
            self.counters.update_counts(self.run.counts(), self.run.duration_s)
            self.show_conclusions()
            self.refresh_charts()

    def export_tables(self):
        if self.run is None and not self.events:
            QMessageBox.information(self, "Export", "Aucune trace à exporter.")
            return
        folder = self.save_results(quiet=True)
        if folder:
            QMessageBox.information(self, "Export", f"Fichiers traces.csv et traces.xlsx écrits dans :\n{folder}")

    # ========================================================== graphiques ==
    def _rows(self):
        if self.run is not None:
            return self.run.table(), self.run.duration_s
        return [e.to_row() for e in self.events if e.valid], self.last_stats.get("duration_s", 0.0)

    def refresh_charts(self):
        try:
            from ..analysis import plots
        except Exception as e:
            for c in self.charts.values():
                c.set_message(f"Module graphique indisponible : {e}")
            return
        rows, dur = self._rows()
        makers = {"lw": lambda: plots.fig_length_width(rows), "time": lambda: plots.fig_rate_timeseries(rows, max(dur, 1.0)),
                  "hist": lambda: plots.fig_length_hist(rows), "dens": lambda: plots.fig_density_width(rows),
                  "energy": lambda: plots.fig_alpha_energy(rows), "orient": lambda: plots.fig_orientation(rows)}
        for k, mk in makers.items():
            try:
                self.charts[k].set_figure(mk())
            except Exception as e:
                self.charts[k].set_message(f"Erreur : {e}")

    # ========================================================= conclusions ==
    def show_conclusions(self):
        run = self.run
        if run is None:
            return
        try:
            s, items = conclude_run(run)
        except Exception as e:
            self.concl.setHtml(f"<p>Erreur lors du calcul des conclusions : {html.escape(str(e))}</p>")
            return
        head = (f"<h2>{html.escape(run.name)}</h2><p style='color:{B.MUTED}'>Source : {html.escape(run.source)} — "
                f"{run.n_frames} images, {run.duration_s / 60:.1f} min analysées — "
                f"{'calibré' if run.calibrated else 'NON calibré'} ({run.px_per_mm:.3f} px/mm) — "
                f"zone analysée ≈ {run.roi_area_mm2 / 100:.0f} cm²</p>")
        rows = "".join(
            f"<tr><td><span style='color:{physics.CLASS_INFO[c]['hex']}'>●</span>&nbsp;"
            f"<b style='color:{B.class_text_hex(c)}'>{physics.CLASS_INFO[c]['nom']}</b></td>"
            f"<td align='right'>{s['classes'][c]['rate'].n}</td><td align='right'>{s['classes'][c]['rate'].fmt()}</td>"
            f"<td align='right'>{s['classes'][c]['per_cm2_min']:.4f}</td></tr>" for c in CLASS_ORDER)
        table = (f"<table cellpadding='5' border='0'><tr><th align='left'>Classe</th><th>N</th><th>Taux</th>"
                 f"<th>/cm²/min</th></tr>{rows}<tr><td><b>Total</b></td><td align='right'><b>{s['total'].n}</b></td>"
                 f"<td align='right'><b>{s['total'].fmt()}</b></td><td></td></tr></table>")
        self.concl.setHtml(head + table + "<h3>Conclusions</h3>" + conclusions_html(items))

    def _out_dir(self) -> Path:
        d = Path(self.cfg.output_dir or default_output_dir())
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_results(self, quiet: bool = False):
        if self.run is None:
            if not quiet:
                QMessageBox.information(self, "Sauvegarde", "Terminez d'abord l'enregistrement (bouton « Arrêter »).")
            return None
        return self._save_run(self.run, quiet)

    def _save_run(self, run, quiet: bool = False):
        saved, self.run = self.run, run      # le code de sauvegarde travaille sur self.run
        try:
            return self._save_current_run(quiet)
        finally:
            self.run = saved

    def _save_current_run(self, quiet: bool):
        try:
            if not self.run.folder:
                base = self._out_dir()
                name = safe_name(self.run.name)
                k = 2
                while (base / name).exists():
                    name = safe_name(f"{self.run.name} ({k})")
                    k += 1
                self.run.name = name
            folder = self.run.save(self._out_dir() if not self.run.folder else Path(self.run.folder).parent)
            if self.run.save_warnings:
                QMessageBox.warning(self, "Sauvegarde incomplète", "\n".join(self.run.save_warnings))
            try:
                from ..analysis.report import build_run_report
                build_run_report(self.run)
            except Exception as e:
                self.statusBar().showMessage(f"Rapport non généré : {e}", 8000)
            self.refresh_runs()
            if not quiet:
                QMessageBox.information(self, "Sauvegarde", f"Résultats enregistrés dans :\n{folder}")
            else:
                self.statusBar().showMessage(f"Résultats enregistrés dans {folder}", 12000)
            return folder
        except Exception as e:
            QMessageBox.critical(self, "Sauvegarde", f"Échec de la sauvegarde : {e}")
            return None

    def make_report(self, ai_text: str | None = None):
        if self.run is None:
            QMessageBox.information(self, "Rapport", "Terminez d'abord un enregistrement.")
            return
        if not self.run.folder:
            if self.save_results(quiet=True) is None:
                return
        try:
            from ..analysis.report import build_run_report
            p = build_run_report(self.run, ai_text=ai_text or None)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception as e:
            QMessageBox.critical(self, "Rapport", f"Impossible de créer le rapport : {e}")

    def open_run_folder(self):
        if self.run is not None and self.run.folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.run.folder))
        else:
            self.open_output_dir()

    def open_output_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._out_dir())))

    def open_results(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier d'un enregistrement (contenant run.json)",
                                             str(self._out_dir()))
        if not d:
            return
        try:
            run = RunResult.load(d)
        except Exception as e:
            QMessageBox.warning(self, "Ouvrir", f"Ce dossier ne contient pas de résultats valides : {e}")
            return
        self._show_loaded_run(run)

    def _show_loaded_run(self, run):
        self.run = run
        self.events = []
        self.table.setRowCount(0)
        self.gallery.clear()
        self._fill_rows_table(run.rows)
        self.counters.update_counts(run.counts(), run.duration_s)
        self.show_conclusions()
        self.refresh_charts()
        self.tabs.setCurrentWidget(self.tab_concl)

    def _fill_rows_table(self, rows):
        self.table.setSortingEnabled(False)
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            vals = [r.get("id"), r.get("classe"), r.get("confiance"), r.get("t_debut_s"), r.get("longueur_mm"),
                    r.get("epaisseur_mm"), r.get("rectitude"), r.get("densite_rel"), r.get("duree_visible_s"),
                    r.get("energie_MeV") or "", r.get("raison")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem()
                it.setData(Qt.ItemDataRole.DisplayRole, v)
                self.table.setItem(i, c, it)
        self.table.setSortingEnabled(True)

    # ========================================================== comparaison =
    def refresh_runs(self):
        dirs = [self._out_dir()] + [Path(p) for p in self.extra_run_dirs]
        found: dict[str, RunResult] = {}
        for base in dirs:
            if not base.exists():
                continue
            cands = [base] if (base / "run.json").exists() else [p for p in base.iterdir() if (p / "run.json").exists()]
            for p in cands:
                try:
                    found[str(p.resolve())] = RunResult.load(p)
                except Exception:
                    pass
        self.runs_index = sorted(found.values(), key=lambda r: r.date, reverse=True)
        self.runs_table.setRowCount(0)
        s_prev, b_prev = self.cmp_sample.currentText(), self.cmp_bg.currentText()
        self.cmp_sample.clear()
        self.cmp_bg.clear()
        for r in self.runs_index:
            i = self.runs_table.rowCount()
            self.runs_table.insertRow(i)
            c = r.counts()
            vals = [r.name, r.condition, r.sample_type, f"{r.duration_s / 60:.1f}", r.total,
                    f"{c.get('alpha', 0)} / {c.get('beta', 0)} / {c.get('muon', 0)}", r.folder]
            for j, v in enumerate(vals):
                self.runs_table.setItem(i, j, QTableWidgetItem(str(v)))
            self.cmp_sample.addItem(r.name, r.folder)
            self.cmp_bg.addItem(r.name, r.folder)
        if s_prev:
            self.cmp_sample.setCurrentText(s_prev)
        if b_prev:
            self.cmp_bg.setCurrentText(b_prev)
        else:
            j = next((k for k, r in enumerate(self.runs_index) if "fond" in r.condition.lower()
                      or "cosmique" in r.condition.lower()), -1)
            if j >= 0:
                self.cmp_bg.setCurrentIndex(j)
        try:
            recent = []
            for r in self.runs_index[:5]:
                recent.append({"name": r.name, "condition": r.condition, "total": r.total, "counts": r.counts(),
                               "duration_min": r.duration_s / 60.0, "folder": r.folder})
            self.home.set_recent_runs(recent)
        except Exception:
            pass

    def add_run_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier contenant des résultats", str(self._out_dir()))
        if d:
            if d not in self.extra_run_dirs:
                self.extra_run_dirs.append(d)
                self.settings.set("extra_run_dirs", self.extra_run_dirs)
            self.refresh_runs()

    def _cmp_pair(self):
        si, bi = self.cmp_sample.currentIndex(), self.cmp_bg.currentIndex()
        if si < 0 or bi < 0 or not self.runs_index:
            QMessageBox.information(self, "Comparer", "Sauvegardez d'abord au moins deux enregistrements.")
            return None
        if si == bi:
            QMessageBox.information(self, "Comparer", "Choisissez deux enregistrements différents.")
            return None
        return self.runs_index[si], self.runs_index[bi]

    def compare_runs(self):
        pair = self._cmp_pair()
        if pair is None:
            return
        sample, bg = pair
        comps, items = conclude_comparison(sample, bg)
        rows = []
        for c in CLASS_ORDER + ["fines", "total"]:
            k = comps[c]
            name = {"fines": "Traces fines (β+μ)", "total": "Total"}.get(c, physics.CLASS_INFO.get(c, {}).get("nom", c))
            sig = "oui" if k.significant else "non"
            rows.append(f"<tr><td>{name}</td><td align='right'>{k.a.fmt()}</td><td align='right'>{k.b.fmt()}</td>"
                        f"<td align='right'>{k.diff:+.2f} ± {k.diff_err:.2f}</td><td align='right'>{k.p_value:.2g}</td>"
                        f"<td align='right'>{k.sigma:.1f}</td><td>{sig}</td></tr>")
        table = ("<table cellpadding='4'><tr><th align='left'>Classe</th><th>Échantillon</th><th>Fond</th>"
                 "<th>Excès (/min)</th><th>p</th><th>σ</th><th>Significatif</th></tr>" + "".join(rows) + "</table>")
        self.cmp_text.setHtml(f"<h3>{html.escape(sample.name)}<br><span style='color:{B.MUTED}'>contre</span> "
                              f"{html.escape(bg.name)}</h3>" + table + "<h3>Conclusions</h3>" + conclusions_html(items))
        try:
            from ..analysis import plots
            self.cmp_chart.set_figure(plots.fig_comparison([bg, sample]))
        except Exception as e:
            self.cmp_chart.set_message(f"Graphique indisponible : {e}")

    def compare_report(self, ai_text: str | None = None):
        pair = self._cmp_pair()
        if pair is None:
            return
        sample, bg = pair
        try:
            from ..analysis.report import build_comparison_report
            out = Path(sample.folder) / f"comparaison avec {safe_name(bg.name)}.html"
            if len(str(out)) > 240:          # limite des chemins Windows (260 caractères)
                out = Path(sample.folder) / "comparaison.html"
            p = build_comparison_report(sample, bg, out, ai_text=ai_text or None)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception as e:
            QMessageBox.critical(self, "Rapport", f"Impossible de créer le rapport : {e}")

    # ======================================================= assistant IA ====
    def _get_assistant(self):
        from ..ai_assistant import AssistantError, ClaudeAssistant
        key = self.api_key.text().strip() or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise AssistantError("Entrez votre clé API Anthropic dans l'onglet « Assistant IA ».")
        if self.remember_key.isChecked():
            self.settings.set("anthropic_api_key", key)
        else:
            self.settings.data.pop("anthropic_api_key", None)
        model = self._ai_model_id()
        effort = self.ai_effort.currentData() or self.ai_effort.currentText()
        self.settings.set("ai_model", model)
        self.settings.set("ai_effort", effort)
        a = self.assistant
        if a is None or getattr(a, "_key", None) != key or a.model != model:
            a = ClaudeAssistant(key, model=model, effort=effort)
            a._key = key
            self.assistant = a
        a.effort = effort
        return a

    def _ai_model_id(self) -> str:
        return str(self.ai_model.currentData() or self.ai_model.currentText() or "claude-opus-5-5")

    def _ai_call(self, label: str, fn, on_done, switch_tab: bool = True, on_error=None):
        try:
            self._get_assistant()
        except Exception as e:
            self.tabs.setCurrentWidget(self.tab_ai)
            QMessageBox.information(self, "Assistant IA", str(e))
            return False
        self.ai_status.setText(f"⏳ Claude réfléchit… ({label})")
        self.statusBar().showMessage(f"⏳ Claude réfléchit… ({label})")
        if switch_tab:
            self.tabs.setCurrentWidget(self.tab_ai)
        w = CallWorker(fn)
        self._calls.append(w)

        def ok(res):
            self.ai_status.setText("")
            self.statusBar().clearMessage()
            on_done(res)
            self._calls.remove(w)

        def ko(msg):
            self.ai_status.setText("")
            self.statusBar().showMessage(f"Claude : {msg}", 10000)
            self._ai_append("Erreur", f"❌ {msg}")
            if on_error is not None:
                on_error(msg)
            self._calls.remove(w)
        w.done.connect(ok)
        w.failed.connect(ko)
        w.start()
        return True

    def _ai_append(self, who: str, md: str):
        self.ai_log.append(f"### {who}\n\n{md}")
        self.ai_out.setMarkdown("\n\n---\n\n".join(self.ai_log))
        sb = self.ai_out.verticalScrollBar()
        sb.setValue(sb.maximum())

    def ai_test(self):
        self._ai_call("test", lambda: self.assistant.test_connection(), lambda r: self._ai_append("Claude", r))

    def ai_reset(self):
        if self.assistant is not None:
            self.assistant.reset()
        self.ai_log = []
        self.ai_out.clear()

    def _verdict_md(self, v) -> str:
        lines = [f"**Résumé :** {v.summary}", "", "| # | Classe (Claude) | Confiance | Justification |", "|---|---|---|---|"]
        for t in v.traces:
            lines.append(f"| {t.get('id')} | {t.get('classe')} | {float(t.get('confiance', 0)):.0%} | "
                         f"{str(t.get('justification', '')).replace('|', '/')} |")
        lines += ["", f"**Traces possiblement manquées :** {v.missed}", "", f"**Qualité de l'image :** {v.quality}"]
        if v.advice:
            lines += ["", "**Conseils pour le montage :**"] + [f"- {a}" for a in v.advice]
        return "\n".join(lines)

    def ai_analyze_frame(self):
        img = None
        if self.worker is not None and self.worker.last_annotated is not None:
            img = self.worker.last_annotated.copy()
        elif self.video.current_image() is not None and not self.video.current_image().isNull():
            img = qimage_to_bgr(self.video.current_image())
        if img is None:
            QMessageBox.information(self, "Assistant IA", "Aucune image à analyser : démarrez une source.")
            return
        t_now = self.last_stats.get("t", 0.0)
        recent = [e for e in self.events if e.valid and t_now - e.t_last <= 1.3]
        rows = [e.to_row() for e in recent]
        ctx = f"Condition : {self.cfg.run.condition}. Échantillon : {self.cfg.run.sample_type}."
        self._ai_append("Vous", f"Analyse de l'image actuelle ({len(rows)} trace(s) numérotée(s)).")
        self._ai_call("image", lambda: self.assistant.analyze_frame(img, rows, ctx),
                      lambda v: self._ai_append("Claude", self._verdict_md(v)))

    def _track_measurements(self, e) -> dict:
        f = e.features
        ph = self.cfg.physics
        return {
            "id": e.id,
            "classe_logiciel": e.final_cls,
            "confiance_logiciel": round(float(e.confidence), 3),
            "methode": e.extra.get("method") or "Règles physiques",
            "longueur_mm": round(float(f.get("length_mm", 0.0)), 1),
            "corde_mm": round(float(f.get("chord_mm", 0.0)), 1),
            "epaisseur_mm": round(float(f.get("width_mm", 0.0)), 2),
            "epaisseur_relative": round(float(f.get("width_rel", 1.0)), 2),
            "densite_ionisation_relative": round(float(f.get("density_rel", 1.0)), 2),
            "rectitude": round(float(f.get("straightness", 0.0)), 3),
            "fleche_relative": round(float(f.get("sagitta_ratio", 0.0)), 3),
            "tortuosite_rad_mm": round(float(f.get("tortuosity", 0.0)), 3),
            "contraste_sigma": round(float(f.get("peak_snr", 0.0)), 1),
            "duree_visible_s": round(float(e.duration_s), 2),
            "touche_le_bord": bool(e.touches_border),
            "calibre": bool(e.extra.get("calibrated", False)),
            "temperature_chambre_c": ph.chamber_temp_c,
        }

    def ai_analyze_track(self):
        e = getattr(self, "_detail_event", None)
        if e is None or e.thumb is None:
            QMessageBox.information(self, "Assistant IA", "Sélectionnez d'abord une trace dans le tableau.")
            return
        try:
            local = self._why(e).to_text()
        except Exception:
            local = e.reason
        meas = self._track_measurements(e)
        ctx = (f"Condition : {self.cfg.run.condition}. Échantillon : {self.cfg.run.sample_type}. "
               f"Image recadrée sur une seule trace (vue de dessus, fond noir, éclairage latéral).")
        thumb = e.thumb.copy()

        def done(op):
            from ..ai_assistant import opinion_markdown
            names = {c: physics.CLASS_INFO[c]["nom"] for c in physics.CLASS_INFO}
            md = opinion_markdown(op, names)
            e.extra["claude_md"] = md
            self._ai_append("Claude", f"**Trace #{e.id}**\n\n{md}")
            if getattr(self, "_detail_event", None) is e:
                self.detail_claude.setMarkdown(md)
                self.detail_claude.show()

        def failed(msg):
            if getattr(self, "_detail_event", None) is e:
                self.detail_claude.setMarkdown(f"❌ {msg}")

        if self._ai_call("trace", lambda: self.assistant.explain_track(thumb, meas, local, ctx), done,
                         switch_tab=False, on_error=failed):
            self._ai_append("Vous", f"Pourquoi la trace #{e.id} est-elle « {e.final_cls} » ? (2e avis demandé)")
            self.detail_claude.setMarkdown("⏳ *Claude examine la trace…*")
            self.detail_claude.show()

    def ai_interpret_run(self):
        if self.run is None:
            QMessageBox.information(self, "Assistant IA", "Terminez d'abord un enregistrement.")
            return
        run = self.run
        try:
            from ..analysis.report import report_text
            txt = report_text(run)
        except Exception:
            _s, items = conclude_run(run)
            txt = "\n".join(c.as_markdown() for c in items)
        self._ai_append("Vous", f"Interprétation de l'enregistrement « {run.name} ».")

        def done(ans):
            self._ai_append("Claude", ans)
            self._last_ai_run_text = ans
        self._ai_call("interprétation", lambda: self.assistant.interpret(txt), done)

    def ai_interpret_comparison(self):
        pair = self._cmp_pair()
        if pair is None:
            return
        sample, bg = pair
        try:
            from ..analysis.report import comparison_text
            txt = comparison_text(sample, bg)
        except Exception:
            _c, items = conclude_comparison(sample, bg)
            txt = "\n".join(c.as_markdown() for c in items)
        self._ai_append("Vous", f"Interprétation de la comparaison « {sample.name} » / « {bg.name} ».")
        self._ai_call("comparaison", lambda: self.assistant.interpret(txt), lambda a: self._ai_append("Claude", a))

    def ai_chat(self):
        q = self.ai_q.text().strip()
        if not q:
            return
        self.ai_q.clear()
        ctx = ""
        if self.run is not None:
            try:
                from ..analysis.report import report_text
                ctx = report_text(self.run)
            except Exception:
                ctx = ""
        self._ai_append("Vous", q)
        self._ai_call("question", lambda: self.assistant.chat(q, ctx), lambda a: self._ai_append("Claude", a))

    # ========================================================== réglages ====
    def _apply_settings(self):
        self.cfg = self.panel.apply_to(self.cfg)
        self.settings.store_config(self.cfg)
        self.settings.save()
        if self.worker is not None:
            self.worker.send("live_config", self.cfg.copy())
        self._update_actions()

    def _start_draw(self, mode: str):
        if self.video.current_image() is None:
            QMessageBox.information(self, "Dessin", "Démarrez d'abord une source pour voir l'image.")
            return
        self.tabs.setCurrentWidget(self.tab_live)
        if self.worker is not None and not self.worker.paused:
            self.worker.set_paused(True)
            self._resume_after_draw = True
        else:
            self._resume_after_draw = False
        self.video.set_mode(mode)
        self.video.setFocus()
        self.video.mode_finished.connect(self._after_draw)

    def _after_draw(self):
        try:
            self.video.mode_finished.disconnect(self._after_draw)
        except (RuntimeError, TypeError):
            pass
        if self.worker is not None and getattr(self, "_resume_after_draw", False):
            self.worker.set_paused(False)

    def set_roi(self, pts: list):
        self.cfg.detection.roi = [[round(x, 4), round(y, 4)] for x, y in pts]
        self.panel.set_roi_text(self.cfg.detection.roi)
        self._apply_settings()
        self._offer_restart("La zone d'analyse a changé.")

    def clear_roi(self):
        self.cfg.detection.roi = []
        self.panel.set_roi_text([])
        self._apply_settings()
        self._offer_restart("La zone d'analyse a été effacée.")

    def set_calibration(self, d_px: float):
        mm, ok = QInputDialog.getDouble(self, "Calibration",
                                        f"Distance mesurée : {d_px:.1f} px.\nLongueur réelle entre les deux points (mm) :",
                                        10.0, 0.1, 5000, 2)
        if not ok or mm <= 0:
            return
        ppm = d_px / mm
        self.panel.px_per_mm.setValue(ppm)
        self._apply_settings()
        self._offer_restart(f"Échelle : {ppm:.3f} px/mm.")

    def _offer_restart(self, why: str):
        if self.worker is None:
            self.statusBar().showMessage(why + " Elle sera utilisée au prochain démarrage.", 8000)
            return
        r = QMessageBox.question(self, "Appliquer", why + "\n\nRedémarrer l'analyse maintenant pour l'appliquer ?\n"
                                 "(les compteurs repartiront de zéro)")
        if r == QMessageBox.StandardButton.Yes:
            kind, arg = self.source_kind, self.source_arg
            self.stop_source(wait=True)
            QTimer.singleShot(200, lambda: self.start_source(kind, arg))

    def save_preset(self):
        self.cfg = self.panel.apply_to(self.cfg)
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer un préréglage", str(self._out_dir() / "preglage.json"),
                                              "Préréglage (*.json)")
        if path:
            self.cfg.save(path)
            self.statusBar().showMessage(f"Préréglage enregistré : {path}", 6000)

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger un préréglage", str(self._out_dir()), "Préréglage (*.json)")
        if path:
            try:
                self.cfg = AppConfig.load(path)
                if not self.cfg.output_dir:
                    self.cfg.output_dir = str(default_output_dir())
                self.panel.load(self.cfg)
                self._apply_settings()
                self._offer_restart("Préréglage chargé.")
            except Exception as e:
                QMessageBox.warning(self, "Préréglage", f"Fichier invalide : {e}")

    def reset_defaults(self):
        out = self.cfg.output_dir
        self.cfg = AppConfig()
        self.cfg.output_dir = out
        self.panel.load(self.cfg)
        self._apply_settings()

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier des résultats", str(self._out_dir()))
        if d:
            self.panel.out_dir.setText(d)
            self._apply_settings()
            self.refresh_runs()

    # ============================================================ outils ====
    def tool_generate_video(self):
        path, _ = QFileDialog.getSaveFileName(self, "Vidéo de démonstration", str(self._out_dir() / "demo.mp4"),
                                              "MP4 (*.mp4)")
        if not path:
            return
        dur, ok = QInputDialog.getInt(self, "Durée", "Durée (secondes) :", 60, 5, 3600)
        if not ok:
            return
        src, ok = QInputDialog.getDouble(self, "Source simulée",
                                         "Traces β supplémentaires par minute (0 = fond seul, 40 ≈ KCl) :", 0, 0, 1000, 1)
        if not ok:
            return
        from ..synth import SynthConfig, generate_video
        sc = SynthConfig(duration_s=dur, source_beta_per_min=src)
        self.statusBar().showMessage("Génération de la vidéo…")
        w = CallWorker(generate_video, path, sc)
        self._calls.append(w)
        w.done.connect(lambda _r: (self.statusBar().showMessage(f"Vidéo générée : {path} "
                                                                 f"(échelle {sc.px_per_mm:.2f} px/mm)", 15000),
                                   self._calls.remove(w)))
        w.failed.connect(lambda m: (QMessageBox.warning(self, "Génération", m), self._calls.remove(w)))
        w.start()

    def tool_validate(self):
        from ..analysis.run import analyze_video
        from ..evaluate import evaluate, format_report
        from ..synth import SynthConfig, generate_video

        def job():
            tmp = settings_dir() / "validation.mp4"
            sc = SynthConfig(duration_s=45, seed=42)
            gt = generate_video(tmp, sc)
            cfg = self.panel.apply_to(self.cfg.copy())
            cfg.detection.roi = []
            cfg.physics.px_per_mm = gt["px_per_mm"]
            run, _det = analyze_video(str(tmp), cfg)
            return format_report(evaluate(run.events, gt))
        self.statusBar().showMessage("Validation en cours (≈ 1–2 min)…")
        w = CallWorker(job)
        self._calls.append(w)
        w.done.connect(lambda r: (QMessageBox.information(self, "Validation de l'algorithme", r), self._calls.remove(w),
                                  self.statusBar().clearMessage()))
        w.failed.connect(lambda m: (QMessageBox.warning(self, "Validation", m), self._calls.remove(w)))
        w.start()

    def about(self):
        QMessageBox.about(self, f"À propos de {APP_NAME}",
                          f"<h3>{APP_NAME} {__version__}</h3>"
                          "<p>Détection, classification et analyse statistique des traces de particules "
                          "dans une chambre de Wilson.</p>"
                          "<p>Projet d'intégration en sciences — Cégep Marie-Victorin.<br>"
                          "Équipe : Yanis Amarouche, Yanees Ait Saadi, Amir Badri, Taha Bich.</p>"
                          "<p>Méthodes : fond adaptatif, filtre de lignes hessien multi-échelle, suivi temporel, "
                          "mesures à mi-hauteur, classification physique expliquée (dE/dx, diffusion multiple, "
                          "portée-énergie) combinée à une IA de référence entraînée sur des traces simulées, "
                          "statistiques de Poisson. Second avis optionnel : Claude Opus 5.5 (Anthropic).</p>")

    def closeEvent(self, ev):
        threads = [t for t in (self.worker2, self.worker, *getattr(self, "_zombies", [])) if t is not None]
        for t in threads:
            t.stop()
        for t in threads:
            t.wait(15000)
        # Un fil encore bloqué (caméra figée dans son pilote) ferait planter Qt à la sortie (SIGABRT) :
        # run_gui quitte alors directement, une fois les réglages enregistrés ci-dessous.
        self.unclean_exit = any(t.isRunning() for t in threads)
        if getattr(self, "batch_ctl", None) is not None:
            try:
                self.batch_ctl.stop()
            except Exception:
                pass
        if self.serial is not None:
            self.serial.close()
        self.cfg = self.panel.apply_to(self.cfg)
        self.settings.store_config(self.cfg)
        self.settings.set("geometry", bytes(self.saveGeometry()).hex())
        if not self.remember_key.isChecked():
            self.settings.data.pop("anthropic_api_key", None)
        self.settings.save()
        ev.accept()

    def showEvent(self, ev):
        super().showEvent(ev)
        QTimer.singleShot(300, self.refresh_runs)


def QApplicationProcess():
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()


def _readable_class_colors(text: str) -> str:
    """Remplace les couleurs de classe utilisées comme couleur de TEXTE par leur version lisible sur blanc."""
    for c, info in physics.CLASS_INFO.items():
        hx = info.get("hex")
        if hx:
            text = text.replace(f"color:{hx}", f"color:{B.class_text_hex(c)}")
    return text


def _reference_model_help() -> str:
    """Section d'aide décrivant l'IA de référence réellement embarquée (chiffres lus dans le modèle)."""
    try:
        from ..detection.reference_model import ReferenceModel
        m = ReferenceModel.load_default()
    except Exception:
        m = None
    if m is None:
        return ("<h3>IA de référence</h3><p>Le modèle de référence n'est pas disponible dans cette installation : "
                "la classification utilise les règles physiques seules.</p>")
    info = getattr(m, "info", {}) or {}
    n = info.get("n_tracks_total") or info.get("n_train") or 0
    met = info.get("metrics") or {}
    acc = None
    for key in ("hybrid_test_balanced_accuracy", "test_hybrid_balanced_accuracy"):
        if isinstance(met, dict) and met.get(key) is not None:
            acc = met.get(key)
    txt = (f"<h3>IA de référence</h3><p>Ce modèle a été entraîné sur <b>{int(n):,}</b> traces simulées ".replace(",", " ")
           + "(vérité terrain connue) dans des conditions très variées : éclairage, bruit du capteur, brouillard, "
             "échelle, température, épaisseur de la couche sensible, avec ou sans calibration.")
    if acc is not None:
        txt += f" Exactitude équilibrée mesurée sur des simulations jamais vues : <b>{100 * float(acc):.0f} %</b>."
    return txt + "</p>"


HELP_HTML = """
<h2>WilsonTrack — guide rapide</h2>
<h3>0. Le menu d'accueil</h3>
<p>L'onglet <b>Accueil</b> regroupe tout : caméra en direct, <b>analyser une vidéo</b> (bouton « Choisir une
vidéo… » ou glisser-déposer du fichier), série de photos, démonstration, comparaison des résultats et assistant Claude.
Avant chaque analyse de vidéo, une fenêtre demande la condition (fond, KCl, banane…), l'échantillon, sa masse et
l'échelle (px/mm). Pour les vidéos d'exemple, l'échelle est remplie automatiquement.</p>
<h3>1. Préparer la chambre et la caméra</h3>
<ul>
<li>Caméra fixée <b>au-dessus</b> de la chambre, immobile (trépied), mise au point sur la couche de brouillard juste
au-dessus de la plaque froide. Fond noir mat, LED sur les côtés, pièce sombre.</li>
<li>Dans l'accueil (ou <b>Réglages › Caméra</b>, bouton « Réglages » de l'en-tête), choisissez la caméra puis
cliquez <b>Démarrer la caméra</b>. La résolution
se règle automatiquement sur celle de la caméra. Ajustez l'exposition et le focus avec « Réglages du pilote »
(désactivez l'exposition automatique si possible). Un seul logiciel à la fois peut utiliser la webcam : fermez Zoom,
Teams ou l'application Caméra de Windows.</li>
</ul>
<h3>2. Délimiter la zone et calibrer</h3>
<ul>
<li><b>Zone analysée</b> : tracez un rectangle ou un polygone qui exclut les parois, reflets et objets brillants.</li>
<li><b>Calibrer</b> : filmez une règle posée sur la plaque, cliquez « Calibrer », puis cliquez deux graduations et
entrez la distance réelle (mm). Indispensable pour des longueurs et énergies en unités physiques.</li>
</ul>
<h3>3. Mesurer</h3>
<ul>
<li>Renseignez la <b>condition</b> (Fond, KCl, Banane…), l'échantillon, sa masse et le numéro d'essai.</li>
<li>Les traces sont entourées en direct : <span style='color:#e5484d'>α alpha</span>,
<span style='color:#2f9bff'>β électron</span>, <span style='color:#30a46c'>μ muon</span>,
<span style='color:#f76b15'>p proton</span>, <span style='color:#8b8d98'>? incertaine</span>.</li>
<li><b>Enregistrer la vidéo</b> conserve la vidéo brute (nommée « date condition essai-NN.mp4 »), réanalysable plus tard.</li>
<li>Enregistrez <b>au moins 10–20 minutes</b> par condition, avec les mêmes réglages, pour des comparaisons valables.</li>
<li>Cliquez <b>Arrêter</b> : les classes sont recalculées avec des références finales, puis les résultats,
le tableau (CSV/Excel), les vignettes et le rapport HTML sont sauvegardés.</li>
</ul>
<h3>4. Conclure</h3>
<ul>
<li><b>Conclusions</b> : taux ± incertitude (Poisson), test du caractère aléatoire, énergie des alphas, muons…</li>
<li><b>Comparer</b> : choisissez un enregistrement « échantillon » (KCl) et une « référence » (fond) : excès net,
signification statistique (σ, p), activité attendue du K-40 et efficacité apparente.</li>
</ul>
<h3>5. Comment l'application reconnaît chaque particule ?</h3>
<p>Dans l'onglet <b>Traces</b>, cliquez une trace : le panneau de droite explique <b>pourquoi</b> elle a été
classée ainsi (mesures, critère par critère, et pourquoi ce n'est pas une autre particule).</p>
<table cellpadding='4'>
<tr><td style='color:#e5484d'><b>α alpha</b></td><td>courte (quelques cm au plus), <b>épaisse</b>, très
<b>dense</b> en gouttelettes, droite. Noyau d'hélium de charge 2, lent : il ionise des centaines de fois plus
qu'un muon (formule de Bethe : ionisation ∝ z²/β²) et s'arrête vite (relation portée-énergie).</td></tr>
<tr><td style='color:#2f9bff'><b>β électron</b></td><td><b>fine</b> et souvent <b>sinueuse</b> ou en zigzag.
Particule très légère : chaque collision la dévie (diffusion multiple). Source typique : K-40 du KCl et de la banane.</td></tr>
<tr><td style='color:#30a46c'><b>μ muon</b></td><td><b>fine, longue et parfaitement droite</b>. Particule cosmique
très énergétique (≈ 4 GeV) au minimum d'ionisation, presque pas déviée. Environ 1 par cm² par minute.</td></tr>
<tr><td style='color:#f76b15'><b>p proton</b></td><td>épaisse et droite comme une alpha, mais plus longue que la
portée maximale d'une alpha naturelle. Rare.</td></tr>
<tr><td style='color:#8b8d98'><b>? incertaine</b></td><td>les mesures ne tranchent pas. Cas classique : une trace
fine, courte et droite peut être un muon presque vertical ou un électron énergétique (ambiguïté physique réelle).</td></tr>
</table>
<p><b>Méthode « Hybride » (par défaut si disponible)</b> : les règles physiques ci-dessus et une <b>IA de référence</b>
votent ensemble ; si elles ne sont pas d'accord, la confiance baisse et l'explication le signale. Le bouton
<b>Pourquoi ? — 2e avis de Claude Opus 5.5</b> fait examiner la vignette par Claude, qui donne sa propre
classification et la justifie (clé API nécessaire).</p>
<p><b>Pourquoi pas « des millions de vidéos en ligne » ?</b> Il n'existe pas de grande banque publique de vidéos de
chambres de Wilson <i>étiquetées</i> (avec la vraie particule pour chaque trace) ; des vidéos non étiquetées
n'apprendraient pas les bonnes réponses, et les télécharger en masse poserait des problèmes de droits d'auteur.
L'IA de référence est donc entraînée sur un très grand nombre de traces <b>simulées</b> à partir de la physique,
dont on connaît la vraie nature, dans des conditions très variées. Vous pouvez ensuite l'adapter à <i>votre</i>
chambre : corrigez quelques traces, puis « Entraîner l'IA locale ».</p>
<h3>6. Si les détections ne sont pas bonnes</h3>
<ul>
<li>Affichage « Contraste (image − fond) » ou « Carte de détection » pour voir ce que voit l'algorithme.</li>
<li>Trop de fausses détections → baissez la <b>sensibilité</b> ou augmentez « Contraste min. ». Traces manquées →
augmentez la sensibilité, améliorez l'éclairage rasant.</li>
<li>Corrigez les classes erronées dans l'onglet <b>Traces</b>, puis « Entraîner l'IA locale » : le logiciel apprend
à partir de vos corrections.</li>
</ul>
<h3>Raccourcis</h3>
<p>Ctrl+H : accueil · Espace : pause · R : enregistrer la vidéo · P : photo · E : expansion · Ctrl+K : caméra ·
Ctrl+O : vidéo · Ctrl+D : démo · Ctrl+, : réglages · Ctrl+I : Claude · F1 : aide</p>
"""
