"""Panneau de réglages : synchronisé avec AppConfig.

Seuls les réglages essentiels sont visibles (Expérience, Caméra, Échelle et zone) ; tout le reste (détection,
classification, physique, chambre à expansion, dossier, préréglages…) est rangé dans une seule section repliée
« Réglages avancés ». Aucun widget n'a été supprimé : la fenêtre principale y accède par leur nom."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QToolButton, QVBoxLayout, QWidget)

from .. import physics
from ..config import AppConfig
from . import brand as B

CONDITIONS = ["Fond (rayons cosmiques)", "KCl", "Banane", "Substitut de sel", "Source de test", "Autre"]
CLS_MODES = [("Hybride : règles + IA de référence", "hybride"), ("Règles physiques seules", "regles"),
             ("IA locale (vos corrections)", "ia")]
PROCESS_WIDTHS = [("640 px (rapide)", 640), ("960 px (recommandé)", 960), ("1280 px (précis)", 1280),
                  ("Résolution native (lent)", 0)]


class Collapsible(QWidget):
    """Section repliable : titre cliquable (chevron) + carte blanche contenant un formulaire."""

    def __init__(self, title: str, expanded: bool = True):
        super().__init__()
        self.btn = QToolButton(text=title, checkable=True, checked=expanded)
        self.btn.setObjectName("sectionToggle")
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn.toggled.connect(self._toggle)
        self.body = QGroupBox()
        self.body.setObjectName("sectionBody")
        self.body.setVisible(expanded)
        self.form = QFormLayout(self.body)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setContentsMargins(14, 12, 14, 14)
        self.form.setHorizontalSpacing(12)
        self.form.setVerticalSpacing(9)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.btn)
        lay.addWidget(self.body)
        self._set_icon(expanded)

    def _set_icon(self, on: bool):
        from PySide6.QtGui import QTransform
        pm = B.icon_pixmap("chevron", B.FOREST, 16, 2.0)
        if on:
            pm = pm.transformed(QTransform().rotate(90), Qt.TransformationMode.SmoothTransformation)
            pm.setDevicePixelRatio(2.0)
        self.btn.setIcon(QIcon(pm))

    def _toggle(self, on: bool):
        self.body.setVisible(on)
        self._set_icon(on)


PANEL_QSS = f"""
QWidget#SettingsRoot {{ background: {B.SOFT}; }}
QToolButton#sectionToggle {{ border: none; background: transparent; color: {B.FOREST}; font-size: 14px;
    font-weight: 700; padding: 6px 2px; text-align: left; }}
QToolButton#sectionToggle:hover {{ color: {B.GREEN}; }}
QToolButton#sectionToggle:focus {{ border: none; color: {B.GREEN}; text-decoration: underline; }}
QGroupBox#sectionBody {{ background: #ffffff; border: 1px solid {B.BORDER}; border-radius: 14px; margin-top: 0px;
    padding-top: 0px; }}
QGroupBox#sectionBody QLabel {{ color: {B.MUTED}; }}
QLabel#subHeader {{ color: {B.GREEN}; font-size: 11px; font-weight: 700; padding-top: 10px; }}
QLabel#hint {{ color: {B.MUTED}; font-size: 12px; }}
"""


def dspin(lo, hi, step, dec, suffix="", tip=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    if suffix:
        w.setSuffix(" " + suffix)
    if tip:
        w.setToolTip(tip)
    return w


def ispin(lo, hi, suffix="", tip=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    if suffix:
        w.setSuffix(" " + suffix)
    if tip:
        w.setToolTip(tip)
    return w


class SettingsPanel(QScrollArea):
    changed = Signal()                  # un paramètre a changé
    restart_needed = Signal()           # un paramètre exige de relancer la détection
    roi_rect_requested = Signal()
    roi_poly_requested = Signal()
    roi_clear_requested = Signal()
    calib_requested = Signal()
    camera_refresh_requested = Signal()
    camera_start_requested = Signal()
    camera_dialog_requested = Signal()
    trigger_requested = Signal()
    save_preset_requested = Signal()
    load_preset_requested = Signal()
    browse_output_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(340)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root = QWidget()
        root.setObjectName("SettingsRoot")
        root.setStyleSheet(PANEL_QSS)
        self.setWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(10)
        self._loading = False
        self.cfg = AppConfig()

        # =====================================================================================
        # Réglages essentiels (visibles) : Expérience, Caméra, Échelle et zone.
        # Tout le reste est rangé dans UNE section repliée « Réglages avancés ».
        # =====================================================================================

        # --- Expérience -----------------------------------------------------
        g = Collapsible("Expérience")
        self.condition = QComboBox()
        self.condition.setEditable(True)
        self.condition.addItems(CONDITIONS)
        self.sample = QComboBox()
        self.sample.addItems(list(physics.SAMPLES.keys()))
        self.mass = dspin(0, 5000, 1, 1, "g", "Masse de l'échantillon placé dans la chambre")
        self.trial = ispin(1, 999, tip="Numéro de l'essai (utilisé dans le nom des fichiers)")
        self.operator = QLineEdit()
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("température de la plaque, quantité d'alcool…")
        g.form.addRow("Condition", self.condition)
        g.form.addRow("Échantillon", self.sample)
        g.form.addRow("Masse", self.mass)
        g.form.addRow("Essai n°", self.trial)
        lay.addWidget(g)
        self.sec_experience = g

        # --- Caméra ----------------------------------------------------------
        g = Collapsible("Caméra")
        self.camera = QComboBox()
        self.camera.setToolTip("Webcam USB, caméra d'un téléphone (via DroidCam/Iriun…), appareil photo en mode webcam")
        rb = QPushButton("↻")
        rb.setFixedWidth(40)
        rb.setToolTip("Rechercher les caméras branchées")
        rb.setAccessibleName("Rechercher les caméras")
        rb.clicked.connect(self.camera_refresh_requested)
        row = QHBoxLayout()
        row.addWidget(self.camera, 1)
        row.addWidget(rb)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        self.resolution = QComboBox()
        for r in ["1920×1080", "1280×720", "3840×2160", "640×480"]:
            self.resolution.addItem(r)
        self.cam_fps = ispin(5, 120, "img/s")
        self.backend = QComboBox()
        self.backend.addItems(["DirectShow", "Media Foundation", "auto"])
        drv = QPushButton("Réglages du pilote (exposition, focus…)")
        drv.clicked.connect(self.camera_dialog_requested)
        self.cam_start_btn = QPushButton("Démarrer cette caméra")
        self.cam_start_btn.setToolTip("Ouvre la caméra sélectionnée et lance la détection en direct")
        self.cam_start_btn.setProperty("primary", True)
        self.cam_start_btn.clicked.connect(self.camera_start_requested)
        g.form.addRow("Caméra", w)
        g.form.addRow("Résolution", self.resolution)
        g.form.addRow(self.cam_start_btn)
        g.form.addRow(drv)
        lay.addWidget(g)
        self.sec_camera = g

        # --- Échelle et zone -------------------------------------------------
        g = Collapsible("Échelle et zone")
        self.px_per_mm = dspin(0, 500, 0.1, 3, "px/mm", "Pixels de l'image par millimètre (0 = non calibré)")
        self.fov = dspin(10, 2000, 5, 0, "mm", "Largeur filmée supposée si non calibré")
        cb = QPushButton("Calibrer avec une règle (2 clics)")
        cb.clicked.connect(self.calib_requested)
        b1 = QPushButton("Rectangle")
        b1.clicked.connect(self.roi_rect_requested)
        b2 = QPushButton("Polygone")
        b2.clicked.connect(self.roi_poly_requested)
        b3 = QPushButton("Effacer")
        b3.clicked.connect(self.roi_clear_requested)
        r = QHBoxLayout()
        r.setSpacing(6)
        for b in (b1, b2, b3):
            r.addWidget(b)
        rw = QWidget()
        rw.setLayout(r)
        r.setContentsMargins(0, 0, 0, 0)
        self.roi_label = QLabel("Image entière")
        self.roi_label.setObjectName("hint")
        g.form.addRow("Échelle", self.px_per_mm)
        g.form.addRow(cb)
        zl = QLabel("Zone analysée")
        g.form.addRow(zl, self.roi_label)
        g.form.addRow(rw)
        lay.addWidget(g)
        self.sec_scale = g

        # --- Réglages avancés (une seule section, repliée) ---------------------------
        adv = Collapsible("Réglages avancés", expanded=False)
        adv.setToolTip("Détection, classification, physique, chambre à expansion, dossier, préréglages…")
        self.sec_advanced = adv
        f = adv.form

        def sub(title: str):
            lab = QLabel(title.upper())
            lab.setObjectName("subHeader")
            f.addRow(lab)

        sub("Expérience")
        f.addRow("Opérateur", self.operator)
        f.addRow("Notes", self.notes)
        sub("Caméra")
        f.addRow("Cadence", self.cam_fps)
        f.addRow("Pilote", self.backend)
        sub("Échelle")
        f.addRow("Champ supposé", self.fov)

        # Détection
        sub("Détection")
        self.sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity.setRange(1, 10)
        self.sensitivity.setToolTip("Plus élevé = traces plus faibles détectées (mais plus de fausses détections)")
        self.sens_label = QLabel("5")
        self.sens_label.setMinimumWidth(18)
        srow = QHBoxLayout()
        srow.addWidget(self.sensitivity, 1)
        srow.addWidget(self.sens_label)
        sw = QWidget()
        sw.setLayout(srow)
        srow.setContentsMargins(0, 0, 0, 0)
        self.k_high = dspin(1.5, 20, 0.1, 2, "σ", "Seuil d'amorçage (écarts-types du bruit)")
        self.k_low = dspin(0.5, 10, 0.1, 2, "σ", "Seuil d'extension (hystérésis)")
        self.proc_width = QComboBox()
        for label, _v in PROCESS_WIDTHS:
            self.proc_width.addItem(label)
        self.bg_time = dspin(0.3, 60, 0.5, 1, "s", "Constante de temps du fond adaptatif")
        self.min_len = dspin(1, 100, 0.5, 1, "mm", "Longueur minimale d'une trace")
        self.min_vis = dspin(0.03, 5, 0.05, 2, "s", "Durée minimale de visibilité (rejette le bruit)")
        self.min_snr = dspin(1, 50, 0.5, 1, "σ", "Contraste minimal le long de la trace")
        self.ridge = QCheckBox("Filtre de lignes (hessienne)")
        self.stabilize = QCheckBox("Stabiliser l'image (vibrations)")
        f.addRow("Sensibilité", sw)
        f.addRow("Seuil haut", self.k_high)
        f.addRow("Seuil bas", self.k_low)
        f.addRow("Largeur de calcul", self.proc_width)
        f.addRow("Mémoire du fond", self.bg_time)
        f.addRow("Longueur min.", self.min_len)
        f.addRow("Visibilité min.", self.min_vis)
        f.addRow("Contraste min.", self.min_snr)
        f.addRow(self.ridge)
        f.addRow(self.stabilize)

        # Classification
        sub("Classification")
        self.cls_mode = QComboBox()
        for label, val in CLS_MODES:
            self.cls_mode.addItem(label, val)
        self.cls_mode.setToolTip("Hybride : les règles physiques et une IA de référence (entraînée sur des milliers "
                                 "de traces simulées) votent ensemble.\nRègles physiques : critères expliqués "
                                 "seulement.\nIA locale : modèle entraîné sur VOS corrections (onglet Traces).")
        self.alpha_w = dspin(0.2, 10, 0.1, 2, "mm", "Épaisseur au-delà de laquelle une trace est « épaisse »")
        self.alpha_L = dspin(10, 300, 5, 0, "mm", "Longueur max. d'une alpha naturelle")
        self.muon_L = dspin(5, 500, 5, 0, "mm", "Longueur min. d'un muon")
        self.straight = dspin(0.5, 1.0, 0.01, 2, "", "Rectitude min. d'un muon (1 = parfaitement droit)")
        self.heavy_d = dspin(1.1, 20, 0.1, 1, "×", "Densité d'ionisation relative d'une particule lourde")
        self.model_label = QLabel("Aucun modèle")
        self.model_label.setObjectName("hint")
        f.addRow("Méthode", self.cls_mode)
        f.addRow("Épaisseur alpha", self.alpha_w)
        f.addRow("Longueur max. alpha", self.alpha_L)
        f.addRow("Longueur min. muon", self.muon_L)
        f.addRow("Rectitude muon", self.straight)
        f.addRow("Densité lourde", self.heavy_d)
        f.addRow("Modèle IA", self.model_label)

        # Physique
        sub("Physique")
        self.temp = dspin(-80, 40, 1, 1, "°C", "Température de la couche sensible (air froid → alphas plus courtes)")
        self.pressure = dspin(50, 120, 0.5, 1, "kPa")
        self.bfield = dspin(0, 3000, 10, 0, "mT", "Champ magnétique (aimant sous la chambre). 0 = aucun")
        f.addRow("Temp. couche sensible", self.temp)
        f.addRow("Pression", self.pressure)
        f.addRow("Champ magnétique B", self.bfield)

        # Chambre à expansion
        sub("Chambre à expansion")
        self.exp_on = QCheckBox("Mode chambre à expansion")
        self.exp_trig = QComboBox()
        self.exp_trig.addItems(["auto (saut de luminosité)", "série (Arduino)", "manuel (touche E)"])
        self.exp_port = QComboBox()
        self.exp_port.setEditable(True)
        self.exp_delay = dspin(0, 5000, 10, 0, "ms")
        self.exp_window = dspin(20, 10000, 50, 0, "ms")
        tb = QPushButton("Déclencher une expansion (E)")
        tb.clicked.connect(self.trigger_requested)
        f.addRow(self.exp_on)
        f.addRow("Déclenchement", self.exp_trig)
        f.addRow("Port série", self.exp_port)
        f.addRow("Délai", self.exp_delay)
        f.addRow("Fenêtre sensible", self.exp_window)
        f.addRow(tb)

        # Préréglages et dossier
        sub("Dossier et préréglages")
        self.out_dir = QLineEdit()
        ob = QPushButton("Parcourir…")
        ob.clicked.connect(self.browse_output_requested)
        self.autosave = QCheckBox("Sauvegarder automatiquement à la fin")
        self.autosave.setChecked(True)
        sp = QPushButton("Enregistrer un préréglage…")
        sp.clicked.connect(self.save_preset_requested)
        lp = QPushButton("Charger un préréglage…")
        lp.clicked.connect(self.load_preset_requested)
        rs = QPushButton("Valeurs par défaut")
        rs.clicked.connect(self.reset_requested)
        f.addRow("Dossier des résultats", self.out_dir)
        f.addRow(ob)
        f.addRow(self.autosave)
        f.addRow(sp)
        f.addRow(lp)
        f.addRow(rs)
        lay.addWidget(adv)
        lay.addStretch(1)

        # --- signaux ------------------------------------------------------------
        self.sensitivity.valueChanged.connect(self._on_sensitivity)
        live_widgets = [self.k_high, self.k_low, self.bg_time, self.min_len, self.min_vis, self.min_snr, self.temp,
                        self.pressure, self.bfield, self.alpha_w, self.alpha_L, self.muon_L, self.straight,
                        self.heavy_d, self.mass, self.trial, self.exp_delay, self.exp_window, self.px_per_mm, self.fov]
        for w in live_widgets:
            w.valueChanged.connect(self._emit_changed)
        for w in (self.condition, self.sample, self.cls_mode, self.exp_trig):
            w.currentIndexChanged.connect(self._emit_changed)
        self.condition.editTextChanged.connect(self._emit_changed)
        for w in (self.operator, self.notes):
            w.editingFinished.connect(self._emit_changed)
        for w in (self.ridge, self.stabilize, self.exp_on):
            w.toggled.connect(self._emit_changed)
        self.proc_width.currentIndexChanged.connect(self._emit_changed)
        for cbx in root.findChildren(QComboBox):          # le panneau tient dans sa largeur
            cbx.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cbx.setMinimumContentsLength(8)
            cbx.view().setMinimumWidth(max(cbx.view().sizeHintForColumn(0) + 36, 160))
        self.load(self.cfg)

    # ------------------------------------------------------------------------
    def _on_sensitivity(self, s: int):
        self.sens_label.setText(str(s))
        if self._loading:
            return
        kh = round(7.0 - 0.5 * s, 2)
        self._loading = True
        self.k_high.setValue(kh)
        self.k_low.setValue(round(kh * 0.49, 2))
        self._loading = False
        self._emit_changed()

    def _emit_changed(self, *_):
        if not self._loading:
            self.changed.emit()

    def set_roi_text(self, roi: list):
        if not roi:
            self.roi_label.setText("Image entière")
        else:
            xs = [p[0] for p in roi]
            ys = [p[1] for p in roi]
            self.roi_label.setText(f"{len(roi)} sommets — {100 * (max(xs) - min(xs)):.0f} % × "
                                   f"{100 * (max(ys) - min(ys)):.0f} % de l'image")

    def load(self, cfg: AppConfig):
        self._loading = True
        self.cfg = cfg
        d, c, p, e, r = cfg.detection, cfg.classification, cfg.physics, cfg.expansion, cfg.run
        i = self.condition.findText(r.condition)
        if i >= 0:
            self.condition.setCurrentIndex(i)
        else:
            self.condition.setEditText(r.condition)
        j = self.sample.findText(r.sample_type)
        self.sample.setCurrentIndex(max(0, j))
        self.mass.setValue(r.sample_mass_g)
        self.trial.setValue(r.trial)
        self.operator.setText(r.operator)
        self.notes.setText(r.notes)
        s = int(round((7.0 - d.k_high) / 0.5))
        self.sensitivity.setValue(max(1, min(10, s)))
        self.sens_label.setText(str(self.sensitivity.value()))
        self.k_high.setValue(d.k_high)
        self.k_low.setValue(d.k_low)
        idx = next((k for k, (_l, v) in enumerate(PROCESS_WIDTHS) if v == d.process_width), 1)
        self.proc_width.setCurrentIndex(idx)
        self.bg_time.setValue(d.bg_time_s)
        self.min_len.setValue(d.min_length_mm)
        self.min_vis.setValue(d.min_visible_s)
        self.min_snr.setValue(d.min_peak_snr)
        self.ridge.setChecked(d.use_ridge)
        self.stabilize.setChecked(d.stabilize)
        self.set_roi_text(d.roi)
        self.px_per_mm.setValue(p.px_per_mm)
        self.fov.setValue(p.default_fov_mm)
        self.temp.setValue(p.chamber_temp_c)
        self.pressure.setValue(p.pressure_kpa)
        self.bfield.setValue(p.b_field_mt)
        k = self.cls_mode.findData(c.mode)
        self.cls_mode.setCurrentIndex(k if k >= 0 else self.cls_mode.findData("regles"))
        self.alpha_w.setValue(c.alpha_width_mm)
        self.alpha_L.setValue(c.alpha_max_length_mm)
        self.muon_L.setValue(c.muon_min_length_mm)
        self.straight.setValue(c.straight_min)
        self.heavy_d.setValue(c.heavy_density_rel)
        self.model_label.setText(c.model_path.split("/")[-1].split("\\")[-1] if c.model_path else "Aucun modèle")
        self.exp_on.setChecked(e.enabled)
        self.exp_trig.setCurrentIndex({"auto": 0, "serie": 1, "manuel": 2}.get(e.trigger, 0))
        self.exp_port.setEditText(e.serial_port)
        self.exp_delay.setValue(e.delay_ms)
        self.exp_window.setValue(e.window_ms)
        self.out_dir.setText(cfg.output_dir)
        self._loading = False

    def apply_to(self, cfg: AppConfig) -> AppConfig:
        d, c, p, e, r = cfg.detection, cfg.classification, cfg.physics, cfg.expansion, cfg.run
        r.condition = self.condition.currentText().strip() or "Sans nom"
        r.sample_type = self.sample.currentText()
        r.sample_mass_g = self.mass.value()
        r.trial = self.trial.value()
        r.operator = self.operator.text()
        r.notes = self.notes.text()
        d.k_high = self.k_high.value()
        d.k_low = min(self.k_low.value(), d.k_high)
        d.process_width = PROCESS_WIDTHS[self.proc_width.currentIndex()][1]
        d.bg_time_s = self.bg_time.value()
        d.min_length_mm = self.min_len.value()
        d.min_visible_s = self.min_vis.value()
        d.min_peak_snr = self.min_snr.value()
        d.use_ridge = self.ridge.isChecked()
        d.stabilize = self.stabilize.isChecked()
        p.px_per_mm = self.px_per_mm.value()
        p.default_fov_mm = self.fov.value()
        p.chamber_temp_c = self.temp.value()
        p.pressure_kpa = self.pressure.value()
        p.b_field_mt = self.bfield.value()
        c.mode = self.cls_mode.currentData() or "regles"
        c.alpha_width_mm = self.alpha_w.value()
        c.alpha_max_length_mm = self.alpha_L.value()
        c.muon_min_length_mm = self.muon_L.value()
        c.straight_min = self.straight.value()
        c.heavy_density_rel = self.heavy_d.value()
        e.enabled = self.exp_on.isChecked()
        e.trigger = ["auto", "serie", "manuel"][self.exp_trig.currentIndex()]
        e.serial_port = self.exp_port.currentText().strip()
        e.delay_ms = self.exp_delay.value()
        e.window_ms = self.exp_window.value()
        cfg.output_dir = self.out_dir.text().strip()
        return cfg

    def camera_settings(self) -> dict:
        w, h = (int(v) for v in self.resolution.currentText().split("×"))
        return {"index": self.camera.currentData() if self.camera.currentData() is not None else 0,
                "width": w, "height": h, "fps": self.cam_fps.value(), "backend": self.backend.currentText()}
