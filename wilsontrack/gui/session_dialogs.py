"""Longues analyses : session de nuit, vidéos reçues (analyse par lot), bandeau d'état et file d'attente.

Ce module ne fait AUCUNE acquisition ni analyse : il fournit
- ``NightSessionDialog`` : réglages d'une session de nuit → dictionnaire « nuit » (voir NIGHT_DEFAULTS) ;
- ``BatchDialog`` : liste de vidéos reçues et/ou dossier surveillé → dictionnaire « lot » (BATCH_DEFAULTS) ;
- ``SessionStatusBar`` : bandeau lisible de loin pendant la nuit (dictionnaire d'état du travailleur) ;
- ``BatchQueueWidget`` : tableau de la file d'analyse par lot (liste de dictionnaires « tâche »).
Le moteur (wilsontrack/longrun.py, wilsontrack/batch.py) consomme et produit ces dictionnaires.
"""
from __future__ import annotations

import datetime as _dt
import html
import math
import os
import shutil
from pathlib import Path

from PySide6.QtCore import QMetaMethod, QRectF, QSize, Qt, QTime, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QProgressBar, QPushButton, QRadioButton, QSizePolicy,
                               QSpinBox, QStyle, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QTimeEdit,
                               QToolTip, QVBoxLayout, QWidget)

from .. import physics
from . import brand as B
from .home import (ACCENT, BORDER, CARD, CONDITIONS, FAINT, MUTED, NBSP, SUCCESS, TEXT, VIDEO_EXTS, VIDEO_FILTER,
                   WINDOW, ElidedLabel, FlowLayout, _fr_locale, _norm, _sample_key, _tinted_icon, class_hex,
                   fmt_duration, fmt_num, fmt_size, is_background, sample_for_condition)

__all__ = ["NightSessionDialog", "BatchDialog", "SessionStatusBar", "BatchQueueWidget", "SampleFields",
           "NIGHT_DEFAULTS", "BATCH_DEFAULTS", "RECORD_MODES", "JOB_STATUSES", "normalize_night",
           "normalize_batch", "normalize_job", "fmt_clock", "fmt_hms"]

NIGHT_TINT = B.FOREST          # accents « nuit » / « lot » : verts du Cégep (petites icônes seulement)
BATCH_TINT = B.GREEN
WARN = B.WARN_TEXT              # couleurs de TEXTE lisibles sur fond blanc (WCAG AA)
DANGER = B.DANGER_TEXT

RECORD_MODES = ("aucune", "reduite", "complete")
SCALE_CHOICES = (("½", 0.5), ("⅓", 1.0 / 3.0), ("¼", 0.25))

# Dictionnaire « nuit » émis par NightSessionDialog / HomePage.night_requested (contrat avec longrun.py).
NIGHT_DEFAULTS: dict = {
    "camera_index": 0,
    "duration_h": 0.0,             # 0 = utiliser end_time
    "end_time": "07:30",           # « HH:MM », heure locale ; plus tôt que maintenant = demain
    "segment_min": 30,
    "record_video": "reduite",     # "aucune" | "reduite" | "complete"
    "record_scale": 0.5,
    "record_fps": 10.0,
    "snapshot_every_min": 10,      # 0 = jamais
    "autosave_every_min": 5,
    "keep_awake": True,
    "dead_after_min": 20,          # alerte si aucune trace valide pendant ce délai (0 = jamais)
    "condition": CONDITIONS[0],
    "sample": "Aucun",
    "mass_g": 0.0,
    "px_per_mm": 0.0,
    "output_dir": "",              # "" = dossier des résultats par défaut
}

# Dictionnaire « lot » émis par BatchDialog / HomePage.batch_requested (contrat avec batch.py).
BATCH_DEFAULTS: dict = {
    "files": [],
    "watch_folder": "",
    "condition": CONDITIONS[0],
    "sample": "Aucun",
    "mass_g": 0.0,
    "px_per_mm": 0.0,
    "output_dir": "",
    "make_reports": True,
}

JOB_STATUSES = ("en attente", "en cours", "terminé", "erreur", "annulé")
_JOB_STYLE = {  # statut → (icône, libellé, couleur)
    "en attente": ("⏳", "En attente", MUTED),
    "en cours": ("▶", "En cours", ACCENT),
    "terminé": ("✓", "Terminé", SUCCESS),
    "erreur": ("✕", "Erreur", DANGER),
    "annulé": ("⊘", "Annulé", FAINT),
}
_MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
           "novembre", "décembre")


# ============================================================================ utilitaires ====
def _parse_hhmm(s) -> tuple[int, int] | None:
    try:
        h, m = str(s).strip().split(":")[:2]
        h, m = int(h), int(m)
    except (ValueError, AttributeError):
        return None
    return (h, m) if 0 <= h <= 23 and 0 <= m <= 59 else None


def _coerce(value, default):
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "vrai", "oui", "yes")
        return bool(value)
    if isinstance(default, int):
        return int(round(float(value)))
    if isinstance(default, float):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError
        return v
    if isinstance(default, list):
        return [str(x) for x in value] if isinstance(value, (list, tuple)) else []
    return str(value)


def _normalize(d, defaults: dict) -> dict:
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in defaults.items()}
    if isinstance(d, dict):
        for k, default in defaults.items():
            if k in d and d[k] is not None:
                try:
                    out[k] = _coerce(d[k], default)
                except (TypeError, ValueError):
                    pass
    return out


def normalize_night(d: dict | None) -> dict:
    """Dictionnaire « nuit » complet : clés inconnues ignorées, clés manquantes ou invalides → défauts."""
    out = _normalize(d, NIGHT_DEFAULTS)
    if out["record_video"] not in RECORD_MODES:
        out["record_video"] = NIGHT_DEFAULTS["record_video"]
    hm = _parse_hhmm(out["end_time"])
    out["end_time"] = f"{hm[0]:02d}:{hm[1]:02d}" if hm else ""
    out["duration_h"] = max(0.0, out["duration_h"])
    if out["duration_h"] <= 0 and not out["end_time"]:
        out["end_time"] = NIGHT_DEFAULTS["end_time"]
    out["record_scale"] = min(1.0, max(0.1, out["record_scale"]))
    out["record_fps"] = min(60.0, max(1.0, out["record_fps"]))
    for k, lo, hi in (("segment_min", 1, 24 * 60), ("snapshot_every_min", 0, 24 * 60),
                      ("autosave_every_min", 1, 24 * 60), ("dead_after_min", 0, 24 * 60)):
        out[k] = min(hi, max(lo, out[k]))
    out["mass_g"] = max(0.0, out["mass_g"])
    out["px_per_mm"] = max(0.0, out["px_per_mm"])
    return out


def normalize_batch(d: dict | None) -> dict:
    """Dictionnaire « lot » complet (mêmes règles que normalize_night)."""
    out = _normalize(d, BATCH_DEFAULTS)
    out["mass_g"] = max(0.0, out["mass_g"])
    out["px_per_mm"] = max(0.0, out["px_per_mm"])
    return out


def normalize_job(j: dict) -> dict:
    """Tâche de la file (affichage) : {"path","name","status","progress","eta_s","result_folder","counts","error"}."""
    j = j if isinstance(j, dict) else {}
    path = str(j.get("path", "") or "")
    status = str(j.get("status", "en attente") or "en attente").strip().lower()
    if status not in JOB_STATUSES:
        status = {"termine": "terminé", "annule": "annulé", "done": "terminé", "running": "en cours",
                  "error": "erreur", "pending": "en attente", "cancelled": "annulé"}.get(status, "en attente")
    try:
        prog = float(j.get("progress", 0.0) or 0.0)
        prog = 0.0 if not math.isfinite(prog) else min(1.0, max(0.0, prog))
    except (TypeError, ValueError):
        prog = 0.0
    if status == "terminé":
        prog = 1.0
    try:
        eta = float(j.get("eta_s", -1))
        eta = eta if math.isfinite(eta) else -1.0
    except (TypeError, ValueError):
        eta = -1.0
    counts = j.get("counts") if isinstance(j.get("counts"), dict) else {}
    return {"path": path, "name": str(j.get("name") or (Path(path).name if path else "Vidéo")), "status": status,
            "progress": prog, "eta_s": eta, "result_folder": str(j.get("result_folder", "") or ""),
            "counts": counts, "error": str(j.get("error", "") or ""), "from_watch": bool(j.get("from_watch")),
            "warnings": [str(w) for w in (j.get("warnings") or []) if str(w).strip()]}


def fmt_hms(sec) -> str:
    """Durée lisible : « 42 s », « 12 min 05 s », « 3 h 12 min »."""
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(sec) or sec < 0:
        return "—"
    if sec < 60:
        return f"{int(sec)} s"
    return fmt_duration(sec)


def _size(n) -> str:
    """Taille lisible sans fausse précision : « 850 Mo », « 4,2 Go », « 231 Go »."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "—"
    if n < 0:
        return "—"
    if n < 1024 ** 3:
        return fmt_size(n) if n < 100 * 1024 ** 2 else f"{fmt_num(n / 1024 ** 2, 0)} Mo"
    g = n / 1024 ** 3
    return f"{fmt_num(g, 1 if g < 100 else 0)} Go"


def fmt_clock(t: _dt.datetime | _dt.time) -> str:
    """Heure à la québécoise : « 7 h 30 », « 23 h 05 »."""
    return f"{t.hour} h {t.minute:02d}"


def _day_word(when: _dt.datetime, now: _dt.datetime) -> str:
    delta = (when.date() - now.date()).days
    if delta == 0:
        return "aujourd'hui"
    if delta == 1:
        return "demain"
    return f"le {when.day} {_MONTHS[when.month - 1]}"


def _existing_ancestor(path: str) -> Path | None:
    p = Path(path)
    for cand in [p, *p.parents]:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    return None


def _free_bytes_fallback(path: str) -> int | None:
    base = _existing_ancestor(path)
    if base is None:
        return None
    try:
        return int(shutil.disk_usage(str(base)).free)
    except OSError:
        return None


def _default_output_dir() -> str:
    try:
        from ..config import default_output_dir
        return str(default_output_dir())
    except Exception:  # pragma: no cover
        return str(Path.home() / "Documents" / "WilsonTrack Résultats")


def _load_longrun():
    """Fonctions d'estimation du moteur de longues sessions (écrit en parallèle) ; None si indisponible."""
    try:
        from ..longrun import NightSessionConfig, estimate_disk_usage, free_disk_bytes
        return NightSessionConfig, estimate_disk_usage, free_disk_bytes
    except Exception:
        return None


def _videos_in(folder: str) -> list[str]:
    try:
        return sorted((str(p) for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
                      key=lambda s: Path(s).name.lower())
    except OSError:
        return []


def _set_state(w: QWidget, name: str, value) -> None:
    """Change une propriété dynamique et réapplique la feuille de style (au widget et à ses enfants, dont
    les règles « parent[prop] enfant » dépendent de cette propriété)."""
    if w.property(name) == value:
        return
    w.setProperty(name, value)
    for x in [w, *w.findChildren(QWidget)]:
        x.style().unpolish(x)
        x.style().polish(x)
    w.update()


def _label(text: str, tip: str = "", name: str = "formLabel") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(name)
    if tip:
        lab.setToolTip(tip)
    return lab


def _help(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("help")
    lab.setWordWrap(True)
    return lab


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    box = QFrame()
    box.setObjectName("dlgPanel")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(18, 14, 18, 16)
    lay.setSpacing(8)
    t = QLabel(title)
    t.setObjectName("panelTitle")
    lay.addWidget(t)
    return box, lay


def _header(icon: str, tint: str, title: str, subtitle: str) -> QHBoxLayout:
    head = QHBoxLayout()
    head.setSpacing(14)
    head.addWidget(_tinted_icon(icon, tint, 46, 13, 22), 0, Qt.AlignmentFlag.AlignTop)
    col = QVBoxLayout()
    col.setSpacing(2)
    t = QLabel(title)
    t.setObjectName("dlgTitle")
    s = QLabel(subtitle)
    s.setObjectName("dlgSub")
    s.setWordWrap(True)
    col.addWidget(t)
    col.addWidget(s)
    head.addLayout(col, 1)
    return head


def _connected(obj, signal) -> bool:
    try:
        return obj.isSignalConnected(QMetaMethod.fromSignal(signal))
    except Exception:  # pragma: no cover
        return False


def _fit_to_screen(dlg: QWidget, natural: QSize, avail=None) -> None:
    """Taille naturelle, réduite si l'écran est plus petit (les panneaux restent lisibles)."""
    if avail is None:
        scr = dlg.screen()
        avail = scr.availableGeometry() if scr is not None else None
    w, h = natural.width(), natural.height()
    if avail is not None and avail.width() >= 640 and avail.height() >= 480:
        w = min(w, avail.width() - 24)
        h = min(h, avail.height() - 40)
    dlg.resize(max(w, 560), max(h, 420))


SESSION_QSS = f"""
QDialog#NightSessionDialog, QDialog#BatchDialog {{ background: {B.SOFT}; }}
QLabel#dlgTitle {{ color: {B.INK}; font-size: 16pt; font-weight: 700; }}
QLabel#dlgSub {{ color: {B.MUTED}; }}
QFrame#dlgPanel {{ background: {B.WHITE}; border: 1px solid {B.BORDER}; border-radius: 16px; }}
QLabel#panelTitle {{ color: {B.GREEN}; font-size: 8pt; font-weight: 700; }}
QLabel#formLabel {{ color: {B.INK}; }}
QLabel#help {{ color: {B.MUTED}; font-size: 9pt; }}
QLabel#activity {{ color: {B.SUCCESS_TEXT}; font-size: 9pt; font-weight: 600; }}
QLabel#endPreview {{ color: {B.FOREST}; font-size: 9pt; font-weight: 700; }}
QLabel#hint {{ color: {B.MUTED}; }}
QLabel#hint[error="true"] {{ color: {B.DANGER_TEXT}; font-weight: 600; }}
QFrame#sep {{ background: {B.BORDER}; border: none; }}
QFrame#estBox {{ background: {B.MINT_SOFT}; border: 1px solid #BFE6C4; border-radius: 14px; }}
QFrame#estBox[state="warn"] {{ background: {B.WARN_BG}; border: 1px solid {B.WARN_BORDER}; }}
QFrame#estBox[state="bad"] {{ background: {B.DANGER_BG}; border: 1px solid {B.DANGER_BORDER}; }}
QFrame#estBox[state="na"] {{ background: {B.ALT_ROW}; border: 1px solid {B.BORDER}; }}
QLabel#estMain {{ color: {B.INK}; font-size: 12pt; font-weight: 700; }}
QLabel#estSub {{ color: {B.MUTED}; font-size: 9pt; }}
QLabel#estWarn {{ color: {B.WARN_TEXT}; font-size: 9pt; font-weight: 600; }}
QFrame#estBox[state="bad"] QLabel#estWarn {{ color: {B.DANGER_TEXT}; }}
QFrame#checklist {{ background: {B.WHITE}; border: 1px solid {B.BORDER}; border-left: 4px solid {B.BRIGHT};
    border-radius: 14px; }}
QLabel#checkTitle {{ color: {B.FOREST}; font-weight: 700; }}
QLabel#checkItem {{ color: {B.INK}; }}
QLabel#fileSummary {{ color: {B.MUTED}; font-size: 9pt; }}
QCheckBox#strongCheck {{ color: {B.INK}; font-weight: 600; }}
"""


# ============================================================================ champs partagés ====
class SampleFields(QWidget):
    """Condition / échantillon / masse / échelle, avec les mêmes règles que la boîte « Analyser une vidéo »."""
    changed = Signal()

    def __init__(self, values: dict | None = None, parent=None, scale_help: str | None = None):
        super().__init__(parent)
        v = dict(values or {})
        loc = _fr_locale()
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        g.setColumnStretch(1, 1)
        self.condition = QComboBox()
        self.condition.setEditable(True)
        self.condition.addItems(CONDITIONS)
        self.condition.setToolTip("Ce qui est placé dans la chambre (nom des fichiers et conclusions)")
        self.sample = QComboBox()
        self.sample.addItems(list(physics.SAMPLES.keys()))
        self.sample.setToolTip("Échantillon radioactif placé dans la chambre (sert à estimer l'activité attendue)")
        self.mass = QDoubleSpinBox()
        self.mass.setLocale(loc)
        self.mass.setRange(0, 5000)
        self.mass.setDecimals(1)
        self.mass.setSuffix(" g")
        self.mass.setToolTip("Masse de l'échantillon placé dans la chambre")
        self.activity = QLabel("")
        self.activity.setObjectName("activity")
        self.activity.setWordWrap(True)
        self.scale = QDoubleSpinBox()
        self.scale.setLocale(loc)
        self.scale.setRange(0, 500)
        self.scale.setDecimals(3)
        self.scale.setSingleStep(0.1)
        self.scale.setSuffix(" px/mm")
        self.scale.setSpecialValueText("0 — inconnue")
        self.scale.setToolTip("Nombre de pixels de l'image par millimètre réel (0 = inconnue)")
        self.scale_help = _help(scale_help if scale_help is not None else
                                "0 = inconnue : les longueurs en millimètres sont alors estimées. Pour calibrer, "
                                "filmez une règle puis « 📏 Calibrer avec une règle » dans les réglages.")
        g.addWidget(_label("Condition"), 0, 0)
        g.addWidget(self.condition, 0, 1)
        g.addWidget(_label("Échantillon"), 1, 0)
        g.addWidget(self.sample, 1, 1)
        g.addWidget(_label("Masse"), 2, 0)
        g.addWidget(self.mass, 2, 1)
        g.addWidget(self.activity, 3, 1)
        g.addWidget(_label("Échelle", "Pixels par millimètre"), 4, 0)
        g.addWidget(self.scale, 4, 1)
        g.addWidget(self.scale_help, 5, 1)
        self.set_values(v)
        self.condition.currentTextChanged.connect(self._on_condition)
        self.sample.currentIndexChanged.connect(self._update_activity)
        self.mass.valueChanged.connect(self._update_activity)
        for sig in (self.condition.currentTextChanged, self.sample.currentIndexChanged, self.mass.valueChanged,
                    self.scale.valueChanged):
            sig.connect(lambda *_: self.changed.emit())

    def set_values(self, d: dict):
        cond = str(d.get("condition") or CONDITIONS[0])
        self.condition.blockSignals(True)
        i = self.condition.findText(cond)
        if i >= 0:
            self.condition.setCurrentIndex(i)
        else:
            self.condition.setEditText(cond)
        self.condition.blockSignals(False)
        sample = d.get("sample")
        j = self.sample.findText(str(sample)) if sample else -1
        if j >= 0:
            self.sample.setCurrentIndex(j)
        try:
            self.mass.setValue(float(d.get("mass_g", 0) or 0))
        except (TypeError, ValueError):
            self.mass.setValue(0)
        suggestion = sample_for_condition(cond)
        if is_background(cond):
            self._set_sample(suggestion)
            self.mass.setValue(0)
        elif suggestion and (j < 0 or sample == _sample_key("aucun")):
            self._set_sample(suggestion)
        try:
            self.scale.setValue(float(d.get("px_per_mm", 0) or 0))
        except (TypeError, ValueError):
            self.scale.setValue(0)
        self._update_activity()

    def _set_sample(self, name: str | None):
        if name:
            j = self.sample.findText(name)
            if j >= 0:
                self.sample.setCurrentIndex(j)

    def _on_condition(self, text: str):
        s = sample_for_condition(text)
        if s is None:
            return
        self._set_sample(s)
        if is_background(text):
            self.mass.setValue(0)
        self._update_activity()

    def _update_activity(self, *_):
        frac = float(physics.SAMPLES.get(self.sample.currentText(), 0.0) or 0.0)
        self.mass.setEnabled(frac > 0)
        if frac <= 0:
            self.activity.hide()
            return
        m = self.mass.value()
        if m <= 0:
            self.activity.setText("Indiquez la masse pour estimer l'activité attendue du K-40.")
        else:
            try:
                a = physics.sample_activity(m, frac).activity_bq
                self.activity.setText(f"Activité attendue du K-40 : ≈ {fmt_num(a, 0) if a >= 100 else fmt_num(a, 1)}"
                                      f"{NBSP}Bq")
            except Exception:
                self.activity.setText("")
        self.activity.show()

    def values(self) -> dict:
        return {"condition": self.condition.currentText().strip() or "Sans nom",
                "sample": self.sample.currentText(),
                "mass_g": float(self.mass.value()),
                "px_per_mm": float(self.scale.value())}


class _FolderField(QWidget):
    """Champ « dossier » : zone de texte + bouton Parcourir (QFileDialog.getExistingDirectory)."""
    changed = Signal(str)

    def __init__(self, title: str, placeholder: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(self.changed)
        self.button = QPushButton("Parcourir…")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self.browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.button)

    def browse(self):
        start = self.path() or str(_existing_ancestor(self.edit.placeholderText().replace("Par défaut : ", ""))
                                   or Path.home())
        d = QFileDialog.getExistingDirectory(self, self._title, start)
        if d:
            self.edit.setText(os.path.normpath(d))

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, p: str):
        self.edit.setText(str(p or ""))


# ============================================================================ session de nuit ====
class NightSessionDialog(QDialog):
    """Réglages d'une session de nuit (filmer et analyser toute une nuit sans surveillance)."""

    def __init__(self, defaults: dict | None = None, cameras: list[dict] | None = None, frame_wh=(1280, 720),
                 fps: float = 30.0, parent=None):
        super().__init__(parent)
        self.setObjectName("NightSessionDialog")
        self.setWindowTitle("Session de nuit")
        self.setStyleSheet(SESSION_QSS)
        self.setModal(True)
        d = normalize_night(defaults)
        self._cams = [c for c in (cameras or []) if isinstance(c, dict) and "index" in c]
        try:
            self._frame_wh = (max(16, int(frame_wh[0])), max(16, int(frame_wh[1])))
        except (TypeError, ValueError, IndexError):
            self._frame_wh = (1280, 720)
        try:
            self._fps = float(fps) if float(fps) > 0 else 30.0
        except (TypeError, ValueError):
            self._fps = 30.0
        self._now = _dt.datetime.now           # remplaçable dans les tests
        self._longrun = _load_longrun()
        self.estimate_bytes: int | None = None
        self.free_bytes: int | None = None
        self.space_ok = True

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(14)
        root.addLayout(_header("🌙", NIGHT_TINT, "Session de nuit",
                               "Filmez et analysez la chambre toute une nuit, sans surveillance : la vidéo est "
                               "découpée en morceaux et les résultats sont sauvegardés régulièrement."))
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._when_panel(d), 1)
        body.addWidget(self._record_panel(d), 1)
        root.addLayout(body, 1)
        root.addWidget(self._checklist())

        row = QHBoxLayout()
        self.error_label = QLabel("")
        self.error_label.setObjectName("hint")
        self.error_label.setWordWrap(True)
        row.addWidget(self.error_label, 1)
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn = QPushButton()
        self.ok_btn.setProperty("primary", True)
        self.ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok_btn.setDefault(True)
        self.ok_btn.setMinimumWidth(230)
        self.ok_btn.clicked.connect(self._on_ok)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.ok_btn)
        root.addLayout(row)

        for sig in (self.duration.valueChanged, self.end_time.timeChanged, self.mode_group.idToggled,
                    self.record_mode.currentIndexChanged, self.record_scale.currentIndexChanged,
                    self.record_fps.valueChanged, self.segment.valueChanged, self.snapshot.valueChanged,
                    self.camera.currentIndexChanged, self.output.changed, self.dead_after.valueChanged):
            sig.connect(self._refresh)
        self._timer = QTimer(self)
        self._timer.setInterval(20000)           # la durée « jusqu'à » diminue avec le temps
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()
        _fit_to_screen(self, self.sizeHint().expandedTo(QSize(940, 0)))
        self.ok_btn.setFocus()

    # ------------------------------------------------------------------ panneaux --
    def _when_panel(self, d: dict) -> QFrame:
        box, lay = _panel("QUAND ET QUOI")
        loc = _fr_locale()
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        g.setColumnStretch(2, 1)
        self.camera = QComboBox()
        self.camera.setToolTip("Caméra fixée au-dessus de la chambre")
        for c in self._cams:
            self.camera.addItem(f"📷  {c.get('label') or 'Caméra ' + str(c['index'])}", int(c["index"]))
        if not self._cams:
            self.camera.addItem("📷  Caméra 0 (par défaut)", 0)
        j = self.camera.findData(int(d["camera_index"]))
        if j < 0 and not self._cams:
            self.camera.setItemText(0, f"📷  Caméra {int(d['camera_index'])}")
            self.camera.setItemData(0, int(d["camera_index"]))
            j = 0
        self.camera.setCurrentIndex(max(0, j))
        g.addWidget(_label("Caméra"), 0, 0)
        g.addWidget(self.camera, 0, 1, 1, 2)

        self.mode_duration = QRadioButton("Pendant")
        self.mode_duration.setToolTip("Durée fixe à partir du lancement")
        self.mode_until = QRadioButton("Jusqu'à")
        self.mode_until.setToolTip("Heure de fin (si elle est déjà passée aujourd'hui : demain)")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_duration, 0)
        self.mode_group.addButton(self.mode_until, 1)
        self.duration = QDoubleSpinBox()
        self.duration.setLocale(loc)
        self.duration.setRange(0.5, 72.0)
        self.duration.setDecimals(1)
        self.duration.setSingleStep(0.5)
        self.duration.setSuffix(" h")
        self.duration.setToolTip("Durée de la session, en heures")
        self.end_time = QTimeEdit()
        self.end_time.setLocale(loc)
        self.end_time.setDisplayFormat("HH' h 'mm")
        self.end_time.setToolTip("Heure locale de fin de la session")
        hm = _parse_hhmm(d["end_time"]) or _parse_hhmm(NIGHT_DEFAULTS["end_time"])
        self.end_time.setTime(QTime(hm[0], hm[1]))
        if d["duration_h"] > 0:
            self.duration.setValue(d["duration_h"])
            self.mode_duration.setChecked(True)
        else:
            self.duration.setValue(10.0)
            self.mode_until.setChecked(True)
        g.addWidget(_label("Durée"), 1, 0)
        g.addWidget(self.mode_duration, 1, 1)
        g.addWidget(self.duration, 1, 2)
        g.addWidget(self.mode_until, 2, 1)
        g.addWidget(self.end_time, 2, 2)
        self.end_preview = QLabel("")
        self.end_preview.setObjectName("endPreview")
        self.end_preview.setWordWrap(True)
        g.addWidget(self.end_preview, 3, 1, 1, 2)
        lay.addLayout(g)
        self.mode_group.idToggled.connect(lambda *_: self._sync_mode())
        self._sync_mode()

        lay.addSpacing(4)
        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFixedHeight(1)
        lay.addWidget(sep)
        lay.addSpacing(2)
        t = QLabel("DANS LA CHAMBRE")
        t.setObjectName("panelTitle")
        lay.addWidget(t)
        self.sample_fields = SampleFields(d, scale_help="0 = inconnue : les longueurs en millimètres sont "
                                                       "estimées. Calibrez d'abord avec une règle si possible.")
        lay.addWidget(self.sample_fields)
        lay.addStretch(1)
        return box

    def _record_panel(self, d: dict) -> QFrame:
        box, lay = _panel("ENREGISTREMENT ET SÉCURITÉ")
        loc = _fr_locale()
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        g.setColumnStretch(1, 1)
        self.record_mode = QComboBox()
        self.record_mode.setToolTip("Les traces détectées (vignettes, mesures, rapport) sont toujours enregistrées ; "
                                    "ce choix concerne seulement la vidéo brute.")
        for key in RECORD_MODES:
            self.record_mode.addItem("", key)
        self.record_mode.setCurrentIndex(RECORD_MODES.index(d["record_video"]))
        g.addWidget(_label("Vidéo"), 0, 0)
        g.addWidget(self.record_mode, 0, 1)

        self.reduced_row = QWidget()
        rr = QHBoxLayout(self.reduced_row)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.setSpacing(8)
        self.record_scale = QComboBox()
        self.record_scale.setToolTip("Taille de l'image enregistrée par rapport à l'image de la caméra")
        for txt, val in SCALE_CHOICES:
            self.record_scale.addItem(f"{txt} résolution", val)
        k = min(range(len(SCALE_CHOICES)), key=lambda i: abs(SCALE_CHOICES[i][1] - d["record_scale"]))
        if abs(SCALE_CHOICES[k][1] - d["record_scale"]) > 1e-3:
            self.record_scale.addItem(f"{fmt_num(100 * d['record_scale'], 0)} % de la résolution", d["record_scale"])
            k = self.record_scale.count() - 1
        self.record_scale.setCurrentIndex(k)
        self.record_fps = QDoubleSpinBox()
        self.record_fps.setLocale(loc)
        self.record_fps.setRange(1, 60)
        self.record_fps.setDecimals(0)
        self.record_fps.setSuffix(" img/s")
        self.record_fps.setValue(d["record_fps"])
        self.record_fps.setToolTip("Cadence de la vidéo enregistrée (l'analyse, elle, utilise toutes les images)")
        rr.addWidget(self.record_scale, 1)
        rr.addWidget(self.record_fps)
        g.addWidget(self.reduced_row, 1, 1)

        self.segment = QSpinBox()
        self.segment.setRange(5, 240)
        self.segment.setSingleStep(5)
        self.segment.setSuffix(" min")
        self.segment.setValue(int(min(240, max(5, d["segment_min"]))))
        self.segment.setToolTip("La vidéo est découpée en fichiers de cette durée : en cas de coupure de courant, "
                                "seul le dernier morceau est perdu.")
        g.addWidget(_label("Fichiers de", "Durée de chaque morceau de vidéo"), 2, 0)
        g.addWidget(self.segment, 2, 1)

        self.snapshot = QSpinBox()
        self.snapshot.setRange(0, 240)
        self.snapshot.setSuffix(" min")
        self.snapshot.setSpecialValueText("jamais")
        self.snapshot.setValue(int(d["snapshot_every_min"]))
        self.snapshot.setToolTip("Une photo de la chambre est enregistrée à cet intervalle (0 = jamais) : "
                                 "pratique pour vérifier le lendemain que tout allait bien.")
        g.addWidget(_label("Photo toutes les", "Photo de contrôle de la chambre"), 3, 0)
        g.addWidget(self.snapshot, 3, 1)

        self.autosave = QSpinBox()
        self.autosave.setRange(1, 60)
        self.autosave.setSuffix(" min")
        self.autosave.setValue(int(min(60, d["autosave_every_min"])))
        self.autosave.setToolTip("Les résultats (traces, compteurs, tableau) sont écrits sur le disque à cet "
                                 "intervalle : rien n'est perdu si l'ordinateur s'arrête.")
        g.addWidget(_label("Sauvegarde toutes les", "Sauvegarde automatique des résultats"), 4, 0)
        g.addWidget(self.autosave, 4, 1)

        self.dead_after = QSpinBox()
        self.dead_after.setRange(0, 240)
        self.dead_after.setSuffix(" min")
        self.dead_after.setSpecialValueText("jamais")
        self.dead_after.setValue(int(min(240, d["dead_after_min"])))
        self.dead_after.setToolTip("Avertir si aucune trace valide n'est détectée pendant ce délai : la chambre n'a "
                                   "probablement plus d'alcool ou de glace sèche.")
        g.addWidget(_label("Alerte sans trace", "Chambre « morte » : plus d'alcool ou de glace sèche ?"), 5, 0)
        g.addWidget(self.dead_after, 5, 1)

        self.keep_awake = QCheckBox("Empêcher la mise en veille de l'ordinateur")
        self.keep_awake.setChecked(bool(d["keep_awake"]))
        self.keep_awake.setToolTip("Windows ne met pas l'ordinateur ni l'écran en veille pendant la session")
        g.addWidget(self.keep_awake, 6, 0, 1, 2)
        g.addWidget(_label("Dossier"), 7, 0)
        self.output = _FolderField("Dossier de la session de nuit", f"Par défaut : {_default_output_dir()}")
        self.output.set_path(d["output_dir"])
        self.output.edit.setToolTip("Un sous-dossier daté est créé pour la session")
        g.addWidget(self.output, 7, 1)
        lay.addLayout(g)
        lay.addSpacing(4)

        self.est_box = QFrame()
        self.est_box.setObjectName("estBox")
        el = QVBoxLayout(self.est_box)
        el.setContentsMargins(14, 10, 14, 10)
        el.setSpacing(3)
        self.est_main = QLabel("")
        self.est_main.setObjectName("estMain")
        self.est_sub = QLabel("")
        self.est_sub.setObjectName("estSub")
        self.est_sub.setWordWrap(True)
        self.est_warn = QLabel("")
        self.est_warn.setObjectName("estWarn")
        self.est_warn.setWordWrap(True)
        el.addWidget(self.est_main)
        el.addWidget(self.est_sub)
        el.addWidget(self.est_warn)
        lay.addWidget(self.est_box)
        lay.addStretch(1)
        self.record_mode.currentIndexChanged.connect(lambda *_: self._sync_record())
        self.record_scale.currentIndexChanged.connect(lambda *_: self._sync_record())
        self.record_fps.valueChanged.connect(lambda *_: self._sync_record())
        self._sync_record()
        return box

    def _checklist(self) -> QFrame:
        box = QFrame()
        box.setObjectName("checklist")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(14)
        t = QLabel("✅  Avant de partir")
        t.setObjectName("checkTitle")
        lay.addWidget(t)
        self.checklist_items = []
        for txt in ("🧊 Assez d'alcool et de glace sèche ?", "📷 Caméra fixée ?", "🔌 Ordinateur branché sur secteur ?"):
            lab = QLabel(txt)
            lab.setObjectName("checkItem")
            lay.addWidget(lab)
            self.checklist_items.append(lab)
        lay.addStretch(1)
        return box

    # ------------------------------------------------------------------ logique --
    def _sync_mode(self):
        until = self.mode_until.isChecked()
        self.end_time.setEnabled(until)
        self.duration.setEnabled(not until)

    def _scale_text(self) -> str:
        v = float(self.record_scale.currentData() or 0.5)
        for txt, val in SCALE_CHOICES:
            if abs(val - v) < 1e-3:
                return f"{txt} résolution"
        return f"{fmt_num(100 * v, 0)} %"

    def _sync_record(self):
        texts = {"aucune": "Aucune (vignettes et photos seulement)",
                 "reduite": f"Réduite ({self._scale_text()}, {fmt_num(self.record_fps.value(), 0)} img/s) — recommandé",
                 "complete": "Complète (lourde)"}
        for i in range(self.record_mode.count()):
            self.record_mode.setItemText(i, texts[self.record_mode.itemData(i)])
        mode = self.record_mode.currentData()
        self.reduced_row.setVisible(mode == "reduite")
        self.segment.setEnabled(mode != "aucune")

    def selected_frame_wh(self) -> tuple[int, int]:
        idx = self.camera.currentData()
        for c in self._cams:
            if int(c.get("index", -1)) == idx:
                try:
                    w, h = int(c.get("width") or 0), int(c.get("height") or 0)
                    if w > 0 and h > 0:
                        return w, h
                except (TypeError, ValueError):
                    pass
        return self._frame_wh

    def end_datetime(self) -> _dt.datetime:
        """Fin prévue si la session était lancée maintenant."""
        now = self._now()
        if self.mode_duration.isChecked():
            return now + _dt.timedelta(hours=float(self.duration.value()))
        t = self.end_time.time()
        end = now.replace(hour=t.hour(), minute=t.minute(), second=0, microsecond=0)
        if end <= now:
            end += _dt.timedelta(days=1)
        return end

    def planned_hours(self) -> float:
        if self.mode_duration.isChecked():
            return float(self.duration.value())
        return max(0.0, (self.end_datetime() - self._now()).total_seconds() / 3600.0)

    def _refresh(self, *_):
        now = self._now()
        end = self.end_datetime()
        hours = self.planned_hours()
        if self.mode_until.isChecked():
            self.end_preview.setText(f"Fin {_day_word(end, now)} à {fmt_clock(end)} — durée ≈ "
                                     f"{fmt_duration(hours * 3600)}")
        else:
            self.end_preview.setText(f"Lancée maintenant, la session se termine {_day_word(end, now)} vers "
                                     f"{fmt_clock(end)}.")
        self._update_estimate(hours)
        dead = self.dead_after.value()
        self.error_label.setText(f"Une alerte s'affichera si aucune trace valide n'est vue pendant {dead} min."
                                 if dead > 0 else "Aucune alerte « chambre sans trace » ne sera affichée.")

    def _update_estimate(self, hours: float):
        path = self.output.path() or _default_output_dir()
        d = self.result_dict()
        d.update(duration_h=hours, end_time="")
        est = free = None
        if self._longrun is not None:
            cfg_cls, estimate, free_fn = self._longrun
            try:
                est = int(estimate(cfg_cls.from_dict(d), self.selected_frame_wh(), self._fps))
                est = est if est >= 0 else None
            except Exception:
                est = None
            try:
                free = int(free_fn(path))
                free = free if free >= 0 else None
            except Exception:
                free = None
        if free is None:
            free = _free_bytes_fallback(path)
        self.estimate_bytes, self.free_bytes = est, free
        anchor = Path(path).anchor or path
        free_txt = f"Espace libre : {_size(free)} sur {anchor}" if free is not None else "Espace libre inconnu"
        state, warn = "ok", ""
        if est is None:
            state = "na"
            self.est_main.setText("💾  Estimation indisponible")
            self.est_sub.setText(free_txt + " · la taille dépendra du réglage « Vidéo » et de la durée.")
        else:
            per_h = est / hours if hours > 0 else 0
            self.est_main.setText(f"💾  ≈ {_size(est)} nécessaires")
            self.est_sub.setText(f"{free_txt} · ≈ {_size(int(per_h))} par heure")
            if free is not None and est > free:
                state = "bad"
                warn = (f"⚠  Espace insuffisant : il manque ≈ {_size(est - free)}. Choisissez « Réduite » ou "
                        "« Aucune », raccourcissez la session ou changez de dossier.")
            elif free is not None and est > 0.8 * free:
                state = "warn"
                warn = "Le disque sera presque plein à la fin de la session."
        self.space_ok = state != "bad"
        self.est_warn.setText(warn)
        self.est_warn.setVisible(bool(warn))
        _set_state(self.est_box, "state", state)
        self.ok_btn.setText("⚠  Lancer quand même" if state == "bad" else "🌙  Lancer la session de nuit")

    def validate(self) -> str:
        """Message d'erreur bloquant, ou "" si la session peut démarrer."""
        out = self.output.path()
        if out:
            anchor = Path(out).anchor
            if not anchor or not os.path.exists(anchor):
                return "Le dossier choisi est sur un disque introuvable (clé USB débranchée ?)."
            if os.path.isfile(out):
                return "Le dossier choisi est un fichier : choisissez un dossier."
        if self.planned_hours() <= 0:
            return "La durée de la session doit être positive."
        return ""

    def _on_ok(self):
        err = self.validate()
        if err:
            self.error_label.setText(f"⚠  {err}")
            _set_state(self.error_label, "error", True)
            return
        self.accept()

    def result_dict(self) -> dict:
        """Dictionnaire « nuit » (toutes les clés de NIGHT_DEFAULTS)."""
        until = self.mode_until.isChecked()
        t = self.end_time.time()
        d = {"camera_index": int(self.camera.currentData() if self.camera.currentData() is not None else 0),
             "duration_h": 0.0 if until else float(self.duration.value()),
             "end_time": f"{t.hour():02d}:{t.minute():02d}" if until else "",
             "segment_min": int(self.segment.value()),
             "record_video": str(self.record_mode.currentData()),
             "record_scale": float(self.record_scale.currentData() or 0.5),
             "record_fps": float(self.record_fps.value()),
             "snapshot_every_min": int(self.snapshot.value()),
             "autosave_every_min": int(self.autosave.value()),
             "keep_awake": bool(self.keep_awake.isChecked()),
             "dead_after_min": int(self.dead_after.value()),
             "output_dir": self.output.path()}
        d.update(self.sample_fields.values())
        return {k: d[k] for k in NIGHT_DEFAULTS}


# ============================================================================ analyse par lot ====
class _FileDelegate(QStyledItemDelegate):
    """Ligne de la liste des vidéos : nom à gauche, taille à droite (grisée)."""

    def sizeHint(self, option, index):  # noqa: N802 (API Qt)
        return QSize(200, 32)

    def paint(self, p: QPainter, opt, index):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(opt.rect).adjusted(3, 1, -3, -1)
        if opt.state & QStyle.StateFlag.State_Selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(B.SELECTION))
            p.drawRoundedRect(r, 6, 6)
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(B.HOVER))
            p.drawRoundedRect(r, 6, 6)
        f = QFont(opt.font)
        p.setFont(f)
        fm = p.fontMetrics()
        size_txt = str(index.data(Qt.ItemDataRole.UserRole + 1) or "")
        sw = fm.horizontalAdvance(size_txt) + 10
        p.setPen(QColor(TEXT))
        p.drawText(r.adjusted(8, 0, 0, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), "🎞")
        name = fm.elidedText(str(index.data(Qt.ItemDataRole.DisplayRole) or ""), Qt.TextElideMode.ElideMiddle,
                             int(r.width() - 40 - sw))
        p.drawText(r.adjusted(34, 0, -sw, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), name)
        p.setPen(QColor(MUTED))
        p.drawText(r.adjusted(0, 0, -8, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                   size_txt)
        p.restore()


class _FileList(QListWidget):
    """Liste des vidéos du lot : glisser-déposer de fichiers ou de dossiers, Suppr pour retirer."""
    paths_dropped = Signal(list)
    delete_pressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("batchFiles")
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setItemDelegate(_FileDelegate(self))
        self.setMouseTracking(True)
        self.setMinimumHeight(190)
        self.setAccessibleName("Vidéos à analyser")

    @staticmethod
    def _paths(mime) -> list[str]:
        if mime is None or not mime.hasUrls():
            return []
        return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]

    def dragEnterEvent(self, e):  # noqa: N802
        if self._paths(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):  # noqa: N802
        if self._paths(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):  # noqa: N802
        paths = self._paths(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.paths_dropped.emit(paths)
        else:
            e.ignore()

    def keyPressEvent(self, e):  # noqa: N802
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_pressed.emit()
            e.accept()
            return
        super().keyPressEvent(e)

    def paintEvent(self, e):  # noqa: N802
        super().paintEvent(e)
        if self.count():
            return
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.viewport().rect()).adjusted(12, 12, -12, -12)
        p.setPen(QPen(QColor(B.BORDER_STRONG), 1.4, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, 10, 10)
        f = QFont(self.font())
        f.setPixelSize(28)
        p.setFont(f)
        p.setPen(QColor(MUTED))
        p.drawText(r.adjusted(0, -40, 0, -40), int(Qt.AlignmentFlag.AlignCenter), "📥")
        f.setPixelSize(13)
        p.setFont(f)
        p.drawText(r.adjusted(16, 30, -16, 30), int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                   "Glissez-déposez ici les vidéos reçues\nou cliquez sur « Ajouter des vidéos… »")


class BatchDialog(QDialog):
    """Vidéos reçues : liste de fichiers à analyser l'un après l'autre et/ou dossier surveillé."""

    def __init__(self, defaults: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("BatchDialog")
        self.setWindowTitle("Vidéos reçues — analyse par lot")
        self.setStyleSheet(SESSION_QSS)
        self.setModal(True)
        self.setAcceptDrops(True)
        d = normalize_batch(defaults)
        self._ignored = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(14)
        root.addLayout(_header("📥", BATCH_TINT, "Vidéos reçues — analyse par lot",
                               "Analysez d'un coup les vidéos qu'on vous a envoyées (courriel, clé USB, OneDrive, "
                               "Google Drive…). Elles sont traitées l'une après l'autre, en arrière-plan."))
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._files_panel(d), 6)
        body.addWidget(self._settings_panel(d), 5)
        root.addLayout(body, 1)

        row = QHBoxLayout()
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        row.addWidget(self.hint, 1)
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn = QPushButton("▶  Analyser")
        self.ok_btn.setProperty("primary", True)
        self.ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok_btn.setDefault(True)
        self.ok_btn.setMinimumWidth(200)
        self.ok_btn.clicked.connect(self._on_ok)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.ok_btn)
        root.addLayout(row)

        self.add_files(d["files"])
        self.watch_check.setChecked(bool(d["watch_folder"]))
        self.watch_field.set_path(d["watch_folder"])
        self._sync()
        _fit_to_screen(self, self.sizeHint().expandedTo(QSize(980, 600)))

    # ------------------------------------------------------------------ panneaux --
    def _files_panel(self, d: dict) -> QFrame:
        box, lay = _panel("VIDÉOS À ANALYSER")
        self.file_list = _FileList()
        self.file_list.paths_dropped.connect(self.add_paths)
        self.file_list.delete_pressed.connect(self.remove_selected)
        self.file_list.itemSelectionChanged.connect(self._sync)
        lay.addWidget(self.file_list, 1)
        r = QHBoxLayout()
        r.setSpacing(8)
        self.add_btn = QPushButton("＋  Ajouter des vidéos…")
        self.add_btn.setToolTip("Choisir un ou plusieurs fichiers vidéo (Ctrl ou Maj pour en choisir plusieurs)")
        self.add_btn.clicked.connect(self.choose_files)
        self.add_folder_btn = QPushButton("📁  Tout un dossier…")
        self.add_folder_btn.setToolTip("Ajouter toutes les vidéos d'un dossier")
        self.add_folder_btn.clicked.connect(self.choose_folder)
        self.remove_btn = QPushButton("Retirer")
        self.remove_btn.setToolTip("Retirer les vidéos sélectionnées de la liste (touche Suppr)")
        self.remove_btn.clicked.connect(self.remove_selected)
        self.clear_btn = QPushButton("Tout retirer")
        self.clear_btn.clicked.connect(self.clear_files)
        for b in (self.add_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        r.addWidget(self.add_btn)
        r.addWidget(self.add_folder_btn)
        r.addStretch(1)
        r.addWidget(self.remove_btn)
        r.addWidget(self.clear_btn)
        lay.addLayout(r)
        self.summary = QLabel("")
        self.summary.setObjectName("fileSummary")
        lay.addWidget(self.summary)
        lay.addSpacing(4)
        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFixedHeight(1)
        lay.addWidget(sep)
        lay.addSpacing(4)
        self.watch_check = QCheckBox("👁  Surveiller un dossier")
        self.watch_check.setObjectName("strongCheck")
        self.watch_check.setToolTip("Analyser automatiquement chaque nouvelle vidéo qui arrive dans un dossier")
        self.watch_check.toggled.connect(self._sync)
        lay.addWidget(self.watch_check)
        self.watch_field = _FolderField("Dossier à surveiller", "Dossier où arrivent les vidéos")
        self.watch_field.changed.connect(self._on_watch_text)
        lay.addWidget(self.watch_field)
        lay.addWidget(_help("Chaque nouvelle vidéo copiée dans ce dossier — par exemple un dossier OneDrive/Google "
                            "Drive où on vous envoie des vidéos — sera analysée automatiquement."))
        return box

    def _settings_panel(self, d: dict) -> QFrame:
        box, lay = _panel("RÉGLAGES POUR CES VIDÉOS")
        self.sample_fields = SampleFields(d)
        lay.addWidget(self.sample_fields)
        lay.addWidget(_help("Ces réglages s'appliquent à toutes les vidéos de la liste et du dossier surveillé."))
        lay.addSpacing(6)
        lay.addWidget(_label("Dossier des résultats"))
        self.output = _FolderField("Dossier des résultats", f"Par défaut : {_default_output_dir()}")
        self.output.set_path(d["output_dir"])
        self.output.edit.setToolTip("Un sous-dossier de résultats est créé pour chaque vidéo")
        lay.addWidget(self.output)
        lay.addSpacing(4)
        self.reports = QCheckBox("Créer un rapport pour chaque vidéo")
        self.reports.setChecked(bool(d["make_reports"]))
        self.reports.setToolTip("Rapport HTML (compteurs, graphiques, conclusions) dans le dossier de chaque vidéo")
        lay.addWidget(self.reports)
        lay.addStretch(1)
        return box

    # ------------------------------------------------------------------ fichiers --
    def files(self) -> list[str]:
        return [str(self.file_list.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.file_list.count())]

    def add_files(self, paths) -> int:
        """Ajoute des fichiers vidéo (doublons, fichiers absents et non-vidéos ignorés) ; renvoie le nombre ajouté."""
        known = {_norm(p) for p in self.files()}
        added = 0
        for p in paths or []:
            p = os.path.normpath(str(p))
            if Path(p).suffix.lower() not in VIDEO_EXTS or not os.path.isfile(p):
                self._ignored += 1
                continue
            k = _norm(p)
            if k in known:
                continue
            known.add(k)
            try:
                size = fmt_size(os.path.getsize(p))
            except OSError:
                size = ""
            it = QListWidgetItem(Path(p).name)
            it.setData(Qt.ItemDataRole.UserRole, p)
            it.setData(Qt.ItemDataRole.UserRole + 1, size)
            it.setToolTip(p)
            self.file_list.addItem(it)
            added += 1
        self._sync()
        return added

    def add_paths(self, paths) -> int:
        """Fichiers et dossiers déposés : les dossiers apportent les vidéos qu'ils contiennent."""
        files = []
        for p in paths or []:
            if os.path.isdir(p):
                files.extend(_videos_in(p))
            else:
                files.append(p)
        return self.add_files(files)

    def choose_files(self):
        start = os.path.dirname(self.files()[-1]) if self.files() else str(Path.home())
        paths, _flt = QFileDialog.getOpenFileNames(self, "Vidéos à analyser", start, VIDEO_FILTER)
        if paths:
            self.add_files(paths)

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Ajouter toutes les vidéos d'un dossier", str(Path.home()))
        if d:
            self.add_files(_videos_in(d))

    def remove_selected(self):
        for it in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(it))
        self._sync()

    def clear_files(self):
        self.file_list.clear()
        self._sync()

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData() is not None and e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):  # noqa: N802
        paths = _FileList._paths(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.add_paths(paths)

    # ------------------------------------------------------------------ validation --
    def _on_watch_text(self, text: str):
        if text.strip() and not self.watch_check.isChecked():
            self.watch_check.setChecked(True)
        self._sync()

    def watch_folder(self) -> str:
        return self.watch_field.path() if self.watch_check.isChecked() else ""

    def validate(self) -> str:
        """Message d'erreur, ou "" si le lot peut démarrer (au moins une vidéo ou un dossier surveillé)."""
        w = self.watch_folder()
        if w and not os.path.isdir(w):
            return "Le dossier à surveiller est introuvable."
        if not self.files() and not w:
            return "Ajoutez au moins une vidéo ou choisissez un dossier à surveiller."
        return ""

    def _sync(self, *_):
        files = self.files()
        n = len(files)
        total = 0
        for p in files:
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
        if n:
            txt = f"{n} vidéo{'s' if n > 1 else ''} · {fmt_size(total)}"
        else:
            txt = "Aucune vidéo pour l'instant"
        if self._ignored:
            txt += f" · {self._ignored} fichier{'s' if self._ignored > 1 else ''} ignoré" \
                   f"{'s' if self._ignored > 1 else ''} (pas une vidéo)"
        self.summary.setText(txt)
        self.remove_btn.setEnabled(bool(self.file_list.selectedItems()))
        self.clear_btn.setEnabled(n > 0)
        self.watch_field.setEnabled(self.watch_check.isChecked())
        err = self.validate()
        self.ok_btn.setEnabled(not err)
        w = self.watch_folder()
        if n and w:
            self.ok_btn.setText(f"▶  Analyser {n} vidéo{'s' if n > 1 else ''} et surveiller")
        elif n:
            self.ok_btn.setText(f"▶  Analyser {n} vidéo{'s' if n > 1 else ''}")
        elif w:
            self.ok_btn.setText("👁  Surveiller le dossier")
        else:
            self.ok_btn.setText("▶  Analyser")
        self.hint.setText(err)
        _set_state(self.hint, "error", bool(err) and (n > 0 or bool(w)))

    def _on_ok(self):
        if not self.validate():
            self.accept()

    def result_dict(self) -> dict:
        """Dictionnaire « lot » : {"files","watch_folder","condition","sample","mass_g","px_per_mm","output_dir",
        "make_reports"}."""
        d = {"files": self.files(), "watch_folder": self.watch_folder(), "output_dir": self.output.path(),
             "make_reports": bool(self.reports.isChecked())}
        d.update(self.sample_fields.values())
        return {k: d[k] for k in BATCH_DEFAULTS}


# ============================================================================ bandeau de nuit ====
STATUS_QSS = f"""
QFrame#SessionStatusBar {{ border: 1px solid {B.FOREST}; border-radius: 20px; background: {B.FOREST}; }}
QFrame#SessionStatusBar[alert="true"] {{ border: 2px solid {B.WARN_BORDER}; }}
QFrame#SessionStatusBar[finished="true"] {{ border: 2px solid {B.BRIGHT};
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {B.FOREST}, stop:1 #1B5C40); }}
QLabel#sbEyebrow {{ color: {B.MINT}; font-size: 8pt; font-weight: 700; }}
QLabel#sbElapsed {{ color: {B.WHITE}; font-size: 24pt; font-weight: 700; }}
QLabel#sbElapsedUnit {{ color: {B.MINT}; font-size: 10pt; }}
QLabel#sbSub {{ color: {B.WHITE}; }}
QFrame#statTile {{ background: {B.WHITE}; border: none; border-radius: 14px; }}
QLabel#statTitle {{ color: {B.GREEN}; font-size: 8pt; font-weight: 700; }}
QLabel#statValue {{ color: {B.INK}; font-size: 14pt; font-weight: 700; }}
QLabel#statSub {{ color: {B.MUTED}; font-size: 8pt; }}
QLabel#sbWarnings {{ color: {B.WARN_TEXT}; background: {B.WARN_BG}; border: 1px solid {B.WARN_BORDER};
    border-radius: 10px; padding: 7px 10px; font-weight: 600; }}
QLabel#sbFooter {{ color: {B.WHITE}; font-size: 9pt; }}
QFrame#SessionStatusBar QProgressBar {{ border: none; border-radius: 3px; background: {B.rgba(B.WHITE, 50)};
    min-height: 6px; max-height: 6px; }}
QFrame#SessionStatusBar QProgressBar::chunk {{ border-radius: 3px; background: {B.MINT}; }}
QFrame#SessionStatusBar QPushButton[link="true"] {{ color: {B.MINT}; background: transparent; border: none;
    padding: 2px 4px; font-weight: 600; }}
QFrame#SessionStatusBar QPushButton[link="true"]:hover {{ color: {B.WHITE}; text-decoration: underline; }}
QFrame#SessionStatusBar QPushButton[link="true"]:focus {{ color: {B.WHITE}; text-decoration: underline; }}
"""


class _StatTile(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("statTile")
        self.setMinimumWidth(128)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        self.title = QLabel(title.upper())
        self.title.setObjectName("statTitle")
        self.value = QLabel("—")
        self.value.setObjectName("statValue")
        self.value.setTextFormat(Qt.TextFormat.RichText)
        self.sub = QLabel("")
        self.sub.setObjectName("statSub")
        lay.addWidget(self.title)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)
        self.setAccessibleName(title)

    def set(self, value: str, sub: str = "", color: str | None = None):
        self.value.setText(value)
        self.value.setStyleSheet(f"color: {color};" if color else "")
        self.sub.setText(sub)
        self.setAccessibleDescription(f"{value} {sub}")


class SessionStatusBar(QFrame):
    """Bandeau de la session de nuit : temps écoulé/restant, traces, rythme, disque, enregistrement, alertes.

    ``update_status(d)`` reçoit le dictionnaire d'état du travailleur (toutes les ~2 s).
    Le bouton « Arrêter » demande une confirmation par un second clic (``confirm_stop``) puis émet
    ``stop_requested``.
    """
    stop_requested = Signal()
    open_folder_requested = Signal(str)

    def __init__(self, parent=None, confirm_stop: bool = True):
        super().__init__(parent)
        self.setObjectName("SessionStatusBar")
        self.setStyleSheet(STATUS_QSS)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.confirm_stop = confirm_stop
        self._now = _dt.datetime.now
        self._armed = False
        self._finished = False
        self._folder = ""
        self.last_status: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 12)
        outer.setSpacing(10)
        top = QHBoxLayout()
        top.setSpacing(16)
        self.icon = _tinted_icon("🌙", B.MINT, 52, 14, 26)
        top.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.eyebrow = QLabel("SESSION DE NUIT EN COURS")
        self.eyebrow.setObjectName("sbEyebrow")
        er = QHBoxLayout()
        er.setSpacing(8)
        self.elapsed = QLabel("0 s")
        self.elapsed.setObjectName("sbElapsed")
        self.elapsed_unit = QLabel("écoulées")
        self.elapsed_unit.setObjectName("sbElapsedUnit")
        er.addWidget(self.elapsed)
        er.addWidget(self.elapsed_unit, 0, Qt.AlignmentFlag.AlignBottom)
        er.addStretch(1)
        self.remaining = QLabel("")
        self.remaining.setObjectName("sbSub")
        col.addWidget(self.eyebrow)
        col.addLayout(er)
        col.addWidget(self.remaining)
        col.addStretch(1)
        head = QWidget()
        head.setLayout(col)
        head.setMinimumWidth(250)
        head.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        top.addWidget(head, 0)
        self.tiles_host = QWidget()
        self.tiles_host.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        flow = FlowLayout(self.tiles_host, 10, 10)
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.tiles_host.setSizePolicy(sp)
        self.tile_tracks = _StatTile("Traces valides")
        self.tile_rate = _StatTile("Rythme")
        self.tile_disk = _StatTile("Disque libre")
        self.tile_rec = _StatTile("Vidéo")
        self.tiles = [self.tile_tracks, self.tile_rate, self.tile_disk, self.tile_rec]
        for t in self.tiles:
            flow.addWidget(t)
        top.addWidget(self.tiles_host, 1)
        self.stop_btn = QPushButton("■  Arrêter la session")
        self.stop_btn.setProperty("danger", True)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setMinimumWidth(190)
        self.stop_btn.setToolTip("Arrêter la session et enregistrer les résultats (deux clics pour confirmer)")
        self.stop_btn.clicked.connect(self._on_stop)
        top.addWidget(self.stop_btn, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setProperty("slim", True)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1000)
        outer.addWidget(self.progress)
        self.warnings = QLabel("")
        self.warnings.setObjectName("sbWarnings")
        self.warnings.setWordWrap(True)
        self.warnings.setTextFormat(Qt.TextFormat.RichText)
        self.warnings.hide()
        outer.addWidget(self.warnings)
        foot = QHBoxLayout()
        foot.setSpacing(10)
        self.footer = QLabel("")
        self.footer.setObjectName("sbFooter")
        self.footer.setWordWrap(True)
        foot.addWidget(self.footer, 1)
        self.folder_btn = QPushButton("📂  Dossier de la session")
        self.folder_btn.setProperty("link", True)
        self.folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.folder_btn.clicked.connect(self._open_folder)
        self.folder_btn.hide()
        foot.addWidget(self.folder_btn)
        outer.addLayout(foot)
        self._disarm_timer = QTimer(self)
        self._disarm_timer.setSingleShot(True)
        self._disarm_timer.setInterval(5000)
        self._disarm_timer.timeout.connect(self._disarm)
        self.update_status({})

    # ------------------------------------------------------------------ état --
    def update_status(self, d: dict):
        """Met à jour le bandeau avec le dictionnaire d'état de la session (clés manquantes tolérées)."""
        d = d if isinstance(d, dict) else {}
        self.last_status = dict(d)

        def num(key, default=0.0):
            try:
                v = float(d.get(key, default))
                return v if math.isfinite(v) else default
            except (TypeError, ValueError):
                return default
        elapsed = max(0.0, num("elapsed_s"))
        remaining = num("remaining_s", -1.0)
        self.elapsed.setText(fmt_hms(elapsed))
        self.elapsed_unit.setText("écoulée" if elapsed < 2 else "écoulées")
        if self._finished:
            self.remaining.setText(f"Durée totale : {fmt_hms(elapsed)}")
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
        elif remaining >= 0:
            end = self._now() + _dt.timedelta(seconds=remaining)
            self.remaining.setText(f"Reste {fmt_hms(remaining)} · fin vers {fmt_clock(end)}")
            total = elapsed + remaining
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(round(1000 * elapsed / total)) if total > 0 else 0)
        else:
            self.remaining.setText("Durée libre : jusqu'à l'arrêt manuel")
            self.progress.setRange(0, 0)               # barre « occupée »
        n = int(num("n_tracks"))
        self.tile_tracks.set(fmt_num(n, 0), "depuis le début")
        self.tile_rate.set(f"{fmt_num(num('rate_per_min'), 1)}<span style='font-size:9pt; color:{MUTED}'>"
                           f"{NBSP}/min</span>", "10 dernières minutes")
        free = d.get("disk_free_bytes")
        try:
            free = int(free) if free is not None else None
        except (TypeError, ValueError):
            free = None
        if free is None or free < 0:
            self.tile_disk.set("—", "inconnu")
        else:
            col = DANGER if free < 2 * 1024 ** 3 else (WARN if free < 10 * 1024 ** 3 else None)
            self.tile_disk.set(_size(free), "attention : presque plein" if col else "sur le disque", col)
        seg = int(num("segment_index", -1))
        if bool(d.get("recording")) and seg >= 0:
            self.tile_rec.set(f"<span style='color:{DANGER}'>●</span>{NBSP}Segment{NBSP}{seg + 1}",
                              "enregistrement en cours")
        elif bool(d.get("recording")):
            self.tile_rec.set(f"<span style='color:{DANGER}'>●</span>{NBSP}En cours", "enregistrement")
        else:
            self.tile_rec.set("Désactivée", "vidéo non enregistrée", MUTED)
        parts = []
        last = str(d.get("last_checkpoint", "") or "")
        if last:
            try:
                parts.append(f"💾 Dernière sauvegarde à {fmt_clock(_dt.datetime.fromisoformat(last))}")
            except ValueError:
                parts.append(f"💾 Dernière sauvegarde : {last}")
        else:
            parts.append("💾 Pas encore de sauvegarde")
        nxt = num("next_autosave_s", -1.0)
        if nxt >= 0 and not self._finished:
            parts.append("prochaine : maintenant" if nxt < 1 else f"prochaine dans {fmt_hms(nxt)}")
        self.footer.setText("  ·  ".join(parts))
        self._folder = str(d.get("session_folder", "") or "")
        self.folder_btn.setVisible(bool(self._folder))
        self.folder_btn.setToolTip(self._folder)
        warns = [str(w) for w in (d.get("warnings") or []) if str(w).strip()]
        self.warnings.setText("<br>".join(f"⚠&nbsp; {html.escape(w)}" for w in warns))
        self.warnings.setVisible(bool(warns))
        _set_state(self, "alert", bool(warns))

    def set_finished(self, message: str = "Session terminée"):
        """État final : bandeau vert, bouton d'arrêt masqué."""
        self._disarm()
        self._finished = True
        self.eyebrow.setText(message.upper())
        self.stop_btn.hide()
        _set_state(self.progress, "success", True)
        _set_state(self, "finished", True)
        self.update_status(self.last_status)

    def reset(self):
        """Nouvelle session : état initial."""
        self._disarm()
        self._finished = False
        self.eyebrow.setText("SESSION DE NUIT EN COURS")
        self.stop_btn.show()
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("■  Arrêter la session")
        _set_state(self.progress, "success", False)
        _set_state(self, "finished", False)
        self.update_status({})

    # ------------------------------------------------------------------ arrêt --
    def _on_stop(self):
        if self.confirm_stop and not self._armed:
            self._armed = True
            self.stop_btn.setText("Cliquer encore pour arrêter")
            _set_state(self.stop_btn, "armed", True)
            self._disarm_timer.start()
            return
        self._disarm()
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Arrêt en cours…")
        self.stop_requested.emit()

    def _disarm(self):
        self._disarm_timer.stop()
        self._armed = False
        _set_state(self.stop_btn, "armed", False)
        if self.stop_btn.isEnabled():
            self.stop_btn.setText("■  Arrêter la session")

    def _open_folder(self):
        if not self._folder:
            return
        if _connected(self, self.open_folder_requested):
            self.open_folder_requested.emit(self._folder)
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._folder))


# ============================================================================ file du lot ====
QUEUE_QSS = f"""
QLabel#queueTitle {{ color: {B.INK}; font-size: 14pt; font-weight: 700; }}
QLabel#queueSummary {{ color: {B.MUTED}; }}
QLabel#queueEmpty {{ color: {B.MUTED}; padding: 26px; background: {B.WHITE};
    border: 1px dashed {B.BORDER_STRONG}; border-radius: 16px; }}
QTableWidget#queueTable {{ background: {B.WHITE}; border: 1px solid {B.BORDER}; border-radius: 14px; }}
QTableWidget#queueTable::item {{ padding: 0 8px; }}
"""

COLS = ["Vidéo", "État", "Progression", "Restant", "α", "β", "μ", "p", "Résultats"]
_COUNT_COLS = {4: "alpha", 5: "beta", 6: "muon", 7: "proton"}


class _ClassHeader(QHeaderView):
    """En-tête de tableau dont les colonnes α β μ p sont écrites dans la couleur de leur classe.

    (La feuille de style impose une couleur unique aux en-têtes : on dessine donc le symbole soi-même.)
    """

    def __init__(self, columns: dict[int, str], parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._cols = dict(columns)

    def paintSection(self, painter, rect, logical):  # noqa: N802 (API Qt)
        super().paintSection(painter, rect, logical)
        cls = self._cols.get(logical)
        if not cls:
            return
        painter.save()
        f = QFont(self.font())
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(QColor(B.class_text_hex(cls)))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), physics.CLASS_INFO.get(cls, {}).get("symbole", cls))
        painter.restore()


class BatchQueueWidget(QWidget):
    """File d'analyse par lot : une ligne par vidéo (état, progression, temps restant, compteurs, résultats)."""
    cancel_requested = Signal()
    open_folder_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BatchQueueWidget")
        self.setStyleSheet(QUEUE_QSS)
        self.jobs: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(_tinted_icon("📥", BATCH_TINT, 38, 10, 18))
        col = QVBoxLayout()
        col.setSpacing(1)
        t = QLabel("File d'analyse des vidéos reçues")
        t.setObjectName("queueTitle")
        self.summary = QLabel("")
        self.summary.setObjectName("queueSummary")
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(t)
        col.addWidget(self.summary)
        head.addLayout(col, 1)
        self.cancel_btn = QPushButton("■  Annuler le lot")
        self.cancel_btn.setProperty("danger", True)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setToolTip("Arrêter la vidéo en cours et annuler les vidéos en attente")
        self.cancel_btn.clicked.connect(self.cancel_requested)
        head.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(head)
        self.overall = QProgressBar()
        self.overall.setProperty("slim", True)
        self.overall.setTextVisible(False)
        self.overall.setRange(0, 1000)
        lay.addWidget(self.overall)

        self.table = QTableWidget(0, len(COLS))
        self.table.setObjectName("queueTable")
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.setMouseTracking(True)
        self.table.setHorizontalHeader(_ClassHeader(_COUNT_COLS, self.table))
        hh = self.table.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for c, name in enumerate(COLS):
            it = QTableWidgetItem("" if c in _COUNT_COLS else name)
            if c in _COUNT_COLS:
                it.setToolTip(physics.CLASS_INFO.get(_COUNT_COLS[c], {}).get("nom", name))
            self.table.setHorizontalHeaderItem(c, it)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c, w in ((1, 120), (2, 150), (3, 96), (8, 120)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(c, w)
        for c in _COUNT_COLS:
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(c, 52)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        lay.addWidget(self.table, 1)
        self.empty = QLabel("📭  Aucune vidéo dans la file. Choisissez « Vidéos reçues / analyse par lot » sur "
                            "la page d'accueil pour en ajouter.")
        self.empty.setObjectName("queueEmpty")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        lay.addWidget(self.empty)
        self.update_jobs([])

    # ------------------------------------------------------------------ mise à jour --
    def update_jobs(self, jobs: list[dict]):
        """Affiche la liste des tâches (réutilise les lignes existantes : appel fréquent sans scintillement)."""
        self.jobs = [normalize_job(j) for j in (jobs or []) if isinstance(j, dict)]
        n = len(self.jobs)
        t = self.table
        t.setUpdatesEnabled(False)
        try:
            if t.rowCount() != n:
                t.setRowCount(n)
            for r, job in enumerate(self.jobs):
                self._set_row(r, job)
        finally:
            t.setUpdatesEnabled(True)
        t.setVisible(n > 0)
        self.empty.setVisible(n == 0)
        self._update_summary()

    def _item(self, r: int, c: int) -> QTableWidgetItem:
        it = self.table.item(r, c)
        if it is None:
            it = QTableWidgetItem()
            self.table.setItem(r, c, it)
        return it

    def _set_row(self, r: int, job: dict):
        icon, label, color = _JOB_STYLE[job["status"]]
        name = self._item(r, 0)
        name.setText(("👁  " if job["from_watch"] else "") + job["name"]
                     + (f"  ⚠{NBSP}{len(job['warnings'])}" if job["warnings"] else ""))
        name.setToolTip((job["path"] or job["name"])
                        + ("\nArrivée dans le dossier surveillé" if job["from_watch"] else "")
                        + "".join(f"\n⚠ {w}" for w in job["warnings"])
                        + (f"\nErreur : {job['error']}" if job["status"] == "erreur" and job["error"] else ""))
        st = self._item(r, 1)
        st.setText(f"{icon}  {label}")
        st.setForeground(QColor(color))
        st.setToolTip(job["error"] if job["status"] == "erreur" and job["error"] else label)
        host = self.table.cellWidget(r, 2)
        bar = host.findChild(QProgressBar) if host is not None else None
        if bar is None:
            host = QWidget()
            hl = QHBoxLayout(host)
            hl.setContentsMargins(6, 0, 10, 0)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFormat("%p %")
            bar.setMaximumHeight(18)
            hl.addWidget(bar)
            self.table.setCellWidget(r, 2, host)
        bar.setValue(int(round(1000 * job["progress"])))
        for prop in ("success", "error", "warning"):
            _set_state(bar, prop, False)
        if job["status"] == "terminé":
            _set_state(bar, "success", True)
        elif job["status"] == "erreur":
            _set_state(bar, "error", True)
        elif job["status"] == "annulé":
            _set_state(bar, "warning", True)
        eta = self._item(r, 3)
        active = job["status"] in ("en cours", "en attente")
        eta.setText(fmt_hms(job["eta_s"]) if active and job["eta_s"] >= 0 else "—")
        eta.setForeground(QColor(TEXT if job["status"] == "en cours" else MUTED))
        has_counts = bool(job["counts"]) and job["status"] in ("en cours", "terminé", "annulé")
        for c, cls in _COUNT_COLS.items():
            it = self._item(r, c)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            try:
                v = int(job["counts"].get(cls, 0) or 0)
            except (TypeError, ValueError):
                v = 0
            it.setText(fmt_num(v, 0) if has_counts else "—")
            it.setForeground(QColor(B.class_text_hex(cls) if has_counts and v else MUTED))
        btn = self.table.cellWidget(r, 8)
        if not isinstance(btn, QPushButton):
            btn = QPushButton()
            btn.setProperty("link", True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, b=btn: self._on_result_button(b))
            self.table.setCellWidget(r, 8, btn)
        folder = job["result_folder"]
        btn.setProperty("folder", folder)
        btn.setProperty("errorText", job["error"] if job["status"] == "erreur" else "")
        if job["status"] == "erreur":
            btn.setText("⚠  Détails")
            btn.setToolTip(job["error"] or "Erreur inconnue")
            btn.setEnabled(True)
        else:
            btn.setText("📂  Ouvrir" if folder else "—")
            btn.setToolTip(folder or "Le dossier de résultats apparaîtra à la fin de l'analyse")
            btn.setEnabled(bool(folder))

    def _update_summary(self):
        counts = {s: 0 for s in JOB_STATUSES}
        for j in self.jobs:
            counts[j["status"]] += 1
        n = len(self.jobs)
        bits = []
        for s in ("terminé", "en cours", "en attente", "erreur", "annulé"):
            if counts[s]:
                icon, label, color = _JOB_STYLE[s]
                word = {"terminé": "terminée", "en cours": "en cours", "en attente": "en attente",
                        "erreur": "erreur", "annulé": "annulée"}[s]
                if counts[s] > 1 and s in ("terminé", "annulé"):
                    word += "s"
                if counts[s] > 1 and s == "erreur":
                    word = "erreurs"
                bits.append(f"<span style='color:{color}'>{icon}</span>&nbsp;{counts[s]}&nbsp;{word}")
        eta = sum(j["eta_s"] for j in self.jobs if j["status"] in ("en cours", "en attente") and j["eta_s"] >= 0)
        if eta > 0:
            bits.append(f"≈&nbsp;{fmt_hms(eta)} restantes")
        self.summary.setText(" &nbsp;·&nbsp; ".join(bits) if bits else "La file est vide.")
        done = sum(1.0 if j["status"] in ("terminé", "erreur", "annulé") else j["progress"] for j in self.jobs)
        self.overall.setValue(int(round(1000 * done / n)) if n else 0)
        self.overall.setVisible(n > 0)
        self.cancel_btn.setEnabled(counts["en cours"] + counts["en attente"] > 0)

    # ------------------------------------------------------------------ actions --
    def _on_result_button(self, btn: QPushButton):
        err = str(btn.property("errorText") or "")
        if err:                                   # erreur : message affiché sous le bouton
            QToolTip.showText(btn.mapToGlobal(btn.rect().bottomLeft()), err, btn)
            return
        self._open(str(btn.property("folder") or ""))

    def _open(self, folder: str):
        if not folder:
            return
        if _connected(self, self.open_folder_requested):
            self.open_folder_requested.emit(folder)
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_double_click(self, r: int, _c: int):
        if 0 <= r < len(self.jobs):
            self._open(self.jobs[r]["result_folder"])
