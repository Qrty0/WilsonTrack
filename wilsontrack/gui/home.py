"""Page d'accueil de WilsonTrack : menu principal, choix d'une vidéo, rappel de la physique des traces.

Contenu :
- grand bandeau vert forêt (style cegepmv.ca) avec la grille de « pilules » menthe du symbole du Cégep,
  animée à l'ouverture (désactivable : WILSONTRACK_NO_ANIM=1) ;
- les 4 cartes essentielles (caméra, vidéo, session de nuit, comparer les résultats) dans une grille
  blanche (2 colonnes, 1 si la fenêtre est étroite), puis une rangée discrète « Plus d'outils »
  (photos, démonstration, lot, centre d'apprentissage de l'IA, assistant Claude, guide) ;
- glisser-déposer d'une vidéo n'importe où sur la page (plusieurs vidéos ou un dossier → analyse par lot) ;
- derniers résultats enregistrés (dans la carte « Comparer ») et pied de page (logo du Cégep, équipe).

La page n'exécute aucune analyse elle-même : elle émet des signaux que la fenêtre principale
relie à ses actions (start_source, open_folder, open_results…). Les boîtes « Session de nuit » et
« Vidéos reçues » sont dans session_dialogs.py, le centre d'apprentissage dans learning.py.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (QEvent, QLocale, QMetaMethod, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl,
                            Signal)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QImage, QPainter, QPainterPath,
                           QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLayout, QPushButton, QScrollArea, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, physics
from . import brand as B

try:  # la liste des conditions est partagée avec le panneau de réglages
    from .settings_panel import CONDITIONS
except Exception:  # pragma: no cover - repli si le panneau est en cours de modification
    CONDITIONS = ["Fond (rayons cosmiques)", "KCl", "Banane", "Substitut de sel", "Source de test", "Autre"]

__all__ = ["HomePage", "VideoSetupDialog", "find_example_videos", "read_video_info", "ground_truth_scale",
           "guess_condition", "reference_model_blurb", "VIDEO_EXTS", "VIDEO_FILTER"]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".m4v")
EXAMPLE_EXTS = (".mp4", ".avi", ".mov")
VIDEO_FILTER = "Vidéos (*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.m4v);;Tous les fichiers (*.*)"
MAX_RECENT_VIDEOS = 8
SHOWN_RECENT_VIDEOS = 4
SHOWN_RECENT_RUNS = 5

# Palette claire du Cégep Marie-Victorin (cohérente avec app.light_palette ; voir brand.py)
WINDOW, BASE, CARD, BORDER = B.SOFT, B.WHITE, B.WHITE, B.BORDER
TEXT, MUTED, FAINT, ACCENT, SUCCESS = B.INK, B.MUTED, B.FAINT, B.GREEN, B.SUCCESS_TEXT
CLAUDE = "#d97757"
NIGHT_TINT, BATCH_TINT, LEARN_TINT = "#7c66dc", "#12a594", "#e93d82"
MODEL_JSON = Path(__file__).resolve().parents[1] / "assets" / "modele_reference.json"
NBSP = "\u00a0"  # espace insécable (milliers, unités)

TEAM = "Yanis Amarouche, Yanees Ait Saadi, Amir Badri, Taha Bich"
TEACHER = "Simon Langlois"


def class_hex(cls: str) -> str:
    fallback = {"alpha": "#e5484d", "beta": "#2f9bff", "muon": "#30a46c", "proton": "#f76b15"}
    return physics.CLASS_INFO.get(cls, {}).get("hex", fallback.get(cls, "#8b8d98"))


# Ce que l'utilisateur voit à l'image, et pourquoi la physique donne cette forme.
PARTICLES = [
    {"cls": "alpha", "nom": "Alpha", "symbole": "α",
     "image": "Courte, épaisse, très dense et bien droite (2 à 7 cm).",
     "pourquoi": "Noyau d'hélium de charge +2, lourd et lent : il arrache énormément d'électrons "
                 "à l'air et s'arrête en quelques centimètres.",
     "meters": (5, 1, 4)},
    {"cls": "beta", "nom": "Bêta (électron)", "symbole": "β",
     "image": "Fine, sinueuse, en zigzag ou enroulée en fin de parcours.",
     "pourquoi": f"L'électron est environ 7{NBSP}300 fois plus léger qu'une alpha : chaque collision "
                 "avec un atome le dévie (diffusion multiple).",
     "meters": (1, 3, 1)},
    {"cls": "muon", "nom": "Muon cosmique", "symbole": "μ",
     "image": "Longue, fine et parfaitement droite ; traverse souvent tout le champ.",
     "pourquoi": "Très énergétique (≈ 4 GeV) et 207 fois plus lourd que l'électron : il n'est presque "
                 "pas dévié et ionise peu.",
     "meters": (1, 5, 5)},
    {"cls": "proton", "nom": "Proton", "symbole": "p",
     "image": "Épaisse et droite, mais plus longue qu'une alpha.",
     "pourquoi": "Charge +1 et lourd : il ionise fortement mais va plus loin qu'une alpha. "
                 "Rare (protons de recul, secondaires cosmiques).",
     "meters": (4, 3, 5)},
]
METER_LABELS = ("Épaisseur", "Longueur", "Rectitude")


# ============================================================================ utilitaires ====
def fmt_num(x: float, dec: int = 1) -> str:
    """Nombre à la québécoise : virgule décimale, espace insécable pour les milliers."""
    try:
        s = f"{float(x):,.{dec}f}"
    except (TypeError, ValueError):
        return "—"
    return s.replace(",", NBSP).replace(".", ",")


def fmt_duration(sec: float) -> str:
    if sec is None or not math.isfinite(sec) or sec <= 0:
        return "—"
    if sec < 60:
        return f"{fmt_num(sec, 1)} s"
    if sec < 3600:
        m, s = divmod(int(round(sec)), 60)
        return f"{m} min {s:02d} s"
    h, rem = divmod(int(round(sec)), 3600)
    return f"{h} h {rem // 60:02d} min"


def fmt_size(n: int) -> str:
    if n < 1024 ** 2:
        return f"{fmt_num(n / 1024, 0)} ko"
    if n < 1024 ** 3:
        return f"{fmt_num(n / 1024 ** 2, 1)} Mo"
    return f"{fmt_num(n / 1024 ** 3, 2)} Go"


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _set_prop(w: QWidget, name: str, value=True) -> QWidget:
    w.setProperty(name, value)
    return w


def find_example_videos() -> list[Path]:
    """Vidéos d'exemple livrées avec l'application (dossier « exemples »)."""
    if getattr(sys, "frozen", False):
        bases = [Path(sys.executable).resolve().parent / "exemples"]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bases.append(Path(meipass) / "exemples")
    else:
        bases = [Path(__file__).resolve().parents[2] / "exemples"]
    for base in bases:
        if base.is_dir():
            return sorted((p for p in base.iterdir() if p.is_file() and p.suffix.lower() in EXAMPLE_EXTS),
                          key=lambda p: p.name.lower())
    return []


def ground_truth_scale(path: str | Path) -> float | None:
    """Échelle (px/mm) lue dans « <vidéo>.gt.json » (vidéos d'exemple simulées), sinon None."""
    p = Path(path)
    for cand in (Path(str(p) + ".gt.json"), p.with_suffix(".gt.json")):
        try:
            if cand.is_file():
                v = float(json.loads(cand.read_text(encoding="utf-8")).get("px_per_mm", 0) or 0)
                if v > 0:
                    return v
        except Exception:
            continue
    return None


def guess_condition(path: str | Path) -> str | None:
    """Devine la condition d'après le nom du fichier (« … fond … », « … KCl … »)."""
    name = Path(path).stem.lower()
    if "fond" in name or "cosmique" in name or "background" in name:
        return CONDITIONS[0]
    if "kcl" in name:
        return next((c for c in CONDITIONS if c.lower() == "kcl"), "KCl")
    if "banane" in name:
        return next((c for c in CONDITIONS if c.lower().startswith("banane")), "Banane")
    if "substitut" in name:
        return next((c for c in CONDITIONS if c.lower().startswith("substitut")), "Substitut de sel")
    return None


def reference_model_blurb(path: str | Path | None = None) -> str:
    """Une ligne sur l'IA de référence installée, lue dans le petit fichier JSON qui accompagne le modèle
    (lecture instantanée : le modèle lui-même n'est pas chargé par la page d'accueil)."""
    p = Path(path) if path else MODEL_JSON
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Aucune IA de référence installée : l'application utilise les règles physiques seules."
    try:
        n = int(d.get("n_tracks_total") or 0)
        jobs = int(d.get("n_jobs") or 0)
    except (TypeError, ValueError, AttributeError):
        n = jobs = 0
    if n <= 0:
        return "IA de référence entraînée sur des traces simulées."
    return (f"Entraînée sur {fmt_num(n, 0)} traces simulées"
            + (f" ({fmt_num(jobs, 0)} enregistrements simulés)" if jobs else "") + ".")


def _sample_key(prefix: str) -> str | None:
    return next((k for k in physics.SAMPLES if k.lower().startswith(prefix)), None)


def is_background(condition: str) -> bool:
    c = (condition or "").strip().lower()
    return c.startswith("fond") or "cosmique" in c


def sample_for_condition(condition: str) -> str | None:
    """Échantillon suggéré pour une condition (None = aucune suggestion)."""
    c = (condition or "").strip().lower()
    if is_background(c):
        return _sample_key("aucun")
    if "kcl" in c:
        return _sample_key("kcl")
    if "banane" in c:
        return _sample_key("banane")
    if "substitut" in c:
        return _sample_key("substitut")
    return None


def _open_capture(path: str):
    cap = cv2.VideoCapture(path)
    if cap is not None and cap.isOpened():
        return cap
    if sys.platform == "win32":  # chemins accentués : OpenCV accepte mieux le nom court 8.3
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024):
                cap2 = cv2.VideoCapture(buf.value)
                if cap2 is not None and cap2.isOpened():
                    return cap2
        except Exception:
            pass
    return cap


def read_video_info(path: str | Path) -> dict:
    """Résolution, cadence, durée et première image d'une vidéo (sans lever d'exception)."""
    info = {"ok": False, "error": "", "width": 0, "height": 0, "fps": 0.0, "n_frames": 0,
            "duration_s": 0.0, "size_bytes": 0, "frame": None}
    p = Path(path)
    if not p.is_file():
        info["error"] = "Fichier introuvable."
        return info
    try:
        info["size_bytes"] = p.stat().st_size
    except OSError:
        pass
    cap = None
    try:
        cap = _open_capture(str(p))
        if cap is None or not cap.isOpened():
            info["error"] = "Format ou codec non pris en charge : ce fichier ne peut pas être ouvert."
            return info
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, fr = cap.read()
        if not ok or fr is None or fr.size == 0:
            info["error"] = "Le fichier ne contient aucune image lisible."
            return info
        fps_ok = fps if 1.0 <= fps <= 1000.0 else 0.0
        info.update(ok=True, width=int(fr.shape[1]), height=int(fr.shape[0]), fps=fps_ok, n_frames=max(0, n),
                    duration_s=(n / fps_ok) if fps_ok and n > 0 else 0.0, frame=fr)
    except Exception as e:  # pragma: no cover - dépend du décodeur
        info["error"] = f"Lecture impossible : {e}"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return info


def _brighten_dark(img):
    """Éclaircit l'aperçu d'une vidéo très sombre (chambre filmée dans le noir) ; n'affecte pas l'analyse."""
    try:
        hi = float(np.percentile(img[::4, ::4], 99.5))
    except Exception:
        return img, False
    if hi >= 90:
        return img, False
    gain = min(4.0, 115.0 / max(hi, 1.0))
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8), True


def _bgr_to_qimage(img) -> QImage:
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8).copy()
    h, w = img.shape[:2]
    return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888).copy()


# ============================================================================ dessins ====
def _pen(color: QColor, width: float) -> QPen:
    return QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)


def _glow_stroke(p: QPainter, path: QPainterPath, color: str, width: float, glow: float = 1.0):
    """Trace lumineuse : halo diffus + cœur plus clair, comme une traînée de gouttelettes éclairée."""
    c = QColor(color)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for mult, alpha in ((5.0, 14), (3.0, 32), (1.8, 70)):
        g = QColor(c)
        g.setAlpha(int(alpha * glow))
        p.setPen(_pen(g, width * mult))
        p.drawPath(path)
    body = QColor(c)
    body.setAlpha(int(255 * min(1.0, 0.55 + 0.45 * glow)))
    p.setPen(_pen(body, width))
    p.drawPath(path)
    core = QColor(c).lighter(165)
    core.setAlpha(int(230 * glow))
    p.setPen(_pen(core, max(0.6, width * 0.35)))
    p.drawPath(path)


def _line(x0, y0, x1, y1) -> QPainterPath:
    path = QPainterPath(QPointF(x0, y0))
    path.lineTo(x1, y1)
    return path


def _electron_path(rng: random.Random, x: float, y: float, ang: float, n: int, step: float,
                   curl: float = 1.0, bounds: QRectF | None = None) -> QPainterPath:
    """Marche aléatoire de plus en plus courbée : un électron perd son énergie et s'enroule.

    Avec `bounds`, plusieurs tirages (déterministes) sont essayés pour que la trace reste dans le cadre.
    """
    best = None
    for attempt in range(60 if bounds is not None else 1):
        r = random.Random(rng.random() * 1e9 + attempt)
        px, py, a, w = x, y, ang, 0.0
        path = QPainterPath(QPointF(px, py))
        for i in range(n):
            f = i / max(1, n - 1)
            w = 0.82 * w + r.gauss(0.0, 0.12)
            a += 0.5 * w + curl * 0.34 * f ** 3
            s = step * (1.0 - 0.55 * f)
            px += math.cos(a) * s
            py += math.sin(a) * s
            path.lineTo(px, py)
        best = path
        if bounds is None or bounds.contains(path.boundingRect()):
            break
    return best


def _mist(p: QPainter, r: QRectF, rng: random.Random, n: int, alpha_max: int = 38):
    p.setPen(Qt.PenStyle.NoPen)
    for _ in range(n):
        x = r.left() + rng.random() * r.width()
        y = r.top() + rng.random() * r.height()
        rad = rng.choice((0.5, 0.7, 0.9, 1.2))
        p.setBrush(QColor(200, 214, 235, rng.randint(5, alpha_max)))
        p.drawEllipse(QPointF(x, y), rad, rad)


def _alpha_track(p: QPainter, x0, y0, ang, length, width, glow=1.0):
    x1, y1 = x0 + math.cos(ang) * length, y0 + math.sin(ang) * length
    _glow_stroke(p, _line(x0, y0, x1, y1), class_hex("alpha"), width, glow)
    # pic de Bragg : l'ionisation est maximale juste avant l'arrêt
    xb, yb = x0 + math.cos(ang) * length * 0.7, y0 + math.sin(ang) * length * 0.7
    c = QColor(class_hex("alpha")).lighter(120)
    c.setAlpha(int(200 * glow))
    p.setPen(_pen(c, width * 1.35))
    p.drawPath(_line(xb, yb, x1, y1))


def paint_signature(p: QPainter, r: QRectF, cls: str, seed: int = 3):
    """Petit dessin de la signature typique d'une classe (cartes « Comment… »)."""
    rng = random.Random(seed)
    _mist(p, r, rng, int(r.width() * r.height() / 230))
    L, T, W, H = r.left(), r.top(), r.width(), r.height()
    if cls == "alpha":  # courte quelle que soit la largeur de la carte
        n = min(0.40 * W, 1.05 * H)
        a = -0.45
        _alpha_track(p, L + 0.42 * W - math.cos(a) * n / 2, T + 0.52 * H - math.sin(a) * n / 2, a, n, 4.6)
        _alpha_track(p, L + 0.80 * W, T + 0.34 * H, 2.45, 0.55 * n, 4.0, glow=0.5)
    elif cls == "beta":
        inner = r.adjusted(6, 6, -6, -6)
        _glow_stroke(p, _electron_path(rng, L + 0.10 * W, T + 0.60 * H, -0.2, 80, min(W, 4.2 * H) / 92, 1.0,
                                       inner), class_hex("beta"), 1.5)
    elif cls == "muon":
        _glow_stroke(p, _line(L - 0.05 * W, T + 0.84 * H, L + 1.05 * W, T + 0.18 * H), class_hex("muon"), 1.5)
    elif cls == "proton":
        _glow_stroke(p, _line(L + 0.08 * W, T + 0.74 * H, L + 0.88 * W, T + 0.28 * H), class_hex("proton"), 3.4)


def paint_chamber(p: QPainter, r: QRectF, seed: int = 11):
    """Vue de la chambre (image sombre de la caméra) : brouillard, muon, alphas d'une source, électron."""
    rng = random.Random(seed)
    L, T, W, H = r.left(), r.top(), r.width(), r.height()
    _mist(p, r, rng, int(W * H / 140), 44)
    _glow_stroke(p, _line(L + 0.30 * W, T - 0.1 * H, L + 0.62 * W, T + 1.1 * H), class_hex("muon"), 1.4)
    sx, sy = L + 0.80 * W, T + 0.62 * H
    for deg, frac in ((200, 0.42), (248, 0.36), (322, 0.38), (20, 0.30)):
        a = math.radians(deg)
        _alpha_track(p, sx + math.cos(a) * 6, sy + math.sin(a) * 6, a, frac * H, 3.6)
    halo = QRadialGradient(QPointF(sx, sy), 7)
    halo.setColorAt(0.0, QColor(255, 240, 225, 235))
    halo.setColorAt(1.0, QColor(255, 120, 110, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(halo)
    p.drawEllipse(QPointF(sx, sy), 7, 7)
    inner = r.adjusted(8, 8, -8, -8)
    _glow_stroke(p, _electron_path(rng, L + 0.06 * W, T + 0.40 * H, 0.15, 90, min(W, 3.4 * H) / 70, 1.0, inner),
                 class_hex("beta"), 1.3)


class NightStrip(QWidget):
    """Frise « une nuit = une suite de vidéos » dessinée avec les pilules du symbole du Cégep."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Illustration : une nuit découpée en vidéos d'une heure")

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        bg = QPainterPath()
        bg.addRoundedRect(r, 14, 14)
        p.fillPath(bg, QColor(B.SOFT))
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(B.MUTED))
        inner = r.adjusted(16, 10, -16, -10)
        p.drawText(QRectF(inner.left(), inner.bottom() - 16, 80, 16), int(Qt.AlignmentFlag.AlignLeft), "21 h")
        p.drawText(QRectF(inner.right() - 80, inner.bottom() - 16, 80, 16), int(Qt.AlignmentFlag.AlignRight), "7 h")
        p.drawText(QRectF(inner.left(), inner.bottom() - 16, inner.width(), 16), int(Qt.AlignmentFlag.AlignHCenter),
                   "10 vidéos d'une heure")
        n = 10
        y = inner.top() + max(0.0, (inner.height() - 24 - 12) / 2)
        gap = 5.0
        sw = (inner.width() - (n - 1) * gap) / n
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            c = QColor(B.FOREST) if i < n - 1 else QColor(B.BRIGHT)
            c.setAlphaF(0.35 + 0.65 * (i + 1) / n)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(inner.left() + i * (sw + gap), y, sw, 12), 6, 6)
        p.end()


class ChamberPreview(QWidget):
    """Petite fenêtre sombre « ce que voit la caméra » (la vidéo reste sombre : c'est l'image réelle)."""

    def __init__(self, seed: int = 11, parent=None):
        super().__init__(parent)
        self.seed = seed
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cache: QPixmap | None = None
        self.setAccessibleName("Illustration : traces de particules vues par la caméra")

    def resizeEvent(self, e):
        self._cache = None
        super().resizeEvent(e)

    def paintEvent(self, _e):
        dpr = self.devicePixelRatioF()
        if self._cache is None or self._cache.size() != self.size() * dpr:
            pm = QPixmap(self.size() * dpr)
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = QRectF(0, 0, self.width(), self.height())
            clip = QPainterPath()
            clip.addRoundedRect(r, 14, 14)
            q.fillPath(clip, QColor(B.VIDEO_BG))
            q.setClipPath(clip)
            paint_chamber(q, r, self.seed)
            q.setClipping(False)
            f = QFont(self.font())
            f.setPixelSize(11)
            f.setBold(True)
            q.setFont(f)
            label = "EN DIRECT"
            tw = q.fontMetrics().horizontalAdvance(label) + 26
            badge = QRectF(12, 12, tw, 22)
            q.setPen(Qt.PenStyle.NoPen)
            q.setBrush(QColor(0, 0, 0, 120))
            q.drawRoundedRect(badge, 11, 11)
            q.setBrush(QColor("#e5484d"))
            q.drawEllipse(QPointF(badge.left() + 11, badge.center().y()), 3.5, 3.5)
            q.setPen(QColor("#ffffff"))
            q.drawText(badge.adjusted(18, 0, 0, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                       label)
            q.end()
            self._cache = pm
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)


# ============================================================================ widgets ====
class Clickable(QFrame):
    """Cadre cliquable et accessible au clavier (Tab puis Entrée / Espace)."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._pressed = False

    def click(self):
        """Comme QAbstractButton.click() : émet `clicked` si le cadre est actif."""
        if self.isEnabled():
            self.clicked.emit()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        was = self._pressed
        self._pressed = False
        if was and e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            e.accept()
            self.clicked.emit()
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            e.accept()
            self.clicked.emit()
            return
        super().keyPressEvent(e)


class ElidedLabel(QLabel):
    """Étiquette d'une ligne qui raccourcit le texte (…) au lieu d'élargir la mise en page."""

    def __init__(self, text: str = "", parent=None, mode=Qt.TextElideMode.ElideMiddle):
        super().__init__(text, parent)
        self._full = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text: str):  # noqa: N802 (API Qt)
        self._full = text
        super().setText(text)
        self.update()

    def fullText(self) -> str:  # noqa: N802
        return self._full

    def minimumSizeHint(self):
        return QSize(24, super().minimumSizeHint().height())

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setFont(self.font())
        p.setPen(self.palette().color(self.foregroundRole()))
        cr = self.contentsRect()
        txt = self.fontMetrics().elidedText(self._full, self._mode, cr.width())
        p.drawText(cr, int(self.alignment() | Qt.AlignmentFlag.AlignVCenter), txt)


class NoWheelComboBox(QComboBox):
    """Liste déroulante qui laisse la molette faire défiler la page tant qu'elle n'a pas le focus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class FlowLayout(QLayout):
    """Disposition « en ligne » qui passe à la ligne suivante quand la place manque (pastilles)."""

    def __init__(self, parent=None, hspacing: int = 8, vspacing: int = 8):
        super().__init__(parent)
        self._items = []
        self._h, self._v = hspacing, vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802 (API Qt)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._layout(QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, dry=False)

    def sizeHint(self):  # noqa: N802
        w = sum(it.sizeHint().width() for it in self._items) + self._h * max(0, len(self._items) - 1)
        h = max((it.sizeHint().height() for it in self._items), default=0)
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):  # noqa: N802
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect: QRect, dry: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for it in self._items:
            if it.isEmpty():
                continue
            hint = it.sizeHint()
            if x > r.x() and x + hint.width() > r.right() + 1:
                x = r.x()
                y += line_h + self._v
                line_h = 0
            if not dry:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def _chips(texts, object_name: str, hspacing: int = 8) -> QWidget:
    box = QWidget()
    box.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    flow = FlowLayout(box, hspacing, 6)
    for t in texts:
        c = QLabel(t)
        c.setObjectName(object_name)
        flow.addWidget(c)
    sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    sp.setHeightForWidth(True)
    box.setSizePolicy(sp)
    return box


def _discard(w: QWidget | None):
    """Retire immédiatement un widget de l'écran (deleteLater seul le laisse affiché jusqu'à la boucle)."""
    if w is not None:
        w.hide()
        w.setParent(None)
        w.deleteLater()


def _tinted_icon(emoji: str, tint: str, size: int = 48, radius: int = 13, font_px: int = 22) -> QLabel:
    """Pastille d'icône. Un nom d'icône de brand.ICONS donne une icône vectorielle vert forêt sur menthe
    pâle (style du Cégep) ; un émoji est affiché sur un fond teinté très clair."""
    lab = QLabel()
    lab.setObjectName("tintIcon")
    lab.setFixedSize(size, size)
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if emoji in B.ICONS:
        lab.setPixmap(B.icon_pixmap(emoji, B.FOREST, max(14, int(size * 0.48)), 2.0))
        lab.setStyleSheet(f"QLabel#tintIcon {{ background: {B.MINT_SOFT}; border: none; border-radius: {radius}px; }}")
        return lab
    lab.setText(emoji)
    c = QColor(tint)
    lab.setStyleSheet(f"QLabel#tintIcon {{ background: rgba({c.red()},{c.green()},{c.blue()},30);"
                      f" border: 1px solid rgba({c.red()},{c.green()},{c.blue()},60); border-radius: {radius}px;"
                      f" font-size: {font_px}px; }}")
    return lab


def _icon_label(name: str, color: str, size: int = 16) -> QLabel:
    lab = QLabel()
    lab.setPixmap(B.icon_pixmap(name, color, size, 2.0))
    lab.setFixedSize(size, size)
    lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    return lab


class ActionCard(Clickable):
    """Carte d'action blanche : pastille d'icône, titre, description et commandes propres à la carte."""

    def __init__(self, icon: str, title: str, text: str, tint: str = ACCENT, wide: bool = False,
                 compact_body: bool = False, badge: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("ActionCard")
        self.setAccessibleName(title)
        self.setAccessibleDescription(text)
        self.icon = _tinted_icon(icon, tint, 48, 15)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.title.setWordWrap(True)
        self.text = QLabel(text)
        self.text.setObjectName("cardText")
        self.text.setWordWrap(True)
        tcol = QVBoxLayout()
        tcol.setSpacing(4)
        self.badge = None
        if badge:
            self.badge = QLabel(badge)
            self.badge.setObjectName("cardBadge")
            tcol.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignLeft)
        tcol.addWidget(self.title)
        tcol.addWidget(self.text)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 24, 26, 24)
        lay.setSpacing(18)
        head = QHBoxLayout()
        head.setSpacing(16)
        head.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        head.addLayout(tcol, 1)
        lay.addLayout(head)
        self.top = QVBoxLayout()                    # contenu facultatif juste sous l'en-tête
        self.top.setSpacing(9)
        lay.addLayout(self.top)
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        if compact_body:
            lay.addLayout(self.body)
            lay.addStretch(1)
        else:
            lay.addStretch(1)
            lay.addLayout(self.body)
        self.tcol = tcol


class ToolChip(Clickable):
    """Petit bouton-pastille de la rangée « Plus d'outils » (icône + texte)."""

    def __init__(self, icon: str, text: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("ToolChip")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(13, 8, 15, 8)
        lay.setSpacing(8)
        lay.addWidget(_icon_label(icon, B.GREEN, 16))
        self.label = QLabel(text)
        self.label.setObjectName("chipText")
        lay.addWidget(self.label)
        self.extra = lay
        if tooltip:
            self.setToolTip(tooltip)
        self.setAccessibleName(text)
        self.setFixedHeight(38)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class ListRow(Clickable):
    """Ligne cliquable compacte (vidéo récente / vidéo d'exemple)."""

    _ICON = {"🎞": "video", "✨": "spark"}

    def __init__(self, emoji: str, text: str, tag: str = "", tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("ListRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(10)
        name = self._ICON.get(emoji, emoji if emoji in B.ICONS else "")
        if name:
            ic = _icon_label(name, B.GREEN, 16)
        else:
            ic = QLabel(emoji)
            ic.setFixedWidth(20)
        ic.setObjectName("rowIcon")
        self.label = ElidedLabel(text)
        self.label.setObjectName("rowText")
        lay.addWidget(ic)
        lay.addWidget(self.label, 1)
        if tag:
            t = QLabel(tag)
            t.setObjectName("rowTag")
            lay.addWidget(t)
        lay.addWidget(_icon_label("chevron", B.FAINT, 14))
        if tooltip:
            self.setToolTip(tooltip)
        self.setAccessibleName(text)


class RunRow(Clickable):
    """Ligne « Derniers résultats »."""
    folder_requested = Signal(str)

    def __init__(self, run: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("RunRow")
        self.run = run
        self.folder = str(run.get("folder", "") or "")
        cond = str(run.get("condition", "") or "")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 6, 8)
        lay.setSpacing(12)
        lay.addWidget(_tinted_icon(_condition_icon(cond), _condition_tint(cond), 36, 11, 17))
        col = QVBoxLayout()
        col.setSpacing(1)
        name = ElidedLabel(str(run.get("name", "") or "Sans nom"), mode=Qt.TextElideMode.ElideRight)
        name.setObjectName("runName")
        dur = run.get("duration_min")
        bits = [cond or "Condition inconnue"]
        if isinstance(dur, (int, float)) and dur > 0:
            bits.append(f"{fmt_num(dur, 1)} min")
        sub = ElidedLabel(" · ".join(bits), mode=Qt.TextElideMode.ElideRight)
        sub.setObjectName("runSub")
        col.addWidget(name)
        col.addWidget(sub)
        lay.addLayout(col, 1)
        counts = run.get("counts", {}) or {}
        chips = []
        for cls, sym in (("alpha", "α"), ("beta", "β"), ("muon", "μ"), ("proton", "p")):
            n = int(counts.get(cls, 0) or 0)
            chips.append(f"<span style='color:{B.class_text_hex(cls)}; font-weight:700'>{sym}</span>"
                         f"<span style='color:{TEXT}'>{NBSP}{n}</span>")
        n_unc = int(counts.get("incertain", 0) or 0)
        if n_unc:
            chips.append(f"<span style='color:{B.class_text_hex('incertain')}; font-weight:700'>?</span>"
                         f"<span style='color:{TEXT}'>{NBSP}{n_unc}</span>")
        cl = QLabel(("&nbsp;&nbsp;").join(chips))
        cl.setObjectName("runCounts")
        cl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(cl)
        total = int(run.get("total", sum(int(v or 0) for v in counts.values())) or 0)
        tl = QLabel(f"<span style='font-size:15px; font-weight:700; color:{TEXT}'>{fmt_num(total, 0)}</span>"
                    f"<br><span style='color:{MUTED}; font-size:11px'>trace{'s' if total > 1 else ''}</span>")
        tl.setObjectName("runTotal")
        tl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tl.setMinimumWidth(50)
        lay.addWidget(tl)
        fb = QToolButton()
        fb.setObjectName("runFolder")
        fb.setIcon(B.line_icon("folder", B.GREEN, 18))
        fb.setIconSize(QSize(18, 18))
        fb.setToolTip("Ouvrir le dossier dans l'Explorateur")
        fb.setAccessibleName("Ouvrir le dossier")
        fb.setCursor(Qt.CursorShape.PointingHandCursor)
        fb.setEnabled(bool(self.folder))
        fb.clicked.connect(lambda: self.folder_requested.emit(self.folder))
        lay.addWidget(fb)
        self.setToolTip(self.folder)
        self.setAccessibleName(str(run.get("name", "")))


def _condition_icon(cond: str) -> str:
    c = cond.lower()
    if "fond" in c or "cosm" in c:
        return "spark"
    if "kcl" in c or "sel" in c or "banane" in c or "source" in c:
        return "flask"
    return "bars"


def _condition_tint(cond: str) -> str:
    c = cond.lower()
    if "fond" in c or "cosm" in c:
        return B.FOREST
    if "kcl" in c or "sel" in c:
        return B.BRIGHT
    if "banane" in c:
        return "#e8b10d"
    return "#8b8d98"


# ----------------------------------------------------------------------------- bandeau --
class HeroBanner(QFrame):
    """Grand bloc vert forêt (style cegepmv.ca) : texte à gauche, grille de « pilules » menthe à droite.

    Animation d'accueil (play_intro) : les pilules montent dans leur case l'une après l'autre (révélation par
    découpe, courbe du site cubic-bezier(0.215, 0.61, 0.355, 1)) et le titre apparaît en glissant. Désactivée
    par WILSONTRACK_NO_ANIM=1 (mouvement réduit, tests).
    """

    PILL_MS = 950          # durée de la montée d'une pilule
    COL_STAGGER = 42       # décalage entre deux colonnes (ms)
    ROW_STAGGER = 120      # décalage entre deux rangées (ms)
    TEXT_DELAY, TEXT_MS = 180, 900

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hero")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._t_ms = 1e9              # temps écoulé de l'animation (très grand = terminée)
        self._total_ms = 1500
        self._anim = None
        self._effect = None
        lay = QHBoxLayout(self)
        self._lay = lay
        self._text = QWidget()
        self._text.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._text.setMaximumWidth(600)
        self._text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        col = QVBoxLayout(self._text)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)
        eyebrow = QLabel("CÉGEP MARIE-VICTORIN  ·  SCIENCES DE LA NATURE")
        eyebrow.setObjectName("heroEyebrow")
        fe = QFont(eyebrow.font())
        fe.setPixelSize(12)
        fe.setBold(True)
        fe.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.4)
        eyebrow.setFont(fe)
        title = QLabel(APP_NAME)
        title.setObjectName("heroTitle")
        tag = QLabel("Voir l'invisible : les particules de la radioactivité.")
        tag.setObjectName("heroTagline")
        tag.setWordWrap(True)
        desc = QLabel("Filmez la chambre de Wilson : WilsonTrack repère chaque trace, reconnaît les particules "
                      "alpha, bêta, muons et protons, explique pourquoi, puis calcule des résultats avec leurs "
                      "incertitudes.")
        desc.setObjectName("heroDesc")
        desc.setWordWrap(True)
        meta = QHBoxLayout()
        meta.setSpacing(10)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("heroVersion")
        info = QLabel("Détection en temps réel  ·  Physique + IA  ·  Fonctionne hors ligne")
        info.setObjectName("heroMeta")
        meta.addWidget(ver)
        meta.addWidget(info)
        meta.addStretch(1)
        col.addWidget(eyebrow)
        col.addSpacing(2)
        col.addWidget(title)
        col.addWidget(tag)
        col.addSpacing(4)
        col.addWidget(desc)
        col.addSpacing(8)
        col.addLayout(meta)
        lay.addWidget(self._text, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        self.title_label, self.tagline_label, self.version_label = title, tag, ver
        self.eyebrow_label, self.desc_label, self.meta_label = eyebrow, desc, info
        self._compact = None
        self.set_compact(False)

    # ------------------------------------------------------------------ mise en page --
    def set_compact(self, on: bool):
        """Bandeau réduit (fenêtre peu haute, p. ex. portable 1366×768) : on garde la place pour les cartes."""
        on = bool(on)
        if on == self._compact:
            return
        self._compact = on
        self._margins = (40, 24, 24, 24) if on else (48, 44, 28, 44)
        self._apply_margins(0.0)
        self.eyebrow_label.setVisible(not on)
        self.desc_label.setVisible(not on)
        self.setMinimumHeight(196 if on else 330)
        f = QFont(self.title_label.font())
        f.setPixelSize(44 if on else 60)
        f.setWeight(QFont.Weight.Bold)
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 97.0)
        self.title_label.setFont(f)
        self.setProperty("compact", on)
        self.style().unpolish(self)
        self.style().polish(self)
        for w in self.findChildren(QLabel):
            w.style().unpolish(w)
            w.style().polish(w)
        self.update()

    @property
    def compact(self) -> bool:
        return bool(self._compact)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._text.setMinimumWidth(max(260, min(580, int(self.width() * 0.46))))

    def _apply_margins(self, slide: float):
        l, t, r, b = self._margins
        d = int(round(slide))
        self._lay.setContentsMargins(l, t + d, r, max(0, b - d))

    # ------------------------------------------------------------------ animation --
    def play_intro(self):
        """Lance l'animation d'accueil (≈ 1,5 s). Sans effet si le mouvement réduit est demandé."""
        if B.reduced_motion():
            self.finish_intro()
            return
        from PySide6.QtCore import QVariantAnimation
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._stop_anim()
        self._effect = QGraphicsOpacityEffect(self._text)
        self._effect.setOpacity(0.0)
        self._text.setGraphicsEffect(self._effect)
        self._t_ms = 0.0
        a = QVariantAnimation(self)
        a.setStartValue(0.0)
        a.setEndValue(float(self._total_ms))
        a.setDuration(int(self._total_ms))
        a.valueChanged.connect(self._on_tick)
        a.finished.connect(self.finish_intro)
        self._anim = a
        self._on_tick(0.0)
        a.start()

    def set_intro_time(self, ms: float):
        """Fige l'animation à l'instant `ms` (aperçus)."""
        self._stop_anim()
        if self._effect is None:
            from PySide6.QtWidgets import QGraphicsOpacityEffect
            self._effect = QGraphicsOpacityEffect(self._text)
            self._text.setGraphicsEffect(self._effect)
        self._on_tick(float(ms))

    def _stop_anim(self):
        if self._anim is not None:
            try:
                self._anim.stop()
                self._anim.deleteLater()
            except RuntimeError:
                pass
            self._anim = None

    def _on_tick(self, v):
        self._t_ms = float(v)
        k = B.ease((self._t_ms - self.TEXT_DELAY) / self.TEXT_MS)
        if self._effect is not None:
            self._effect.setOpacity(k)
        self._apply_margins(18.0 * (1.0 - k))
        self.update()

    def finish_intro(self):
        self._stop_anim()
        self._t_ms = 1e9
        if self._effect is not None:
            self._text.setGraphicsEffect(None)      # détruit l'effet : rendu direct, plus net
            self._effect = None
        self._apply_margins(0.0)
        self.update()

    @property
    def intro_running(self) -> bool:
        return self._anim is not None

    # ------------------------------------------------------------------ dessin --
    def _pill_grid(self):
        """Positions des pilules : [(kind, QRectF, delay_ms, alpha)]."""
        W, H = float(self.width()), float(self.height())
        text_right = self._text.geometry().right() if self._text.isVisible() else 0
        left = max(text_right + 48.0, W * 0.47)
        top = 34.0 if not self.compact else 24.0
        rows = 3 if not self.compact else 2
        # la dernière rangée sort à moitié par le bas du bandeau (effet de « montée »)
        ph = (H - top) / ((rows - 0.45) + (rows - 1) * 0.20)
        pw = ph * (10.62 / 24.0)
        gap = pw * 0.173
        vgap = ph * 0.20
        right = W - 40.0
        n = int((right - left + gap) // (pw + gap))
        if n <= 0:
            return []
        x0 = right - n * (pw + gap) + gap
        out = []
        for r in range(rows):
            y = top + r * (ph + vgap)
            if y > H:
                break
            for c in range(n):
                x = x0 + c * (pw + gap)
                delay = c * self.COL_STAGGER + r * self.ROW_STAGGER
                alpha = (0.28, 0.6)[c] if c < 2 and n >= 6 else 1.0
                out.append(((c - n) % 4, QRectF(x, y, pw, ph), delay, alpha))
        if out:
            self._total_ms = max(1200, int(max(d for _k, _r, d, _a in out) + self.PILL_MS))
        return out

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(r, 24, 24)
        p.fillPath(clip, QColor(B.FOREST))
        p.setClipPath(clip)
        mint = QColor(B.MINT)
        for kind, cell, delay, alpha in self._pill_grid():
            k = B.ease((self._t_ms - delay) / self.PILL_MS)
            if k <= 0.0:
                continue
            p.save()
            p.setClipRect(cell.adjusted(-1, -1, 1, 1), Qt.ClipOperation.IntersectClip)
            p.translate(0, (1.0 - k) * cell.height())
            c = QColor(mint)
            c.setAlphaF(alpha)
            p.fillPath(B.pill_shape(kind, cell), c)
            p.restore()
        p.end()


class _DropOverlay(QWidget):
    """Voile affiché pendant le glisser-déposer d'une vidéo."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.name = ""
        self.title = "Déposez la vidéo pour l'analyser"
        self.hide()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(241, 245, 241, 236))
        r = QRectF(self.rect()).adjusted(24, 24, -24, -24)
        p.setPen(QPen(QColor(B.GREEN), 2, Qt.PenStyle.DashLine))
        p.setBrush(QColor(B.MINT_SOFT))
        p.drawRoundedRect(r, 24, 24)
        p.drawPixmap(QPointF(r.center().x() - 28, r.center().y() - 110), B.icon_pixmap("drop", B.FOREST, 56, 2.0))
        f = QFont(self.font())
        f.setPixelSize(24)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(B.FOREST))
        p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), self.title)
        if self.name:
            f.setPixelSize(13)
            f.setBold(False)
            p.setFont(f)
            p.setPen(QColor(MUTED))
            p.drawText(r.adjusted(40, 54, -40, 54), int(Qt.AlignmentFlag.AlignCenter),
                       self.fontMetrics().elidedText(self.name, Qt.TextElideMode.ElideMiddle, int(r.width() - 80)))


PAGE_QSS = f"""
QWidget#homeContent {{ background: {B.SOFT}; }}
QScrollArea#homeScroll {{ background: {B.SOFT}; border: none; }}
QFrame#hero {{ background: transparent; border: none; }}
QLabel#heroEyebrow {{ color: {B.MINT}; font-size: 12px; font-weight: 700; }}
QLabel#heroTitle {{ color: #ffffff; font-size: 62px; font-weight: 700; }}
QFrame#hero[compact="true"] QLabel#heroTitle {{ font-size: 44px; }}
QLabel#heroTagline {{ color: {B.MINT}; font-size: 21px; font-weight: 600; }}
QFrame#hero[compact="true"] QLabel#heroTagline {{ font-size: 17px; }}
QLabel#heroDesc {{ color: rgba(255,255,255,225); font-size: 14px; }}
QLabel#heroVersion {{ color: {B.MINT}; background: rgba(152,242,157,22); border: 1px solid rgba(152,242,157,130);
    border-radius: 11px; padding: 2px 10px; font-size: 12px; font-weight: 700; }}
QLabel#heroMeta {{ color: rgba(255,255,255,205); font-size: 12px; }}
QLabel#sectionTitle {{ color: {B.INK}; font-size: 22px; font-weight: 700; }}
QLabel#sectionSub {{ color: {B.MUTED}; font-size: 13px; }}
QFrame#ActionCard {{ background: #ffffff; border: 1px solid {B.BORDER}; border-radius: 20px; }}
QFrame#ActionCard:hover {{ border: 1px solid {B.GREEN}; }}
QFrame#ActionCard:focus {{ border: 2px solid {B.BRIGHT}; }}
QLabel#cardTitle {{ color: {B.INK}; font-size: 18px; font-weight: 700; }}
QLabel#cardText {{ color: {B.MUTED}; font-size: 13px; }}
QLabel#cardBadge {{ color: {B.FOREST}; background: {B.MINT}; border-radius: 8px; padding: 1px 8px;
    font-size: 10px; font-weight: 700; }}
QLabel#cardHint {{ color: {B.FAINT}; font-size: 12px; }}
QLabel#miniHeader {{ color: {B.GREEN}; font-size: 11px; font-weight: 700; }}
QLabel#featureItem {{ color: {B.INK}; }}
QFrame#ListRow {{ background: {B.SOFT}; border: 1px solid transparent; border-radius: 11px; }}
QFrame#ListRow:hover {{ background: {B.HOVER}; border: 1px solid {B.BORDER}; }}
QFrame#ListRow:focus {{ border: 2px solid {B.BRIGHT}; }}
QLabel#rowText {{ color: {B.INK}; }}
QLabel#rowTag {{ color: {B.FOREST}; background: {B.MINT_SOFT}; border-radius: 9px; padding: 2px 8px;
    font-size: 11px; font-weight: 600; }}
QFrame#runsPanel {{ background: transparent; border: none; }}
QFrame#RunRow {{ background: {B.SOFT}; border: 1px solid transparent; border-radius: 12px; }}
QFrame#RunRow:hover {{ background: {B.HOVER}; border: 1px solid {B.BORDER}; }}
QFrame#RunRow:focus {{ border: 2px solid {B.BRIGHT}; }}
QLabel#runName {{ color: {B.INK}; font-weight: 600; }}
QLabel#runSub {{ color: {B.MUTED}; font-size: 12px; }}
QToolButton#runFolder {{ background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 5px; }}
QToolButton#runFolder:hover {{ background: #ffffff; border: 1px solid {B.BORDER}; }}
QLabel#emptyState {{ color: {B.MUTED}; padding: 14px 16px; background: {B.SOFT}; border-radius: 12px; }}
QLabel#toolsTitle {{ color: {B.INK}; font-size: 15px; font-weight: 700; }}
QFrame#ToolChip {{ background: #ffffff; border: 1px solid {B.BORDER}; border-radius: 18px; }}
QFrame#ToolChip:hover {{ background: {B.HOVER}; border: 1px solid {B.GREEN}; }}
QFrame#ToolChip:focus {{ border: 2px solid {B.BRIGHT}; }}
QLabel#chipText {{ color: {B.FOREST}; font-weight: 600; }}
QLabel#modelBlurb {{ color: {B.MUTED}; font-size: 12px; }}
QFrame#footerLine {{ background: {B.BORDER}; border: none; }}
QLabel#footer {{ color: {B.MUTED}; font-size: 12px; }}
"""


# ============================================================================ page ====
class HomePage(QWidget):
    """Page d'accueil (menu principal) de WilsonTrack."""

    camera_requested = Signal(int)          # index de la caméra choisie sur la page
    camera_refresh_requested = Signal()
    video_requested = Signal(dict)          # {"path", "condition", "sample", "mass_g", "px_per_mm", "realtime"}
    photos_requested = Signal()
    demo_requested = Signal(str)            # "demo" ou "demo_kcl"
    compare_requested = Signal()
    results_requested = Signal()            # ouvrir des résultats enregistrés
    assistant_requested = Signal()
    guide_requested = Signal()
    run_open_requested = Signal(str)        # (facultatif) clic sur un « dernier résultat » → dossier du run
    night_requested = Signal(dict)          # dictionnaire « nuit » (session_dialogs.NIGHT_DEFAULTS)
    batch_requested = Signal(dict)          # dictionnaire « lot » (session_dialogs.BATCH_DEFAULTS)
    learning_requested = Signal()           # ouvrir le Centre d'apprentissage de l'IA

    ESSENTIAL = ("camera", "video", "night", "results")          # cartes blanches de la grille
    TOOLS = ("photos", "demo", "batch", "learning", "assistant")  # rangée discrète « Plus d'outils »

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setObjectName("HomePage")
        self.setStyleSheet(PAGE_QSS)
        self.setAcceptDrops(True)
        self._cams: list[dict] = []
        self._runs: list[dict] = []
        self._cols = 0
        self._intro_played = False
        self._pending_drop: str | list[str] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("homeScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        content = QWidget()
        content.setObjectName("homeContent")
        self.content = content
        self.scroll.setWidget(content)
        hl = QHBoxLayout(content)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addStretch(1)
        center = QWidget()
        center.setObjectName("homeCenter")
        center.setMaximumWidth(1280)
        center.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        hl.addWidget(center, 100)
        hl.addStretch(1)
        self._center = center
        self._center.installEventFilter(self)
        v = QVBoxLayout(center)
        v.setContentsMargins(32, 28, 32, 26)
        v.setSpacing(16)
        self._v = v

        self.hero = HeroBanner()
        v.addWidget(self.hero)
        v.addSpacing(14)
        v.addLayout(self._section_header("Que voulez-vous faire ?",
                                         "Filmez en direct ou toute une nuit, analysez une vidéo, puis comparez "
                                         "vos résultats."))
        v.addSpacing(2)
        self._cards_host = QWidget()
        self._cards_grid = QGridLayout(self._cards_host)
        self._cards_grid.setContentsMargins(0, 0, 0, 0)
        self._cards_grid.setHorizontalSpacing(20)
        self._cards_grid.setVerticalSpacing(20)
        v.addWidget(self._cards_host)
        self._build_cards()
        self._place_cards(2)

        v.addSpacing(16)
        v.addWidget(self._build_tools())

        v.addSpacing(18)
        line = QFrame()
        line.setObjectName("footerLine")
        line.setFixedHeight(1)
        v.addWidget(line)
        v.addWidget(self._build_footer())

        self._overlay = _DropOverlay(self)
        self.set_cameras([])
        self.set_recent_runs([])
        self.refresh_videos()

    # ------------------------------------------------------------------ construction --
    def _section_header(self, title: str, subtitle: str = "", right: QWidget | None = None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(3)
        t = QLabel(title)
        t.setObjectName("sectionTitle")
        col.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("sectionSub")
            s.setWordWrap(True)
            col.addWidget(s)
        row.addLayout(col, 1)
        if right is not None:
            row.addWidget(right, 0, Qt.AlignmentFlag.AlignBottom)
        return row

    def _button(self, text: str, slot=None, tip: str = "", **props) -> QPushButton:
        b = QPushButton(text)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        for k, val in props.items():
            b.setProperty(k, val)
        if tip:
            b.setToolTip(tip)
        if slot is not None:
            b.clicked.connect(slot)
        return b

    def _feature(self, text: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(9)
        row.addWidget(_icon_label("check", B.BRIGHT, 16), 0, Qt.AlignmentFlag.AlignTop)
        lab = QLabel(text)
        lab.setObjectName("featureItem")
        lab.setWordWrap(True)
        row.addWidget(lab, 1)
        return row

    def _build_cards(self):
        # 1. Caméra en direct ------------------------------------------------------------
        cam = ActionCard("camera", "Caméra en direct",
                         "Webcam USB ou téléphone (DroidCam, Iriun…) fixé au-dessus de la chambre : "
                         "les traces sont détectées en temps réel.")
        self.cam_combo = NoWheelComboBox()
        self.cam_combo.setToolTip("Caméra à utiliser")
        self.cam_combo.setMinimumWidth(120)
        self.cam_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cam_refresh_btn = self._button("↻", self._on_camera_refresh, "Rechercher les caméras branchées")
        self.cam_refresh_btn.setFixedWidth(44)
        self.cam_refresh_btn.setStyleSheet("QPushButton { font-size: 17px; padding: 3px 0px; }")
        self.cam_refresh_btn.setAccessibleName("Rechercher les caméras")
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.cam_combo, 1)
        row.addWidget(self.cam_refresh_btn)
        self.cam_hint = QLabel("Branchez la caméra puis cliquez ↻ pour la rechercher.")
        self.cam_hint.setObjectName("cardHint")
        self.cam_hint.setWordWrap(True)
        self.cam_start_btn = self._button("Démarrer la caméra", self._emit_camera,
                                          "Ouvrir la caméra et lancer la détection", success=True)
        self.cam_start_btn.setIcon(B.line_icon("camera", "#ffffff", 18))
        self.cam_preview = ChamberPreview()
        cam.top.addWidget(self.cam_preview, 1)
        cam.body.addLayout(row)
        cam.body.addWidget(self.cam_hint)
        cam.body.addWidget(self.cam_start_btn)
        cam.clicked.connect(self._emit_camera)

        # 2. Analyser une vidéo ----------------------------------------------------------
        vid = ActionCard("video", "Analyser une vidéo",
                         "Un fichier MP4, AVI, MOV… filmé pendant l'expérience : indiquez la condition et "
                         "l'échelle, puis lancez l'analyse.", compact_body=True)
        self.video_btn = self._button("Choisir une vidéo…", self.choose_video, "Ouvrir un fichier vidéo (Ctrl+O)",
                                      primary=True)
        self.video_btn.setIcon(B.line_icon("video", "#ffffff", 18))
        drop = QLabel("ou glissez-déposez un fichier vidéo n'importe où sur cette page")
        drop.setObjectName("cardHint")
        drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop.setWordWrap(True)
        self.recent_header = QLabel("VIDÉOS RÉCENTES")
        self.recent_header.setObjectName("miniHeader")
        self._recent_box = QVBoxLayout()
        self._recent_box.setSpacing(4)
        self.examples_header = QLabel("VIDÉOS D'EXEMPLE")
        self.examples_header.setObjectName("miniHeader")
        self._examples_box = QVBoxLayout()
        self._examples_box.setSpacing(4)
        vid.body.addWidget(self.video_btn)
        vid.body.addWidget(drop)
        vid.body.addSpacing(2)
        vid.body.addWidget(self.recent_header)
        vid.body.addLayout(self._recent_box)
        vid.body.addWidget(self.examples_header)
        vid.body.addLayout(self._examples_box)
        vid.clicked.connect(self.choose_video)
        self.recent_rows: list[ListRow] = []
        self.example_rows: list[ListRow] = []

        # 3. Session de nuit -------------------------------------------------------------
        night = ActionCard("moon", "Session de nuit",
                           "Laissez la chambre filmer toute une nuit sans surveillance.")
        for t in ("Vidéo découpée en morceaux faciles à relire",
                  "Résultats sauvegardés automatiquement",
                  "Alerte si la chambre s'épuise (plus de traces)"):
            night.top.addLayout(self._feature(t))
        night.top.addSpacing(4)
        self.night_strip = NightStrip()
        night.top.addWidget(self.night_strip, 1)
        self.night_btn = self._button("Préparer la nuit…", self.open_night_setup,
                                      "Choisir la durée, l'enregistrement vidéo et le dossier, puis lancer")
        self.night_btn.setIcon(B.line_icon("moon", B.FOREST, 16))
        night.body.addWidget(self.night_btn)
        night.clicked.connect(self.open_night_setup)

        # 4. Comparer les résultats -------------------------------------------------------
        res = ActionCard("bars", "Comparer les résultats",
                         "Fond contre échantillon (KCl, banane…) : excès net, incertitudes de Poisson et "
                         "signification statistique.", compact_body=True)
        self.compare_btn = self._button("Comparer", self.compare_requested,
                                        "Comparer un enregistrement « échantillon » à un « fond »", primary=True)
        self.compare_btn.setIcon(B.line_icon("bars", "#ffffff", 16))
        self.results_btn = self._button("Ouvrir des résultats…", self.results_requested,
                                        "Ouvrir des résultats enregistrés")
        self.results_btn.setIcon(B.line_icon("folder", B.FOREST, 16))
        r3 = QHBoxLayout()
        r3.setSpacing(10)
        r3.addWidget(self.compare_btn, 1)
        r3.addWidget(self.results_btn, 1)
        res.body.addLayout(r3)
        hrow = QHBoxLayout()
        hrow.setContentsMargins(0, 6, 0, 0)
        lab = QLabel("DERNIERS RÉSULTATS")
        lab.setObjectName("miniHeader")
        self.results_btn2 = _set_prop(QPushButton("Tout voir  →"), "link")
        self.results_btn2.setCursor(Qt.CursorShape.PointingHandCursor)
        self.results_btn2.setToolTip("Ouvrir des résultats enregistrés")
        self.results_btn2.clicked.connect(self.results_requested)
        hrow.addWidget(lab)
        hrow.addStretch(1)
        hrow.addWidget(self.results_btn2)
        res.body.addLayout(hrow)
        self.runs_panel = QFrame()
        self.runs_panel.setObjectName("runsPanel")
        self._runs_lay = QVBoxLayout(self.runs_panel)
        self._runs_lay.setContentsMargins(0, 0, 0, 0)
        self._runs_lay.setSpacing(4)
        res.body.addWidget(self.runs_panel)
        res.clicked.connect(self.compare_requested)
        self.run_rows: list[RunRow] = []

        # 5…9. Rangée « Plus d'outils » : petits boutons-pastilles -------------------------
        pho = ToolChip("images", "Série de photos", "Analyser un dossier de photos : rafale ou time-lapse (Ctrl+Maj+O)")
        pho.clicked.connect(self.photos_requested)
        demo = ToolChip("flask", "Démonstration",
                        "Chambre de Wilson simulée en temps réel, sans matériel (Ctrl+D)")
        demo.clicked.connect(lambda: self.demo_requested.emit("demo"))
        self.demo_kcl_btn = _set_prop(QPushButton("avec KCl"), "link")
        self.demo_kcl_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.demo_kcl_btn.setToolTip("Démonstration avec une source β simulée (K-40 du chlorure de potassium)")
        self.demo_kcl_btn.clicked.connect(lambda: self.demo_requested.emit("demo_kcl"))
        sep = QLabel("·")
        sep.setObjectName("cardHint")
        demo.extra.addWidget(sep)
        demo.extra.addWidget(self.demo_kcl_btn)
        demo.extra.setContentsMargins(13, 8, 10, 8)
        self.demo_kcl_btn.setStyleSheet("QPushButton { padding: 0px 2px; }")
        batch = ToolChip("inbox", "Vidéos reçues / lot",
                         "Plusieurs vidéos (courriel, clé USB, OneDrive…) analysées l'une après l'autre, ou un "
                         "dossier surveillé")
        batch.clicked.connect(self.open_batch_setup)
        learn = ToolChip("book", "Centre d'apprentissage de l'IA",
                         "Comment l'IA de référence a appris à reconnaître α, β, μ et p, et ce que valent ses résultats")
        learn.clicked.connect(self.learning_requested)
        ai = ToolChip("chat", "Assistant Claude",
                      "Second avis sur une trace, interprétation des résultats, questions (clé API facultative)")
        ai.clicked.connect(self.assistant_requested)
        guide = ToolChip("help", "Guide", "Guide rapide : critères de classification et méthode (F1)")
        guide.clicked.connect(self.guide_requested)

        self.photos_btn, self.demo_btn, self.batch_btn = pho, demo, batch
        self.learning_btn, self.assistant_btn, self.guide_btn = learn, ai, guide
        self.cards = {"camera": cam, "video": vid, "night": night, "results": res, "photos": pho, "demo": demo,
                      "batch": batch, "learning": learn, "assistant": ai}

    def _build_tools(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        t = QLabel("Plus d'outils")
        t.setObjectName("toolsTitle")
        lay.addWidget(t)
        host = QWidget()
        host.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        flow = FlowLayout(host, 10, 10)
        for k in self.TOOLS:
            flow.addWidget(self.cards[k])
        flow.addWidget(self.guide_btn)
        sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        host.setSizePolicy(sp)
        self._tools_host = host
        lay.addWidget(host)
        self.model_blurb = QLabel(reference_model_blurb())
        self.model_blurb.setObjectName("modelBlurb")
        self.model_blurb.setWordWrap(True)
        self.model_blurb.setToolTip("IA de référence installée (voir le Centre d'apprentissage de l'IA)")
        lay.addWidget(self.model_blurb)
        return box

    def _build_footer(self) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(18)
        logo = QLabel()
        logo.setObjectName("footerLogo")
        logo.setPixmap(B.svg_pixmap(B.LOGO_LIGHT_BG, 30, 2.0))
        logo.setAccessibleName("Cégep Marie-Victorin")
        logo.setToolTip("Cégep Marie-Victorin")
        row.addWidget(logo, 0, Qt.AlignmentFlag.AlignVCenter)
        foot = QLabel(f"<b style='color:{B.INK}'>Projet d'intégration en sciences de la nature — Cégep "
                      f"Marie-Victorin</b><br>Équipe : {TEAM} · enseignant : {TEACHER} · {APP_NAME} {__version__}")
        foot.setObjectName("footer")
        foot.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        foot.setWordWrap(True)
        foot.setTextFormat(Qt.TextFormat.RichText)
        row.addWidget(foot, 1)
        self.footer_label = foot
        return box

    # ------------------------------------------------------------------ disposition --
    CARD_ORDER = ("camera", "video", "night", "results", "photos", "demo", "batch", "learning", "assistant")

    def _place_cards(self, cols: int):
        g = self._cards_grid
        for k in self.ESSENTIAL:
            g.removeWidget(self.cards[k])
        for c in range(3):
            g.setColumnStretch(c, 0)
        if cols >= 2:
            pos = {"camera": (0, 0), "video": (0, 1), "night": (1, 0), "results": (1, 1)}
        else:
            pos = {k: (i, 0) for i, k in enumerate(self.ESSENTIAL)}
        for k in self.ESSENTIAL:                     # ordre de lecture = ordre de tabulation
            g.addWidget(self.cards[k], *pos[k])
        for c in range(max(1, min(cols, 2))):
            g.setColumnStretch(c, 1)
        self._cols = cols

    @property
    def columns(self) -> int:
        """Nombre de colonnes actuel de la grille des cartes essentielles."""
        return self._cols

    def _reflow(self):
        if getattr(self, "hero", None) is not None:
            self.hero.set_compact(0 < self.height() < 720)
        m = self._v.contentsMargins()
        w = self._center.width() - m.left() - m.right()
        cols = 2 if w >= 760 else 1
        if cols != self._cols:
            self._place_cards(cols)

    def eventFilter(self, obj, ev):
        if obj is self._center and ev.type() == QEvent.Type.Resize:
            self._reflow()
        return super().eventFilter(obj, ev)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._overlay.setGeometry(self.rect())
        self._reflow()

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh_videos()
        self._reflow()
        if not self._intro_played:
            self._intro_played = True
            self.hero.play_intro()

    # ------------------------------------------------------------------ caméras --
    def set_cameras(self, cams: list[dict]):
        """Remplit la liste des caméras ([{"index", "label", "width", "height"}], éventuellement vide)."""
        self._cams = [c for c in (cams or []) if isinstance(c, dict) and "index" in c]
        combo = self.cam_combo
        combo.blockSignals(True)
        combo.clear()
        for c in self._cams:
            label = c.get("label") or f"Caméra {c['index']}"
            combo.addItem(str(label), int(c["index"]))
        if self._cams:
            want = self.settings.get("cam_index", self._cams[0]["index"])
            j = combo.findData(want)
            combo.setCurrentIndex(j if j >= 0 else 0)
            combo.setEnabled(True)
            self.cam_hint.hide()
            self.cam_start_btn.setToolTip("Ouvrir la caméra choisie et lancer la détection")
        else:
            combo.setPlaceholderText("Aucune caméra détectée")
            combo.setCurrentIndex(-1)
            combo.setEnabled(False)
            self.cam_hint.show()
            self.cam_start_btn.setToolTip("Aucune caméra détectée : la caméra 0 (par défaut) sera essayée")
        combo.blockSignals(False)
        self.cam_refresh_btn.setEnabled(True)
        self.cam_refresh_btn.setText("↻")

    def selected_camera(self) -> int:
        if self.cam_combo.count() and self.cam_combo.currentIndex() >= 0:
            d = self.cam_combo.currentData()
            if d is not None:
                return int(d)
        return 0

    def _on_camera_refresh(self):
        self.cam_refresh_btn.setEnabled(False)
        self.cam_refresh_btn.setText("…")
        QTimer.singleShot(12000, self, self._camera_refresh_timeout)
        self.camera_refresh_requested.emit()

    def _camera_refresh_timeout(self):
        if not self.cam_refresh_btn.isEnabled():
            self.cam_refresh_btn.setEnabled(True)
            self.cam_refresh_btn.setText("↻")

    def _emit_camera(self):
        idx = self.selected_camera()
        self.settings.set("cam_index", idx)
        self.camera_requested.emit(idx)

    # ------------------------------------------------------------------ vidéos --
    def recent_videos(self) -> list[str]:
        """Vidéos récentes existantes (la plus récente d'abord)."""
        items = self.settings.get("recent_videos", []) or []
        out, seen = [], set()
        for x in items:
            if not isinstance(x, str) or not x:
                continue
            k = _norm(x)
            if k in seen or not os.path.isfile(x):
                continue
            seen.add(k)
            out.append(x)
        return out[:MAX_RECENT_VIDEOS]

    def add_recent_video(self, path: str):
        """Ajoute une vidéo en tête des récentes (8 au maximum, sans doublon, fichiers absents ignorés)."""
        p = str(Path(path))
        key = _norm(p)
        items = [x for x in self.recent_videos() if _norm(x) != key]
        if os.path.isfile(p):
            items.insert(0, os.path.abspath(p))
        items = items[:MAX_RECENT_VIDEOS]
        self.settings.set("recent_videos", items)
        self.settings.save()
        self.refresh_videos()

    def refresh_videos(self):
        """Reconstruit les listes « Vidéos récentes » et « Vidéos d'exemple »."""
        for rows, box in ((self.recent_rows, self._recent_box), (self.example_rows, self._examples_box)):
            for r in rows:
                box.removeWidget(r)
                _discard(r)
            rows.clear()
        examples = find_example_videos()
        ex_keys = {_norm(str(e)) for e in examples}
        recent = [r for r in self.recent_videos() if _norm(r) not in ex_keys][:SHOWN_RECENT_VIDEOS]
        for path in recent:
            p = Path(path)
            row = ListRow("🎞", p.name, tooltip=f"{path}\nCliquer pour analyser cette vidéo")
            row.clicked.connect(lambda path=path: self.open_video_setup(path))
            self._recent_box.addWidget(row)
            self.recent_rows.append(row)
        for e in examples:
            known = ground_truth_scale(e) is not None
            row = ListRow("✨", e.stem, "échelle connue" if known else "",
                          tooltip=f"{e}\nVidéo simulée fournie avec l'application"
                                  + (" — échelle et vérité terrain connues" if known else ""))
            row.clicked.connect(lambda path=str(e): self.open_video_setup(path))
            self._examples_box.addWidget(row)
            self.example_rows.append(row)
        self.recent_header.setVisible(bool(recent))
        self.examples_header.setVisible(bool(examples))
        prev = self.video_btn                     # Tab : bouton « Choisir une vidéo… » puis les lignes
        for row in self.recent_rows + self.example_rows:
            QWidget.setTabOrder(prev, row)
            prev = row

    def choose_video(self):
        start = self.settings.get("last_dir", "") or ""
        if not start or not os.path.isdir(start):
            ex = find_example_videos()
            start = str(ex[0].parent) if ex else str(Path.home())
        path, _flt = QFileDialog.getOpenFileName(self, "Choisir une vidéo", start, VIDEO_FILTER)
        if not path:
            return None
        self.settings.set("last_dir", str(Path(path).parent))
        return self.open_video_setup(path)

    def _video_defaults(self, path: str) -> dict:
        d = dict(self.settings.get("video_defaults") or {})
        scales = self.settings.get("video_scales") or {}
        known = float(scales.get(_norm(path), 0) or 0) if isinstance(scales, dict) else 0.0
        if known > 0:  # échelle déjà choisie pour cette vidéo
            d["px_per_mm"], d["scale_source"] = known, "video"
        else:          # sinon : calibration actuelle (vidéos enregistrées avec la même caméra)
            try:
                cal = float(((self.settings.get("config") or {}).get("physics") or {}).get("px_per_mm", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                cal = 0.0
            d["px_per_mm"], d["scale_source"] = cal, ("calibration" if cal > 0 else "")
        g = guess_condition(path)
        if g and g != d.get("condition"):
            d["condition"] = g
            d.pop("sample", None)
            d.pop("mass_g", None)
        return d

    def _run_dialog(self, dlg: QDialog) -> int:
        """Affiche la boîte modale (remplaçable dans les tests)."""
        return dlg.exec()

    def open_video_setup(self, path: str):
        """Boîte « Analyser une vidéo » puis émission de video_requested(dict) si l'utilisateur confirme."""
        path = os.path.normpath(str(path))  # QFileDialog / glisser-déposer renvoient des « / » sous Windows
        dlg = VideoSetupDialog(path, self._video_defaults(path), self)
        try:
            res = self._run_dialog(dlg)
            if int(getattr(res, "value", res)) != int(QDialog.DialogCode.Accepted.value):
                return None
            d = dlg.result_dict()
        finally:
            dlg.deleteLater()
        self.settings.set("video_defaults", {k: d[k] for k in ("condition", "sample", "mass_g", "realtime")})
        scales = dict(self.settings.get("video_scales") or {})
        if d["px_per_mm"] > 0:
            scales[_norm(path)] = d["px_per_mm"]
            while len(scales) > 40:
                scales.pop(next(iter(scales)))
            self.settings.set("video_scales", scales)
        self.add_recent_video(path)
        self.video_requested.emit(d)
        return d

    # ------------------------------------------------------------------ nuit et lot --
    def _calibration_scale(self) -> float:
        try:
            return float(((self.settings.get("config") or {}).get("physics") or {}).get("px_per_mm", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def open_night_setup(self):
        """Boîte « Session de nuit » puis émission de night_requested(dict) si l'utilisateur confirme."""
        from .session_dialogs import NightSessionDialog
        defaults = dict(self.settings.get("night_defaults") or {})
        if self._cams:                                    # caméra choisie sur la carte « Caméra en direct »
            defaults["camera_index"] = self.selected_camera()
        else:
            defaults.setdefault("camera_index", int(self.settings.get("cam_index", 0) or 0))
        if not float(defaults.get("px_per_mm", 0) or 0):
            defaults["px_per_mm"] = self._calibration_scale()
        cam = next((c for c in self._cams if c.get("index") == defaults["camera_index"]), {})
        try:
            wh = (int(cam.get("width") or 1280), int(cam.get("height") or 720))
        except (TypeError, ValueError):
            wh = (1280, 720)
        dlg = NightSessionDialog(defaults, self._cams, wh, 30.0, self)
        try:
            res = self._run_dialog(dlg)
            if int(getattr(res, "value", res)) != int(QDialog.DialogCode.Accepted.value):
                return None
            d = dlg.result_dict()
        finally:
            dlg.deleteLater()
        self.settings.set("night_defaults", d)
        self.settings.set("cam_index", d["camera_index"])
        self.settings.save()
        self.night_requested.emit(d)
        return d

    def open_batch_setup(self, files: list[str] | None = None):
        """Boîte « Vidéos reçues / analyse par lot » (éventuellement pré-remplie) puis batch_requested(dict)."""
        from .session_dialogs import BatchDialog
        defaults = dict(self.settings.get("batch_defaults") or {})
        defaults["files"] = [str(f) for f in (files or []) if isinstance(f, (str, Path))]
        dlg = BatchDialog(defaults, self)
        try:
            res = self._run_dialog(dlg)
            if int(getattr(res, "value", res)) != int(QDialog.DialogCode.Accepted.value):
                return None
            d = dlg.result_dict()
        finally:
            dlg.deleteLater()
        self.settings.set("batch_defaults", {k: v for k, v in d.items() if k != "files"})
        self.settings.save()
        self.batch_requested.emit(d)
        return d

    def set_learning_summary(self, text: str | None = None):
        """Ligne d'information de la carte « Centre d'apprentissage » (None = relire le fichier du modèle)."""
        self.model_blurb.setText(text if text else reference_model_blurb())

    # ------------------------------------------------------------------ glisser-déposer --
    @staticmethod
    def _videos_from_mime(mime) -> list[str]:
        """Vidéos déposées (les dossiers apportent les vidéos qu'ils contiennent, sans sous-dossiers)."""
        if mime is None or not mime.hasUrls():
            return []
        out = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            f = url.toLocalFile()
            if os.path.isdir(f):
                try:
                    out.extend(sorted((str(p) for p in Path(f).iterdir()
                                       if p.is_file() and p.suffix.lower() in VIDEO_EXTS), key=str.lower))
                except OSError:
                    pass
            elif Path(f).suffix.lower() in VIDEO_EXTS and os.path.isfile(f):
                out.append(f)
        return out

    @classmethod
    def _video_from_mime(cls, mime) -> str | None:
        vids = cls._videos_from_mime(mime)
        return vids[0] if vids else None

    @staticmethod
    def _has_folder(mime) -> bool:
        return bool(mime is not None and mime.hasUrls()
                    and any(u.isLocalFile() and os.path.isdir(u.toLocalFile()) for u in mime.urls()))

    def dragEnterEvent(self, e):
        vids = self._videos_from_mime(e.mimeData())
        if vids:
            e.acceptProposedAction()
            batch = len(vids) > 1 or self._has_folder(e.mimeData())
            self._overlay.title = (f"Déposez les {len(vids)} vidéos : analyse par lot" if batch
                                   else "Déposez la vidéo pour l'analyser")
            self._overlay.name = ", ".join(Path(v).name for v in vids[:3]) + (" …" if len(vids) > 3 else "")
            self._overlay.setGeometry(self.rect())
            self._overlay.raise_()
            self._overlay.show()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if self._videos_from_mime(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self._overlay.hide()
        e.accept()

    def dropEvent(self, e):
        self._overlay.hide()
        vids = self._videos_from_mime(e.mimeData())
        if not vids:
            e.ignore()
            return
        e.acceptProposedAction()
        # la boîte modale est ouverte après le dépôt (sinon l'Explorateur reste bloqué)
        batch = len(vids) > 1 or self._has_folder(e.mimeData())
        self._pending_drop = list(vids) if batch else vids[0]
        QTimer.singleShot(0, self, self._open_pending_drop)

    def _open_pending_drop(self):
        f, self._pending_drop = self._pending_drop, None
        if isinstance(f, list):
            self.open_batch_setup(f)
        elif f:
            self.open_video_setup(f)

    # ------------------------------------------------------------------ derniers résultats --
    def set_recent_runs(self, runs: list[dict]):
        """Affiche les 5 derniers enregistrements ([{"name", "condition", "total", "counts", "duration_min",
        "folder"}], le plus récent d'abord)."""
        self._runs = [r for r in (runs or []) if isinstance(r, dict)]
        while self._runs_lay.count():
            it = self._runs_lay.takeAt(0)
            _discard(it.widget())
        self.run_rows = []
        shown = self._runs[:SHOWN_RECENT_RUNS]
        if not shown:
            lab = QLabel("Aucun résultat enregistré pour l'instant. Lancez une analyse : à la fin, les résultats "
                         "sont sauvegardés et apparaissent ici.")
            lab.setObjectName("emptyState")
            lab.setWordWrap(True)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._runs_lay.addWidget(lab)
            return
        for run in shown:
            row = RunRow(run)
            row.clicked.connect(lambda folder=row.folder: self._on_run_clicked(folder))
            row.folder_requested.connect(self._open_folder)
            self._runs_lay.addWidget(row)
            self.run_rows.append(row)

    def _on_run_clicked(self, folder: str):
        if folder and self.isSignalConnected(QMetaMethod.fromSignal(self.run_open_requested)):
            self.run_open_requested.emit(folder)
        elif folder and os.path.isdir(folder):
            self._open_folder(folder)
        else:
            self.results_requested.emit()

    @staticmethod
    def _open_folder(folder: str):
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))


# ============================================================================ boîte vidéo ====
DIALOG_QSS = f"""
QDialog#VideoSetupDialog {{ background: {B.SOFT}; }}
QLabel#dlgTitle {{ color: {B.INK}; font-size: 20px; font-weight: 700; }}
QLabel#dlgFile {{ color: {B.MUTED}; }}
QFrame#dlgPanel {{ background: #ffffff; border: 1px solid {B.BORDER}; border-radius: 16px; }}
QLabel#infoKey {{ color: {B.MUTED}; font-size: 12px; }}
QLabel#infoVal {{ color: {B.INK}; font-weight: 600; }}
QLabel#formLabel {{ color: {B.INK}; font-weight: 600; }}
QLabel#help {{ color: {B.MUTED}; font-size: 12px; }}
QLabel#okNote {{ color: {B.SUCCESS_TEXT}; font-size: 12px; font-weight: 600; }}
QLabel#warnNote {{ color: {B.WARN_TEXT}; font-size: 12px; }}
QLabel#activity {{ color: {B.FOREST}; font-size: 12px; font-weight: 600; }}
QLabel#errorBox {{ background: {B.DANGER_BG}; border: 1px solid {B.DANGER_BORDER}; border-radius: 10px;
    color: {B.DANGER_TEXT}; padding: 9px 12px; }}
QLabel#panelTitle {{ color: {B.GREEN}; font-size: 11px; font-weight: 700; }}
"""


def _fr_locale() -> QLocale:
    loc = QLocale(QLocale.Language.French, QLocale.Country.Canada)
    loc.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator)
    return loc


class VideoSetupDialog(QDialog):
    """Choix des paramètres d'analyse d'une vidéo, avec aperçu et informations du fichier."""

    PREVIEW = QSize(432, 243)

    def __init__(self, path: str, defaults: dict | None = None, parent=None):
        super().__init__(parent)
        self.path = str(path)
        d = dict(defaults or {})
        self.setObjectName("VideoSetupDialog")
        self.setWindowTitle("Analyser une vidéo")
        self.setStyleSheet(DIALOG_QSS)
        self.setModal(True)
        self.info = read_video_info(self.path)
        self.gt_scale = ground_truth_scale(self.path)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # --- en-tête -----------------------------------------------------------
        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(_tinted_icon("video", ACCENT, 46, 14))
        tcol = QVBoxLayout()
        tcol.setSpacing(1)
        t = QLabel("Analyser une vidéo")
        t.setObjectName("dlgTitle")
        self.file_label = ElidedLabel(Path(self.path).name, mode=Qt.TextElideMode.ElideMiddle)
        self.file_label.setObjectName("dlgFile")
        self.file_label.setToolTip(self.path)
        tcol.addWidget(t)
        tcol.addWidget(self.file_label)
        head.addLayout(tcol, 1)
        root.addLayout(head)

        # --- corps : aperçu + informations | paramètres --------------------------
        body = QHBoxLayout()
        body.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(12)
        self.preview = QLabel()
        self.preview.setObjectName("preview")
        self.preview.setFixedSize(self.PREVIEW)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setPixmap(self._make_preview())
        left.addWidget(self.preview)
        left.addWidget(self._info_panel())
        left.addStretch(1)
        body.addLayout(left)
        body.addWidget(self._form_panel(d), 1)
        root.addLayout(body, 1)

        self.error_label = QLabel()
        self.error_label.setObjectName("errorBox")
        self.error_label.setWordWrap(True)
        root.addWidget(self.error_label)

        # --- boutons -----------------------------------------------------------
        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn = QPushButton("▶  Analyser")
        self.ok_btn.setProperty("primary", True)
        self.ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok_btn.setDefault(True)
        self.ok_btn.setMinimumWidth(140)
        self.ok_btn.clicked.connect(self.accept)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.ok_btn)
        root.addLayout(row)

        if self.info["ok"]:
            self.error_label.hide()
        else:
            self.error_label.setText(f"⚠  Impossible de lire cette vidéo : {self.info['error']}<br>"
                                     "Formats conseillés : MP4 (H.264) ou AVI (MJPG). Vous pouvez convertir le "
                                     "fichier avec un logiciel comme VLC ou HandBrake.")
            self.ok_btn.setEnabled(False)
            self.cancel_btn.setDefault(True)
        if self.info["ok"]:
            self.ok_btn.setFocus()
        else:
            self.cancel_btn.setFocus()

    # ------------------------------------------------------------------------
    def _make_preview(self) -> QPixmap:
        size = self.PREVIEW
        dpr = self.devicePixelRatioF()
        pm = QPixmap(size * dpr)
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(0, 0, size.width(), size.height())
        clip = QPainterPath()
        clip.addRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        p.fillPath(clip, QColor(10, 11, 14))
        p.setClipPath(clip)
        fr = self.info.get("frame")
        if fr is not None:
            fr, brightened = _brighten_dark(fr)
            img = _bgr_to_qimage(fr)
            s = min(r.width() / img.width(), r.height() / img.height())
            w, h = img.width() * s, img.height() * s
            p.drawImage(QRectF((r.width() - w) / 2, (r.height() - h) / 2, w, h), img)
            p.setClipping(False)
            badge = "Première image" + (" (éclaircie pour l'aperçu)" if brightened else "")
            f = QFont(self.font())
            f.setPixelSize(11)
            p.setFont(f)
            bw = p.fontMetrics().horizontalAdvance(badge) + 16
            br = QRectF(10, r.height() - 30, bw, 20)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(br, 10, 10)
            p.setPen(QColor(220, 225, 235))
            p.drawText(br, int(Qt.AlignmentFlag.AlignCenter), badge)
        else:
            f = QFont(self.font())
            f.setPixelSize(40)
            p.setFont(f)
            p.setPen(QColor(MUTED))
            p.drawText(r.adjusted(0, -30, 0, -30), int(Qt.AlignmentFlag.AlignCenter), "⚠")
            f.setPixelSize(14)
            p.setFont(f)
            p.drawText(r.adjusted(0, 40, 0, 40), int(Qt.AlignmentFlag.AlignCenter), "Aperçu indisponible")
        p.setClipping(False)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(clip)
        p.end()
        return pm

    def _info_panel(self) -> QFrame:
        box = QFrame()
        box.setObjectName("dlgPanel")
        box.setFixedWidth(self.PREVIEW.width())
        g = QGridLayout(box)
        g.setContentsMargins(16, 12, 16, 12)
        g.setHorizontalSpacing(16)
        g.setVerticalSpacing(6)
        i = self.info
        ok = i["ok"]
        vals = [("Résolution", f"{i['width']}{NBSP}×{NBSP}{i['height']}" if ok else "—"),
                ("Images/s", fmt_num(i["fps"], 2 if i["fps"] and abs(i["fps"] - round(i["fps"])) > 0.005 else 0)
                 if ok and i["fps"] else "—"),
                ("Durée", fmt_duration(i["duration_s"]) if ok else "—"),
                ("Nombre d'images", fmt_num(i["n_frames"], 0) if ok and i["n_frames"] else "—"),
                ("Taille du fichier", fmt_size(i["size_bytes"]) if i["size_bytes"] else "—")]
        self.info_values = {}
        for k, (key, val) in enumerate(vals):
            r, c = divmod(k, 2)
            kl = QLabel(key)
            kl.setObjectName("infoKey")
            vl = QLabel(val)
            vl.setObjectName("infoVal")
            g.addWidget(kl, r * 2, c)
            g.addWidget(vl, r * 2 + 1, c)
            self.info_values[key] = vl
        return box

    def _form_panel(self, d: dict) -> QFrame:
        box = QFrame()
        box.setObjectName("dlgPanel")
        box.setMinimumWidth(340)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(6)
        title = QLabel("PARAMÈTRES DE L'ANALYSE")
        title.setObjectName("panelTitle")
        lay.addWidget(title)
        lay.addSpacing(4)
        loc = _fr_locale()

        def label(text):
            lab = QLabel(text)
            lab.setObjectName("formLabel")
            return lab

        self.condition = QComboBox()
        self.condition.setEditable(True)
        self.condition.addItems(CONDITIONS)
        self.condition.setToolTip("Ce qui est placé dans la chambre (sert au nom des fichiers et aux conclusions)")
        self.sample = QComboBox()
        self.sample.addItems(list(physics.SAMPLES.keys()))
        self.mass = QDoubleSpinBox()
        self.mass.setLocale(loc)
        self.mass.setRange(0, 5000)
        self.mass.setDecimals(1)
        self.mass.setSingleStep(1)
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
        self.scale_note = QLabel("✓  échelle connue (vidéo d'exemple)")
        self.scale_note.setObjectName("okNote")
        self.scale_help = QLabel("Pixels par millimètre. 0 = inconnue : les longueurs sont alors estimées. Pour "
                                 "calibrer, filmez une règle posée sur la plaque, puis « 📏 Calibrer avec une "
                                 "règle » (2 clics) dans les réglages.")
        self.scale_help.setObjectName("help")
        self.scale_help.setWordWrap(True)
        self.realtime = QCheckBox("Analyser à vitesse réelle")
        self.realtime.setToolTip("Décoché : la vidéo est analysée aussi vite que l'ordinateur le permet")
        rt_help = QLabel("Décoché : analyse aussi rapide que possible (recommandé).")
        rt_help.setObjectName("help")
        rt_help.setWordWrap(True)

        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setColumnStretch(1, 1)
        form.addWidget(label("Condition"), 0, 0)
        form.addWidget(self.condition, 0, 1)
        form.addWidget(label("Échantillon"), 1, 0)
        form.addWidget(self.sample, 1, 1)
        form.addWidget(label("Masse"), 2, 0)
        form.addWidget(self.mass, 2, 1)
        form.addWidget(self.activity, 3, 1)
        form.addWidget(label("Échelle"), 4, 0)
        form.addWidget(self.scale, 4, 1)
        form.addWidget(self.scale_note, 5, 1)
        form.addWidget(self.scale_help, 6, 1)
        lay.addLayout(form)
        lay.addSpacing(6)
        lay.addWidget(self.realtime)
        lay.addWidget(rt_help)
        lay.addStretch(1)

        # --- valeurs initiales -------------------------------------------------
        cond = str(d.get("condition") or guess_condition(self.path) or CONDITIONS[0])
        i = self.condition.findText(cond)
        if i >= 0:
            self.condition.setCurrentIndex(i)
        else:
            self.condition.setEditText(cond)
        sample = d.get("sample")
        if sample and self.sample.findText(str(sample)) >= 0:
            self.sample.setCurrentIndex(self.sample.findText(str(sample)))
        try:
            self.mass.setValue(float(d.get("mass_g", 0) or 0))
        except (TypeError, ValueError):
            self.mass.setValue(0)
        suggestion = sample_for_condition(cond)
        if is_background(cond):
            self._set_sample(suggestion)
            self.mass.setValue(0)
        elif suggestion and (not sample or sample == _sample_key("aucun")):
            self._set_sample(suggestion)
        if self.gt_scale:
            self.scale.setValue(self.gt_scale)
        else:
            try:
                self.scale.setValue(float(d.get("px_per_mm", 0) or 0))
            except (TypeError, ValueError):
                self.scale.setValue(0)
            src = d.get("scale_source", "")
            if self.scale.value() > 0 and src == "video":
                self.scale_note.setText("✓  échelle déjà utilisée pour cette vidéo")
            elif self.scale.value() > 0 and src == "calibration":
                self.scale_note.setObjectName("warnNote")
                self.scale_note.setText("Échelle de la calibration actuelle : vérifiez qu'elle correspond à "
                                        "cette vidéo.")
                self.scale_note.setWordWrap(True)
        self._note_value = self.scale.value()
        self.scale_note.setVisible(bool(self.gt_scale) or (self._note_value > 0 and d.get("scale_source") in
                                                            ("video", "calibration")))
        self.realtime.setChecked(bool(d.get("realtime", False)))

        self.condition.currentTextChanged.connect(self._on_condition_changed)
        self.sample.currentIndexChanged.connect(self._update_activity)
        self.mass.valueChanged.connect(self._update_activity)
        self.scale.valueChanged.connect(self._on_scale_changed)
        self._update_activity()
        return box

    # ------------------------------------------------------------------------
    def _set_sample(self, name: str | None):
        if name:
            j = self.sample.findText(name)
            if j >= 0:
                self.sample.setCurrentIndex(j)

    def _on_condition_changed(self, text: str):
        s = sample_for_condition(text)
        if s is None:
            return
        self._set_sample(s)
        if is_background(text):
            self.mass.setValue(0)
        self._update_activity()

    def _on_scale_changed(self, v: float):
        # la note décrit la valeur proposée : elle disparaît si l'utilisateur la modifie
        if self._note_value > 0 and abs(v - self._note_value) > 1e-6:
            self.scale_note.hide()

    def _update_activity(self, *_):
        name = self.sample.currentText()
        frac = float(physics.SAMPLES.get(name, 0.0) or 0.0)
        none = frac <= 0
        self.mass.setEnabled(not none)
        if none:
            self.activity.setText("")
            self.activity.hide()
            return
        m = self.mass.value()
        if m <= 0:
            self.activity.setText("Indiquez la masse pour estimer l'activité attendue du K-40.")
        else:
            try:
                a = physics.sample_activity(m, frac).activity_bq
                txt = fmt_num(a, 0) if a >= 100 else fmt_num(a, 1)
                self.activity.setText(f"Activité attendue du K-40 : ≈ {txt} Bq")
            except Exception:
                self.activity.setText("")
        self.activity.show()

    def result_dict(self) -> dict:
        """Paramètres choisis (mêmes clés que HomePage.video_requested)."""
        return {"path": self.path,
                "condition": self.condition.currentText().strip() or "Sans nom",
                "sample": self.sample.currentText(),
                "mass_g": float(self.mass.value()),
                "px_per_mm": float(self.scale.value()),
                "realtime": bool(self.realtime.isChecked())}
