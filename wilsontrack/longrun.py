"""Sessions de longue durée (« filmer toute la nuit »).

Tout ce qu'il faut pour laisser la chambre filmée plusieurs heures sans surveillance :

- ``NightSessionConfig`` : réglages de la session (dict de la boîte de dialogue ↔ dataclass) ;
- ``estimate_disk_usage`` / ``free_disk_bytes`` : espace disque nécessaire et disponible ;
- ``KeepAwake`` : empêche la mise en veille de Windows (pas l'écran, sauf demande) ;
- ``SessionJournal`` : journal texte ``journal.txt`` (démarrage, santé chaque minute, points de
  sauvegarde, avertissements, reconnexions de la caméra) ;
- ``ChamberDeadDetector`` : avertit si aucune trace valide n'est vue pendant longtemps (alcool
  épuisé, glace sèche fondue…) ;
- ``SessionClock`` : temps écoulé / restant, arrêt automatique ;
- ``SessionController`` : le « chef d'orchestre » appelé à chaque image par la boucle de
  traitement (gui/worker.py ou ``run_session``) : vidéo segmentée, photos périodiques, points de
  sauvegarde (run.json « partiel » + traces.csv, écrits dans un fil séparé), délestage des
  vignettes (mémoire bornée), état toutes les ~2 s, sauvegarde finale + rapport ;
- ``run_session`` : boucle complète sans interface (commande « nuit » de la ligne de commande,
  essai d'endurance tools/night_soak.py, tests).

Dossier d'une session :
    <sortie>/<date> <condition> nuit <HHhMM>/
        run.json, traces.csv (points de sauvegarde puis résultat final), traces.xlsx, rapport.html
        vignettes/   photos/   video/   journal.txt   session.json

Honnêteté scientifique : les points de sauvegarde sont PROVISOIRES (traces encore actives non
comptées, classification non réévaluée sur l'ensemble de la nuit) et marqués « partiel » ; seul le
résultat final (fin de session) fait foi. Aucun import de Qt dans ce module.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import traceback
import unicodedata
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from . import __version__

VIDEO_MODES = ("aucune", "reduite", "complete")
VIDEO_MODE_LABELS = {"aucune": "aucune vidéo", "reduite": "vidéo réduite", "complete": "vidéo complète"}

# --------------------------------------------------------------------------- #
# Hypothèses de l'estimation d'espace disque (mesurées, voir docstring de estimate_disk_usage)
# --------------------------------------------------------------------------- #
VIDEO_BITS_PER_PIXEL = 0.35       # bit par pixel et par image (mesuré 0,17–0,26 ; marge ≈ ×1,7)
THUMB_BYTES = 20_000              # vignette PNG d'une trace (mesuré : ≈ 12 Ko en moyenne, 22 Ko au plus)
ASSUMED_TRACKS_PER_MIN = 60.0     # hypothèse prudente (chambre de démonstration : 10–40 traces/min)
SNAPSHOT_BYTES_PER_PIXEL = 1.5    # PNG d'une image de chambre bruitée (mesuré ≈ 1,41 octet/px)
ROW_BYTES = 700                   # traces.csv ≈ 270 o + traces.xlsx ≈ 230 o par trace (mesuré), marge
FIXED_BYTES = 30 * 1024 ** 2      # rapport HTML, run.json, journal, session.json, marge
UNLIMITED_ASSUMED_H = 12.0        # session sans fin prévue : on suppose 12 h


# =========================================================================== #
# Mise en forme (français : virgule décimale, espace insécable des milliers)
# =========================================================================== #
def fmt_num(v: float, d: int = 1) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(v):
        return "?"
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def fmt_duration(s: float) -> str:
    """« 8 h 05 min », « 12 min 30 s », « 45 s »."""
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(s) or s < 0:
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {sec:02d} s"
    return f"{sec} s"


def fmt_bytes(n: float) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    if n < 0:
        return "inconnu"
    for unit, k in (("Go", 1024 ** 3), ("Mo", 1024 ** 2), ("Ko", 1024)):
        if n >= k:
            return f"{fmt_num(n / k, 1)} {unit}"
    return f"{int(n)} o"


def fmt_min(v: float) -> str:
    """Nombre de minutes : « 30 », « 0,5 »."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "?"
    return fmt_num(v, 0) if abs(v - round(v)) < 1e-9 else fmt_num(v, 1)


def _hms(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).casefold().strip()


def normalize_video_mode(v) -> str:
    f = _fold(v or "")
    if f in ("aucune", "aucun", "non", "none", "no", "0", "false", ""):
        return "aucune" if f else "reduite"
    if f.startswith("compl") or f in ("full", "pleine"):
        return "complete"
    if f.startswith("redu") or f in ("reduced", "light", "legere"):
        return "reduite"
    return "reduite"


_HHMM = re.compile(r"^\s*(\d{1,2})\s*(?:[:hH]\s*(\d{2})?(?:\s*:\s*(\d{2}))?)?\s*$")


def parse_end_time(txt: str) -> tuple[int, int] | None:
    """« 07:00 », « 7:00 », « 7h », « 07h30 », « 23:59:00 » → (heure, minute). Chaîne vide → None.

    Lève ValueError si le format est invalide."""
    if txt is None or not str(txt).strip():
        return None
    m = _HHMM.match(str(txt))
    if not m or (m.group(2) is None and not re.search(r"[hH:]", str(txt))):
        raise ValueError(f"heure de fin invalide « {txt} » (format attendu HH:MM, ex. 07:00)")
    h = int(m.group(1))
    mi = int(m.group(2) or 0)
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        raise ValueError(f"heure de fin invalide « {txt} » (heures 0–23, minutes 0–59)")
    return h, mi


def _coerce(value, default):
    """Convertit ``value`` vers le type de ``default`` ; garde ``default`` si impossible."""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return _fold(value) in ("1", "true", "oui", "vrai", "yes", "on")
            return bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(round(float(str(value).replace(",", "."))))
        if isinstance(default, float):
            v = float(str(value).replace(",", ".").replace(" ", "").replace(" ", ""))
            return v if math.isfinite(v) else default
        if isinstance(default, str):
            return "" if value is None else str(value)
    except (TypeError, ValueError):
        return default
    return value


# =========================================================================== #
# Configuration d'une session
# =========================================================================== #
@dataclass
class NightSessionConfig:
    """Réglages d'une session de longue durée (clés = celles du dict de la boîte de dialogue).

    Les durées « *_min » acceptent des fractions de minute (essais accélérés)."""
    camera_index: int = 0
    duration_h: float = 8.0          # 0 = utiliser end_time
    end_time: str = ""               # « HH:MM » heure locale ; plus tôt que maintenant = demain
    segment_min: float = 30.0        # durée d'un segment vidéo (0 = un seul fichier)
    record_video: str = "reduite"    # « aucune » | « reduite » | « complete »
    record_scale: float = 0.5        # échelle de la vidéo réduite
    record_fps: float = 10.0         # cadence maximale de la vidéo réduite (0 = toutes les images)
    snapshot_every_min: float = 10.0  # photo brute + annotée (0 = jamais)
    autosave_every_min: float = 5.0  # point de sauvegarde
    keep_awake: bool = True          # empêcher la mise en veille (pas l'écran)
    dead_after_min: float = 20.0     # avertir si aucune trace valide pendant ce temps (0 = jamais)
    condition: str = ""
    sample: str = ""
    mass_g: float = 0.0
    px_per_mm: float = 0.0
    output_dir: str = ""             # dossier racine des sessions ("" = dossier de résultats par défaut)
    keep_display_on: bool = False    # garder aussi l'ÉCRAN allumé (seulement si demandé)

    _INT_KEYS = ("camera_index", "segment_min", "snapshot_every_min", "autosave_every_min", "dead_after_min")

    @classmethod
    def from_dict(cls, d: dict | None) -> "NightSessionConfig":
        """Clés inconnues ignorées, clés absentes → valeurs par défaut, valeurs normalisées."""
        out = cls()
        d = dict(d or {})
        for f in fields(cls):
            if f.name in d:
                setattr(out, f.name, _coerce(d[f.name], getattr(out, f.name)))
        out.normalize()
        return out

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in self._INT_KEYS and isinstance(v, float) and v.is_integer():
                v = int(v)
            out[f.name] = v
        return out

    def normalize(self) -> "NightSessionConfig":
        self.camera_index = max(0, int(self.camera_index))
        self.duration_h = min(max(0.0, float(self.duration_h)), 7 * 24.0)
        self.end_time = (self.end_time or "").strip()
        self.segment_min = max(0.0, float(self.segment_min))
        self.record_video = normalize_video_mode(self.record_video)
        self.record_scale = min(1.0, max(0.1, float(self.record_scale or 0.5)))
        self.record_fps = min(240.0, max(0.0, float(self.record_fps)))
        self.snapshot_every_min = max(0.0, float(self.snapshot_every_min))
        a = float(self.autosave_every_min)
        self.autosave_every_min = 5.0 if a <= 0 else max(0.05, a)    # jamais désactivé : c'est la sécurité
        self.dead_after_min = max(0.0, float(self.dead_after_min))
        self.mass_g = max(0.0, float(self.mass_g))
        self.px_per_mm = max(0.0, float(self.px_per_mm))
        self.condition = (self.condition or "").strip()
        self.sample = (self.sample or "").strip()
        self.output_dir = (self.output_dir or "").strip()
        return self

    # ------------------------------------------------------------ horaires --
    def end_datetime(self, start: datetime | None = None) -> datetime | None:
        """Instant de fin prévu (None = jusqu'à l'arrêt manuel). Une heure de fin plus tôt que
        ``start`` (ou égale) désigne le lendemain."""
        start = start or datetime.now()
        if self.duration_h > 0:
            return start + timedelta(hours=self.duration_h)
        try:
            hm = parse_end_time(self.end_time)
        except ValueError:
            return None
        if hm is None:
            return None
        end = start.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
        if end <= start:
            end += timedelta(days=1)
        return end

    def planned_seconds(self, start: datetime | None = None) -> float | None:
        start = start or datetime.now()
        end = self.end_datetime(start)
        return None if end is None else max(0.0, (end - start).total_seconds())

    @property
    def segment_s(self) -> float:
        return self.segment_min * 60.0

    def video_params(self) -> tuple[float, float] | None:
        """(échelle, cadence max) de la vidéo, ou None si pas de vidéo."""
        if self.record_video == "aucune":
            return None
        if self.record_video == "complete":
            return 1.0, 0.0
        return self.record_scale, self.record_fps

    def validate(self) -> list[str]:
        """Problèmes bloquants (messages en français) ; liste vide = réglages utilisables."""
        errs = []
        if self.duration_h <= 0 and self.end_time:
            try:
                parse_end_time(self.end_time)
            except ValueError as e:
                errs.append(str(e).capitalize() + ".")
        return errs

    def apply_to(self, app_cfg) -> None:
        """Reporte condition / échantillon / masse / calibration dans une AppConfig (en place)."""
        if self.condition:
            app_cfg.run.condition = self.condition
        if self.sample:
            app_cfg.run.sample_type = self.sample
        if self.sample or self.mass_g > 0:
            app_cfg.run.sample_mass_g = self.mass_g
        if self.px_per_mm > 0:
            app_cfg.physics.px_per_mm = self.px_per_mm

    def describe(self, start: datetime | None = None) -> str:
        """Résumé d'une ligne (affichage)."""
        start = start or datetime.now()
        end = self.end_datetime(start)
        if end is None:
            fin = "jusqu'à l'arrêt manuel"
        else:
            day = "aujourd'hui" if end.date() == start.date() else ("demain" if end.date() == start.date()
                                                                        + timedelta(days=1) else end.strftime("%d/%m"))
            fin = f"fin {day} à {end:%H:%M} (dans {fmt_duration((end - start).total_seconds())})"
        vp = self.video_params()
        if vp is None:
            vid = "sans vidéo"
        elif self.record_video == "complete":
            vid = "vidéo complète"
        else:
            vid = (f"vidéo réduite ({fmt_num(100 * vp[0], 0)} %"
                   + (f", {fmt_num(vp[1], 0)} img/s" if vp[1] > 0 else "") + ")")
        if vp is not None and self.segment_min > 0:
            vid += f" en segments de {fmt_min(self.segment_min)} min"
        photo = (f"photo toutes les {fmt_min(self.snapshot_every_min)} min" if self.snapshot_every_min > 0
                 else "pas de photo")
        return f"{fin} ; {vid} ; {photo} ; sauvegarde toutes les {fmt_min(self.autosave_every_min)} min"


# =========================================================================== #
# Espace disque
# =========================================================================== #
def _video_size_out(cfg: NightSessionConfig, frame_wh: tuple[int, int], fps: float) -> tuple[int, int, float]:
    w, h = int(frame_wh[0]), int(frame_wh[1])
    vp = cfg.video_params()
    if vp is None:
        return 0, 0, 0.0
    scale, max_fps = vp
    if scale < 1.0:
        w = max(2, int(round(w * scale)) // 2 * 2)
        h = max(2, int(round(h * scale)) // 2 * 2)
    out_fps = min(fps, max_fps) if max_fps > 0 else fps
    return w, h, float(out_fps)


def estimate_disk_breakdown(cfg: NightSessionConfig, frame_wh: tuple[int, int], fps: float) -> dict:
    """Détail de ``estimate_disk_usage`` : {"video", "vignettes", "photos", "donnees", "total"} en
    octets, plus "hours" (durée supposée) et "unlimited" (True si aucune fin n'est prévue)."""
    planned = cfg.planned_seconds()
    unlimited = planned is None
    secs = UNLIMITED_ASSUMED_H * 3600.0 if unlimited else planned
    fps = float(fps) if fps and fps > 0 else 30.0
    w, h, out_fps = _video_size_out(cfg, frame_wh, fps)
    video = VIDEO_BITS_PER_PIXEL * w * h * out_fps * secs / 8.0
    n_tracks = ASSUMED_TRACKS_PER_MIN * secs / 60.0
    thumbs = n_tracks * THUMB_BYTES
    n_snap = int(secs // (cfg.snapshot_every_min * 60.0)) + 1 if cfg.snapshot_every_min > 0 else 0
    photos = n_snap * 2 * SNAPSHOT_BYTES_PER_PIXEL * int(frame_wh[0]) * int(frame_wh[1])
    data = n_tracks * ROW_BYTES + FIXED_BYTES
    parts = {"video": int(video), "vignettes": int(thumbs), "photos": int(photos), "donnees": int(data)}
    parts["total"] = sum(parts.values())
    parts["hours"] = secs / 3600.0
    parts["unlimited"] = unlimited
    return parts


def estimate_disk_usage(cfg: NightSessionConfig, frame_wh: tuple[int, int], fps: float) -> int:
    """Espace disque (octets) nécessaire à toute la session — estimation PRUDENTE (majorée).

    Hypothèses (mesurées sur des images de chambre simulées avec le bruit d'un capteur,
    OpenCV 5.0 / FFmpeg, octobre 2026) :
    - vidéo : mp4v/XVID ≈ 0,17–0,21 bit par pixel et par image (MJPG 0,21–0,26) ; on compte
      0,35 bit/px/image (marge ≈ ×1,7 pour une caméra plus bruitée). Exemple : vidéo réduite
      480×270 à 10 img/s ≈ 0,2 Go/h ; vidéo complète 1280×720 à 30 img/s ≈ 4,3 Go/h ;
    - vignettes : 20 Ko par trace (mesuré ≈ 12 Ko en moyenne) à 60 traces/min (prudent) ;
    - photos : 2 PNG (brute + annotée) de 1,5 octet/pixel chacune (mesuré ≈ 1,41) ;
    - données : 700 octets par trace (CSV + Excel) + 30 Mo (rapport, journal, marge) ;
    - session sans fin prévue : 12 h supposées.
    """
    return int(estimate_disk_breakdown(cfg, frame_wh, fps)["total"])


def free_disk_bytes(path: str | Path) -> int:
    """Espace libre (octets) sur le disque de ``path`` (ou de son premier parent existant) ; −1 si inconnu."""
    try:
        p = Path(path) if path else Path.cwd()
        p = p.absolute()
        while not p.exists() and p.parent != p:
            p = p.parent
        return int(shutil.disk_usage(str(p)).free)
    except OSError:
        return -1


# =========================================================================== #
# Empêcher la mise en veille (Windows)
# =========================================================================== #
class KeepAwake:
    """Contexte qui empêche la mise en veille AUTOMATIQUE de Windows pendant la session.

    ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` : l'écran peut s'éteindre et
    l'économiseur d'écran démarrer (sauf ``display=True``) ; l'état est rétabli à la sortie.
    Attention : cet état est lié au FIL d'exécution — entrer et sortir dans le même fil, qui
    doit vivre toute la session (le fil de traitement). Sans effet hors Windows. Ne peut pas
    empêcher une mise en veille demandée par l'utilisateur (capot fermé, bouton).
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def __init__(self, enabled: bool = True, display: bool = False):
        self.enabled = bool(enabled)
        self.display = bool(display)
        self.active = False
        self.error = ""
        self._fn = None

    def _func(self):
        if self._fn is None:
            fn = ctypes.windll.kernel32.SetThreadExecutionState
            fn.restype = ctypes.c_uint32
            fn.argtypes = [ctypes.c_uint32]
            self._fn = fn
        return self._fn

    @property
    def flags(self) -> int:
        f = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
        if self.display:
            f |= self.ES_DISPLAY_REQUIRED
        return f

    def __enter__(self) -> "KeepAwake":
        if self.enabled and sys.platform == "win32":
            try:
                prev = self._func()(self.flags)
                self.active = True
                if not prev:
                    self.error = "SetThreadExecutionState a échoué"
                    self.active = False
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                self.active = False
        elif self.enabled and sys.platform == "darwin":
            # macOS : l'outil système « caffeinate » empêche la veille tant que ce processus vit
            try:
                import os
                import subprocess
                args = ["caffeinate", "-i", "-s"] + (["-d"] if self.display else []) + ["-w", str(os.getpid())]
                self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.active = True
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                self.active = False
        return self

    def refresh(self) -> None:
        """Réaffirme l'état (sans effet si inactif)."""
        if self.active and getattr(self, "_proc", None) is None:
            try:
                self._func()(self.flags)
            except Exception:  # noqa: BLE001
                pass

    def __exit__(self, *exc) -> bool:
        if self.active:
            proc = getattr(self, "_proc", None)
            try:
                if proc is not None:
                    proc.terminate()
                    self._proc = None
                else:
                    self._func()(self.ES_CONTINUOUS)
            except Exception:  # noqa: BLE001
                pass
            self.active = False
        return False


# =========================================================================== #
# Horloge, journal, chambre « éteinte », taux récent
# =========================================================================== #
class SessionClock:
    """Temps écoulé / restant d'une session.

    simulated=False : horloge murale (time.monotonic) — les pauses et coupures de caméra comptent,
    l'heure de fin est absolue. simulated=True : temps de la source (SynthSource accélérée,
    fichier) cumulé à chaque ``tick(t)`` ; ``now()`` = début + temps écoulé simulé."""

    def __init__(self, planned_s: float | None = None, simulated: bool = False,
                 started_at: datetime | None = None, monotonic: Callable[[], float] = time.monotonic):
        self.planned_s = planned_s
        self.simulated = bool(simulated)
        self.started_at = started_at or datetime.now()
        self._mono = monotonic
        self._t0 = monotonic()
        self._sim = 0.0
        self._last_src: float | None = None

    def start(self, started_at: datetime | None = None, planned_s: float | None = None) -> None:
        self.started_at = started_at or datetime.now()
        self._t0 = self._mono()
        self._sim = 0.0
        self._last_src = None
        if planned_s is not None:
            self.planned_s = planned_s

    def tick(self, source_t: float | None) -> None:
        if not self.simulated or source_t is None:
            return
        if self._last_src is not None:
            d = source_t - self._last_src
            if d > 0:
                self._sim += d
        self._last_src = source_t

    @property
    def elapsed_s(self) -> float:
        return self._sim if self.simulated else max(0.0, self._mono() - self._t0)

    @property
    def remaining_s(self) -> float:
        if self.planned_s is None:
            return -1.0
        return max(0.0, self.planned_s - self.elapsed_s)

    def should_stop(self) -> bool:
        return self.planned_s is not None and self.elapsed_s >= self.planned_s

    def now(self) -> datetime:
        if self.simulated:
            return self.started_at + timedelta(seconds=self.elapsed_s)
        return datetime.now()

    @property
    def end_at(self) -> datetime | None:
        if self.planned_s is None:
            return None
        return self.started_at + timedelta(seconds=self.planned_s)


class SessionJournal:
    """Journal texte de la session (``journal.txt``, UTF-8, ajout seulement, une ligne par événement).

    Chaque ligne est écrite puis le fichier refermé : rien n'est perdu en cas de plantage.
    Utilisable depuis plusieurs fils (verrou)."""

    LEVELS = {"info": "", "attention": "ATTENTION : ", "erreur": "ERREUR : "}

    def __init__(self, folder: str | Path, clock: SessionClock | None = None, name: str = "journal.txt"):
        self.path = Path(folder) / name
        self.clock = clock
        self.last_error = ""
        self._lock = threading.Lock()

    def write(self, msg: str, level: str = "info") -> None:
        clock = self.clock
        when = clock.now() if clock is not None else datetime.now()
        el = f"  [+{_hms(clock.elapsed_s)}]" if clock is not None else ""
        prefix = self.LEVELS.get(level, "")
        text = str(msg).replace("\r", " ").replace("\n", " ")
        line = f"{when:%Y-%m-%d %H:%M:%S}{el}  {prefix}{text}\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line)
            except OSError as e:
                self.last_error = str(e)

    def lines(self) -> list[str]:
        try:
            return self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []


class ChamberDeadDetector:
    """Chambre « éteinte » : aucune trace valide pendant ``dead_after_s`` secondes (d'ANALYSE :
    les pauses et coupures de caméra ne comptent pas). Avertit une seule fois ; l'alerte est levée
    dès que des traces reviennent."""

    def __init__(self, dead_after_s: float):
        self.dead_after_s = float(dead_after_s)
        self.last_track_s = 0.0
        self.dead = False
        self._n = 0

    def update(self, t_s: float, n_valid: int) -> str | None:
        """Renvoie « eteinte » (nouvelle alerte), « reprise » (fin d'alerte) ou None."""
        if n_valid > self._n:
            self._n = n_valid
            self.last_track_s = t_s
            if self.dead:
                self.dead = False
                return "reprise"
            return None
        if self.dead_after_s > 0 and not self.dead and t_s - self.last_track_s >= self.dead_after_s:
            self.dead = True
            return "eteinte"
        return None

    def silence_s(self, t_s: float) -> float:
        return max(0.0, t_s - self.last_track_s)


class RateWindow:
    """Taux de traces valides sur une fenêtre glissante (temps d'analyse, en secondes)."""

    def __init__(self, window_s: float = 600.0):
        self.window_s = float(window_s)
        self._t: deque[float] = deque()

    def add(self, t_s: float) -> None:
        self._t.append(float(t_s))

    def count(self, now_s: float) -> int:
        lo = now_s - self.window_s
        while self._t and self._t[0] <= lo:
            self._t.popleft()
        return len(self._t)

    def rate_per_min(self, now_s: float) -> float:
        n = self.count(now_s)
        span = min(self.window_s, max(0.0, now_s))
        return n / (span / 60.0) if span > 1.0 else 0.0


class TimelineFallback:
    """Chronologie (une entrée par minute d'analyse) quand le détecteur n'a pas ``timeline``.

    Même contenu que ``TrackDetector.timeline`` : t_s, wall_s, frames, disturbed_frames, noise,
    lit, counts (traces valides cumulées par classe), rejected."""

    def __init__(self, period_s: float = 60.0):
        self.period_s = float(period_s) if period_s and period_s > 0 else 60.0
        self.entries: list[dict] = []
        self.counts: dict[str, int] = {}
        self._next = self.period_s
        self._noise: list[float] = []
        self._lit: list[float] = []
        self._wall = 0.0

    def on_frame(self, det, res) -> None:
        for e in res.finished:
            if getattr(e, "valid", True):
                self.counts[e.final_cls] = self.counts.get(e.final_cls, 0) + 1
        self._wall = float(getattr(res, "t", 0.0) or 0.0)
        if getattr(res, "warming", False):
            return
        if not getattr(res, "disturbed", False):
            self._noise.append(float(res.noise_sigma))
            self._lit.append(float(res.lit_frac))
        if det.analyzed_s >= self._next:
            self._append(det)
            while self._next <= det.analyzed_s:
                self._next += self.period_s

    def _append(self, det) -> None:
        import numpy as np
        self.entries.append({
            "t_s": float(det.analyzed_s), "wall_s": self._wall, "frames": int(det.n_frames_processed),
            "disturbed_frames": int(det.n_frames_disturbed),
            "noise": float(np.median(self._noise)) if self._noise else 0.0,
            "lit": float(np.median(self._lit)) if self._lit else 0.0,
            "counts": dict(self.counts), "rejected": int(det.rejected),
        })
        self._noise, self._lit = [], []

    def finish(self, det) -> None:
        if not self.entries or self.entries[-1]["frames"] != det.n_frames_processed:
            self._append(det)


class CheckpointWriter:
    """Exécute les écritures des points de sauvegarde dans un fil séparé : l'analyse ne marque
    pas de pause (sinon des traces courtes seraient perdues pendant l'écriture)."""

    def __init__(self):
        self._th: threading.Thread | None = None
        self.last_error = ""

    @property
    def busy(self) -> bool:
        return self._th is not None and self._th.is_alive()

    def submit(self, fn: Callable[[], None]) -> bool:
        """Lance ``fn`` en arrière-plan ; False si une écriture précédente est encore en cours."""
        if self.busy:
            return False

        def run():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__} : {e}"

        self._th = threading.Thread(target=run, name="wilsontrack-sauvegarde", daemon=True)
        self._th.start()
        return True

    def join(self, timeout: float | None = None) -> None:
        th = self._th
        if th is not None:
            th.join(timeout)


class ReconnectPolicy:
    """Que faire quand la caméra ne renvoie plus d'images (session de nuit : on insiste).

    fail() → « attendre », « perdue » (une seule fois, après ``grace_s`` secondes sans image : un
    simple hoquet de webcam n'est pas signalé), « reconnecter » ou « abandon » ;
    ok() → durée de la coupure (s) si une coupure SIGNALÉE vient de se terminer, sinon None."""

    def __init__(self, first_retry_s: float = 3.0, retry_every_s: float = 10.0, slow_after_s: float = 120.0,
                 slow_retry_s: float = 30.0, give_up_after_s: float = 900.0, grace_s: float = 2.0,
                 monotonic: Callable[[], float] = time.monotonic):
        self.first_retry_s = first_retry_s
        self.retry_every_s = retry_every_s
        self.slow_after_s = slow_after_s
        self.slow_retry_s = slow_retry_s
        self.give_up_after_s = give_up_after_s
        self.grace_s = grace_s
        self._mono = monotonic
        self.lost_at: float | None = None
        self.declared = False
        self.next_try = 0.0
        self.attempts = 0

    @property
    def outage_s(self) -> float:
        return 0.0 if self.lost_at is None else self._mono() - self.lost_at

    def fail(self) -> str:
        now = self._mono()
        if self.lost_at is None:
            self.lost_at = now
            self.declared = False
            self.attempts = 0
            self.next_try = now + self.first_retry_s
        out = now - self.lost_at
        if not self.declared:
            if out >= self.grace_s:
                self.declared = True
                return "perdue"
            return "attendre"
        if out >= self.give_up_after_s:
            return "abandon"
        if now >= self.next_try:
            self.attempts += 1
            wait = self.slow_retry_s if out > self.slow_after_s else self.retry_every_s
            self.next_try = now + wait
            return "reconnecter"
        return "attendre"

    def ok(self) -> float | None:
        if self.lost_at is None:
            return None
        gap = self._mono() - self.lost_at
        declared = self.declared
        self.lost_at = None
        self.declared = False
        self.attempts = 0
        return gap if declared else None


class SourceTime:
    """Horodatage croissant malgré les reconnexions (une caméra rouverte repart de t = 0) :
    après une coupure de ``gap`` secondes, le temps reprend à « dernier temps + gap »."""

    def __init__(self):
        self.offset = 0.0
        self.last_out: float | None = None
        self._gap: float | None = None

    def gap(self, seconds: float) -> None:
        self._gap = max(0.0, float(seconds))

    def __call__(self, t: float | None) -> float | None:
        if t is None:
            return None
        if self.last_out is not None:
            if self._gap is not None:
                self.offset = self.last_out + max(self._gap, 1e-3) - t
            elif t + self.offset <= self.last_out:
                self.offset = self.last_out + 1.0 / 30.0 - t
        self._gap = None
        out = t + self.offset
        self.last_out = out
        return out


def source_finished(src) -> bool:
    """Une source à durée finie (fichier, simulation) est-elle arrivée au bout ?"""
    n = int(getattr(src, "n_frames", 0) or 0)
    return n > 0 and int(getattr(src, "position", 0) or 0) >= n


def make_session_folder(cfg: NightSessionConfig, app_cfg, started_at: datetime | None = None) -> Path:
    """« <sortie>/<date> <condition> nuit <HHhMM> » (suffixe « (2) », « (3) »… si le dossier existe)."""
    from .analysis.run import safe_name
    from .config import default_output_dir
    started_at = started_at or datetime.now()
    if cfg.output_dir:
        root = Path(cfg.output_dir)
    elif getattr(app_cfg, "output_dir", ""):
        root = Path(app_cfg.output_dir)
    else:
        root = default_output_dir()
    cond = (cfg.condition or app_cfg.run.condition or "Session").strip()
    name = safe_name(f"{started_at:%Y-%m-%d} {cond} nuit {started_at:%Hh%M}")
    folder, k = root / name, 2
    while folder.exists():
        folder = root / f"{name} ({k})"
        k += 1
    return folder


# =========================================================================== #
# Chef d'orchestre d'une session
# =========================================================================== #
class SessionController:
    """Appelé par la boucle de traitement ; ne lit pas lui-même la caméra.

    Utilisation (voir ``run_session`` et gui/worker.py) ::

        ctl = SessionController(cfg, app_cfg, source_name="camera:0")
        det = TrackDetector(app_cfg, src.fps, src.shape)
        ctl.begin(det, src.fps, src.shape)
        while ...:
            res = det.process(frame, t)
            status = ctl.on_frame(frame, t, res, det)     # dict toutes les ~2 s, sinon None
            if ctl.should_stop(): break
        run = ctl.finish(det, "durée prévue atteinte")    # finalisation + sauvegarde + rapport
    """

    STATUS_EVERY_S = 2.0          # temps réel
    HEALTH_EVERY_S = 60.0         # temps de session
    SPILL_EVERY_S = 60.0
    FIRST_SNAPSHOT_S = 30.0
    KEEP_THUMBS = 300
    DISK_WARN_BYTES = 2 * 1024 ** 3
    DISK_STOP_VIDEO_BYTES = 500 * 1024 ** 2
    RATE_DROP_MIN_ANALYZED_S = 30 * 60.0

    def __init__(self, cfg: NightSessionConfig | dict, app_cfg, session_folder: str | Path = "",
                 source_name: str = "", simulated_clock: bool = False,
                 on_warning: Callable[[str], None] | None = None, on_info: Callable[[str], None] | None = None,
                 started_at: datetime | None = None):
        from .capture import VideoRecorder
        self.cfg = cfg if isinstance(cfg, NightSessionConfig) else NightSessionConfig.from_dict(cfg)
        self.app_cfg = app_cfg
        self.cfg.apply_to(self.app_cfg)
        self.started_at = started_at or datetime.now()
        self.folder = Path(session_folder) if session_folder else make_session_folder(self.cfg, app_cfg,
                                                                                       self.started_at)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.name = self.folder.name
        self.source_name = source_name or f"camera:{self.cfg.camera_index}"
        self.simulated = bool(simulated_clock)
        self.clock = SessionClock(self.cfg.planned_seconds(self.started_at), self.simulated, self.started_at)
        self.journal = SessionJournal(self.folder, self.clock)
        self.recorder = VideoRecorder()
        self.dead = ChamberDeadDetector(self.cfg.dead_after_min * 60.0)
        self.rate = RateWindow(600.0)
        self.writer = CheckpointWriter()
        self.timeline_fb: TimelineFallback | None = None
        self.on_warning = on_warning
        self.on_info = on_info
        self.n_valid = 0
        self.counts: dict[str, int] = {}
        self.checkpoints = 0
        self.last_checkpoint = ""
        self.photos = 0
        self.reconnections = 0
        self.spilled = 0
        self.warnings_active: dict[str, str] = {}
        self.warning_log: list[dict] = []
        self.stop_reason = ""
        self.ended_at: datetime | None = None
        self.final_run = None
        self.finished = False
        self._fps = 30.0
        self._frame_hw = (0, 0)
        self._det = None
        self._next_status_mono = 0.0
        self._next_health = self.HEALTH_EVERY_S
        self._next_spill = self.SPILL_EVERY_S
        self._next_ckpt = self.cfg.autosave_every_min * 60.0
        self._next_snap = (min(self.FIRST_SNAPSHOT_S, self.cfg.snapshot_every_min * 60.0)
                           if self.cfg.snapshot_every_min > 0 else math.inf)
        self._video_started = False
        self._video_error_reported = False
        self._h_prev = (0.0, 0, 0)          # (temps de session, images, traces) à la dernière ligne de santé
        self._noise_min: list[float] = []
        self._rate_drop = False
        self._last_res = None
        self._last_frame = None

    # ------------------------------------------------------------ messages --
    def info(self, msg: str) -> None:
        self.journal.write(msg)
        if self.on_info is not None:
            try:
                self.on_info(msg)
            except Exception:  # noqa: BLE001
                pass

    def warn(self, key: str, msg: str) -> None:
        if self.warnings_active.get(key) == msg:
            return
        self.warnings_active[key] = msg
        self.warning_log.append({"heure": self.clock.now().isoformat(timespec="seconds"),
                                 "temps_s": round(self.clock.elapsed_s, 1), "message": msg})
        self.journal.write(msg, "attention")
        if self.on_warning is not None:
            try:
                self.on_warning(msg)
            except Exception:  # noqa: BLE001
                pass

    def clear_warning(self, key: str, msg: str | None = None) -> None:
        if self.warnings_active.pop(key, None) is not None and msg:
            self.info(msg)

    # ---------------------------------------------------------------- début --
    def begin(self, det, fps: float, frame_hw: tuple[int, int]) -> None:
        self._det = det
        self._fps = float(fps) if fps and fps > 0 else 30.0
        self._frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
        if not self.simulated:
            self.started_at = datetime.now()
        self.clock.start(self.started_at, self.cfg.planned_seconds(self.started_at))
        for sub in ("vignettes", "photos"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        if self.cfg.video_params() is not None:
            (self.folder / "video").mkdir(parents=True, exist_ok=True)
        if not hasattr(det, "timeline"):
            self.timeline_fb = TimelineFallback(float(getattr(det, "timeline_period_s", 60.0) or 60.0))
        if hasattr(det, "keep_invalid_events"):
            det.keep_invalid_events = False       # rejets comptés mais non conservés (mémoire bornée)
        h, w = self._frame_hw
        rc = self.app_cfg.run
        end = self.clock.end_at
        self.journal.write(f"===== WilsonTrack {__version__} — session de longue durée =====")
        if self.simulated:
            self.journal.write("Horloge SIMULÉE (source accélérée) : les heures indiquées sont fictives.")
        self.journal.write(f"Dossier : {self.folder}")
        self.journal.write(f"Source : {self.source_name} — image {w}×{h} à {fmt_num(self._fps, 1)} img/s")
        self.journal.write(f"Condition : {rc.condition} ; échantillon : {rc.sample_type}"
                           + (f" ({fmt_num(rc.sample_mass_g, 1)} g)" if rc.sample_mass_g > 0 else ""))
        ppm = self.app_cfg.physics.px_per_mm
        self.journal.write("Calibration : " + (f"{fmt_num(ppm, 3)} px/mm" if ppm > 0 else "aucune (échelle supposée)"))
        self.journal.write("Fin prévue : " + (f"{end:%Y-%m-%d %H:%M} (durée {fmt_duration(self.clock.planned_s)})"
                                              if end is not None else "à l'arrêt manuel"))
        self.journal.write("Réglages : " + self.cfg.describe(self.started_at))
        try:
            est = estimate_disk_breakdown(self.cfg, (w, h), self._fps)
            free = free_disk_bytes(self.folder)
            self.journal.write(f"Espace disque estimé (prudent) : {fmt_bytes(est['total'])} "
                               f"(vidéo {fmt_bytes(est['video'])}) ; libre : {fmt_bytes(free)}")
            if 0 <= free < est["total"]:
                self.warn("disque_estimation",
                          f"L'espace libre ({fmt_bytes(free)}) est inférieur à l'estimation prudente "
                          f"({fmt_bytes(est['total'])}) : la vidéo pourrait s'arrêter avant la fin "
                          "(les mesures, elles, continuent).")
        except Exception:  # noqa: BLE001
            pass
        self._write_session_json()
        self.info(f"Session démarrée : {self.folder.name}")

    # ------------------------------------------------------------ par image --
    def on_frame(self, frame, t: float | None, res, det) -> dict | None:
        """À appeler après chaque ``det.process``. Renvoie l'état (dict) toutes les ~2 s."""
        self._det = det
        self.clock.tick(t)
        el = self.clock.elapsed_s
        self._last_res, self._last_frame = res, frame
        if self.timeline_fb is not None:
            self.timeline_fb.on_frame(det, res)
        for e in res.finished:
            if getattr(e, "valid", True):
                self.n_valid += 1
                self.counts[e.final_cls] = self.counts.get(e.final_cls, 0) + 1
                self.rate.add(det.analyzed_s)
        if not res.warming and not res.disturbed:
            self._noise_min.append(float(res.noise_sigma))
        # Vidéo : démarrée après l'apprentissage du fond et, avec une vraie caméra, après 3 s (la
        # cadence réelle, souvent inférieure à la cadence annoncée, est alors mesurée)
        if (self.cfg.video_params() is not None and not self._video_started and not res.warming
                and (self.simulated or el >= 3.0)):
            self._start_video(det)
        if self.recorder.active:
            self.recorder.write(frame, el)
        elif self._video_started and self.recorder.last_error and not self._video_error_reported:
            self._video_error_reported = True
            self.warn("video", f"Enregistrement vidéo arrêté : {self.recorder.last_error} "
                               "Les mesures continuent.")
        if el >= self._next_snap:
            self.snapshot(frame, res, det)
            period = self.cfg.snapshot_every_min * 60.0
            self._next_snap = max(self._next_snap + period, el + 0.5 * period)
        if el >= self._next_health:
            self._health(det, el)
            self._next_health = max(self._next_health + self.HEALTH_EVERY_S, el + 0.5 * self.HEALTH_EVERY_S)
        if el >= self._next_spill and not self.writer.busy:
            self._spill(det)
            self._next_spill = el + self.SPILL_EVERY_S
        if el >= self._next_ckpt:
            self.checkpoint(det)
            self._next_ckpt = el + self.cfg.autosave_every_min * 60.0
        return self._maybe_status(det)

    def idle_tick(self, det=None) -> dict | None:
        """Sans image (pause, caméra coupée) : l'état continue d'être envoyé."""
        return self._maybe_status(det if det is not None else self._det)

    def should_stop(self) -> bool:
        return self.clock.should_stop()

    def _maybe_status(self, det) -> dict | None:
        now = time.monotonic()
        if now < self._next_status_mono:
            return None
        self._next_status_mono = now + self.STATUS_EVERY_S
        return self.status(det)

    def status(self, det=None) -> dict:
        """État de la session (clés du contrat GUI + extras)."""
        det = det if det is not None else self._det
        el = self.clock.elapsed_s
        ana = float(getattr(det, "analyzed_s", 0.0) or 0.0) if det is not None else 0.0
        end = self.clock.end_at
        return {
            "elapsed_s": el,
            "remaining_s": self.clock.remaining_s,
            "next_autosave_s": max(0.0, self._next_ckpt - el),
            "disk_free_bytes": free_disk_bytes(self.folder),
            "segment_index": self.recorder.segment_index,
            "recording": self.recorder.active,
            "n_tracks": self.n_valid,
            "rate_per_min": self.rate.rate_per_min(ana),
            "last_checkpoint": self.last_checkpoint,
            "warnings": list(self.warnings_active.values()),
            "session_folder": str(self.folder),
            # extras
            "planned_s": self.clock.planned_s if self.clock.planned_s is not None else -1.0,
            "end_at": end.isoformat(timespec="seconds") if end is not None else "",
            "checkpoints": self.checkpoints,
            "video_bytes": self.recorder.bytes_written,
            "chamber_dead": self.dead.dead,
            "simulated_clock": self.simulated,
            "analyzed_s": ana,
            "photos": self.photos,
            "counts": dict(self.counts),
        }

    # ------------------------------------------------------------- caméra --
    def on_camera_lost(self) -> None:
        self.warn("camera", "La caméra ne renvoie plus d'images : tentatives de reconnexion en cours.")

    def on_camera_retry(self, attempt: int, ok: bool) -> None:
        self.reconnections += 1 if ok else 0
        self.journal.write(f"Reconnexion à la caméra (tentative {attempt}) : {'réussie' if ok else 'échec'}.")

    def on_camera_back(self, gap_s: float) -> None:
        self.clear_warning("camera", f"Caméra de nouveau disponible après {fmt_duration(gap_s)} de coupure "
                                     "(ce temps n'est pas compté dans la durée analysée).")

    def on_camera_gave_up(self, outage_s: float) -> None:
        self.warn("camera", f"Caméra perdue depuis {fmt_duration(outage_s)} : fin de la session.")

    # ------------------------------------------------------------- tâches --
    def _start_video(self, det) -> None:
        self._video_started = True
        vp = self.cfg.video_params()
        if vp is None:
            return
        scale, max_fps = vp
        h, w = self._frame_hw
        fps_in = float(getattr(det, "fps", 0) or 0) or self._fps
        # AVI : un fichier interrompu (coupure de courant) reste lisible, contrairement au MP4.
        # Nom court : les chemins Windows de plus de 260 caractères posent problème.
        path = self.folder / "video" / f"{self.started_at:%Y-%m-%d_%Hh%M}.avi"
        try:
            self.recorder.start(path, fps_in, (w, h), segment_s=self.cfg.segment_s, scale=scale, max_fps=max_fps)
            ow, oh = self.recorder.size_out
            self.journal.write(f"Enregistrement vidéo : {VIDEO_MODE_LABELS[self.cfg.record_video]} {ow}×{oh} à "
                               f"{fmt_num(self.recorder.out_fps, 1)} img/s, codec {self.recorder.codec}"
                               + (f", segments de {fmt_min(self.cfg.segment_min)} min" if self.cfg.segment_s > 0
                                  else ""))
        except Exception as e:  # noqa: BLE001
            self._video_error_reported = True
            self.warn("video", f"Impossible de démarrer l'enregistrement vidéo : {e} Les mesures continuent.")

    def snapshot(self, frame, res, det) -> None:
        import cv2

        from .capture import imwrite
        from .render import draw_overlay
        if frame is None:
            return
        free = free_disk_bytes(self.folder)
        if 0 <= free < self.DISK_STOP_VIDEO_BYTES:
            return
        now = self.clock.now()
        stamp = now.strftime("%Y-%m-%d %H-%M-%S")
        base = self.folder / "photos" / stamp
        try:
            ok1 = imwrite(str(base) + " brute.png", frame)
            recent = [e for e in det.events[-60:] if e.valid]
            ann = draw_overlay(frame, res, det, recent, keep_s=3.0)
            # Horodatage incrusté (ASCII : cv2.putText n'affiche pas les accents)
            txt = f"{now:%Y-%m-%d %H:%M:%S}  +{_hms(self.clock.elapsed_s)}  {self.n_valid} traces"
            fs = max(0.5, ann.shape[1] / 1600.0)
            th = max(1, int(round(fs * 2)))
            cv2.putText(ann, txt, (10, int(28 * fs) + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(ann, txt, (10, int(28 * fs) + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th,
                        cv2.LINE_AA)
            ok2 = imwrite(str(base) + " annotee.png", ann)
            if ok1 and ok2:
                self.photos += 1
                self.clear_warning("photo")
            else:
                self.warn("photo", "Photo périodique non enregistrée (disque ?).")
        except Exception as e:  # noqa: BLE001
            self.warn("photo", f"Photo périodique non enregistrée : {e}")

    def _spill(self, det) -> None:
        from .analysis.run import limit_memory
        try:
            n = limit_memory(det, self.folder / "vignettes", self.KEEP_THUMBS)
            self.spilled += n
        except Exception as e:  # noqa: BLE001
            self.warn("vignettes", f"Délestage des vignettes impossible : {e}")

    def _health(self, det, el: float) -> None:
        import numpy as np
        t_prev, f_prev, n_prev = self._h_prev
        frames = int(det.frame_idx + 1)
        dt = max(1e-6, el - t_prev)
        ips = (frames - f_prev) / dt
        noise = float(np.median(self._noise_min)) if self._noise_min else float("nan")
        self._noise_min = []
        dist = 100.0 * det.n_frames_disturbed / max(1, frames)
        rate10 = self.rate.rate_per_min(det.analyzed_s)
        free = free_disk_bytes(self.folder)
        vid = ""
        if self.recorder.active:
            vid = f", vidéo {fmt_bytes(self.recorder.bytes_written)} (segment {self.recorder.segment_index})"
        self.journal.write(
            f"Santé : {fmt_num(ips, 1)} img/s, bruit σ = {fmt_num(noise, 2)}, {self.n_valid} traces valides "
            f"(+{self.n_valid - n_prev} depuis la dernière ligne), taux {fmt_num(rate10, 1)} /min "
            f"(10 dernières min d'analyse), analysé {fmt_duration(det.analyzed_s)}, images perturbées "
            f"{fmt_num(dist, 1)} %, rejets {det.rejected}{vid}, disque libre {fmt_bytes(free)}")
        self._h_prev = (el, frames, self.n_valid)
        # Cadence réellement mesurée → appliquée au prochain segment vidéo
        if not self.simulated and self.recorder.active:
            self.recorder.set_fps(float(getattr(det, "fps", 0) or 0) or self._fps)
        # Chambre « éteinte »
        ev = self.dead.update(det.analyzed_s, self.n_valid)
        if ev == "eteinte":
            self.warn("chambre", f"Aucune trace valide depuis {fmt_duration(self.dead.silence_s(det.analyzed_s))} "
                                 "d'analyse : la chambre ne fonctionne peut-être plus (alcool épuisé, glace sèche "
                                 "fondue, éclairage ?).")
        elif ev == "reprise":
            self.clear_warning("chambre", "Les traces sont revenues : fin de l'alerte « chambre inactive ».")
        self._check_rate_drop(det)
        # Disque
        if 0 <= free < self.DISK_STOP_VIDEO_BYTES and self.recorder.active:
            self.recorder.stop()
            self._video_error_reported = True
            self.warn("disque", f"Disque presque plein ({fmt_bytes(free)} libres) : enregistrement vidéo arrêté. "
                                "Les mesures continuent.")
        elif 0 <= free < self.DISK_WARN_BYTES:
            self.warn("disque", f"Espace disque faible : {fmt_bytes(free)} libres.")
        else:
            self.clear_warning("disque")

    def _check_rate_drop(self, det) -> None:
        """Baisse statistiquement nette du taux (test de Poisson unilatéral, p < 0,001) : la
        sensibilité de la chambre diminue probablement (alcool, froid). Simple avertissement."""
        ana = float(det.analyzed_s)
        if ana < self.RATE_DROP_MIN_ANALYZED_S:
            return
        n10 = self.rate.count(ana)
        win_min = self.rate.window_s / 60.0
        before_min = (ana - self.rate.window_s) / 60.0
        if before_min <= 0:
            return
        ref = (self.n_valid - n10) / before_min
        expected = ref * win_min
        if expected < 20:
            return
        try:
            from scipy.stats import poisson
            p = float(poisson.cdf(n10, expected))
        except Exception:  # noqa: BLE001
            return
        if p < 1e-3 and not self._rate_drop:
            self._rate_drop = True
            self.warn("taux", f"Le taux de traces a nettement baissé : {fmt_num(n10 / win_min, 1)} /min sur les "
                              f"10 dernières minutes contre {fmt_num(ref, 1)} /min avant (p < 0,001). Vérifiez "
                              "l'alcool et le refroidissement ; pour l'analyse, utilisez la période stable.")
        elif self._rate_drop and p > 0.05:
            self._rate_drop = False
            self.clear_warning("taux", "Le taux de traces est revenu à la normale.")

    # ---------------------------------------------------- points de sauvegarde --
    def session_info(self, partial: bool) -> dict:
        segs = []
        for p in self.recorder.segments:
            try:
                segs.append(str(Path(p).relative_to(self.folder)))
            except ValueError:
                segs.append(str(p))
        end = self.clock.end_at
        d = self.cfg.to_dict()
        d.update({
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": self.ended_at.isoformat(timespec="seconds") if self.ended_at else "",
            "planned_end": end.isoformat(timespec="seconds") if end is not None else "",
            "planned_s": self.clock.planned_s,
            "elapsed_s": round(self.clock.elapsed_s, 1),
            "stop_reason": self.stop_reason,
            "partial": bool(partial),
            "checkpoints": self.checkpoints,
            "video_segments": segs,
            "video_codec": self.recorder.codec,
            "photos": self.photos,
            "reconnections": self.reconnections,
            "warnings": list(self.warning_log),
            "simulated_clock": self.simulated,
            "folder": str(self.folder),
            "journal": "journal.txt",
        })
        return d

    def build_run(self, det, partial: bool):
        from .analysis.run import build_run_result
        n = int(det.frame_idx + 1)
        run = build_run_result(det, self.app_cfg, self.source_name, n, self._fps, self._frame_hw,
                               name=self.name, session=self.session_info(partial), valid_only=partial)
        if self.timeline_fb is not None:
            run.timeline = [dict(e, counts=dict(e["counts"])) for e in self.timeline_fb.entries]
        return run

    def checkpoint(self, det, wait: bool = False) -> bool:
        """Point de sauvegarde provisoire (run.json « partiel » + traces.csv + vignettes), écrit en
        arrière-plan. False si le précédent est encore en cours d'écriture."""
        if self.writer.busy:
            self.journal.write("Point de sauvegarde reporté : le précédent est encore en cours d'écriture.")
            return False
        self._spill(det)
        run = self.build_run(det, partial=True)
        k = self.checkpoints + 1

        def job():
            run.save(folder=self.folder, partial=True, xlsx=False, thumbnails=True)
            self.checkpoints = k
            self.last_checkpoint = self.clock.now().isoformat(timespec="seconds")
            self.journal.write(f"Point de sauvegarde n° {k} : {run.total} traces valides, "
                               f"{fmt_duration(run.duration_s)} analysées (résultats provisoires).")
            for p in run.save_warnings:
                self.journal.write(f"Sauvegarde : {p}", "attention")
            self._write_session_json(partial=True)

        self.writer.submit(job)
        if wait:
            self.writer.join()
            if self.writer.last_error:
                self.warn("sauvegarde", f"Point de sauvegarde impossible : {self.writer.last_error}")
        return True

    def _write_session_json(self, partial: bool = True) -> None:
        p = self.folder / "session.json"
        tmp = p.with_name("session.tmp.json")
        try:
            tmp.write_text(json.dumps(self.session_info(partial), indent=2, ensure_ascii=False, default=str),
                           encoding="utf-8")
            os.replace(tmp, p)
        except OSError:
            pass

    # ------------------------------------------------------------------ fin --
    def finish(self, det, reason: str = "", finalize: bool = True, make_report: bool = True):
        """Fin de session : attend la sauvegarde en cours, ferme la vidéo, finalise le détecteur
        (reclassification sur toute la nuit), sauvegarde le résultat final et le rapport HTML.
        Renvoie le RunResult final (déjà enregistré dans le dossier de la session)."""
        if self.finished:
            return self.final_run
        self.stop_reason = reason or self.stop_reason or "arrêt"
        self.writer.join()
        if self.writer.last_error:
            self.journal.write(f"Dernier point de sauvegarde en erreur : {self.writer.last_error}", "attention")
        if self.recorder.active:
            self.recorder.stop()
        if finalize:
            det.finalize_run()
        if self.timeline_fb is not None:
            self.timeline_fb.finish(det)
        self.ended_at = self.clock.now()
        run = self.build_run(det, partial=False)
        try:
            run.save(folder=self.folder, partial=False)
            for p in run.save_warnings:
                self.journal.write(f"Sauvegarde finale : {p}", "attention")
        except Exception as e:  # noqa: BLE001
            self.journal.write(f"Sauvegarde finale impossible : {e}", "erreur")
            try:
                run.save(folder=self.folder, partial=False, xlsx=False)
            except Exception:  # noqa: BLE001
                pass
        report = ""
        if make_report:
            try:
                from .analysis.report import build_run_report
                report = str(build_run_report(run))
            except Exception as e:  # noqa: BLE001
                self.journal.write(f"Rapport HTML non généré : {type(e).__name__} : {e}", "attention")
        self._write_session_json(partial=False)
        cnt = run.counts()
        parts = ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) or "aucune"
        self.journal.write(f"Fin de la session : {self.stop_reason}. Durée {fmt_duration(self.clock.elapsed_s)}, "
                           f"analysée {fmt_duration(run.duration_s)} ; {run.total} traces valides ({parts}).")
        if self.recorder.segments:
            self.journal.write(f"Vidéo : {len(self.recorder.segments)} fichier(s), "
                               f"{fmt_bytes(self.recorder.bytes_written)}.")
        if report:
            self.journal.write(f"Rapport : {report}")
        self.final_run = run
        self.finished = True
        return run


# =========================================================================== #
# Boucle complète sans interface
# =========================================================================== #
def _reopen(src, factory):
    try:
        src.release()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.3)
    try:
        return factory()
    except Exception:  # noqa: BLE001
        return None


def run_session(cfg: NightSessionConfig | dict, app_cfg=None, source_factory: Callable | None = None,
                session_folder: str | Path = "", source_name: str = "",
                on_status: Callable[[dict], None] | None = None, on_info: Callable[[str], None] | None = None,
                on_warning: Callable[[str], None] | None = None, stop_event: threading.Event | None = None,
                simulated_clock: bool | None = None, make_report: bool = True,
                max_outage_s: float = 900.0, on_frame: Callable | None = None):
    """Session complète sans interface ; renvoie le RunResult final (déjà enregistré).

    source_factory() → FrameSource (défaut : CameraSource(cfg.camera_index)). Une SynthSource
    ``realtime=False`` donne une session ACCÉLÉRÉE (horloge simulée). stop_event : arrêt demandé.
    Caméra : reconnexions répétées pendant ``max_outage_s`` secondes avant d'abandonner."""
    from .config import AppConfig
    from .detection import TrackDetector
    cfg = cfg if isinstance(cfg, NightSessionConfig) else NightSessionConfig.from_dict(cfg)
    app_cfg = (app_cfg or AppConfig()).copy()
    if source_factory is None:
        from .capture import CameraSource
        idx = cfg.camera_index
        source_factory = lambda: CameraSource(idx)  # noqa: E731
    src = source_factory()
    live = bool(getattr(src, "live", False))
    if simulated_clock is None:
        simulated_clock = (not live) or (not getattr(src, "realtime", True))
    cfg.apply_to(app_cfg)
    if hasattr(src, "px_per_mm") and app_cfg.physics.px_per_mm <= 0:
        app_cfg.physics.px_per_mm = float(src.px_per_mm)
    ctl = SessionController(cfg, app_cfg, session_folder, source_name or f"camera:{cfg.camera_index}",
                            simulated_clock=simulated_clock, on_warning=on_warning, on_info=on_info)
    det = TrackDetector(app_cfg, src.fps, src.shape)
    ctl.begin(det, src.fps, src.shape)
    policy = ReconnectPolicy(give_up_after_s=max_outage_s)
    stime = SourceTime()
    reason = "arrêt demandé"
    run = None
    with KeepAwake(cfg.keep_awake and not simulated_clock, display=cfg.keep_display_on) as ka:
        if cfg.keep_awake and not simulated_clock and not ka.active and sys.platform == "win32":
            ctl.journal.write(f"Mise en veille non bloquée : {ka.error}", "attention")
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    reason = "arrêt demandé"
                    break
                try:
                    ok, frame, t = src.read()
                except Exception:  # noqa: BLE001
                    ok, frame, t = False, None, 0.0
                if not ok or frame is None:
                    if not live or source_finished(src):
                        reason = "fin de la source"
                        break
                    act = policy.fail()
                    if act == "perdue":
                        ctl.on_camera_lost()
                    elif act == "reconnecter":
                        new = _reopen(src, source_factory)
                        ctl.on_camera_retry(policy.attempts, new is not None)
                        if new is not None:
                            src = new
                    elif act == "abandon":
                        ctl.on_camera_gave_up(policy.outage_s)
                        reason = "caméra perdue"
                        break
                    st = ctl.idle_tick(det)
                    if st and on_status:
                        on_status(st)
                    if ctl.should_stop():
                        reason = "durée prévue atteinte"
                        break
                    time.sleep(0.05)
                    continue
                gap = policy.ok()
                if gap is not None:
                    ctl.on_camera_back(gap)
                    stime.gap(gap)
                tt = stime(t) if live else None
                res = det.process(frame, tt)
                if on_frame is not None:
                    on_frame(frame, res, det)
                st = ctl.on_frame(frame, tt if tt is not None else t, res, det)
                if st and on_status:
                    on_status(st)
                if ctl.should_stop():
                    reason = "durée prévue atteinte"
                    break
        except KeyboardInterrupt:
            reason = "arrêt manuel (Ctrl+C)"
        except Exception as e:  # noqa: BLE001
            reason = f"erreur : {e}"
            ctl.journal.write(f"Erreur de traitement : {e} — {traceback.format_exc(limit=3)}", "erreur")
        finally:
            try:
                src.release()
            except Exception:  # noqa: BLE001
                pass
            run = ctl.finish(det, reason, make_report=make_report)
    return run
