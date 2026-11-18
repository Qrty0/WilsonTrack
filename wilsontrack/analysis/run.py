"""Un « enregistrement » analysé (une vidéo = une condition expérimentale = un essai).

Les résultats sont sauvegardés dans un dossier :
    <sortie>/<date> <condition> essai-NN/
        run.json          métadonnées + résumé + configuration utilisée
        traces.csv        une ligne par trace
        traces.xlsx       idem (Excel)
        vignettes/        images des traces
        rapport.html      rapport illustré avec conclusions

Longues durées (nuit entière, vidéo de plusieurs heures) : les vignettes des traces anciennes
sont écrites sur disque au fil de l'eau (« délestage », voir ``limit_memory``) pour que la
mémoire reste bornée ; ``RunResult.save`` les recopie (ou les déplace) dans ``vignettes/``.
"""
from __future__ import annotations

import atexit
import copy
import csv
import inspect
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Callable, ClassVar

import cv2
import numpy as np

from .. import __version__
from ..config import AppConfig
from ..detection import TrackDetector
from ..detection.events import EXPORT_COLUMNS, TrackEvent

# Au-delà de cette durée de vidéo, analyze_video passe en mode « mémoire bornée »
LONG_VIDEO_S = 20 * 60.0
# Nombre de vignettes de traces valides gardées en mémoire en mode « mémoire bornée »
KEEP_RECENT_THUMBS = 300
# Nombre de détections rejetées gardées en mémoire (repli si le détecteur n'a pas keep_invalid_events)
KEEP_INVALID = 200
# Délestage des vignettes en mode « mémoire bornée » : une fois par minute de vidéo
SPILL_EVERY_S = 60.0


@dataclass
class RunResult:
    name: str
    source: str
    condition: str
    sample_type: str
    sample_mass_g: float
    trial: int
    date: str
    fps: float
    n_frames: int
    duration_s: float            # durée réellement analysée (hors échauffement / perturbations)
    frame_w: int
    frame_h: int
    px_per_mm: float
    calibrated: bool
    roi_area_mm2: float
    config: dict
    events: list = field(default_factory=list)       # list[TrackEvent] (en mémoire)
    rows: list = field(default_factory=list)         # list[dict] (chargé depuis disque)
    expansion_times: list = field(default_factory=list)
    rejected: int = 0
    processing_s: float = 0.0
    folder: str = ""
    version: str = __version__
    notes: str = ""
    save_warnings: list = field(default_factory=list)
    # Évolution au cours du temps (une entrée par minute d'analyse, voir TrackDetector.timeline) :
    # {"t_s", "wall_s", "frames", "disturbed_frames", "noise", "lit", "counts", "rejected"}
    timeline: list = field(default_factory=list)
    # Session de nuit (vide sinon) : configuration + déroulement (voir wilsontrack.longrun)
    session: dict = field(default_factory=dict)
    # Problèmes rencontrés pendant l'analyse (images illisibles sautées, vidéo tronquée…)
    analysis_warnings: list = field(default_factory=list)

    # Résultat provisoire (point de sauvegarde d'une longue session) : « partiel » dans run.json
    partial: ClassVar[bool] = False

    # ----------------------------------------------------------------------
    def table(self) -> list[dict]:
        if self.events:
            return [e.to_row() for e in self.events if e.valid]
        return self.rows

    def counts(self) -> dict:
        c: dict[str, int] = {}
        for r in self.table():
            c[r["classe"]] = c.get(r["classe"], 0) + 1
        return c

    @property
    def total(self) -> int:
        return len(self.table())

    def meta(self) -> dict:
        # Pas de asdict() : il copierait en profondeur toutes les traces (vignettes, masques)
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("events", "rows")}
        d["config"] = copy.deepcopy(self.config)
        return d

    # ----------------------------------------------------------------------
    def save(self, out_dir: str | Path | None = None, thumbnails: bool = True, partial: bool = False,
             xlsx: bool = True, folder: str | Path | None = None) -> Path:
        """Enregistre dans ``<out_dir>/<nom>/`` (ou directement dans ``folder``) et renvoie le dossier.

        partial=True : point de sauvegarde provisoire (« partiel »: true dans run.json) ; les
        vignettes déjà présentes ne sont pas réécrites. xlsx=False : pas de traces.xlsx (plus
        rapide pour les sauvegardes intermédiaires). Toutes les écritures de fichiers de données
        sont atomiques (fichier temporaire puis remplacement)."""
        if folder is not None:
            folder = Path(folder)
        elif out_dir is not None:
            folder = Path(out_dir) / safe_name(self.name)
        else:
            raise ValueError("Indiquez le dossier de sortie (out_dir ou folder).")
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = str(folder)
        rows = self.table()
        problems: list[str] = []

        def atomic(path: Path, writer):
            """Écrit dans un fichier temporaire puis remplace : jamais de fichier à moitié écrit."""
            tmp = path.with_name(path.stem + ".tmp" + path.suffix)
            try:
                writer(tmp)
                os.replace(tmp, path)
            except PermissionError:
                problems.append(f"« {path.name} » est ouvert dans un autre logiciel (Excel ?) : fermez-le puis "
                                "sauvegardez de nouveau.")
            except Exception as ex:
                problems.append(f"« {path.name} » : {ex}")
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

        def write_csv(p: Path):
            with open(p, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

        atomic(folder / "traces.csv", write_csv)
        if xlsx:
            atomic(folder / "traces.xlsx", lambda p: _write_xlsx(p, rows, self))
        if thumbnails and self.events:
            try:
                self._save_thumbnails(folder / "vignettes", skip_existing=partial, problems=problems)
            except Exception as ex:  # noqa: BLE001 - les données principales doivent être sauvegardées
                problems.append(f"Vignettes : {ex}")
        meta = self.meta()
        meta["summary"] = self.counts()
        meta["partiel"] = bool(partial)
        atomic(folder / "run.json", lambda p: p.write_text(json.dumps(_jsonable(meta), indent=2, ensure_ascii=False),
                                                           encoding="utf-8"))
        self.partial = bool(partial)
        self.save_warnings = problems
        return folder

    def _save_thumbnails(self, td: Path, skip_existing: bool, problems: list[str]) -> None:
        """Vignettes des traces valides : encodées depuis la mémoire, ou recopiées / déplacées si elles
        ont été délestées sur disque (``extra['thumb_path']``). Une vignette dont la classe a changé
        (reclassification finale, correction manuelle) est renommée ; l'ancien nom est supprimé."""
        td.mkdir(parents=True, exist_ok=True)
        existing: dict[int, set[str]] = {}
        for p in td.iterdir():
            m = re.match(r"trace_(\d+)_.+\.png$", p.name)
            if m:
                existing.setdefault(int(m.group(1)), set()).add(p.name)
        failed = 0
        for e in self.events:
            if not e.valid or e.id is None or e.id < 0:
                continue
            name = f"trace_{e.id:05d}_{e.final_cls}.png"
            target = td / name
            have = existing.get(e.id, set())
            th = e.thumb
            try:
                if th is not None:
                    if not (skip_existing and name in have):
                        ok, buf = cv2.imencode(".png", th)
                        if not ok:
                            raise IOError("encodage PNG impossible")
                        buf.tofile(str(target))
                else:
                    src = (e.extra or {}).get("thumb_path")
                    if src and os.path.isfile(src):
                        srcp = Path(src)
                        if not (target.exists() and os.path.samefile(srcp, target)):
                            if os.path.samefile(srcp.parent, td):
                                os.replace(srcp, target)                # même dossier : renommage
                            elif _in_spill_dir(srcp):
                                shutil.move(str(srcp), str(target))     # dossier temporaire : déplacement
                                _forget_empty_spill_dir(srcp.parent)
                            else:
                                shutil.copy2(srcp, target)
                        e.extra["thumb_path"] = str(target)
                    elif name not in have:
                        continue
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            for old in have - {name}:
                try:
                    (td / old).unlink()
                except OSError:
                    pass
        if failed:
            problems.append(f"{failed} vignette(s) n'ont pas pu être enregistrées.")

    @classmethod
    def load(cls, folder: str | Path) -> "RunResult":
        folder = Path(folder)
        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        meta.pop("summary", None)
        partial = bool(meta.pop("partiel", False))
        known = {f for f in cls.__dataclass_fields__}
        r = cls(**{k: v for k, v in meta.items() if k in known})
        r.partial = partial
        if not isinstance(r.timeline, list):
            r.timeline = []
        if not isinstance(r.session, dict):
            r.session = {}
        r.folder = str(folder)
        rows = []
        with open(folder / "traces.csv", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                rows.append(_parse_row(row))
        r.rows = rows
        return r


def _parse_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if v is None:
            continue
        if k in ("classe", "classe_manuelle", "raison", "pourquoi", "methode"):
            out[k] = v
            continue
        try:
            fv = float(v)
            out[k] = int(fv) if k in ("id", "images_vues", "embranchements", "touche_bord") else fv
        except ValueError:
            out[k] = v if v != "" else None
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _write_xlsx(path: Path, rows: list[dict], run: RunResult):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "Résumé"
    ws.append(["WilsonTrack — résumé de l'enregistrement"])
    ws["A1"].font = Font(bold=True, size=14)
    for k, v in [("Nom", run.name), ("Condition", run.condition), ("Échantillon", run.sample_type),
                 ("Masse (g)", run.sample_mass_g), ("Essai", run.trial), ("Date", run.date),
                 ("Durée analysée (s)", round(run.duration_s, 2)), ("Images", run.n_frames),
                 ("Calibré", "oui" if run.calibrated else "non (échelle supposée)"),
                 ("px/mm", round(run.px_per_mm, 4)), ("Surface analysée (cm²)", round(run.roi_area_mm2 / 100, 1)),
                 ("Nombre total de traces", len(rows))]:
        ws.append([k, v])
    ws.append([])
    ws.append(["Classe", "Nombre", "Taux (/min)", "Incertitude (/min)"])
    for c in ws[ws.max_row]:
        c.font = Font(bold=True)
    cnt = run.counts()
    dmin = max(run.duration_s, 1e-9) / 60.0
    for k, n in sorted(cnt.items()):
        ws.append([k, n, round(n / dmin, 3), round(math.sqrt(n) / dmin, 3)])
    ws2 = wb.create_sheet("Traces")
    ws2.append(EXPORT_COLUMNS)
    for c in ws2[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2F4F7F")
    for r in rows:
        ws2.append([r.get(k, "") for k in EXPORT_COLUMNS])
    ws2.freeze_panes = "A2"
    wb.save(path)


def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "-", s).strip()
    return s or "enregistrement"


def default_run_name(condition: str, trial: int, date: datetime | None = None) -> str:
    d = (date or datetime.now()).strftime("%Y-%m-%d")
    return f"{d} {condition} essai-{trial:02d}"


# =========================================================================== #
# Mémoire bornée pour les longues durées (nuit entière, vidéo de plusieurs heures)
# =========================================================================== #
_SPILL_LOCK = threading.Lock()
_SPILL_DIRS: set[str] = set()


def make_spill_dir() -> Path:
    """Dossier temporaire pour les vignettes délestées (supprimé à la fermeture du programme ;
    ``RunResult.save`` y DÉPLACE les vignettes vers le dossier de l'enregistrement)."""
    d = Path(tempfile.mkdtemp(prefix="wilsontrack_vignettes_"))
    with _SPILL_LOCK:
        _SPILL_DIRS.add(os.path.normcase(str(d.resolve())))
    return d


def _in_spill_dir(p: Path) -> bool:
    try:
        parent = os.path.normcase(str(Path(p).parent.resolve()))
    except OSError:
        return False
    with _SPILL_LOCK:
        return parent in _SPILL_DIRS


def _forget_empty_spill_dir(d: Path) -> None:
    try:
        if not any(Path(d).iterdir()):
            Path(d).rmdir()
            with _SPILL_LOCK:
                _SPILL_DIRS.discard(os.path.normcase(str(Path(d).resolve())))
    except OSError:
        pass


@atexit.register
def _cleanup_spill_dirs() -> None:  # pragma: no cover - exécuté à la sortie du programme
    with _SPILL_LOCK:
        dirs = list(_SPILL_DIRS)
        _SPILL_DIRS.clear()
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def spill_thumbnails(det, folder: str | Path, keep_recent: int = KEEP_RECENT_THUMBS) -> int:
    """Écrit sur disque les vignettes des traces valides anciennes et les libère de la mémoire.

    Utilise ``det.spill_thumbnails`` (moteur) s'il existe ; sinon fait le même travail :
    ``<folder>/trace_{id:05d}_{classe}.png``, ``e.thumb = e.thumb_mask = None`` et
    ``e.extra['thumb_path']``. Les ``keep_recent`` traces les plus récentes gardent leur vignette.
    Renvoie le nombre de vignettes délestées."""
    fn = getattr(det, "spill_thumbnails", None)
    if callable(fn):
        try:
            return int(fn(folder, keep_recent=keep_recent) or 0)
        except TypeError:
            return int(fn(folder, keep_recent) or 0)
    return spill_event_thumbnails(det.events, folder, keep_recent)


def spill_event_thumbnails(events, folder: str | Path, keep_recent: int = KEEP_RECENT_THUMBS) -> int:
    cands = [e for e in events if e.valid and e.id is not None and e.id >= 0 and e.thumb is not None]
    if len(cands) <= keep_recent:
        return 0
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in cands[:len(cands) - max(0, keep_recent)]:
        th = e.thumb
        if th is None:
            continue
        p = folder / f"trace_{e.id:05d}_{e.final_cls}.png"
        try:
            if not p.exists():
                ok, buf = cv2.imencode(".png", th)
                if not ok:
                    continue
                buf.tofile(str(p))
        except Exception:  # noqa: BLE001 - disque plein : on garde la vignette en mémoire
            break
        e.extra["thumb_path"] = str(p)
        e.thumb = None
        e.thumb_mask = None
        n += 1
    return n


def prune_invalid_events(det, keep: int = KEEP_INVALID) -> int:
    """Ne garde en mémoire que les ``keep`` détections rejetées les plus récentes (elles restent
    comptées dans ``det.rejected``). Renvoie le nombre de détections retirées."""
    evs = det.events
    n_inv = sum(1 for e in evs if not e.valid)
    drop = n_inv - max(0, keep)
    if drop <= 0:
        return 0
    out = []
    k = drop
    for e in evs:
        if k > 0 and not e.valid:
            k -= 1
            continue
        out.append(e)
    det.events = out
    return drop


def limit_memory(det, folder: str | Path, keep_recent: int | None = None, keep_invalid: int | None = None) -> int:
    """Borne la mémoire d'un long enregistrement : détections rejetées non conservées
    (``det.keep_invalid_events = False`` si le moteur le permet, sinon élagage) et vignettes
    anciennes délestées dans ``folder``. Renvoie le nombre de vignettes délestées.
    Par défaut : KEEP_RECENT_THUMBS vignettes et KEEP_INVALID rejets gardés en mémoire."""
    keep_recent = KEEP_RECENT_THUMBS if keep_recent is None else keep_recent
    keep_invalid = KEEP_INVALID if keep_invalid is None else keep_invalid
    if hasattr(det, "keep_invalid_events"):
        det.keep_invalid_events = False
    else:
        prune_invalid_events(det, keep_invalid)
    # Les rejets déjà gardés (diagnostic) n'ont pas besoin de leur image : c'est le plus lourd
    for e in det.events:
        if not e.valid and (e.thumb is not None or e.thumb_mask is not None):
            e.thumb = None
            e.thumb_mask = None
    return spill_thumbnails(det, folder, keep_recent)


def _progress_caller(cb):
    """progress(fraction) ou progress(fraction, eta_s) selon la signature de la fonction."""
    if cb is None:
        return None
    two = False
    try:
        ps = list(inspect.signature(cb).parameters.values())
        pos = [p for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        two = any(p.kind == p.VAR_POSITIONAL for p in ps) or len(pos) >= 2
    except (TypeError, ValueError):
        two = False
    if two:
        return lambda f, eta: cb(f, eta)
    return lambda f, eta: cb(f)


def analyze_video(path: str, cfg: AppConfig, progress: Callable[..., bool | None] | None = None,
                  on_frame: Callable | None = None, max_frames: int = 0,
                  start_s: float = 0.0, end_s: float = 0.0, spill_dir: str | Path | None = None,
                  long_video_s: float = LONG_VIDEO_S) -> tuple[RunResult, TrackDetector]:
    """Analyse complète d'un fichier vidéo (sans interface).

    progress(fraction) — ou progress(fraction, eta_s) si la fonction accepte deux arguments
    (eta_s = temps restant estimé en secondes, −1 si inconnu) — est appelée toutes les 10 images ;
    elle peut renvoyer True pour interrompre l'analyse (les traces déjà vues sont gardées).
    on_frame(frame, result, detector) est appelé pour chaque image (aperçu).

    Vidéo de plus de ``long_video_s`` secondes (20 min) : mémoire bornée — détections rejetées
    non conservées et vignettes anciennes délestées dans ``spill_dir`` (dossier temporaire par
    défaut, vidé par ``RunResult.save``). Images illisibles au milieu du fichier : sautées et
    signalées dans ``run.analysis_warnings``.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 <= fps <= 1000.0):
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise IOError("La vidéo ne contient aucune image lisible.")
    start_frame = 0
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise IOError("Le début demandé dépasse la fin de la vidéo.")
        start_frame = int(round(start_s * fps))
    if end_s and end_s <= start_s:
        end_s = 0.0
    h, w = frame.shape[:2]
    det = TrackDetector(cfg, fps, (h, w))
    t0 = time.perf_counter()
    i = 0
    n_limit = max(0, n_total - start_frame) if n_total > 0 else 0
    if end_s > 0:
        n_limit = int((end_s - start_s) * fps)
    if max_frames:
        n_limit = min(n_limit, max_frames) if n_limit else max_frames
    call = _progress_caller(progress)
    expected = n_limit if n_limit else max(0, n_total - start_frame)
    long_mode = expected > 0 and expected / fps > long_video_s
    spill = None
    spill_every = max(1, int(round(SPILL_EVERY_S * fps)))   # une fois par minute de vidéo
    skipped = 0
    cancelled = False
    warnings: list[str] = []
    while ok:
        t = start_s + i / fps
        res = det.process(frame, t)
        if on_frame is not None:
            on_frame(frame, res, det)
        i += 1
        if not long_mode and i / fps > long_video_s:
            long_mode = True                           # durée inconnue au départ : bascule en cours de route
        if long_mode and i % spill_every == 0:
            if spill is None:
                spill = Path(spill_dir) if spill_dir else make_spill_dir()
            limit_memory(det, spill)
        if call is not None and i % 10 == 0:
            frac = min(1.0, i / n_limit) if n_limit else 0.0
            el = time.perf_counter() - t0
            eta = el * (1.0 - frac) / frac if frac > 0 else -1.0
            if call(frac, eta):
                cancelled = True
                break
        if n_limit and i >= n_limit:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            # Image illisible au milieu d'un fichier (paquet corrompu) : on essaie de continuer
            pos = start_frame + i + skipped
            if n_total > 0 and pos < n_total - 1:
                for _ in range(25):
                    skipped += 1
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
                ok = bool(ok and frame is not None)
    cap.release()
    if skipped:
        warnings.append(f"{skipped} image(s) illisible(s) sautée(s) : fichier vidéo peut-être endommagé.")
    if not cancelled and n_limit and n_total > 0 and not max_frames and not end_s:
        frac_read = (i + skipped) / max(1, n_limit)
        if frac_read < 0.98 and (n_limit - i - skipped) / fps > 2.0:
            warnings.append(f"La lecture s'est arrêtée à {100 * frac_read:.0f} % de la durée annoncée de la vidéo "
                            "(fichier tronqué ou enregistrement interrompu ?).")
    det.finalize_run()
    run = build_run_result(det, cfg, str(path), i, fps, (h, w))
    run.processing_s = time.perf_counter() - t0
    run.analysis_warnings = warnings
    return run, det


def build_run_result(det: TrackDetector, cfg: AppConfig, source: str, n_frames: int, fps: float,
                     frame_hw: tuple[int, int], *, name: str | None = None, session: dict | None = None,
                     valid_only: bool = False) -> RunResult:
    """RunResult à partir d'un détecteur (finalisé ou non : sans finalize_run(), c'est un état
    provisoire — traces encore actives non comptées, classes non réévaluées).

    name : nom de l'enregistrement (défaut « <date> <condition> essai-NN ») ; session : infos de
    session de nuit ; valid_only : ne copier que les traces valides (points de sauvegarde)."""
    ri = cfg.run
    # Temps réellement analysé (horloge d'analyse : hors apprentissage du fond, perturbations
    # et pauses ; pour une caméra, mesuré avec les horodatages réels et non la cadence nominale)
    dur = det.analyzed_s if det.analyzed_s > 0 else det.n_frames_processed / fps
    roi_px_full = det.pre.roi_area_px / (det.scale ** 2)
    roi_mm2 = roi_px_full / det.px_per_mm_full ** 2
    events = [e for e in det.events if e.valid] if valid_only else list(det.events)
    timeline = []
    for entry in getattr(det, "timeline", None) or []:
        try:
            d = dict(entry)
            if isinstance(d.get("counts"), dict):
                d["counts"] = dict(d["counts"])
            timeline.append(d)
        except (TypeError, ValueError):
            continue
    return RunResult(
        name=name or default_run_name(ri.condition, ri.trial), source=source, condition=ri.condition,
        sample_type=ri.sample_type, sample_mass_g=ri.sample_mass_g, trial=ri.trial,
        date=datetime.now().isoformat(timespec="seconds"),
        fps=fps, n_frames=n_frames, duration_s=dur, frame_w=frame_hw[1], frame_h=frame_hw[0],
        px_per_mm=det.px_per_mm_full, calibrated=det.calibrated, roi_area_mm2=roi_mm2,
        config=cfg.to_dict(), events=events, expansion_times=list(det.exp_times),
        rejected=det.rejected, notes=ri.notes, timeline=timeline, session=copy.deepcopy(session or {}),
    )
