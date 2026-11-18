"""Analyse automatique des vidéos REÇUES : lot de fichiers et dossier surveillé.

- ``BatchConfig`` : réglages du lot (dict de la boîte de dialogue) ;
- ``BatchJob`` / ``VideoQueue`` : file d'attente (une vidéo = un travail) ;
- ``FolderWatcher`` : surveille un dossier (toutes les 5 s) ; une vidéo est prise quand sa taille
  n'a pas changé depuis 10 s et qu'elle peut être ouverte ; les fichiers déjà traités sont
  mémorisés dans ``<dossier>/.wilsontrack_traitees.json`` (rien n'est analysé deux fois) ; les
  noms temporaires (*.part, *.crdownload, ~*…) sont ignorés ;
- ``BatchRunner`` (fil d'exécution, sans Qt) : analyse les vidéos une par une
  (``analyze_video`` → ``RunResult.save`` → rapport HTML) et tient à jour
  ``<sortie>/resume_lot.csv`` (+ ``resume_lot.xlsx``) : une ligne par vidéo ;
- ``QtBatchController`` (QObject, créé à la demande : ce module n'importe pas Qt) : signaux
  ``jobs_changed(list)``, ``job_finished(dict)``, ``all_finished()``, ``error(str)``, ``info(str)``
  et méthodes ``start(batch_dict, app_cfg=None)``, ``stop()``, ``add_files(list)``.
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Callable

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm")
STATE_FILE = ".wilsontrack_traitees.json"
SUMMARY_CSV = "resume_lot.csv"
SUMMARY_XLSX = "resume_lot.xlsx"

STATUS_WAITING = "en attente"
STATUS_RUNNING = "en cours"
STATUS_DONE = "terminé"
STATUS_ERROR = "erreur"
STATUS_CANCELLED = "annulé"

# Fichiers en cours de téléchargement / de copie / temporaires
PARTIAL_SUFFIXES = (".part", ".crdownload", ".download", ".partial", ".tmp", ".temp", ".!ut", ".opdownload",
                    ".filepart", ".crswap")

CLASSES = ("alpha", "beta", "muon", "proton", "fine", "incertain")
SUMMARY_COLUMNS = (
    ["video", "fichier", "statut", "condition", "echantillon", "masse_g", "duree_analysee_min", "images",
     "calibre", "px_par_mm", "surface_cm2"]
    + list(CLASSES)
    + ["total", "taux_total_par_min", "incertitude_total_par_min"]
    + [f"taux_{c}_par_min" for c in CLASSES]
    + ["dossier", "date_analyse", "erreur", "remarques"]
)


# =========================================================================== #
# Noms de fichiers
# =========================================================================== #
def is_video_file(name: str | Path) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTS


def is_partial_name(name: str | Path) -> bool:
    """Nom d'un fichier temporaire ou incomplet (à ignorer) ?"""
    n = Path(name).name
    low = n.lower()
    if not n or n.startswith(("~", ".", "~$")):
        return True
    if low.endswith(PARTIAL_SUFFIXES):
        return True
    return ".tmp." in low or ".part." in low or ".crdownload" in low


def video_probe(path: str | Path) -> bool:
    """La vidéo peut-elle être ouverte et sa première image lue ?"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ok, fr = cap.read()
        return bool(ok and fr is not None)
    except Exception:  # noqa: BLE001
        return False
    finally:
        cap.release()


def _can_open(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# =========================================================================== #
# Réglages d'un lot et travaux
# =========================================================================== #
@dataclass
class BatchConfig:
    files: list = field(default_factory=list)
    watch_folder: str = ""
    condition: str = ""
    sample: str = ""
    mass_g: float = 0.0
    px_per_mm: float = 0.0
    output_dir: str = ""
    make_reports: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "BatchConfig":
        """Clés inconnues ignorées, clés absentes → valeurs par défaut."""
        d = dict(d or {})
        out = cls()
        files = d.get("files") or []
        if isinstance(files, (str, Path)):
            files = [files]
        out.files = [str(f) for f in files if str(f).strip()]
        for k in ("watch_folder", "condition", "sample", "output_dir"):
            if d.get(k) is not None:
                setattr(out, k, str(d[k]).strip())
        for k in ("mass_g", "px_per_mm"):
            try:
                v = float(str(d.get(k, 0.0)).replace(",", "."))
                setattr(out, k, v if math.isfinite(v) and v > 0 else 0.0)
            except (TypeError, ValueError):
                pass
        if "make_reports" in d:
            out.make_reports = bool(d["make_reports"])
        return out

    def to_dict(self) -> dict:
        return {f.name: (list(getattr(self, f.name)) if f.name == "files" else getattr(self, f.name))
                for f in fields(self)}


@dataclass
class BatchJob:
    path: str
    status: str = STATUS_WAITING
    progress: float = 0.0
    eta_s: float = -1.0
    result_folder: str = ""
    counts: dict = field(default_factory=dict)
    error: str = ""
    total: int = 0
    duration_s: float = 0.0
    from_watch: bool = False
    started_at: str = ""
    ended_at: str = ""
    warnings: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> dict:
        """Dict d'affichage : path, name, status, progress, eta_s, result_folder, counts, error
        (+ total, duration_s, from_watch, warnings, started_at, ended_at)."""
        return {"path": self.path, "name": self.name, "status": self.status, "progress": float(self.progress),
                "eta_s": float(self.eta_s), "result_folder": self.result_folder, "counts": dict(self.counts),
                "error": self.error, "total": int(self.total), "duration_s": float(self.duration_s),
                "from_watch": self.from_watch, "warnings": list(self.warnings), "started_at": self.started_at,
                "ended_at": self.ended_at}


class VideoQueue:
    """File d'attente des vidéos à analyser (utilisable depuis plusieurs fils)."""

    def __init__(self):
        self._jobs: list[BatchJob] = []
        self._lock = threading.RLock()

    def add(self, paths, from_watch: bool = False) -> list[BatchJob]:
        """Ajoute des vidéos ; une vidéo déjà en attente ou en cours n'est pas ajoutée deux fois."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        added = []
        with self._lock:
            busy = {_norm(j.path) for j in self._jobs if j.status in (STATUS_WAITING, STATUS_RUNNING)}
            for p in paths:
                if not str(p).strip():
                    continue
                key = _norm(p)
                if key in busy:
                    continue
                busy.add(key)
                job = BatchJob(path=str(Path(p)), from_watch=from_watch)
                self._jobs.append(job)
                added.append(job)
        return added

    def next_waiting(self) -> BatchJob | None:
        with self._lock:
            for j in self._jobs:
                if j.status == STATUS_WAITING:
                    return j
        return None

    def jobs(self) -> list[BatchJob]:
        with self._lock:
            return list(self._jobs)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs]

    def pending(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs if j.status in (STATUS_WAITING, STATUS_RUNNING))

    def cancel_waiting(self) -> int:
        n = 0
        with self._lock:
            for j in self._jobs:
                if j.status == STATUS_WAITING:
                    j.status = STATUS_CANCELLED
                    n += 1
        return n

    def clear_finished(self) -> None:
        with self._lock:
            self._jobs = [j for j in self._jobs if j.status in (STATUS_WAITING, STATUS_RUNNING)]


# =========================================================================== #
# Dossier surveillé
# =========================================================================== #
class FolderWatcher:
    """Repère les nouvelles vidéos COMPLÈTES d'un dossier (pas les sous-dossiers).

    scan() renvoie les vidéos prêtes : extension vidéo, nom non temporaire, taille (et date de
    modification) inchangée depuis ``stable_s`` secondes, fichier ouvrable et lisible par
    ``probe`` (une vidéo encore illisible après ``give_up_unreadable_s`` est quand même confiée à
    l'analyse, qui la signalera en erreur). Une vidéo déjà traitée (même nom, même taille) n'est
    jamais reprise ; si elle est remplacée par une version différente, elle est analysée de nouveau.
    """

    def __init__(self, folder: str | Path, poll_s: float = 5.0, stable_s: float = 10.0,
                 state_path: str | Path | None = None, probe: Callable[[Path], bool] | None = video_probe,
                 give_up_unreadable_s: float = 120.0, clock: Callable[[], float] = time.monotonic):
        self.folder = Path(folder)
        self.poll_s = float(poll_s)
        self.stable_s = float(stable_s)
        self.state_path = Path(state_path) if state_path else self.folder / STATE_FILE
        self.probe = probe
        self.give_up_unreadable_s = float(give_up_unreadable_s)
        self._clock = clock
        self._seen: dict[str, list] = {}          # nom → [taille, date modif., stable depuis, illisible depuis]
        self._handed: set[str] = set()
        self.last_scan = -math.inf
        self.last_error = ""
        self.state = self._load_state()

    # ----------------------------------------------------------------- état --
    def _load_state(self) -> dict:
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("fichiers"), dict):
                return d
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            self.last_error = f"Fichier d'état illisible ({self.state_path.name}) : {e}"
        return {"version": 1, "fichiers": {}}

    def _save_state(self) -> None:
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError as e:
            self.last_error = f"Impossible d'enregistrer la liste des vidéos traitées : {e}"
            try:
                tmp.unlink()
            except OSError:
                pass

    def is_processed(self, name: str, size: int | None = None) -> bool:
        rec = self.state["fichiers"].get(Path(name).name)
        if not rec:
            return False
        if size is not None and rec.get("taille") is not None and int(rec["taille"]) != int(size):
            return False                                  # fichier remplacé par une autre version
        return True

    def mark_processed(self, path: str | Path, status: str = STATUS_DONE, result_folder: str = "",
                       error: str = "") -> None:
        p = Path(path)
        try:
            st = p.stat()
            size, mtime = int(st.st_size), float(st.st_mtime)
        except OSError:
            size, mtime = None, None
        self.state["fichiers"][p.name] = {"taille": size, "modifie": mtime, "statut": status,
                                          "dossier": str(result_folder), "erreur": error, "date": _now_iso()}
        self._handed.discard(p.name)
        self._seen.pop(p.name, None)
        self._save_state()

    def release(self, path: str | Path) -> None:
        """Vidéo rendue (analyse annulée) : elle sera proposée de nouveau."""
        self._handed.discard(Path(path).name)

    @property
    def pending_count(self) -> int:
        """Vidéos vues mais pas encore prêtes (taille qui change, fichier pas encore lisible)."""
        return sum(1 for k, v in self._seen.items() if k not in self._handed and v[0] > 0)

    # ------------------------------------------------------------- balayage --
    def due(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        return now - self.last_scan >= self.poll_s

    def scan(self, now: float | None = None) -> list[Path]:
        now = self._clock() if now is None else now
        self.last_scan = now
        try:
            entries = list(os.scandir(self.folder))
        except OSError as e:
            self.last_error = f"Dossier surveillé inaccessible : {e}"
            return []
        ready: list[Path] = []
        present = set()
        for de in entries:
            try:
                if not de.is_file():
                    continue
                name = de.name
                if is_partial_name(name) or not is_video_file(name):
                    continue
                present.add(name)
                if name in self._handed:
                    continue
                st = de.stat()
                size, mtime = int(st.st_size), float(st.st_mtime)
            except OSError:
                continue
            if self.is_processed(name, size):
                self._seen.pop(name, None)
                continue
            prev = self._seen.get(name)
            if prev is None or prev[0] != size or abs(prev[1] - mtime) > 1e-6:
                self._seen[name] = [size, mtime, now, None]
                continue
            if size <= 0 or now - prev[2] < self.stable_s:
                continue
            path = Path(de.path)
            if not _can_open(path):
                continue
            if self.probe is not None:
                try:
                    readable = bool(self.probe(path))
                except Exception:  # noqa: BLE001
                    readable = False
                if not readable:
                    if prev[3] is None:
                        prev[3] = now
                    if now - prev[3] < self.give_up_unreadable_s:
                        continue
            self._handed.add(name)
            ready.append(path)
        for k in list(self._seen):
            if k not in present:
                del self._seen[k]
        return sorted(ready, key=lambda p: p.name.lower())


# =========================================================================== #
# Exécution d'un lot
# =========================================================================== #
def _unique_name(out_dir: Path, name: str) -> str:
    from .analysis.run import safe_name
    cand, k = name, 2
    while (out_dir / safe_name(cand)).exists():
        cand = f"{name} ({k})"
        k += 1
    return cand


def _fmt_dec(v, d: int = 3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return round(v, d) if math.isfinite(v) else ""


def summary_row(job: BatchJob, run=None) -> dict:
    """Ligne de resume_lot.csv pour une vidéo (analysée ou en erreur)."""
    row = {k: "" for k in SUMMARY_COLUMNS}
    row.update({"video": job.name, "fichier": job.path, "statut": job.status, "dossier": job.result_folder,
                "date_analyse": job.ended_at or _now_iso(), "erreur": job.error,
                "remarques": " | ".join(job.warnings)})
    if run is not None:
        from .analysis.stats import rate
        cnt = run.counts()
        dur = float(run.duration_s or 0.0)
        row.update({"condition": run.condition, "echantillon": run.sample_type, "masse_g": run.sample_mass_g,
                    "duree_analysee_min": _fmt_dec(dur / 60.0, 3), "images": run.n_frames,
                    "calibre": "oui" if run.calibrated else "non", "px_par_mm": _fmt_dec(run.px_per_mm, 4),
                    "surface_cm2": _fmt_dec(run.roi_area_mm2 / 100.0, 1)})
        tot = 0
        for c in CLASSES:
            n = int(cnt.get(c, 0))
            tot += n
            row[c] = n
            row[f"taux_{c}_par_min"] = _fmt_dec(rate(n, dur).per_min, 3)
        extra = sum(v for k, v in cnt.items() if k not in CLASSES)
        tot += extra
        r = rate(tot, dur)
        row.update({"total": tot, "taux_total_par_min": _fmt_dec(r.per_min, 3),
                    "incertitude_total_par_min": _fmt_dec(r.err, 3)})
    return row


class BatchRunner(threading.Thread):
    """Analyse séquentielle des vidéos d'un lot (et d'un dossier surveillé), sans Qt.

    Rappels (appelés depuis CE fil) : on_jobs_changed(list[dict]) (au plus 2 fois/s pendant une
    analyse), on_job_finished(dict), on_all_finished(), on_error(str), on_info(str).
    Avec un dossier surveillé, le fil tourne jusqu'à ``stop()`` ; sinon il s'arrête quand la file
    est vide. ``stop()`` interrompt l'analyse en cours (statut « annulé », rien n'est enregistré)."""

    EMIT_EVERY_S = 0.5

    def __init__(self, batch: BatchConfig | dict, app_cfg=None, queue: VideoQueue | None = None,
                 on_jobs_changed: Callable[[list], None] | None = None,
                 on_job_finished: Callable[[dict], None] | None = None,
                 on_all_finished: Callable[[], None] | None = None,
                 on_error: Callable[[str], None] | None = None, on_info: Callable[[str], None] | None = None,
                 poll_s: float = 5.0, stable_s: float = 10.0, probe: Callable | None = video_probe,
                 analyze: Callable | None = None, watch_once: bool = False):
        """watch_once : dossier surveillé balayé jusqu'à ce que toutes les vidéos présentes soient
        traitées, puis arrêt (au lieu d'attendre indéfiniment de nouvelles vidéos)."""
        super().__init__(name="wilsontrack-lot", daemon=True)
        self.watch_once = bool(watch_once)
        from .config import AppConfig, default_output_dir
        self.batch = batch if isinstance(batch, BatchConfig) else BatchConfig.from_dict(batch)
        self.app_cfg = (app_cfg or AppConfig()).copy()
        if self.batch.output_dir:
            self.out_dir = Path(self.batch.output_dir)
        elif self.app_cfg.output_dir:
            self.out_dir = Path(self.app_cfg.output_dir)
        else:
            self.out_dir = default_output_dir()
        self.queue = queue or VideoQueue()
        self.on_jobs_changed = on_jobs_changed
        self.on_job_finished = on_job_finished
        self.on_all_finished = on_all_finished
        self.on_error = on_error
        self.on_info = on_info
        self._analyze = analyze
        self._stop_evt = threading.Event()
        self._last_emit = 0.0
        self.current: BatchJob | None = None
        self.watcher: FolderWatcher | None = None
        if self.batch.watch_folder:
            self.watcher = FolderWatcher(self.batch.watch_folder, poll_s=poll_s, stable_s=stable_s, probe=probe)
        self._summary: dict[str, dict] | None = None
        self.queue.add(self.batch.files)

    # ------------------------------------------------------------- commandes --
    def stop(self) -> None:
        self._stop_evt.set()

    @property
    def stopping(self) -> bool:
        return self._stop_evt.is_set()

    def add_files(self, paths) -> list[dict]:
        added = self.queue.add(paths)
        self._emit_jobs(force=True)
        return [j.to_dict() for j in added]

    # -------------------------------------------------------------- rappels --
    def _call(self, fn, *args):
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - un rappel défaillant ne doit pas arrêter le lot
            pass

    def _emit_jobs(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < self.EMIT_EVERY_S:
            return
        self._last_emit = now
        self._call(self.on_jobs_changed, self.queue.snapshot())

    def _info(self, msg: str) -> None:
        self._call(self.on_info, msg)

    def _error(self, msg: str) -> None:
        self._call(self.on_error, msg)

    # --------------------------------------------------------------- boucle --
    def run(self) -> None:
        try:
            self._emit_jobs(force=True)
            if self.watcher is not None:
                if not self.watcher.folder.is_dir():
                    self._error(f"Dossier surveillé introuvable : {self.watcher.folder}")
                    self.watcher = None
                else:
                    poll = f"{self.watcher.poll_s:g}".replace(".", ",")
                    self._info(f"Surveillance du dossier « {self.watcher.folder} » (vidéos complètes, "
                               f"vérification toutes les {poll} s).")
                    if self.watcher.last_error:
                        self._error(self.watcher.last_error)
            t_start = time.monotonic()
            while not self._stop_evt.is_set():
                job = self.queue.next_waiting()
                if job is not None:
                    self._process(job)
                    continue
                if self.watcher is None:
                    break
                if (self.watch_once and self.watcher.pending_count == 0
                        and time.monotonic() - t_start > self.watcher.stable_s + self.watcher.poll_s
                        and self.watcher.last_scan > t_start):
                    break
                if self.watcher.due():
                    err0 = self.watcher.last_error
                    ready = self.watcher.scan()
                    if self.watcher.last_error and self.watcher.last_error != err0:
                        self._error(self.watcher.last_error)
                    if ready:
                        added = self.queue.add(ready, from_watch=True)
                        if added:
                            self._info(f"{len(added)} nouvelle(s) vidéo(s) reçue(s) : "
                                       + ", ".join(j.name for j in added))
                            self._emit_jobs(force=True)
                        continue
                self._stop_evt.wait(0.2)
            if self.queue.cancel_waiting():
                self._emit_jobs(force=True)
        except Exception as e:  # noqa: BLE001
            self._error(f"Erreur inattendue du traitement par lot : {type(e).__name__} : {e}")
        finally:
            self.current = None
            self._emit_jobs(force=True)
            self._call(self.on_all_finished)

    def _job_config(self, job: BatchJob):
        cfg = self.app_cfg.copy()
        b = self.batch
        if b.condition:
            cfg.run.condition = b.condition
        if b.sample:
            cfg.run.sample_type = b.sample
        if b.sample or b.mass_g > 0:
            cfg.run.sample_mass_g = b.mass_g
        if b.px_per_mm > 0:
            cfg.physics.px_per_mm = b.px_per_mm
        elif cfg.physics.px_per_mm <= 0:
            # Vidéo synthétique : l'échelle exacte est dans la vérité terrain (<vidéo>.gt.json)
            gt = Path(str(job.path) + ".gt.json")
            try:
                if gt.is_file():
                    ppm = float(json.loads(gt.read_text(encoding="utf-8")).get("px_per_mm") or 0)
                    if ppm > 0:
                        cfg.physics.px_per_mm = ppm
                        job.warnings.append("Échelle lue dans la vérité terrain de la vidéo simulée (.gt.json).")
            except (OSError, ValueError, TypeError):
                pass
        return cfg

    def _finish_job(self, job: BatchJob, run=None) -> None:
        job.ended_at = _now_iso()
        if job.status in (STATUS_DONE, STATUS_ERROR):
            self._update_summary(job, run)
            if self.watcher is not None and job.from_watch:
                self.watcher.mark_processed(job.path, job.status, job.result_folder, job.error)
        elif job.status == STATUS_CANCELLED and self.watcher is not None:
            self.watcher.release(job.path)
        self._emit_jobs(force=True)
        self._call(self.on_job_finished, job.to_dict())

    def _process(self, job: BatchJob) -> None:
        from .analysis.run import analyze_video
        analyze = self._analyze or analyze_video
        self.current = job
        job.status = STATUS_RUNNING
        job.started_at = _now_iso()
        job.progress, job.eta_s, job.error = 0.0, -1.0, ""
        self._emit_jobs(force=True)
        self._info(f"Analyse de « {job.name} »…")
        cfg = self._job_config(job)

        def progress(frac, eta=-1.0):
            job.progress = float(frac)
            job.eta_s = float(eta)
            self._emit_jobs()
            return self._stop_evt.is_set()

        try:
            if not Path(job.path).is_file():
                raise FileNotFoundError(f"vidéo introuvable : {job.path}")
            run, _det = analyze(job.path, cfg, progress=progress)
        except Exception as e:  # noqa: BLE001
            job.status = STATUS_ERROR
            job.error = str(e) if isinstance(e, OSError) else f"{type(e).__name__} : {e}"
            self.current = None
            self._finish_job(job)
            return
        if self._stop_evt.is_set():
            job.status, job.error = STATUS_CANCELLED, "analyse interrompue"
            self.current = None
            self._finish_job(job)
            return
        stem = Path(job.path).stem
        date = datetime.now().strftime("%Y-%m-%d")
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            run.name = _unique_name(self.out_dir, f"{date} {run.condition} {stem}")
            folder = run.save(self.out_dir)
        except Exception as e:  # noqa: BLE001
            job.status, job.error = STATUS_ERROR, f"Enregistrement impossible : {e}"
            self.current = None
            self._finish_job(job, run)
            return
        job.result_folder = str(folder)
        job.warnings += list(run.analysis_warnings) + list(run.save_warnings)
        if self.batch.make_reports:
            try:
                from .analysis.report import build_run_report
                build_run_report(run)
            except Exception as e:  # noqa: BLE001
                job.warnings.append(f"Rapport HTML non généré : {type(e).__name__} : {e}")
        job.counts = run.counts()
        job.total = run.total
        job.duration_s = float(run.duration_s)
        job.progress, job.eta_s = 1.0, 0.0
        job.status = STATUS_DONE
        self.current = None
        self._finish_job(job, run)

    # ------------------------------------------------------------- résumé --
    def _load_summary(self) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        p = self.out_dir / SUMMARY_CSV
        try:
            with open(p, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("fichier"):
                        rows[_norm(r["fichier"])] = {k: r.get(k, "") for k in SUMMARY_COLUMNS}
        except FileNotFoundError:
            pass
        except (OSError, ValueError, csv.Error) as e:
            self._error(f"{SUMMARY_CSV} illisible ({e}) : il sera réécrit.")
        return rows

    def _update_summary(self, job: BatchJob, run=None) -> None:
        if self._summary is None:
            self._summary = self._load_summary()
        self._summary[_norm(job.path)] = summary_row(job, run)
        rows = list(self._summary.values())
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            write_summary(self.out_dir, rows)
        except PermissionError:
            self._error(f"« {SUMMARY_CSV} » est ouvert dans un autre logiciel (Excel ?) : fermez-le ; il sera mis "
                        "à jour à la prochaine vidéo.")
        except OSError as e:
            self._error(f"Impossible d'écrire {SUMMARY_CSV} : {e}")


def write_summary(out_dir: str | Path, rows: list[dict]) -> Path:
    """Écrit resume_lot.csv (et resume_lot.xlsx) de façon atomique ; renvoie le chemin du CSV."""
    out_dir = Path(out_dir)
    p = out_dir / SUMMARY_CSV
    tmp = p.with_name("resume_lot.tmp.csv")
    try:
        with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        wb = Workbook()
        ws = wb.active
        ws.title = "Résumé du lot"
        ws.append(SUMMARY_COLUMNS)
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="2F4F7F")
        for r in rows:
            vals = []
            for k in SUMMARY_COLUMNS:
                v = r.get(k, "")
                if isinstance(v, str) and v and k not in ("video", "fichier", "statut", "condition", "echantillon",
                                                          "calibre", "dossier", "date_analyse", "erreur",
                                                          "remarques"):
                    try:
                        v = float(v)
                        v = int(v) if v.is_integer() and k in CLASSES + ("total", "images") else v
                    except ValueError:
                        pass
                vals.append(v)
            ws.append(vals)
        ws.freeze_panes = "A2"
        xt = out_dir / "resume_lot.tmp.xlsx"
        wb.save(xt)
        os.replace(xt, out_dir / SUMMARY_XLSX)
    except PermissionError:
        raise
    except Exception:  # noqa: BLE001 - l'Excel est un plus ; le CSV fait foi
        pass
    return p


# =========================================================================== #
# Adaptateur Qt (créé à la demande)
# =========================================================================== #
_QT_CLASS = None


def _make_qt_controller():
    from PySide6.QtCore import QObject, Signal

    class QtBatchController(QObject):
        """Pilote un BatchRunner depuis l'interface. Les signaux sont émis depuis le fil du lot ;
        Qt les remet dans le fil de l'interface (connexion automatique en file d'attente)."""

        jobs_changed = Signal(list)       # list[dict] (voir BatchJob.to_dict)
        job_finished = Signal(dict)
        all_finished = Signal()
        error = Signal(str)
        info = Signal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.runner: BatchRunner | None = None

        def start(self, batch_dict: dict, app_cfg=None, poll_s: float = 5.0, stable_s: float = 10.0) -> bool:
            """Démarre un lot. Si un lot tourne déjà, ses fichiers sont ajoutés à la file (False)."""
            if self.is_running():
                self.add_files(list((batch_dict or {}).get("files") or []))
                return False
            self.runner = BatchRunner(batch_dict, app_cfg, on_jobs_changed=self.jobs_changed.emit,
                                      on_job_finished=self.job_finished.emit, on_all_finished=self.all_finished.emit,
                                      on_error=self.error.emit, on_info=self.info.emit, poll_s=poll_s,
                                      stable_s=stable_s)
            self.runner.start()
            return True

        def stop(self) -> None:
            if self.runner is not None:
                self.runner.stop()

        def add_files(self, paths: list) -> list[dict]:
            if self.runner is not None and self.runner.is_alive():
                return self.runner.add_files(paths)
            return []

        def is_running(self) -> bool:
            return self.runner is not None and self.runner.is_alive()

        def jobs(self) -> list[dict]:
            return self.runner.queue.snapshot() if self.runner is not None else []

        def wait(self, timeout_s: float | None = None) -> bool:
            if self.runner is None:
                return True
            self.runner.join(timeout_s)
            return not self.runner.is_alive()

    return QtBatchController


def __getattr__(name: str):
    global _QT_CLASS
    if name == "QtBatchController":
        if _QT_CLASS is None:
            _QT_CLASS = _make_qt_controller()
        return _QT_CLASS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
