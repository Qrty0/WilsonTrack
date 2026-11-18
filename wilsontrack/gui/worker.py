"""Fils d'exécution : traitement vidéo en continu et appels réseau (assistant IA).

Le traitement tourne dans un QThread : l'interface reste fluide même si la
détection prend quelques dizaines de millisecondes par image.

Session de nuit : ``ProcessingWorker(..., session=NightSessionConfig(...))`` délègue à
``longrun.SessionController`` la vidéo segmentée, les photos périodiques, les points de
sauvegarde, le journal, l'arrêt automatique et la sauvegarde finale (+ rapport) ; l'état est
émis toutes les ~2 s par le signal ``session_status(dict)``.
"""
from __future__ import annotations

import queue
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..analysis.run import LONG_VIDEO_S, build_run_result, limit_memory, make_spill_dir
from ..capture import FrameSource, VideoRecorder, imwrite
from ..config import AppConfig
from ..detection import TrackDetector
from ..longrun import (KeepAwake, NightSessionConfig, ReconnectPolicy, SessionController, SourceTime,
                       fmt_duration, source_finished)
from ..render import draw_overlay, enhance_for_display
from ..synth import SynthConfig, SyntheticChamber
from .widgets import bgr_to_qimage

# Enregistrement « normal » (sans session) plus long que ceci : mémoire bornée (vignettes délestées)
MEMORY_CAP_AFTER_S = 3600.0

VIEW_MODES = {
    "overlay": "Traces annotées",
    "raw": "Image brute",
    "diff": "Contraste (image − fond)",
    "evidence": "Carte de détection (SNR)",
    "mask": "Masque des détections",
}


class SynthSource(FrameSource):
    """Chambre de Wilson simulée en temps réel (démonstration / formation)."""
    live = True

    def __init__(self, cfg: SynthConfig | None = None, realtime: bool = True):
        self.cfg = cfg or SynthConfig(duration_s=3600, rates_per_min={"alpha": 8, "beta": 14, "muon": 16,
                                                                       "proton": 0.3})
        self.sim = SyntheticChamber(self.cfg)
        self.fps = self.cfg.fps
        self.shape = (self.cfg.height, self.cfg.width)
        self.n_frames = self.sim.n_frames
        self.realtime = realtime
        self._t0 = time.perf_counter()
        self._i = 0

    @property
    def px_per_mm(self) -> float:
        return self.cfg.px_per_mm

    def read(self):
        if self._i >= self.sim.n_frames:
            return False, None, self._i / self.fps
        if self.realtime:
            target = self._t0 + self._i / self.fps
            dt = target - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
        fr = self.sim.render_next()
        t = self._i / self.fps
        self._i += 1
        return True, fr, t

    @property
    def position(self) -> int:
        return self._i


def _false_color(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)


class ProcessingWorker(QThread):
    frame_ready = Signal(object, dict)       # QImage, statistiques
    track_finished = Signal(object)          # TrackEvent
    run_finished = Signal(object)            # RunResult (session : déjà enregistré, run.folder renseigné)
    error = Signal(str)
    info = Signal(str)
    snapshot_saved = Signal(str)
    session_status = Signal(dict)            # session de nuit : état toutes les ~2 s
    session_warning = Signal(str)            # session de nuit : nouvel avertissement (aussi dans le journal)

    def __init__(self, source_factory, cfg: AppConfig, source_name: str, realtime_file: bool = False,
                 display_max_w: int = 1280, session: NightSessionConfig | dict | None = None,
                 session_folder: str = ""):
        super().__init__()
        self.source_factory = source_factory
        self.cfg = cfg.copy()
        self.source_name = source_name
        self.realtime_file = realtime_file
        self.display_max_w = display_max_w
        self.cmd: queue.Queue = queue.Queue()
        self.view_mode = "overlay"
        self.enhance = 1.0
        self.show_labels = True
        self._stop = False
        self._paused = False
        self.detector: TrackDetector | None = None
        self.recorder = VideoRecorder()
        self.last_frame: np.ndarray | None = None
        self.last_annotated: np.ndarray | None = None
        self.source: FrameSource | None = None
        # Session de nuit (None = traitement ordinaire, comportement inchangé)
        if isinstance(session, dict):
            session = NightSessionConfig.from_dict(session)
        self.session_cfg: NightSessionConfig | None = session
        self.session_folder = str(session_folder or "")
        self.session_ctl: SessionController | None = None
        self._stop_reason = ""
        self._spill_dir = None
        self._counts_cache: tuple[float, int, dict] = (0.0, -1, {})
        # Contre-pression vers l'interface : on n'envoie une nouvelle image que lorsque la
        # précédente a été affichée. Sinon, si l'écran est plus lent que la caméra (Mac Retina…),
        # les images s'empilent dans la file Qt et l'application gèle en dévorant la mémoire.
        self._ui_pending_since: float | None = None

    def frame_consumed(self):
        """Appelé par l'interface quand l'image envoyée a été affichée."""
        self._ui_pending_since = None

    def _ui_ready(self, now: float) -> bool:
        p = self._ui_pending_since
        return p is None or now - p > 2.0  # filet de sécurité si l'accusé se perd

    # ------------------------------------------------------------ commandes --
    def stop(self):
        self._stop = True
        self.cmd.put(("noop", None))

    def set_paused(self, paused: bool):
        self._paused = paused

    @property
    def paused(self) -> bool:
        return self._paused

    def send(self, name: str, arg=None):
        self.cmd.put((name, arg))

    # ------------------------------------------------------------------ run --
    def run(self):
        try:
            self.source = self.source_factory()
        except Exception as e:
            self.error.emit(str(e))
            return
        src = self.source
        keep_awake = None
        try:
            if self.session_cfg is not None:
                # Condition / échantillon / calibration de la session AVANT de créer le détecteur
                self.session_cfg.apply_to(self.cfg)
            if hasattr(src, "px_per_mm") and self.cfg.physics.px_per_mm <= 0:
                self.cfg.physics.px_per_mm = src.px_per_mm
            if self.session_cfg is not None:
                live = bool(getattr(src, "live", False))
                simulated = (not live) or (not getattr(src, "realtime", True))
                self.session_ctl = SessionController(
                    self.session_cfg, self.cfg, self.session_folder, self.source_name, simulated_clock=simulated,
                    on_warning=self.session_warning.emit, on_info=None)
                self.session_folder = str(self.session_ctl.folder)
                # Lié au fil d'exécution : entré ET quitté dans ce fil (QThread.run)
                keep_awake = KeepAwake(self.session_cfg.keep_awake and not simulated,
                                       display=self.session_cfg.keep_display_on).__enter__()
                if self.session_cfg.keep_awake and not simulated and not keep_awake.active:
                    self.session_ctl.journal.write(f"Mise en veille non bloquée : {keep_awake.error or 'hors Windows'}",
                                                   "attention")
            self.detector = TrackDetector(self.cfg, src.fps, src.shape)
            if self.session_ctl is not None:
                self.session_ctl.begin(self.detector, src.fps, src.shape)
                self.info.emit(f"Session de nuit démarrée — dossier : {self.session_ctl.folder}")
                self.session_status.emit(self.session_ctl.status(self.detector))
            self._loop(src)
        except Exception as e:  # erreur inattendue : ne jamais faire planter l'interface
            self._stop_reason = f"erreur : {e}"
            if self.session_ctl is not None:
                self.session_ctl.journal.write(f"Erreur de traitement : {e} — {traceback.format_exc(limit=3)}",
                                               "erreur")
            self.error.emit(f"Erreur de traitement : {e}\n\n{traceback.format_exc(limit=4)}")
        finally:
            if self.recorder.active:
                p = self.recorder.stop()
                self.info.emit(f"Vidéo enregistrée : {p}")
            try:
                (self.source or src).release()       # la source a pu être rouverte (reconnexion)
            except Exception:
                pass
            if self.detector is not None:
                if self.session_ctl is not None:
                    try:
                        run = self.session_ctl.finish(self.detector, self._stop_reason or "arrêt manuel")
                        self.info.emit(f"Session terminée ({self.session_ctl.stop_reason}) : résultats, vidéo et "
                                       f"rapport dans {self.session_ctl.folder}")
                        self.run_finished.emit(run)
                    except Exception as e:
                        self.error.emit(f"Impossible de finaliser la session : {e}\n\n{traceback.format_exc(limit=4)}")
                else:
                    try:
                        self.detector.finalize_run()
                        n = self.detector.frame_idx + 1
                        run = build_run_result(self.detector, self.cfg, self.source_name, n, src.fps, src.shape)
                        self.run_finished.emit(run)
                    except Exception as e:
                        self.error.emit(f"Impossible de finaliser l'enregistrement : {e}")
            if keep_awake is not None:
                keep_awake.__exit__(None, None, None)

    def _handle_commands(self, src):
        while True:
            try:
                name, arg = self.cmd.get_nowait()
            except queue.Empty:
                return
            det = self.detector
            if name == "record_start":
                h, w = src.shape
                try:
                    self.recorder.start(arg, src.fps, (w, h))
                    self.info.emit(f"Enregistrement vidéo démarré : {arg}")
                except Exception as e:
                    self.error.emit(str(e))
            elif name == "record_stop":
                p = self.recorder.stop()
                if p:
                    self.info.emit(f"Vidéo enregistrée : {p}")
            elif name == "snapshot":
                folder = arg
                if self.last_frame is not None:
                    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    cond = self.cfg.run.condition
                    base = f"{folder}/{stamp} {cond}"
                    imwrite(base + " brute.png", self.last_frame)
                    if self.last_annotated is not None:
                        imwrite(base + " annotee.png", self.last_annotated)
                    self.snapshot_saved.emit(base + " annotee.png")
            elif name == "trigger" and det is not None:
                det.trigger_expansion()
            elif name == "live_config" and det is not None:
                self._apply_live(arg)
            elif name == "reset_background" and det is not None:
                det.bg.reset()
            elif name == "camera_set" and hasattr(src, "set"):
                prop, val = arg
                src.set(prop, val)
            elif name == "camera_dialog" and hasattr(src, "open_driver_settings"):
                src.open_driver_settings()
            elif name == "session_checkpoint" and det is not None and self.session_ctl is not None:
                # « Sauvegarder maintenant » pendant une session de nuit
                if self.session_ctl.checkpoint(det):
                    self.info.emit("Point de sauvegarde en cours d'écriture…")

    def _reopen(self):
        """Ferme puis rouvre la source (caméra) après une coupure passagère du flux."""
        try:
            self.source.release()
        except Exception:
            pass
        time.sleep(0.3)
        try:
            self.source = self.source_factory()
            return self.source
        except Exception:
            return None

    def _apply_live(self, cfg: AppConfig):
        """Applique en direct les paramètres qui ne nécessitent pas de redémarrer la détection."""
        det = self.detector
        d_new, d = cfg.detection, det.cfg.detection
        for k in ("k_high", "k_low", "close_px", "min_area_px", "merge_gap_mm", "merge_angle_deg", "max_blob_frac",
                  "min_visible_s", "min_frames", "max_missed", "match_dilate_px", "min_length_mm",
                  "min_elongation", "min_peak_snr"):
            setattr(d, k, getattr(d_new, k))
        det.bg.time_s = max(0.1, d_new.bg_time_s)
        det.cfg.classification = cfg.classification
        det.cfg.physics.chamber_temp_c = cfg.physics.chamber_temp_c
        det.cfg.physics.pressure_kpa = cfg.physics.pressure_kpa
        det.cfg.physics.b_field_mt = cfg.physics.b_field_mt
        det.cfg.run = cfg.run
        det.cfg.expansion.delay_ms = cfg.expansion.delay_ms
        det.cfg.expansion.window_ms = cfg.expansion.window_ms
        det.cfg.expansion.jump_sigma = cfg.expansion.jump_sigma
        self.cfg.run = cfg.run

    def _emit_status(self, st):
        if st:
            self.session_status.emit(st)

    def _loop(self, src):
        det = self.detector
        ctl = self.session_ctl
        t_start = time.perf_counter()
        last_emit = 0.0
        fps_hist: list[float] = []
        n_fail = 0
        n_reopen = 0
        live = bool(getattr(src, "live", False))
        warned_slow = False
        policy = ReconnectPolicy() if ctl is not None else None
        stime = SourceTime() if ctl is not None else None
        # Mémoire bornée hors session : vidéo longue dès le départ, sinon au-delà d'une heure
        n_src = int(getattr(src, "n_frames", 0) or 0)
        long_file = (not live) and n_src > 0 and n_src / max(float(src.fps or 30.0), 1.0) > LONG_VIDEO_S
        next_cap_check = t_start + 60.0
        self._stop_reason = "arrêt manuel"
        while not self._stop:
            self._handle_commands(src)
            if self._paused:
                if ctl is not None:
                    self._emit_status(ctl.idle_tick(det))
                    if ctl.should_stop():
                        self._stop_reason = "durée prévue atteinte"
                        self.info.emit("Durée prévue atteinte : fin de la session.")
                        break
                time.sleep(0.03)
                continue
            t0 = time.perf_counter()
            try:
                ok, frame, t = src.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                if ctl is not None and live and not source_finished(src):
                    # Session de nuit : on insiste (reconnexions espacées pendant 15 min)
                    act = policy.fail()
                    if act == "perdue":
                        ctl.on_camera_lost()
                        self.info.emit("La caméra ne renvoie plus d'images : reconnexion automatique…")
                    elif act == "reconnecter":
                        new_src = self._reopen()
                        ctl.on_camera_retry(policy.attempts, new_src is not None)
                        if new_src is not None:
                            src = new_src
                    elif act == "abandon":
                        ctl.on_camera_gave_up(policy.outage_s)
                        self._stop_reason = "caméra perdue"
                        self.error.emit(f"La caméra est perdue depuis {fmt_duration(policy.outage_s)} : fin de la "
                                        "session. Les résultats obtenus jusqu'ici sont sauvegardés.")
                        break
                    self._emit_status(ctl.idle_tick(det))
                    if ctl.should_stop():
                        self._stop_reason = "durée prévue atteinte"
                        break
                    time.sleep(0.05)
                    continue
                if live and ctl is None:
                    n_fail += 1
                    # Une webcam USB peut mettre un instant à démarrer ou hoqueter :
                    # on patiente, on prévient, puis on tente de la rouvrir avant d'abandonner.
                    if n_fail == 25 and not warned_slow:
                        warned_slow = True
                        self.info.emit("La caméra tarde à envoyer des images… patientez (démarrage / mise au point).")
                    if n_fail % 75 == 0 and n_reopen < 3:
                        n_reopen += 1
                        self.info.emit(f"Reconnexion à la caméra (tentative {n_reopen})…")
                        new_src = self._reopen()
                        if new_src is not None:
                            src = new_src
                            n_fail = 0
                        continue
                    if n_fail < 400:  # ~8 s de tolérance avant de renoncer
                        time.sleep(0.02)
                        continue
                    self.error.emit("La caméra ne renvoie pas d'images. Fermez les autres logiciels qui l'utilisent "
                                    "(Zoom, Teams, l'app Caméra de Windows), essayez l'autre pilote "
                                    "(DirectShow / Media Foundation), puis relancez.")
                self._stop_reason = "fin de la source"
                break
            if n_fail:
                n_fail = 0
                self.info.emit("Caméra connectée.")
            t_det = t if live else None
            if ctl is not None:
                gap = policy.ok()
                if gap is not None:
                    ctl.on_camera_back(gap)
                    stime.gap(gap)
                    self.info.emit("Caméra reconnectée : la session continue.")
                t_det = stime(t) if live else None
            if self.recorder.active:
                self.recorder.write(frame)
            want_dbg = self.view_mode in ("diff", "evidence", "mask")
            res = det.process(frame, t_det, want_debug=want_dbg)
            self.last_frame = frame
            for e in res.finished:
                self.track_finished.emit(e)
            if ctl is not None:
                self._emit_status(ctl.on_frame(frame, t_det if t_det is not None else t, res, det))
                if ctl.should_stop():
                    self._stop_reason = "durée prévue atteinte"
                    self.info.emit("Durée prévue atteinte : fin de la session, sauvegarde finale et rapport…")
                    break
            dt = time.perf_counter() - t0
            fps_hist.append(dt)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            now = time.perf_counter()
            if ctl is None and now >= next_cap_check:
                # Enregistrement oublié pendant des heures : la mémoire ne doit pas exploser
                next_cap_check = now + 60.0
                if long_file or now - t_start > MEMORY_CAP_AFTER_S or det.analyzed_s > MEMORY_CAP_AFTER_S:
                    try:
                        if self._spill_dir is None:
                            self._spill_dir = make_spill_dir()
                        limit_memory(det, self._spill_dir)
                    except Exception:
                        pass
            # Affichage limité à ~25 images/s (et plus rarement en analyse rapide de fichier)
            min_gap = 0.04 if (live or self.realtime_file) else 0.12
            if (now - last_emit >= min_gap or res.finished) and self._ui_ready(now):
                last_emit = now
                img = self._render(frame, res, det)
                stats = self._stats(det, res, src, fps_hist, now - t_start)
                self._ui_pending_since = now
                self.frame_ready.emit(bgr_to_qimage(img), stats)
            if self.realtime_file and not live:
                target = 1.0 / max(src.fps, 1.0)
                spent = time.perf_counter() - t0
                if spent < target:
                    time.sleep(target - spent)

    def _render(self, frame, res, det):
        mode = self.view_mode
        h0, w0 = frame.shape[:2]
        if w0 > self.display_max_w:  # on dessine sur l'image déjà réduite (plus rapide en 1080p / 4K)
            f = self.display_max_w / w0
            frame = cv2.resize(frame, (self.display_max_w, int(round(h0 * f))), interpolation=cv2.INTER_AREA)
        if mode == "raw" or res.warming:
            img = frame
        elif mode == "diff" and "diff" in res.debug:
            d = res.debug["diff"]
            s = max(res.noise_sigma, 1e-3)
            img = _false_color(d, -2 * s, 12 * s)
        elif mode == "evidence" and "evidence" in res.debug:
            img = _false_color(res.debug["evidence"], 0, 12)
        elif mode == "mask" and "labels" in res.debug:
            lab = res.debug["labels"]
            img = np.zeros(lab.shape + (3,), np.uint8)
            img[lab > 0] = (255, 255, 255)
        else:
            img = frame
        if img is not frame and img.shape[:2] != frame.shape[:2]:
            img = cv2.resize(img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.enhance > 1.0 and mode in ("overlay", "raw"):
            img = enhance_for_display(img, self.enhance)
        recent = [e for e in det.events[-60:] if e.valid]
        ann = draw_overlay(img, res, det, recent, show_labels=self.show_labels) if mode != "raw" else img
        self.last_annotated = ann
        return ann

    def _stats(self, det, res, src, fps_hist, wall):
        # Comptage par classe : recalculé à chaque nouvelle trace, sinon au plus 2 fois par seconde
        # (une nuit = des dizaines de milliers de traces à parcourir)
        now = time.perf_counter()
        ts, _n, counts = self._counts_cache
        if res.finished or now - ts > 0.5 or _n < 0:
            counts = {}
            for e in det.events:
                if e.valid:
                    counts[e.final_cls] = counts.get(e.final_cls, 0) + 1
            self._counts_cache = (now, len(det.events), counts)
        counts = dict(counts)
        mean_dt = sum(fps_hist) / max(len(fps_hist), 1)
        n = getattr(src, "n_frames", 0) or 0
        return {
            "t": res.t, "frame": res.frame_idx, "n_frames": n,
            "progress": (res.frame_idx + 1) / n if n and not getattr(src, "live", False) else None,
            "counts": counts, "duration_s": det.n_frames_processed / det.fps,
            "active": len(res.active), "noise": res.noise_sigma, "warming": res.warming,
            "disturbed": res.disturbed, "in_window": res.in_window, "proc_fps": 1.0 / mean_dt if mean_dt > 0 else 0.0,
            "wall_s": wall, "native_wh": (src.shape[1], src.shape[0]), "recording": self.recorder.active,
            "expansions": len(det.exp_times), "rejected": det.rejected, "lit": res.lit_frac,
            "calibrated": det.calibrated, "px_per_mm": det.px_per_mm_full,
        }


class CallWorker(QThread):
    """Exécute une fonction bloquante (appel réseau) hors du fil de l'interface."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, **self.kwargs))
        except Exception as e:
            self.failed.emit(str(e))
