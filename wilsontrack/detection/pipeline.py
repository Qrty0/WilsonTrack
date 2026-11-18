"""Pipeline complet : image → traces détectées, suivies dans le temps, mesurées et classées.

Chaque trace physique reste visible environ 0.5 à 3 s dans une chambre à diffusion.
Le suivi temporel associe les détections d'une image à l'autre pour que chaque
trace ne soit comptée qu'UNE seule fois ; ses mesures sont prises à l'instant où
elle est la plus nette (avant que les gouttelettes ne diffusent).

Version 2.0 — performance et sessions longues (résultats identiques à la version 1.1) :
- tampons d'images réutilisés (préparation, évidence, segmentation, carte des traces) ;
- carte d'association construite en une seule érosion d'une image d'étiquettes quand les
  traces actives sont nombreuses (au lieu d'une dilatation par trace) ;
- mesures « en direct » et contours calculés seulement quand ils sont affichés ou utilisés
  (mêmes valeurs : on mémorise l'instant, les références et les paramètres du calcul) ;
- sessions longues : rejets bornés en mémoire (``keep_invalid_events``, ``max_invalid_kept``,
  ``invalid_counts``), vignettes délestées sur disque (``spill_thumbnails``) et chronologie
  minute par minute (``timeline``).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..config import AppConfig
from .classifier import (MLClassifier, References, classify_rules, compute_references,
                         physics_annotations)
from .events import TrackEvent
from .features import compute_features
from .preprocess import BackgroundModel, EvidenceBuilder, Preprocessor, Workspace, ellipse_kernel, rect_kernel
from .segment import Detection, _angle_diff, _moments, segment, segment_evidence  # noqa: F401

THUMB_MAX = 200
# Motif de rejet d'une structure vue sur trop peu d'images (aucun TrackEvent n'est créé)
REJECT_TOO_FEW_FRAMES = "vue sur trop peu d'images"
_EVMAP_LABEL_MAX = 65534              # étiquettes uint16 de la carte d'association


@dataclass
class _Best:
    frame_idx: int
    t: float
    score: float
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray
    diff: np.ndarray
    sigma: float
    thumb: np.ndarray | None
    thumb_origin: tuple   # (x, y, facteur) du crop pleine résolution


class _GhostView:
    """Vue booléenne (compteur > 0) de la carte des fantômes : g[y0:y1, x0:x1] → tableau bool."""
    __slots__ = ("cnt", "shape")

    def __init__(self, cnt: np.ndarray):
        self.cnt = cnt
        self.shape = cnt.shape

    def __getitem__(self, key):
        return self.cnt[key] > 0


class _PendingFeats:
    """Mesures « en direct » demandées mais pas encore calculées (calcul différé).

    Garde tout ce dont dépend le calcul d'origine (meilleur instant, références et paramètres
    de classification du moment) : le résultat, calculé plus tard, est identique."""
    __slots__ = ("best", "refs", "cp")

    def __init__(self, best: _Best, refs: References, cp):
        self.best, self.refs, self.cp = best, refs, cp


@dataclass
class _Active:
    uid: int
    first_frame: int
    t_first: float
    last_frame: int
    t_last: float
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray
    theta: float
    cx: float
    cy: float
    length_q: float
    width_q: float
    n_seen: int = 1
    missed: int = 0
    best: _Best | None = None
    feats: dict | _PendingFeats | None = None
    feats_score: float = 0.0
    cls: str = "incertain"
    conf: float = 0.0
    cycle: int = -1
    contour: np.ndarray | None = None
    t_ana: float = 0.0        # instant d'apparition sur l'horloge d'analyse (s)
    max_len: float = 0.0      # plus grande longueur rapide observée (px)
    max_elong: float = 0.0    # plus grand allongement observé
    max_mean_e: float = 0.0   # plus grand contraste moyen observé
    best_len: float = 0.0     # longueur / largeur rapides au « meilleur instant »
    best_wid: float = 0.0
    ref_len: float = 0.0      # dimensions figées à la fin de la formation
    ref_wid: float = 0.0
    last_mean_e: float = 0.0  # contraste moyen à la dernière image
    contour_src: tuple | None = None   # (masque, x0, y0) de la dernière mise à jour (contour différé)


@dataclass
class FrameResult:
    frame_idx: int
    t: float
    warming: bool = False
    disturbed: bool = False
    in_window: bool = True
    expansion_triggered: bool = False
    noise_sigma: float = 0.0
    lit_frac: float = 0.0
    active: list = field(default_factory=list)       # [(contour_proc, cls, conf, uid)]
    finished: list = field(default_factory=list)     # [TrackEvent] finalisés à cette image
    debug: dict = field(default_factory=dict)


class TrackDetector:
    def __init__(self, cfg: AppConfig, fps: float, frame_shape_hw: tuple[int, int]):
        self.cfg = cfg
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_hw = frame_shape_hw
        dp = cfg.detection
        self.pre = Preprocessor(frame_shape_hw, dp.process_width, dp.blur_sigma, dp.temporal_avg,
                                dp.roi, dp.roi_margin_frac, dp.stabilize)
        self.scale = self.pre.scale
        self.pre.track_motion = dp.max_motion_px > 0
        self.bg = BackgroundModel(self.fps, dp.bg_time_s, dp.bg_warmup_frames)
        self.evb = EvidenceBuilder(dp.use_ridge, dp.ridge_scales)
        ph = cfg.physics
        self.calibrated = ph.px_per_mm > 0
        self.px_per_mm_full = ph.px_per_mm if self.calibrated else frame_shape_hw[1] / max(ph.default_fov_mm, 1.0)
        self.px_per_mm_proc = self.px_per_mm_full * self.scale
        # Distance au bord de la zone d'intérêt ET de l'image (une trace qui les touche est coupée)
        m = self.pre.roi_mask.copy()
        m[[0, -1], :] = 0
        m[:, [0, -1]] = 0
        self.roi_dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        self.active: list[_Active] = []
        self.ghosts: list[tuple[int, int, int, int, np.ndarray, int]] = []
        self.events: list[TrackEvent] = []
        self.rejected = 0
        self.next_uid = 1
        self.next_id = 1
        self.frame_idx = -1
        self.t = 0.0
        self.refs = References()
        self._refs_count = 0
        self.ml: MLClassifier | None = None
        if cfg.classification.mode == "ia" and cfg.classification.model_path:
            try:
                self.ml = MLClassifier.load(cfg.classification.model_path)
            except Exception:
                self.ml = None
        # IA de référence (mode hybride) : chargée une seule fois (cache du module) ; None si le
        # fichier manque ou est illisible → l'application revient aux règles physiques seules.
        self.ref_model = None
        if cfg.classification.mode == "hybride":
            try:
                from .reference_model import ReferenceModel
                self.ref_model = ReferenceModel.load_default()
            except Exception:
                self.ref_model = None
        # Chambre à expansion
        ex = cfg.expansion
        self.exp_enabled = ex.enabled
        self.exp_times: list[float] = []
        self._mean_hist: deque[float] = deque(maxlen=90)
        self._pending_trigger: float | None = None
        self.n_frames_processed = 0
        self.n_frames_disturbed = 0
        self.last_frame_small: np.ndarray | None = None
        # Horloge d'analyse : temps réellement analysé (hors apprentissage du fond,
        # perturbations et pauses). Pour une caméra, la cadence réelle est estimée à
        # partir des horodatages (une webcam peut livrer moins d'images que annoncé).
        self.nominal_fps = self.fps
        self.analyzed_s = 0.0
        self._t_prev: float | None = None
        self._dt_ema: float | None = None
        # --- Performance : tampons réutilisés d'une image à l'autre (voir preprocess) -----------
        self.pre.reuse_buffers = True
        self.bg.reuse_buffers = True
        self.evb.reuse_buffers = True
        self._ws = Workspace()
        self._evmap_dirty: tuple | None = None       # zone de la carte d'association à remettre à 0
        self._ghost_counted: dict[int, tuple] = {}   # fantômes comptés dans le compteur de recouvrement
        # --- Sessions longues (une nuit entière) ------------------------------------------------
        # Détections REJETÉES conservées dans self.events (diagnostic) : False → seulement comptées.
        self.keep_invalid_events = True
        # Nombre maximal de détections rejetées gardées dans self.events (None : illimité). Au-delà,
        # les plus anciennes sont retirées (par lots de 10 %) ; elles restent comptées dans
        # self.rejected et self.invalid_counts.
        self.max_invalid_kept: int | None = 2000
        self.invalid_counts: dict[str, int] = {}     # motif de rejet → nombre (cumul)
        self._invalid_fifo: deque[TrackEvent] = deque()
        self.n_valid = 0                             # traces valides produites (cumul)
        self.valid_counts: dict[str, int] = {}       # classe au moment de la fin de la trace → nombre
        # Chronologie : une entrée toutes les timeline_period_s secondes d'horloge d'analyse
        # (+ une entrée finale dans finalize_run). Voir _timeline_entry pour les clés.
        self.timeline: list[dict] = []
        self.timeline_period_s = 60.0
        self._tl_next: float | None = None
        self._tl_noise: list[float] = []
        self._tl_lit: list[float] = []

    # ------------------------------------------------------------------ API --
    @property
    def ghost_frames(self) -> int:
        return int(round(1.2 * self.fps))

    @property
    def min_track_px(self) -> float:
        """Longueur (px de traitement) à partir de laquelle une structure est une trace."""
        dp = self.cfg.detection
        return max(2 * dp.match_dilate_px + 4, 0.8 * dp.min_length_mm * self.px_per_mm_proc)

    @property
    def growth_frames(self) -> int:
        """Durée de formation d'une trace (croissance des gouttelettes) ≈ 0.3 s."""
        return max(3, int(round(0.3 * self.fps)))

    def trigger_expansion(self, t: float | None = None):
        """Déclenchement externe (Arduino / bouton) d'une expansion."""
        self._pending_trigger = self.t if t is None else t

    def _clock(self, t: float | None) -> tuple[float, float]:
        """Renvoie (t, dt). dt = durée représentée par cette image (bornée : une pause
        ou un gel de la caméra n'est pas compté comme du temps d'observation)."""
        nominal_dt = 1.0 / self.nominal_fps
        if t is None:
            return self.frame_idx / self.nominal_fps, nominal_dt
        prev, self._t_prev = self._t_prev, t
        if prev is None:
            return t, nominal_dt
        dt = t - prev
        if not (0.0 < dt < 1.0):
            return t, nominal_dt if dt >= 1.0 else 0.0
        self._dt_ema = dt if self._dt_ema is None else 0.95 * self._dt_ema + 0.05 * dt
        self.fps = float(np.clip(1.0 / self._dt_ema, 1.0, 240.0))
        self.bg.fps = self.fps
        return t, dt

    def process(self, frame: np.ndarray, t: float | None = None, want_debug: bool = False) -> FrameResult:
        self.frame_idx += 1
        fi = self.frame_idx
        t, dt = self._clock(t)
        self.t = t
        dp = self.cfg.detection
        g = self.pre(frame)
        res = FrameResult(fi, t)

        # --- Chambre à expansion : détection de l'instant d'expansion -------
        triggered = False
        if self.exp_enabled:
            triggered = self._expansion_check(g, t)
            res.expansion_triggered = triggered
            if triggered:
                self.exp_times.append(t)
                self.bg.bg = g.copy()
                self.bg.n = max(self.bg.n, self.bg.warmup)
            res.in_window = self._in_window(t)

        if not self.bg.ready:
            self.bg.update(g, None)
            res.warming = True
            return res

        raw_diff = self.bg.diff(g)
        # Évidence (seed, ext) : évaluée à la demande — masques par seuils équivalents, valeurs
        # seulement dans les cadres des détections (voir preprocess.Evidence) ; E[y0:y1, x0:x1]
        # renvoie les mêmes valeurs que l'image complète.
        E = self.evb.compute(raw_diff, self.pre)
        sigma = E.sig
        # Les MESURES (épaisseur, densité) se font sur le contraste non filtré (le passe-haut
        # réduirait les traces épaisses) ; la ligne de base locale est retirée dans features.
        diff = raw_diff
        res.noise_sigma = sigma
        merge_gap_px = dp.merge_gap_mm * self.px_per_mm_proc
        dets, lab, lit = segment_evidence(E, self.pre.roi_bool, dp.k_low, dp.k_high, dp.close_px,
                                          dp.min_area_px, merge_gap_px, dp.merge_angle_deg,
                                          ws=self._ws, want_labels=want_debug)
        # Image très « peuplée » (bruit) : l'évidence complète sera calculée d'emblée à l'image
        # suivante (moins coûteux que zone par zone). Aucun effet sur les résultats.
        self.evb.eager_hint = len(dets) > 24
        res.lit_frac = lit
        if want_debug:
            # Copies : les tampons internes sont réécrits à l'image suivante
            res.debug = {"diff": diff.copy(), "evidence": E.ext_full().copy(), "seed": E.seed_full().copy(),
                         "labels": lab, "gray": g.copy()}

        moved = dp.max_motion_px > 0 and self.pre.last_motion > dp.max_motion_px
        # Saut uniforme de luminosité (lumière de la salle, flash) : invisible après le filtre
        # passe-haut, on le mesure donc sur le contraste brut (médiane sur la zone analysée,
        # déjà calculée par l'évidence : evb.last_diff_median = np.median(sample_stats(raw_diff))).
        jump = (dp.max_global_jump > 0 and not (self.exp_enabled and res.in_window)
                and abs(self.evb.last_diff_median) > dp.max_global_jump)
        if (lit > dp.max_blob_frac or moved or jump) and not triggered:
            # Perturbation globale : lumière allumée, caméra bougée, expansion…
            self.n_frames_disturbed += 1
            res.disturbed = True
            for ev in self.active:
                ev.missed = dp.max_missed + 1
            res.finished = self._finalize_stale(force=True)
            self.bg.bg = g.copy()
            if self.exp_enabled and self.cfg.expansion.trigger == "auto":
                ex = self.cfg.expansion
                refractory = self.exp_times and (t - self.exp_times[-1]) < (ex.delay_ms + ex.window_ms) / 1000.0
                if not refractory:     # une expansion physique = un seul déclenchement
                    self.exp_times.append(t)
                    res.expansion_triggered = True
            return res

        self.n_frames_processed += 1
        self.analyzed_s += dt
        allow_new = res.in_window
        self._associate(dets, E.shape, diff, E, sigma, frame, fi, t, allow_new)

        # Durée de vie physique : une trace de chambre de Wilson reste visible 0,3 à 3 s. Un
        # événement plus long absorberait les traces qui passent ensuite au même endroit.
        max_frames = int(dp.max_track_s * self.fps)
        for ev in self.active:
            if fi - ev.first_frame > max_frames:
                ev.missed = dp.max_missed + 1
        self._update_background(g, dets)
        res.finished = self._finalize_stale()
        min_frames = self.min_frames
        active = []
        for ev in self.active:
            if ev.n_seen < min_frames:
                continue
            c = self._contour(ev)
            if c is None:
                continue
            if isinstance(ev.feats, _PendingFeats):
                self._resolve_feats(ev, classify=True)
            active.append((c, ev.cls, ev.conf, ev.uid))
        res.active = active
        self._timeline_tick(res)
        return res

    def _update_background(self, g: np.ndarray, dets: list[Detection]) -> None:
        """Fond : les pixels des détections (dilatés de 2 px) sont appris très lentement.
        Équivaut à fg = dilate(étiquettes > 0, carré 5×5) de la version d'origine."""
        if not dets:
            self.bg.update(g, None)
            return
        H, W = g.shape
        ws = self._ws
        fg = ws.zeros("bg_fg", (H, W), np.uint8)
        for d in dets:
            fg[d.y0:d.y1, d.x0:d.x1][d.mask] = 1
        fgd = cv2.dilate(fg, rect_kernel(5), dst=ws.get("bg_fgd", (H, W), np.uint8))
        bgm = np.subtract(np.uint8(1), fgd, out=ws.get("bg_bgm", (H, W), np.uint8))
        for d in dets:
            fg[d.y0:d.y1, d.x0:d.x1] = 0
        self.bg.update(g, fgd, bgm)

    def flush(self) -> list[TrackEvent]:
        """Termine toutes les traces encore actives (fin de vidéo)."""
        for ev in self.active:
            ev.missed = 10 ** 6
        return self._finalize_stale(force=True)

    def finalize_run(self) -> list[TrackEvent]:
        """Fin d'enregistrement : références définitives puis reclassification de tout."""
        self.flush()
        self.reclassify_all()
        self._timeline_final()
        return self.valid_events

    @property
    def valid_events(self) -> list[TrackEvent]:
        return [e for e in self.events if e.valid]

    def reclassify_all(self):
        valid = self.valid_events
        self.refs = compute_references([e.features for e in valid], self.cfg.classification.ref_percentile)
        # Les détections rejetées comme artefacts par l'IA sont réexaminées avec les références finales
        cands = valid + [e for e in self.events if not e.valid and e.extra.get("ml_artefact")]
        for e in cands:
            if e.extra.get("ml_artefact"):
                e.extra["was_artefact"] = True
            self._classify_event(e)
            e.extra.pop("was_artefact", None)
            if e.valid:
                e.extra.pop("ml_artefact", None)
        # Comptes par classe exacts après la reclassification finale (classes et artefacts ont pu changer)
        counts: dict[str, int] = {}
        for e in self.events:
            if e.valid:
                counts[e.final_cls] = counts.get(e.final_cls, 0) + 1
        if hasattr(self, "valid_counts"):
            self.valid_counts = counts
        if hasattr(self, "n_valid"):
            self.n_valid = sum(counts.values())

    # ------------------------------------------------------ sessions longues ---
    def spill_thumbnails(self, folder, keep_recent: int = 300) -> int:
        """Écrit sur disque les vignettes des traces valides anciennes et les libère de la mémoire.

        Les ``keep_recent`` traces valides les plus récentes qui ont encore leur vignette la
        gardent. Fichier ``<folder>/trace_{id:05d}_{classe}.png`` (encodage PNG en mémoire puis
        ``tofile`` : chemins accentués acceptés) ; ensuite ``e.thumb = e.thumb_mask = None`` et
        ``e.extra['thumb_path'] = str(chemin)``. Un fichier déjà présent est réutilisé. Arrêt sans
        erreur si le disque refuse l'écriture (les vignettes restantes restent en mémoire).
        À appeler depuis le fil qui appelle process(). Renvoie le nombre de vignettes délestées."""
        cands = [e for e in self.events
                 if e.valid and e.thumb is not None and e.id is not None and e.id >= 0]
        keep = max(0, int(keep_recent))
        if len(cands) <= keep:
            return 0
        folder = Path(folder)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            return 0
        n = 0
        for e in cands[:len(cands) - keep]:
            p = folder / f"trace_{e.id:05d}_{e.final_cls}.png"
            try:
                if not p.exists():
                    ok, buf = cv2.imencode(".png", e.thumb)
                    if not ok:
                        continue
                    buf.tofile(str(p))
            except Exception:  # noqa: BLE001 - disque plein / accès refusé : on garde la vignette
                break
            e.extra["thumb_path"] = str(p)
            e.thumb = None
            e.thumb_mask = None
            n += 1
        return n

    def memory_report(self) -> dict:
        """Taille (approximative) des structures internes — diagnostic des longues sessions."""
        n_thumb = sum(1 for e in self.events if e.thumb is not None)
        thumb_b = sum((e.thumb.nbytes if e.thumb is not None else 0)
                      + (e.thumb_mask.nbytes if e.thumb_mask is not None else 0) for e in self.events)
        return {"events": len(self.events), "valid": self.n_valid, "invalid_kept": len(self._invalid_fifo),
                "active": len(self.active), "ghosts": len(self.ghosts), "thumbs_in_memory": n_thumb,
                "thumb_mb": thumb_b / 2 ** 20, "workspace_mb": (self._ws.nbytes + self.pre.ws.nbytes
                                                                  + self.evb.ws.nbytes) / 2 ** 20,
                "timeline": len(self.timeline), "exp_times": len(self.exp_times)}

    def _timeline_tick(self, res: FrameResult) -> None:
        """Image analysée : échantillons de bruit / allumage, entrée de chronologie si besoin."""
        self._tl_noise.append(float(res.noise_sigma))
        self._tl_lit.append(float(res.lit_frac))
        period = float(self.timeline_period_s or 0.0)
        if period <= 0:
            return
        if self._tl_next is None:
            self._tl_next = period
        if self.analyzed_s >= self._tl_next:
            self.timeline.append(self._timeline_entry(final=False))
            self._tl_next = (math.floor(self.analyzed_s / period) + 1) * period

    def _timeline_entry(self, final: bool) -> dict:
        """Entrée de chronologie (valeurs cumulées depuis le début, sauf noise / lit) :
        t_s (horloge d'analyse, s), wall_s (instant t de la source : secondes de vidéo / caméra),
        frames (images ANALYSÉES : hors apprentissage du fond et perturbations), disturbed_frames,
        noise (médiane du sigma de bruit depuis l'entrée précédente), lit (médiane de la fraction
        allumée), counts (traces valides par classe, classe au moment où la trace s'est terminée),
        rejected (détections rejetées), final (True : entrée ajoutée par finalize_run)."""
        last = self.timeline[-1] if self.timeline else None
        noise = float(np.median(self._tl_noise)) if self._tl_noise else (last["noise"] if last else 0.0)
        lit = float(np.median(self._tl_lit)) if self._tl_lit else (last["lit"] if last else 0.0)
        self._tl_noise, self._tl_lit = [], []
        return {"t_s": float(self.analyzed_s), "wall_s": float(self.t), "frames": int(self.n_frames_processed),
                "disturbed_frames": int(self.n_frames_disturbed), "noise": noise, "lit": lit,
                "counts": dict(self.valid_counts), "rejected": int(self.rejected), "final": bool(final)}

    def _timeline_final(self) -> None:
        last = self.timeline[-1] if self.timeline else None
        if (last is not None and last["frames"] == self.n_frames_processed
                and last["disturbed_frames"] == self.n_frames_disturbed and last["counts"] == self.valid_counts
                and last["rejected"] == self.rejected and last["t_s"] == self.analyzed_s):
            last["final"] = True         # rien de nouveau depuis la dernière entrée (ou 2e appel)
            return
        self.timeline.append(self._timeline_entry(final=True))

    # --------------------------------------------------------- expansion ---
    def _expansion_check(self, g: np.ndarray, t: float) -> bool:
        ex = self.cfg.expansion
        if self._pending_trigger is not None:
            self._pending_trigger = None
            return True
        if ex.trigger != "auto":
            return False
        m = float(self.pre.sample_stats(g).mean())
        self._mean_hist.append(m)
        if len(self._mean_hist) < 8:
            return False
        d = np.diff(np.array(self._mean_hist))
        base = d[:-1]
        s = 1.4826 * np.median(np.abs(base - np.median(base))) + 0.05
        jump = d[-1]
        refractory = self.exp_times and (t - self.exp_times[-1]) < (ex.delay_ms + ex.window_ms) / 1000.0
        return bool(jump > max(ex.jump_sigma * s, 1.0) and not refractory)

    def _in_window(self, t: float) -> bool:
        if not self.exp_enabled:
            return True
        if not self.exp_times:
            return False
        ex = self.cfg.expansion
        dt = (t - self.exp_times[-1]) * 1000.0
        return ex.delay_ms <= dt <= ex.delay_ms + ex.window_ms

    # ------------------------------------------------------- association ---
    def _event_parts(self) -> list[tuple]:
        win = 0.7 * self.fps
        fi = self.frame_idx
        parts = []
        for k, ev in enumerate(self.active):
            # Masque courant + masque complet du « meilleur instant » : si une partie de la
            # trace (qui s'estompe) disparaît une image puis réapparaît, elle reste associée.
            parts.append((k, ev.x0, ev.y0, ev.x1, ev.y1, ev.mask))
            b = ev.best
            if b is not None and (fi - b.frame_idx) <= win:
                parts.append((k, b.x0, b.y0, b.x1, b.y1, b.mask))
        return parts

    def _event_map(self, shape) -> np.ndarray:
        """Carte des traces actives : chaque pixel reçoit 1 + l'indice de la PREMIÈRE trace dont
        le masque courant (ou celui du « meilleur instant » récent) dilaté le couvre ; 0 sinon.
        (Image complète : utilisée quand peu de traces sont actives, et pour le diagnostic.)"""
        H, W = shape
        ws = self._ws
        evmap = ws.zeros("evmap", (H, W), np.int32)
        if self._evmap_dirty is not None:
            y0, y1, x0, x1 = self._evmap_dirty
            evmap[y0:y1, x0:x1] = 0
            self._evmap_dirty = None
        if not self.active:
            return evmap
        dil = self.cfg.detection.match_dilate_px
        kern = ellipse_kernel(2 * dil + 1)
        parts = self._event_parts()
        self._evmap_dirty = (max(0, min(p[2] for p in parts) - dil), min(H, max(p[4] for p in parts) + dil),
                             max(0, min(p[1] for p in parts) - dil), min(W, max(p[3] for p in parts) + dil))
        for k, px0, py0, px1, py1, pm in parts:
            x0, y0 = max(0, px0 - dil), max(0, py0 - dil)
            x1, y1 = min(W, px1 + dil), min(H, py1 + dil)
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            m[py0 - y0:py1 - y0, px0 - x0:px1 - x0] = pm
            m = cv2.dilate(m, kern)
            reg = evmap[y0:y1, x0:x1]
            reg[(m > 0) & (reg == 0)] = k + 1
        return evmap

    def _det_event_ids(self, dets: list[Detection], shape) -> list[np.ndarray]:
        """Carte des traces actives (voir _event_map) lue sur le cadre de chaque détection.

        Avec beaucoup de traces actives, calcul équivalent plus rapide : les masques sont peints
        dans une image d'étiquettes (la plus petite étiquette l'emporte) puis érodés par le même
        élément elliptique, seulement autour de chaque détection. min sur le voisinage = plus
        petit indice de trace dont le masque dilaté couvre le pixel : même résultat au pixel près."""
        if not dets:
            return []
        if not self.active:
            return [np.zeros((d.y1 - d.y0, d.x1 - d.x0), np.int32) for d in dets]
        parts = self._event_parts()
        if len(self.active) > _EVMAP_LABEL_MAX or len(parts) * 14 <= len(dets) * 10 + 20:
            evmap = self._event_map(shape)
            return [evmap[d.y0:d.y1, d.x0:d.x1] for d in dets]
        H, W = shape
        dil = self.cfg.detection.match_dilate_px
        kern = ellipse_kernel(2 * dil + 1)
        lab = self._ws.full("evlab", (H, W), 65535, np.uint16)
        for k, px0, py0, px1, py1, pm in reversed(parts):
            lab[py0:py1, px0:px1][pm] = k + 1
        out = []
        for d in dets:
            x0, y0 = max(0, d.x0 - dil), max(0, d.y0 - dil)
            x1, y1 = min(W, d.x1 + dil), min(H, d.y1 + dil)
            e = cv2.erode(lab[y0:y1, x0:x1], kern)
            ids = e[d.y0 - y0:d.y1 - y0, d.x0 - x0:d.x1 - x0].astype(np.int32)
            ids[ids == 65535] = 0
            out.append(ids)
        for _k, px0, py0, px1, py1, _pm in parts:
            lab[py0:py1, px0:px1] = 65535
        return out

    def _ghost_map(self, shape) -> "_GhostView | None":
        """Zones « fantômes » (traces récemment terminées) : compteur de recouvrement tenu à jour
        de façon incrémentale (ajouts / expirations) ; zone fantôme = compteur > 0."""
        fi = self.frame_idx
        keep = [gh for gh in self.ghosts if gh[5] >= fi]
        cnt = self._ws.zeros("ghostcnt", shape, np.uint16)
        counted = self._ghost_counted
        if len(counted) > len(keep) or any(id(gh) not in counted for gh in keep):
            keep_ids = {id(gh) for gh in keep}
            for gid in [g for g in counted if g not in keep_ids]:
                x0, y0, x1, y1, m, _exp = counted.pop(gid)
                cnt[y0:y1, x0:x1][m] -= 1
            for gh in keep:
                if id(gh) not in counted:
                    x0, y0, x1, y1, m, _exp = gh
                    cnt[y0:y1, x0:x1][m] += 1
                    counted[id(gh)] = gh
        self.ghosts = keep
        if not keep:
            return None
        return _GhostView(cnt)

    def _associate(self, dets: list[Detection], shape, diff, E, sigma, frame, fi, t, allow_new: bool):
        if not isinstance(shape, tuple):           # ancienne signature : image d'étiquettes
            shape = shape.shape
        all_ids = self._det_event_ids(dets, shape)
        ghost = self._ghost_map(shape)
        nact = len(self.active)
        pairs = []
        det_ids = []
        for j, d in enumerate(dets):
            ids = all_ids[j]
            det_ids.append(ids)
            v = ids[d.mask]
            v = v[v > 0]
            if v.size:
                cnt = np.bincount(v, minlength=nact + 1)
                for k in np.nonzero(cnt)[0]:
                    pairs.append((int(cnt[k]), j, int(k) - 1))
        pairs.sort(reverse=True)
        det_match: dict[int, int] = {}
        ev_match: dict[int, int] = {}
        for c, j, k in pairs:
            if j in det_match or k in ev_match:
                continue
            det_match[j] = k
            ev_match[k] = j
        main_ev: dict[int, int] = {}
        for c, j, k in pairs:
            main_ev.setdefault(j, k)
        # Fragments d'une même trace en formation : deux traces « jeunes » qui se
        # rejoignent dans une même détection ne forment qu'une seule trace.
        absorbed: set[int] = set()
        gf = self.growth_frames
        for c, j, k in pairs:
            if j in det_match and det_match[j] != k and k not in ev_match:
                k0 = det_match[j]
                a, b = self.active[k0], self.active[k]
                if (fi - b.first_frame) <= gf and (fi - a.first_frame) <= gf + 2:
                    absorbed.add(k)

        min_new_len = max(2 * self.cfg.detection.match_dilate_px + 4,
                          0.8 * self.cfg.detection.min_length_mm * self.px_per_mm_proc)
        by_det: dict[int, list[int]] = {}
        for c, j, k in pairs:
            by_det.setdefault(j, []).append(k)
        min_share = max(self.cfg.detection.min_area_px, 6)
        updated: set[int] = set(ev_match)
        spawn: list[Detection] = []
        for j, d in enumerate(dets):
            ids = det_ids[j]
            if j not in det_match and not allow_new:
                continue
            idz = ids == 0
            new = d.mask & idz
            if ghost is not None:
                new &= ~ghost[d.y0:d.y1, d.x0:d.x1]
            ref_ev = self.active[main_ev[j]] if j in main_ev else None
            if j in det_match:
                k0 = det_match[j]
                ev = self.active[k0]
                # Une trace nettement plus brillante apparaît sur une structure pâle existante :
                # c'est une NOUVELLE trace (la structure pâle est close et jugée à part).
                if (allow_new and ev.n_seen >= 2 and d.mean_e > 3.0 * max(ev.last_mean_e, 1e-3)
                        and d.length_q >= self.min_track_px and d.elongation >= 2.5):
                    ev.missed = self.cfg.detection.max_missed + 1
                    updated.discard(k0)
                    spawn.append(d)
                    continue
                # Une trace « jeune » est encore en train de se former (les gouttelettes
                # grossissent) : tout ce qui lui est connexe en fait partie.
                growing = (fi - ev.first_frame) <= self.growth_frames
                subs = self._new_structures(d, new, ev, min_new_len) if (allow_new and not growing) else []
                upd = d.mask.copy()
                for sm in subs:
                    upd &= ~sm
                # Une même composante peut contenir plusieurs traces déjà suivies :
                # - morceaux ALIGNÉS d'une même trace (trace pâle apparue par fragments) → fusion,
                #   sinon une seule trace physique serait comptée plusieurs fois ;
                # - traces qui se CROISENT → chacune reçoit les pixels de sa zone (carte des traces)
                #   pour qu'aucune ne soit « affamée » puis perdue.
                crossing = []
                for k in by_det.get(j, []):
                    if k == k0 or k in ev_match or k in absorbed:
                        continue
                    other = self.active[k]
                    if self._overlap(ev, other) > 0.3 or self._collinear(ev, other, self.min_track_px):
                        self._merge_into(ev, other, min_new_len)
                        absorbed.add(k)
                    else:
                        crossing.append(k)
                if crossing:
                    # Chaque pixel de la composante va à la trace la plus proche (propagation des
                    # étiquettes de la carte des traces) : la trace principale ne « rampe » pas
                    # le long de la trace croisée.
                    seeds = np.where(d.mask, ids, 0)
                    keep_ids = [k0 + 1] + [k + 1 for k in crossing]
                    seeds = np.where(np.isin(seeds, keep_ids), seeds, 0)
                    if seeds.any():
                        inds = distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
                        near = seeds[inds[0], inds[1]]
                    else:
                        near = seeds
                    for k in crossing:
                        share = upd & (near == k + 1)
                        if int(share.sum()) >= min_share:
                            self._update_event(self.active[k], d, share, diff, E, sigma, frame, fi, t)
                            updated.add(k)
                            upd &= ~share
                self._update_event(ev, d, upd, diff, E, sigma, frame, fi, t)
                spawn.extend(self._sub_detection(d, sm, E) for sm in subs)
            else:
                cov = int(np.count_nonzero(d.mask)) - int(np.count_nonzero(d.mask & idz))
                cov = cov / max(1, d.area)
                if cov > 0.25:
                    subs = self._new_structures(d, new, ref_ev, min_new_len)
                    spawn.extend(self._sub_detection(d, sm, E) for sm in subs)
                    continue
                if ghost is not None:
                    gcov = (ghost[d.y0:d.y1, d.x0:d.x1] & d.mask).sum() / max(1, d.area)
                    if gcov > 0.4:
                        continue
                spawn.append(d)
        for k, ev in enumerate(self.active):
            if k not in updated:
                ev.missed += 1
        if absorbed:
            self.active = [ev for k, ev in enumerate(self.active) if k not in absorbed]
        spawn = [d for d in spawn if d is not None and d.area >= self.cfg.detection.min_area_px]
        for d in self._group_simultaneous(spawn, E):
            self._spawn(d, diff, E, sigma, frame, fi, t)

    def _group_simultaneous(self, dets: list[Detection], E: np.ndarray) -> list[Detection]:
        """Regroupe les nouvelles détections voisines apparues à la même image
        (une trace en pointillés ou sinueuse peut naître en plusieurs morceaux)."""
        if len(dets) < 2:
            return dets
        r = int(np.clip(0.5 * self.cfg.detection.merge_gap_mm * self.px_per_mm_proc, 2, 10))
        boxes = [(d.x0 - r, d.y0 - r, d.x1 + r, d.y1 + r) for d in dets]
        parent = list(range(len(dets)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        kern = ellipse_kernel(2 * r + 1)
        dil = [None] * len(dets)

        def dilated(i):
            if dil[i] is None:
                d = dets[i]
                m = np.zeros((d.y1 - d.y0 + 2 * r, d.x1 - d.x0 + 2 * r), np.uint8)
                m[r:r + d.mask.shape[0], r:r + d.mask.shape[1]] = d.mask
                dil[i] = cv2.dilate(m, kern) > 0
            return dil[i]
        for a in range(len(dets)):
            for b in range(a + 1, len(dets)):
                A, B = boxes[a], boxes[b]
                x0, y0 = max(A[0], B[0]), max(A[1], B[1])
                x1, y1 = min(A[2], B[2]), min(A[3], B[3])
                if x1 <= x0 or y1 <= y0:
                    continue
                ma = dilated(a)[y0 - A[1]:y1 - A[1], x0 - A[0]:x1 - A[0]]
                mb = dilated(b)[y0 - B[1]:y1 - B[1], x0 - B[0]:x1 - B[0]]
                if (ma & mb).any():
                    parent[find(b)] = find(a)
        groups: dict[int, list[int]] = {}
        for i in range(len(dets)):
            groups.setdefault(find(i), []).append(i)
        out = []
        for members in groups.values():
            if len(members) == 1:
                out.append(dets[members[0]])
                continue
            x0 = min(dets[m].x0 for m in members)
            y0 = min(dets[m].y0 for m in members)
            x1 = max(dets[m].x1 for m in members)
            y1 = max(dets[m].y1 for m in members)
            mm = np.zeros((y1 - y0, x1 - x0), bool)
            for m in members:
                d = dets[m]
                mm[d.y0 - y0:d.y1 - y0, d.x0 - x0:d.x1 - x0] |= d.mask
            cx, cy, th, l1, l2 = _moments(mm)
            e = E[y0:y1, x0:x1][mm]
            out.append(Detection(x0, y0, x1, y1, mm, int(mm.sum()), cx + x0, cy + y0, th, l1, l2,
                                 float(e.sum()), float(e.mean()) if e.size else 0.0))
        return out

    @staticmethod
    def _overlap(a: _Active, b: _Active) -> float:
        """Recouvrement des masques précédents : |A∩B| / min(|A|, |B|).

        Deux événements qui se recouvrent largement suivent le même objet (doublon) ;
        deux traces qui se croisent ne se recouvrent qu'au point de croisement."""
        x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
        x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        ma = a.mask[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
        mb = b.mask[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]
        inter = int((ma & mb).sum())
        return inter / max(1, min(int(a.mask.sum()), int(b.mask.sum())))

    @staticmethod
    def _collinear(a: _Active, b: _Active, min_track_px: float) -> bool:
        """b est-il un fragment de la même trace que a (prolongement aligné, ou simple gouttelette) ?"""
        ea = a.length_q / max(a.width_q, 1.0)
        eb = b.length_q / max(b.width_q, 1.0)
        if ea < 2.5:
            return False                      # a n'est pas (encore) une trace allongée : on ne juge pas
        if eb >= 2.5 and b.length_q >= min_track_px:
            if _angle_diff(a.theta, b.theta) > math.radians(20):
                return False                  # deux traces allongées d'orientations différentes : croisement
        elif b.length_q >= min_track_px:
            return False                      # grosse tache non allongée : autre objet
        ux, uy = math.cos(a.theta), math.sin(a.theta)
        dx, dy = b.cx - a.cx, b.cy - a.cy
        lateral = abs(-dx * uy + dy * ux)
        along = abs(dx * ux + dy * uy)
        if lateral > max(4.0, 1.2 * max(a.width_q, b.width_q)):
            return False
        # Fragments d'une même trace : bout à bout (pas côte à côte comme deux traces parallèles)
        return b.length_q < min_track_px or along > 0.25 * a.length_q

    def _merge_into(self, dst: _Active, src: _Active, min_len: float):
        """Fusionne l'événement src dans dst (fragments d'une même trace)."""
        if src.max_len >= min_len and src.max_elong >= 2.2 and src.first_frame < dst.first_frame:
            dst.first_frame, dst.t_first, dst.t_ana = src.first_frame, src.t_first, src.t_ana
        dst.n_seen = max(dst.n_seen, src.n_seen)
        dst.max_len = max(dst.max_len, src.max_len)
        dst.max_elong = max(dst.max_elong, src.max_elong)
        dst.max_mean_e = max(dst.max_mean_e, src.max_mean_e)
        # Deux morceaux comparables d'une même trace : les mesures d'un seul morceau ne
        # décrivent pas la trace entière → on les reprend sur la structure fusionnée (mise à
        # jour qui suit immédiatement). Absorption d'une simple gouttelette : on garde tout.
        if src.max_len >= 0.4 * max(dst.max_len, 1.0) and src.max_len >= min_len:
            dst.best, dst.feats, dst.feats_score = None, None, 0.0
            dst.best_len = dst.best_wid = dst.ref_len = dst.ref_wid = 0.0

    def _new_structures(self, d: Detection, new: np.ndarray, ev: _Active | None, min_len: float):
        """Parties nouvelles d'une détection qui ne prolongent PAS la trace existante."""
        if not new.any():
            return []
        n, lab = cv2.connectedComponents(new.view(np.uint8), connectivity=8)
        out = []
        min_a = max(self.cfg.detection.min_area_px, 8)
        for i in range(1, n):
            m = lab == i
            area = int(np.count_nonzero(m))
            if area < min_a:
                continue
            cx, cy, th, l1, l2 = _moments(m)
            L = math.sqrt(12 * l1)
            if L < min_len:
                continue
            if ev is not None:
                ux, uy = math.cos(ev.theta), math.sin(ev.theta)
                dx, dy = (cx + d.x0) - ev.cx, (cy + d.y0) - ev.cy
                lateral = abs(-dx * uy + dy * ux)
                if _angle_diff(th, ev.theta) < math.radians(20) and lateral < max(5.0, 2.0 * ev.width_q):
                    continue  # croissance de la même trace
            out.append(m)
        return out

    @staticmethod
    def _sub_detection(d: Detection, m: np.ndarray, E: np.ndarray) -> Detection | None:
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            return None
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        mm = m[y0:y1, x0:x1]
        cx, cy, th, l1, l2 = _moments(mm)
        X0, Y0 = d.x0 + x0, d.y0 + y0
        e = E[Y0:Y0 + (y1 - y0), X0:X0 + (x1 - x0)][mm]
        return Detection(X0, Y0, X0 + (x1 - x0), Y0 + (y1 - y0), mm, int(mm.sum()), cx + X0, cy + Y0,
                         th, l1, l2, float(e.sum()), float(e.mean()) if e.size else 0.0)

    def _spawn(self, d: Detection, diff, E, sigma, frame, fi, t):
        ev = _Active(self.next_uid, fi, t, fi, t, d.x0, d.y0, d.x1, d.y1, d.mask.copy(), d.theta,
                     d.cx, d.cy, d.length_q, d.width_q)
        ev.cycle = len(self.exp_times) - 1 if self.exp_enabled else -1
        ev.t_ana = self.analyzed_s
        self.next_uid += 1
        ev.n_seen = 0
        self.active.append(ev)
        self._update_event(ev, d, d.mask, diff, E, sigma, frame, fi, t)

    def _update_event(self, ev: _Active, d: Detection, mask: np.ndarray, diff, E, sigma, frame, fi, t):
        if mask is not d.mask and mask.sum() != d.area:
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                ev.missed += 1
                return
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            mask = mask[y0:y1, x0:x1]
            X0, Y0 = d.x0 + x0, d.y0 + y0
            cx, cy, th, l1, l2 = _moments(mask)
            e = E[Y0:Y0 + mask.shape[0], X0:X0 + mask.shape[1]][mask]
            d = Detection(X0, Y0, X0 + mask.shape[1], Y0 + mask.shape[0], mask, int(mask.sum()),
                          cx + X0, cy + Y0, th, l1, l2, float(e.sum()), float(e.mean()))
        # Une « gouttelette » (événement resté minuscule) qui devient soudain une vraie trace :
        # la trace commence MAINTENANT (on ne garde pas l'historique de la gouttelette voisine).
        # Historique « faible » : jamais une vraie trace (trop petite, jamais allongée, pas encore
        # confirmée ou bien moins contrastée). Si la détection devient soudain au moins 2× plus
        # grande, c'est une nouvelle trace qui apparaît à cet endroit : elle commence maintenant.
        weak_history = (ev.max_len < self.min_track_px or ev.max_elong < 2.2
                        or ev.n_seen < self.min_frames or ev.max_mean_e < 0.5 * d.mean_e)
        if (weak_history and ev.n_seen > 0 and d.length_q >= self.min_track_px and d.elongation >= 3.0
                and d.length_q >= 2.0 * max(ev.max_len, 1.0)):
            ev.first_frame, ev.t_first, ev.t_ana = fi, t, self.analyzed_s
            ev.n_seen = 0
            ev.best, ev.feats, ev.feats_score = None, None, 0.0
            ev.ref_len = ev.ref_wid = 0.0
        ev.max_len = max(ev.max_len, d.length_q)
        ev.max_elong = max(ev.max_elong, d.elongation)
        ev.max_mean_e = max(ev.max_mean_e, d.mean_e)
        ev.x0, ev.y0, ev.x1, ev.y1 = d.x0, d.y0, d.x1, d.y1
        ev.mask = d.mask
        ev.theta, ev.cx, ev.cy = d.theta, d.cx, d.cy
        ev.length_q, ev.width_q = d.length_q, d.width_q
        ev.n_seen += 1
        ev.missed = 0
        ev.last_frame, ev.t_last = fi, t
        # Contour (affichage) : calculé seulement si la trace est affichée (voir _contour)
        ev.contour_src = (d.mask, d.x0, d.y0)
        ev.last_mean_e = d.mean_e
        # « Meilleur instant » : trace complète ET nette. La netteté est maximale peu après la
        # formation (0,1–0,3 s) ; ensuite les gouttelettes diffusent (trace plus longue et floue).
        age = fi - ev.first_frame
        score = d.score / (1.0 + age / max(1.0, 0.5 * self.fps))
        if age > self.growth_frames and ev.ref_len <= 0 and ev.best is not None:
            ev.ref_len, ev.ref_wid = ev.best_len, ev.best_wid      # dimensions figées après la formation
        ref_len = ev.ref_len if ev.ref_len > 0 else ev.best_len
        ref_wid = ev.ref_wid if ev.ref_wid > 0 else ev.best_wid
        # Après la formation, une détection nettement plus longue ou plus large est presque
        # toujours la trace diffusée ou fusionnée avec une structure voisine.
        merged_look = (ev.best is not None and age > self.growth_frames
                       and (d.length_q > 1.25 * ref_len or d.width_q > 1.8 * max(ref_wid, 1.0)))
        if ev.best is None or (score > ev.best.score and not merged_look):
            self._store_best(ev, d, diff, sigma, frame, fi, t, score)
            ev.best_len, ev.best_wid = d.length_q, d.width_q
        # Mesures complètes (coûteuses) seulement pour les traces qui persistent,
        # et seulement à quelques moments clés (classification provisoire affichée en direct).
        # Le calcul lui-même est différé jusqu'à l'affichage de la trace (ou sa fin) : une
        # demande remplacée avant d'avoir servi n'est jamais calculée.
        if ev.n_seen >= 3 and ev.best is not None and ev.feats_score != ev.best.score:
            if ev.feats is None or ev.n_seen in (5, 9, 15) or ev.best.score > 1.4 * ev.feats_score:
                self._schedule_live(ev)

    def _contour(self, ev: _Active) -> np.ndarray | None:
        """Contour extérieur (coordonnées de traitement) du masque de la dernière mise à jour."""
        src = ev.contour_src
        if src is not None:
            ev.contour_src = None
            mask, x0, y0 = src
            cs, _ = cv2.findContours(mask.view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                c = max(cs, key=cv2.contourArea)
                ev.contour = c + np.array([[x0, y0]], np.int32)
        return ev.contour

    def _store_best(self, ev: _Active, d: Detection, diff, sigma, frame, fi, t, score):
        H, W = diff.shape
        pad = 10
        x0, y0 = max(0, d.x0 - pad), max(0, d.y0 - pad)
        x1, y1 = min(W, d.x1 + pad), min(H, d.y1 + pad)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[d.y0 - y0:d.y1 - y0, d.x0 - x0:d.x1 - x0] = d.mask
        dcrop = diff[y0:y1, x0:x1].copy()
        # Vignette couleur en pleine résolution
        s = self.scale
        fh, fw = frame.shape[:2]
        mg = 12
        X0, Y0 = max(0, int(x0 / s) - mg), max(0, int(y0 / s) - mg)
        X1, Y1 = min(fw, int(math.ceil(x1 / s)) + mg), min(fh, int(math.ceil(y1 / s)) + mg)
        crop = frame[Y0:Y1, X0:X1]
        f = 1.0
        if crop.size:
            f = min(1.0, THUMB_MAX / max(crop.shape[:2]))
            thumb = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else crop.copy()
        else:
            thumb = None
        ev.best = _Best(fi, t, score, x0, y0, x1, y1, m, dcrop, sigma, thumb, (X0, Y0, f))

    def _schedule_live(self, ev: _Active):
        """Demande des mesures « en direct » au meilleur instant actuel (calcul différé)."""
        ev.feats = _PendingFeats(ev.best, self.refs, self.cfg.classification)
        ev.feats_score = ev.best.score

    def _resolve_feats(self, ev: _Active, classify: bool) -> dict:
        """Effectue un calcul différé (mêmes entrées que s'il avait été fait à la demande)."""
        p = ev.feats
        if isinstance(p, _PendingFeats):
            b = p.best
            f = compute_features(b.mask, b.diff, self.px_per_mm_proc, b.sigma)
            ev.feats = f
            if classify:
                cls, conf, _sc, _r = classify_rules(dict(f), p.refs, p.cp, self.calibrated, self.px_per_mm_proc)
                ev.cls, ev.conf = cls, conf
        return ev.feats

    def _compute_live(self, ev: _Active):
        b = ev.best
        f = compute_features(b.mask, b.diff, self.px_per_mm_proc, b.sigma)
        ev.feats = f
        ev.feats_score = b.score
        cls, conf, _sc, _r = classify_rules(dict(f), self.refs, self.cfg.classification,
                                            self.calibrated, self.px_per_mm_proc)
        ev.cls, ev.conf = cls, conf

    # --------------------------------------------------------- finalisation --
    def _finalize_stale(self, force: bool = False) -> list[TrackEvent]:
        dp = self.cfg.detection
        keep, done = [], []
        for ev in self.active:
            if ev.missed > dp.max_missed or force and ev.missed > dp.max_missed:
                done.append(ev)
            else:
                keep.append(ev)
        self.active = keep
        out = []
        if not done:
            return out
        H, W = self.pre.roi_bool.shape
        r = self.cfg.detection.match_dilate_px + 3        # tolérance : dérive + diffusion de la trace
        kern = ellipse_kernel(2 * r + 1)
        min_track_px = self.min_track_px
        for ev in done:
            # « Fantôme » (masque élargi) : une trace qui s'estompe, se fragmente puis
            # réapparaît au même endroit n'est pas recomptée.
            if ev.max_len >= min_track_px and ev.max_elong >= 2.2:
                x0, y0 = max(0, ev.x0 - r), max(0, ev.y0 - r)
                x1, y1 = min(W, ev.x1 + r), min(H, ev.y1 + r)
                m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                m[ev.y0 - y0:ev.y1 - y0, ev.x0 - x0:ev.x1 - x0] = ev.mask
                self.ghosts.append((x0, y0, x1, y1, cv2.dilate(m, kern) > 0, self.frame_idx + self.ghost_frames))
            te = self._to_event(ev)
            if te is not None:
                if te.valid:
                    self.events.append(te)
                    self.n_valid += 1
                    self.valid_counts[te.final_cls] = self.valid_counts.get(te.final_cls, 0) + 1
                    out.append(te)
                else:
                    self._keep_invalid(te)
        # Références « traces fines » : recalculées toutes les 10 nouvelles traces valides
        # (compte incrémental ; recomptage exact au moment du recalcul, au cas où l'interface
        # aurait marqué des traces comme artefacts entre-temps).
        if self.n_valid >= 6 and self.n_valid - self._refs_count >= 10:
            valid = [e for e in self.events if e.valid]
            self.n_valid = len(valid)
            if len(valid) >= 6 and len(valid) - self._refs_count >= 10:
                self._refs_count = len(valid)
                self.refs = compute_references([e.features for e in valid], self.cfg.classification.ref_percentile)
        return out

    def _keep_invalid(self, te: TrackEvent) -> None:
        """Détection rejetée : gardée (diagnostic) dans la limite de max_invalid_kept."""
        if not self.keep_invalid_events:
            return
        self.events.append(te)
        self._invalid_fifo.append(te)
        cap = self.max_invalid_kept
        if cap is None or cap < 0 or len(self._invalid_fifo) <= cap:
            return
        # Retrait des plus anciennes par lots (10 % de la limite) : coût amorti négligeable
        target = max(0, cap - max(1, cap // 10)) if cap > 0 else 0
        drop = set()
        while len(self._invalid_fifo) > target:
            drop.add(id(self._invalid_fifo.popleft()))
        self.events[:] = [e for e in self.events if id(e) not in drop]

    def _count_rejection(self, reason: str) -> None:
        self.rejected += 1
        self.invalid_counts[reason] = self.invalid_counts.get(reason, 0) + 1

    @property
    def min_frames(self) -> int:
        dp = self.cfg.detection
        return max(dp.min_frames, int(math.ceil(dp.min_visible_s * self.fps)))

    def _to_event(self, ev: _Active) -> TrackEvent | None:
        dp = self.cfg.detection
        if ev.best is None or ev.n_seen < self.min_frames:
            self._count_rejection(REJECT_TOO_FEW_FRAMES)
            return None
        b = ev.best
        if ev.feats is None or ev.feats_score != b.score:
            # Mesures au meilleur instant (la classification « en direct » ne sert plus)
            ev.feats = compute_features(b.mask, b.diff, self.px_per_mm_proc, b.sigma)
            ev.feats_score = b.score
        f = dict(self._resolve_feats(ev, classify=False))
        path = f.pop("_path")
        # Touche le bord de la zone analysée ?
        ys, xs = np.nonzero(b.mask)
        border = bool((self.roi_dist[ys + b.y0, xs + b.x0] <= 2.5).any()) if xs.size else False
        path_full = np.column_stack([(path[:, 1] + b.x0) / self.scale, (path[:, 0] + b.y0) / self.scale]) \
            if len(path) else np.zeros((0, 2))
        thumb_mask = None
        if b.thumb is not None:
            X0, Y0, fac = b.thumb_origin
            th, tw = b.thumb.shape[:2]
            tm = np.zeros((th, tw), np.uint8)
            if len(path_full) >= 2:
                pts = np.round((path_full - [X0, Y0]) * fac).astype(np.int32)
                cv2.polylines(tm, [pts], False, 255, 1)
            thumb_mask = tm
        valid = True
        why = ""
        if f["length_mm"] < dp.min_length_mm:
            valid, why = False, "trop courte"
        elif f["elongation"] < dp.min_elongation:
            valid, why = False, "pas assez allongée (tache/gouttelette)"
        elif f["peak_snr"] < dp.min_peak_snr:
            valid, why = False, "contraste trop faible (bruit / brouillard)"
        elif f["elongation"] < 4.0 and f["length_mm"] < 10.0 and f["peak_snr"] < 2.0 * dp.min_peak_snr:
            valid, why = False, "tache floue courte (bouffée de brume / amas de gouttelettes)"
        te = TrackEvent(
            id=self.next_id if valid else -1,
            first_frame=ev.first_frame, last_frame=ev.last_frame, best_frame=b.frame_idx,
            t_first=ev.t_first, t_best=b.t, t_last=ev.t_last, n_seen=ev.n_seen,
            bbox_proc=(b.x0, b.y0, b.x1, b.y1),
            centroid_proc=(float(np.mean(xs) + b.x0) if xs.size else 0.0,
                           float(np.mean(ys) + b.y0) if ys.size else 0.0),
            features=f, path_full=path_full, thumb=b.thumb, thumb_mask=thumb_mask,
            touches_border=border, valid=valid, reject_reason=why,
            expansion_cycle=ev.cycle, px_per_mm_full=self.px_per_mm_full,
            extra={"scale": self.scale, "calibrated": self.calibrated, "t_ana": ev.t_ana},
        )
        if valid:
            self.next_id += 1
            self._classify_event(te)
        else:
            self._count_rejection(why)
        return te

    def _classify_event(self, te: TrackEvent):
        cp = self.cfg.classification
        cls, conf, scores, reason = classify_rules(te.features, self.refs, cp, self.calibrated,
                                                   self.px_per_mm_proc)
        method = "Règles physiques"
        hybrid_extra = None
        if self.ml is not None and cp.mode == "ia":
            try:
                cls_ml, conf_ml, sc_ml = self.ml.predict(te.features)
                reason = f"IA ({conf_ml:.0%}) — " + reason
                cls, conf, scores = cls_ml, conf_ml, sc_ml
                method = "IA locale (vos corrections)"
            except Exception:
                pass
        elif cp.mode == "hybride":
            if self.ref_model is None:
                try:
                    from .reference_model import ReferenceModel
                    self.ref_model = ReferenceModel.load_default()
                except Exception:
                    self.ref_model = None
            if self.ref_model is not None:
                # Règles physiques + IA de référence calibrée (5 classes, abstention sous le seuil)
                try:
                    from .reference_model import classify_hybrid_v2
                    cls, conf, scores, reason, hybrid_extra = classify_hybrid_v2(
                        self.ref_model, te.features, (cls, conf, scores, reason), cp,
                        te.duration_s, te.touches_border, self.calibrated)
                    method = hybrid_extra.get("method", method)
                except Exception:
                    hybrid_extra = None
        for k in ("rules_cls", "rules_conf", "rules_scores", "ml_proba", "w_ml", "agree", "capped", "abstained",
                  "competitors", "is_artefact", "threshold", "decision_method", "probs", "family",
                  "family_confidence", "ml_cls", "ml_conf", "ml_artefact"):
            te.extra.pop(k, None)            # informations d'une classification précédente
        if hybrid_extra:
            te.extra.update(hybrid_extra)
            proba = hybrid_extra.get("ml_proba") or {}
            if proba:
                top = max(proba, key=proba.get)
                te.extra["ml_cls"], te.extra["ml_conf"] = top, float(proba[top])
            fam_conf = float(hybrid_extra.get("family_confidence") or 0.0)
            thr = float(hybrid_extra.get("threshold") or cp.ml_threshold)
            # Abstention entre électron et muon alors que la famille est sûre : « trace fine (β ou μ) »
            comp = [c for c, _p in (hybrid_extra.get("competitors") or [])]
            if (cls == "incertain" and hybrid_extra.get("family") == "fine" and fam_conf >= min(thr, 0.85)
                    and all(c in ("beta", "muon") for c in comp[:2])):
                cls = "fine"
                conf = fam_conf
                reason = f"Trace fine (électron ou muon), famille sûre à {fam_conf:.0%} — " + reason
            # Adaptateur appris sur de VRAIES vidéos : rejette les artefacts, confirme les traces fines
            try:
                from .real_adapter import RealAdapter, apply as apply_adapter
                ad = RealAdapter.load_default()
                if ad is not None:
                    ap = ad.predict_proba(te.features, te.duration_s, te.touches_border,
                                          hybrid_extra.get("rules_scores") or {}, self.ref_model)
                    te.extra["adapter_proba"] = ap
                    cls, conf, reason, changed = apply_adapter(cls, conf, reason, ap)
                    if changed:
                        method = method + " + adaptateur vidéos réelles"
            except Exception:
                pass
        else:
            from .events import track_family
            te.extra["family"] = track_family(te.features)
        te.extra["method"] = method
        te.extra["cls_params"] = cp
        te.extra["phys_params"] = self.cfg.physics
        te.extra["calibrated"] = self.calibrated
        if cls == "artefact":
            # Détection jugée non physique par l'IA de référence : exclue des comptages (conservée
            # pour l'affichage et pour la reclassification finale).
            te.extra["ml_artefact"] = True
            if te.valid:
                te.valid = False
                tp = te.extra.pop("thumb_path", None)      # vignette déjà écrite sur disque : retirée
                if tp:
                    try:
                        import os
                        os.remove(tp)
                    except OSError:
                        pass
                te.reject_reason = f"artefact probable (IA de référence, {conf:.0%})"
                try:
                    self._count_rejection(te.reject_reason)
                except Exception:
                    self.rejected += 1
        elif te.extra.get("was_artefact"):
            te.valid = True
            te.reject_reason = ""
        te.cls, te.confidence, te.scores, te.reason = cls, conf, scores, reason
        physics_annotations(te.features, te.final_cls, self.cfg.physics, te.touches_border)
