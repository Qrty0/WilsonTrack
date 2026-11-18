"""Banc d'essai du moteur de détection : vitesse, mémoire et non-régression.

Compare le moteur ACTUEL (``wilsontrack/detection/*``, ``wilsontrack/render.py``) à un moteur de
RÉFÉRENCE figé : une copie des fichiers ``pipeline.py``, ``preprocess.py``, ``segment.py``,
``features.py`` (dossier ``<ref>/wilsontrack/detection``) et ``render.py`` (``<ref>/wilsontrack``).
Le moteur de référence réutilise les modules ACTUELS de tout le reste (configuration,
classification, événements, IA de référence) ainsi que la méthode actuelle
``TrackDetector._classify_event`` : la comparaison isole donc uniquement le moteur d'images.

Les deux moteurs traitent les MÊMES images, en alternance (mêmes conditions de charge du
processeur) ; seul ``TrackDetector.process`` est chronométré (lecture / rendu des images exclus).

Usage :
  python tools/bench_engine.py [--inputs video,s720,s1080] [--engines ref,live] [--reps 3]
         [--ref-dir DOSSIER] [--cache DOSSIER] [--max-frames N] [--check] [--render]
         [--profile] [--mem] [--long MINUTES] [--json FICHIER]

Entrées :
  video  = exemples/exemple fond (rayons cosmiques).mp4 (960×540, décodée à la volée) ;
  s720   = SyntheticChamber(SynthConfig(width=1280, height=720, duration_s=30, seed=5)) ;
  s1080  = idem en 1920×1080. Les images simulées sont rendues UNE fois puis gardées dans un
           fichier .npy (memmap) du dossier --cache.
"""
from __future__ import annotations

import argparse
import cProfile
import ctypes
import importlib
import io
import json
import math
import os
import pstats
import statistics
import subprocess
import sys
import time
import types
import zlib
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import wilsontrack  # noqa: E402,F401  (cv2.setNumThreads)
from wilsontrack.config import AppConfig  # noqa: E402

SCRATCH = Path(os.environ.get(
    "WILSONTRACK_SCRATCH",
    r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-Desktop-Chambre-de-wilson-programme"
    r"\d6e18416-39fc-45d6-957c-9b0e27e49c12\scratchpad"))
# Moteur de référence : copie figée du moteur 1.1 (avant optimisation) ; à défaut, la copie 1.0
# (identique pour le moteur d'images : seules l'IA de référence et _classify_event diffèrent,
# et _classify_event est de toute façon remplacée par la version actuelle).
DEFAULT_REF_DIRS = [SCRATCH / "engine_ref_v1_1", SCRATCH / "baseline_v1_0"]
VIDEO = ROOT / "exemples" / "exemple fond (rayons cosmiques).mp4"
STAGES = ("pretraitement", "fond_diff", "evidence", "segmentation", "association", "mesures",
          "fond_maj", "finalisation")


# --------------------------------------------------------------------------- #
# Chargement des moteurs
# --------------------------------------------------------------------------- #
def find_ref_dir(explicit: str | None = None) -> Path | None:
    cands = [Path(explicit)] if explicit else []
    env = os.environ.get("WILSONTRACK_ENGINE_REF")
    if env:
        cands.insert(0, Path(env))
    cands += DEFAULT_REF_DIRS
    for c in cands:
        if (c / "wilsontrack" / "detection" / "pipeline.py").exists():
            return c
    return None


def live_engine() -> types.SimpleNamespace:
    from wilsontrack import render
    from wilsontrack.detection import features, pipeline, preprocess, segment
    return types.SimpleNamespace(name="live", pipeline=pipeline, preprocess=preprocess, segment=segment,
                                 features=features, render=render, TrackDetector=pipeline.TrackDetector)


_REF_CACHE: dict[str, types.SimpleNamespace] = {}


def load_reference_engine(ref_root: str | Path, name: str = "wt_engine_ref") -> types.SimpleNamespace:
    """Charge les fichiers figés du moteur sous le nom de paquet ``name`` ; tous les autres
    modules (config, physics, classifier, events, reference_model) sont ceux de ``wilsontrack``."""
    ref_root = Path(ref_root)
    key = f"{name}|{ref_root}"
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    pkg_dir = ref_root / "wilsontrack"
    det_dir = pkg_dir / "detection"
    for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[k]
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(pkg_dir)]
    sys.modules[name] = pkg
    det_pkg = types.ModuleType(name + ".detection")
    det_pkg.__path__ = [str(det_dir)]
    sys.modules[name + ".detection"] = det_pkg
    pkg.detection = det_pkg
    for sub in ("config", "physics"):
        sys.modules[f"{name}.{sub}"] = importlib.import_module(f"wilsontrack.{sub}")
    for sub in ("classifier", "events", "reference_model"):
        try:
            sys.modules[f"{name}.detection.{sub}"] = importlib.import_module(f"wilsontrack.detection.{sub}")
        except Exception:      # pragma: no cover - module absent : la copie figée l'importera elle-même
            pass
    mods = {}
    for sub in ("segment", "preprocess", "features", "pipeline"):
        mods[sub] = importlib.import_module(f"{name}.detection.{sub}")
    render = importlib.import_module(f"{name}.render")
    from wilsontrack.detection.pipeline import TrackDetector as LiveTD
    RefTD = mods["pipeline"].TrackDetector
    # Classification ACTUELLE (le chef d'intégration la réécrit) et méthodes ajoutées depuis.
    RefTD._classify_event = LiveTD._classify_event
    for attr, val in LiveTD.__dict__.items():
        if attr not in RefTD.__dict__ and not attr.startswith("__"):
            setattr(RefTD, attr, val)
    orig_init = RefTD.__init__

    def __init__(self, cfg, fps, frame_shape_hw):
        orig_init(self, cfg, fps, frame_shape_hw)
        # Attributs d'instance ajoutés depuis (ex. chargement de l'IA de référence réécrit)
        live = LiveTD(cfg, fps, frame_shape_hw)
        for k, v in live.__dict__.items():
            if k not in self.__dict__:
                setattr(self, k, v)
        if hasattr(live, "ref_model"):
            self.ref_model = live.ref_model
        if hasattr(live, "ml"):
            self.ml = live.ml
    RefTD.__init__ = __init__
    eng = types.SimpleNamespace(name="ref", pipeline=mods["pipeline"], preprocess=mods["preprocess"],
                                segment=mods["segment"], features=mods["features"], render=render,
                                TrackDetector=RefTD, root=str(ref_root))
    _REF_CACHE[key] = eng
    return eng


# --------------------------------------------------------------------------- #
# Sources d'images
# --------------------------------------------------------------------------- #
class FrameSet:
    """Images d'entrée rejouables (mêmes images pour tous les moteurs)."""

    def __init__(self, name: str, fps: float, hw: tuple[int, int], n: int, getter=None, video: Path | None = None,
                 stream=None):
        self.name, self.fps, self.hw, self.n = name, fps, hw, n
        self._get = getter
        self.video = video
        self._stream = stream          # fabrique d'un itérateur (images rendues à la volée)

    def __iter__(self):
        if self._stream is not None:
            yield from self._stream()
        elif self.video is not None:
            cap = cv2.VideoCapture(str(self.video))
            for _ in range(self.n):
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr
            cap.release()
        else:
            for i in range(self.n):
                yield np.array(self._get(i))       # copie hors chronométrage (memmap → RAM)


def video_frames(path: Path = VIDEO, max_frames: int = 0) -> FrameSet:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Vidéo introuvable : {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    if max_frames:
        n = min(n, max_frames)
    return FrameSet("video", fps, hw, n, video=path)


def synth_frames(width: int, height: int, duration_s: float = 30.0, seed: int = 5, cache: Path | None = None,
                 max_frames: int = 0, stream: bool = False, **synth_kw) -> FrameSet:
    """Chambre simulée. ``cache`` : rendue une fois dans un .npy ; ``stream`` : rendue à la volée
    à chaque passage (aucun stockage, mêmes images à chaque fois) ; sinon gardée en mémoire."""
    from wilsontrack.synth import SynthConfig, SyntheticChamber
    sc = SynthConfig(width=width, height=height, duration_s=duration_s, seed=seed, **synth_kw)
    n = int(duration_s * sc.fps)
    name = f"synth{width}x{height}"
    if stream:
        nn = min(n, max_frames) if max_frames else n

        def gen():
            sim = SyntheticChamber(SynthConfig(width=width, height=height, duration_s=duration_s, seed=seed,
                                               **synth_kw))
            for _ in range(nn):
                yield sim.render_next()
        return FrameSet(name, sc.fps, (height, width), nn, stream=gen)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        extra = f"_{zlib.crc32(repr(sorted(synth_kw.items())).encode()):08x}" if synth_kw else ""
        fn = cache / f"synth_{width}x{height}_d{duration_s:g}_s{seed}{extra}.npy"
        if fn.exists():
            mm = np.load(fn, mmap_mode="r")
        else:
            print(f"  rendu de {n} images {width}×{height} → {fn.name} …", flush=True)
            tmp = fn.with_suffix(".tmp.npy")
            mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, height, width, 3))
            sim = SyntheticChamber(sc)
            for i in range(n):
                mm[i] = sim.render_next()
            mm.flush()
            del mm
            os.replace(tmp, fn)
            mm = np.load(fn, mmap_mode="r")
        if max_frames:
            n = min(n, max_frames)
        return FrameSet(name, sc.fps, (height, width), n, getter=lambda i: mm[i])
    sim = SyntheticChamber(sc)
    frames = [sim.render_next() for _ in range(n if not max_frames else min(n, max_frames))]
    return FrameSet(name, sc.fps, (height, width), len(frames), getter=lambda i: frames[i])


def get_input(key: str, cache: Path, max_frames: int = 0) -> FrameSet:
    if key == "video":
        return video_frames(VIDEO, max_frames)
    if key == "s720":
        return synth_frames(1280, 720, 30.0, 5, cache, max_frames)
    if key == "s1080":
        return synth_frames(1920, 1080, 30.0, 5, cache, max_frames)
    raise ValueError(key)


# --------------------------------------------------------------------------- #
# Chronométrage par étape
# --------------------------------------------------------------------------- #
@contextmanager
def stage_timers(eng: types.SimpleNamespace, acc: dict):
    """Remplace temporairement les points d'entrée des étapes par des versions chronométrées.
    ``acc[étape]`` cumule le temps (s) ; les étapes imbriquées (mesures ⊂ association /
    finalisation) sont des temps INCLUSIFS."""
    evb_cls = eng.preprocess.EvidenceBuilder
    targets = [
        (eng.preprocess.Preprocessor, "__call__", "pretraitement"),
        (eng.preprocess.BackgroundModel, "diff", "fond_diff"),
        (evb_cls, "compute" if "compute" in evb_cls.__dict__ else "__call__", "evidence"),
        (eng.pipeline, "segment_evidence" if hasattr(eng.pipeline, "segment_evidence") else "segment",
         "segmentation"),
        (eng.pipeline.TrackDetector, "_associate", "association"),
        (eng.pipeline, "compute_features", "mesures"),
        (eng.preprocess.BackgroundModel, "update", "fond_maj"),
        (eng.pipeline.TrackDetector, "_finalize_stale", "finalisation"),
    ]
    saved = []
    pc = time.perf_counter
    for obj, attr, stage in targets:
        orig = obj.__dict__.get(attr) if isinstance(obj, type) else getattr(obj, attr, None)
        if orig is None:
            continue
        acc.setdefault(stage, 0.0)

        def make(fn, st):
            def wrapper(*a, **k):
                t0 = pc()
                try:
                    return fn(*a, **k)
                finally:
                    acc[st] += pc() - t0
            return wrapper
        saved.append((obj, attr, orig))
        setattr(obj, attr, make(orig, stage))
    try:
        yield acc
    finally:
        for obj, attr, orig in saved:
            setattr(obj, attr, orig)


# --------------------------------------------------------------------------- #
# Exécution
# --------------------------------------------------------------------------- #
def frame_summary(res) -> tuple:
    act = tuple((int(uid), cls, float(conf), (c.tobytes() if c is not None else b""))
                for c, cls, conf, uid in res.active)
    fin = tuple((e.id, e.first_frame) for e in res.finished)
    return (res.frame_idx, float(res.t), res.warming, res.disturbed, res.in_window, res.expansion_triggered,
            float(res.noise_sigma), float(res.lit_frac), act, fin)


def run_engine(eng, frames: FrameSet, cfg: AppConfig | None = None, *, collect: bool = True,
               timers: bool = False, render: bool = False, render_check: bool = False,
               debug_every: int = 0, display_max_w: int = 1280, unlimited_invalid: bool = True) -> dict:
    """Traite toutes les images ; renvoie détecteur, temps par image, résumés par image…

    ``unlimited_invalid`` : toutes les détections rejetées sont gardées (comme la version 1.1),
    pour comparer la liste complète des événements."""
    if cfg is None:
        cfg = AppConfig()
        cfg.classification.mode = "regles"   # comparaison du MOTEUR (détection), pas de la décision IA
    det = eng.TrackDetector(cfg, frames.fps, frames.hw)
    if unlimited_invalid and hasattr(det, "max_invalid_kept"):
        det.max_invalid_kept = None
    times: list[float] = []
    rtimes: list[float] = []
    summ: list[tuple] = []
    rcrc: list[int] = []
    dbg: list[tuple] = []
    acc: dict[str, float] = {}
    pc = time.perf_counter
    ctx = stage_timers(eng, acc) if timers else _null()
    with ctx:
        for i, fr in enumerate(frames):
            want = bool(debug_every) and i % debug_every == 0
            t0 = pc()
            res = det.process(fr, i / frames.fps, want_debug=want)
            times.append(pc() - t0)
            if collect:
                summ.append(frame_summary(res))
            if want:
                dbg.append(tuple((k, zlib.crc32(np.ascontiguousarray(v).tobytes()), v.dtype.str, v.shape)
                                 for k, v in sorted(res.debug.items())))
            if render:
                img = fr
                h0, w0 = img.shape[:2]
                if w0 > display_max_w:      # comme ProcessingWorker._render
                    f = display_max_w / w0
                    img = cv2.resize(img, (display_max_w, int(round(h0 * f))), interpolation=cv2.INTER_AREA)
                recent = [e for e in det.events[-60:] if e.valid]
                t1 = pc()
                ann = eng.render.draw_overlay(img, res, det, recent, show_labels=True)
                rtimes.append(pc() - t1)
                if render_check:
                    rcrc.append(zlib.crc32(ann.tobytes()))
        t0 = pc()
        det.finalize_run()
        t_final = pc() - t0
    return {"det": det, "times": times, "render_times": rtimes, "summary": summ, "render_crc": rcrc,
            "debug": dbg, "stages": acc, "t_finalize": t_final, "n": len(times)}


@contextmanager
def _null():
    yield


# --------------------------------------------------------------------------- #
# Comparaison (non-régression)
# --------------------------------------------------------------------------- #
def _close(a, b, rel: float) -> bool:
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    if fa == fb:
        return True
    return abs(fa - fb) <= rel * max(abs(fa), abs(fb))


def _cmp_value(a, b, rel: float, where: str, out: list[str]):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            out.append(f"{where}: type {type(b).__name__}")
            return
        for k, v in a.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            if k not in b:
                out.append(f"{where}.{k}: absent")
                continue
            _cmp_value(v, b[k], rel, f"{where}.{k}", out)
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a2, b2 = np.asarray(a), np.asarray(b)
        if a2.shape != b2.shape:
            out.append(f"{where}: forme {a2.shape} ≠ {b2.shape}")
        elif a2.dtype.kind in "fc":
            if not np.allclose(a2, b2, rtol=rel, atol=1e-12, equal_nan=True):
                out.append(f"{where}: écart max {np.nanmax(np.abs(a2 - b2)):.3g}")
        elif not np.array_equal(a2, b2):
            out.append(f"{where}: tableaux différents")
        return
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            out.append(f"{where}: longueur {len(a)} ≠ {len(b) if hasattr(b, '__len__') else '?'}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp_value(x, y, rel, f"{where}[{i}]", out)
        return
    if isinstance(a, (int, float, np.integer, np.floating, bool, np.bool_)):
        if not _close(a, b, rel):
            out.append(f"{where}: {a!r} ≠ {b!r}")
        return
    if a != b:
        out.append(f"{where}: {a!r} ≠ {b!r}")


EVENT_FIELDS = ("id", "valid", "reject_reason", "first_frame", "last_frame", "best_frame", "t_first", "t_best",
                "t_last", "n_seen", "bbox_proc", "centroid_proc", "touches_border", "expansion_cycle", "cls",
                "confidence", "scores", "reason", "manual_cls", "px_per_mm_full", "features", "extra",
                "path_full", "thumb", "thumb_mask")


def compare_events(ref_events, live_events, rel: float = 1e-9, max_report: int = 30) -> list[str]:
    out: list[str] = []
    if len(ref_events) != len(live_events):
        out.append(f"nombre d'événements : {len(ref_events)} ≠ {len(live_events)}")
    for i, (a, b) in enumerate(zip(ref_events, live_events)):
        for fld in EVENT_FIELDS:
            va, vb = getattr(a, fld), getattr(b, fld)
            if fld in ("thumb", "thumb_mask"):
                if (va is None) != (vb is None) or (va is not None and not np.array_equal(va, vb)):
                    out.append(f"ev[{i}].{fld}: différent")
                continue
            _cmp_value(va, vb, rel, f"ev[{i}](id={a.id}).{fld}", out)
        if len(out) > max_report:
            break
    return out


def compare_runs(ref: dict, live: dict, rel: float = 1e-9) -> list[str]:
    out: list[str] = []
    da, db = ref["det"], live["det"]
    for k in ("rejected", "n_frames_processed", "n_frames_disturbed", "analyzed_s", "next_id", "frame_idx",
              "exp_times", "fps"):
        _cmp_value(getattr(da, k), getattr(db, k), rel, f"det.{k}", out)
    for k in ("width_px", "density", "n", "method"):
        _cmp_value(getattr(da.refs, k), getattr(db.refs, k), rel, f"det.refs.{k}", out)
    out += compare_events(da.events, db.events, rel)
    sa, sb = ref["summary"], live["summary"]
    if len(sa) != len(sb):
        out.append(f"images : {len(sa)} ≠ {len(sb)}")
    nbad = 0
    for x, y in zip(sa, sb):
        d: list[str] = []
        _cmp_value(x[:8], y[:8], rel, f"image {x[0]}", d)
        if x[8] != y[8]:
            # contours / classes affichées en direct : comparaison exacte des octets
            d.append(f"image {x[0]}: traces actives différentes")
        if x[9] != y[9]:
            d.append(f"image {x[0]}: traces finies {x[9]} ≠ {y[9]}")
        if d:
            nbad += 1
            if nbad <= 5:
                out += d
    if nbad > 5:
        out.append(f"… {nbad} images différentes au total")
    if ref["render_crc"] and live["render_crc"] and ref["render_crc"] != live["render_crc"]:
        nd = sum(1 for p, q in zip(ref["render_crc"], live["render_crc"]) if p != q)
        out.append(f"rendu (draw_overlay) différent sur {nd} images")
    if ref["debug"] and live["debug"] and ref["debug"] != live["debug"]:
        out.append("images de diagnostic (want_debug) différentes")
    return out


# --------------------------------------------------------------------------- #
# Empreinte « moteur » (indépendante de la classification) : fixture JSON durable
# --------------------------------------------------------------------------- #
FIXTURE_SYNTH = dict(width=1280, height=720, duration_s=6.0, seed=21,
                     rates_per_min={"alpha": 40.0, "beta": 60.0, "muon": 60.0, "proton": 2.0})
# Clés produites par features.compute_features (les autres sont ajoutées par la classification)
FEATURE_KEYS = ("area_px", "length_px", "chord_px", "length_mm", "chord_mm", "width_mask_px", "width_eff_px",
                "width_fwhm_px", "width_px", "width_mm", "width_mask_mm", "straightness", "line_rms_mm",
                "lateral_mm", "elongation", "turn_total_rad", "tortuosity", "curl_consistency", "sagitta_ratio",
                "radius_mm", "circle_rms_mm", "density", "peak_snr", "total_signal", "asymmetry", "profile_cv",
                "branches", "orientation_deg")


def fixture_frames(cache: Path | None = None) -> FrameSet:
    """Entrée de la fixture JSON (rendue à la volée si ``cache`` est None)."""
    cfg = dict(FIXTURE_SYNTH)
    return synth_frames(cfg.pop("width"), cfg.pop("height"), cfg.pop("duration_s"), cfg.pop("seed"), cache,
                        stream=cache is None, rates_per_min=cfg.pop("rates_per_min"))


def _num(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def engine_fingerprint(run: dict) -> dict:
    """Résultats du moteur d'images seulement (pas de classe ni de confiance) — JSON."""
    det = run["det"]
    evs = []
    for e in det.events:
        evs.append({
            "id": int(e.id), "valid": bool(e.valid), "reject_reason": e.reject_reason,
            "first_frame": int(e.first_frame), "last_frame": int(e.last_frame), "best_frame": int(e.best_frame),
            "t_first": float(e.t_first), "t_best": float(e.t_best), "t_last": float(e.t_last), "n_seen": int(e.n_seen),
            "bbox_proc": [int(v) for v in e.bbox_proc], "centroid_proc": [float(v) for v in e.centroid_proc],
            "touches_border": bool(e.touches_border), "expansion_cycle": int(e.expansion_cycle),
            "features": {k: _num(e.features[k]) for k in FEATURE_KEYS if k in e.features},
            "path_full": np.asarray(e.path_full, np.float64).tolist(),
            "thumb": [list(e.thumb.shape), zlib.crc32(np.ascontiguousarray(e.thumb).tobytes())]
            if e.thumb is not None else None,
            "thumb_mask": zlib.crc32(np.ascontiguousarray(e.thumb_mask).tobytes()) if e.thumb_mask is not None else None,
        })
    frames = [[s[0], s[1], s[2], s[3], s[6], s[7], [[a[0], zlib.crc32(a[3])] for a in s[8]], [list(f) for f in s[9]]]
              for s in run["summary"]]
    return {"detector": {"rejected": int(det.rejected), "n_frames_processed": int(det.n_frames_processed),
                         "n_frames_disturbed": int(det.n_frames_disturbed), "analyzed_s": float(det.analyzed_s),
                         "frame_idx": int(det.frame_idx), "next_id": int(det.next_id)},
            "events": evs, "frames": frames}


def compare_fingerprints(ref: dict, live: dict, rel: float = 1e-9) -> list[str]:
    out: list[str] = []
    _cmp_value(ref["detector"], live["detector"], rel, "det", out)
    if len(ref["events"]) != len(live["events"]):
        out.append(f"nombre d'événements : {len(ref['events'])} ≠ {len(live['events'])}")
    for i, (a, b) in enumerate(zip(ref["events"], live["events"])):
        _cmp_value(a, b, rel, f"ev[{i}]", out)
        if len(out) > 30:
            return out
    if len(ref["frames"]) != len(live["frames"]):
        out.append(f"images : {len(ref['frames'])} ≠ {len(live['frames'])}")
    nbad = 0
    for a, b in zip(ref["frames"], live["frames"]):
        d: list[str] = []
        _cmp_value(a, b, rel, f"image {a[0]}", d)
        if d:
            nbad += 1
            if nbad <= 5:
                out += d
    return out


def make_fixture(path: Path, ref_dir: str | None = None, cache: Path | None = None) -> dict:
    rd = find_ref_dir(ref_dir)
    if rd is None:
        raise SystemExit("Moteur de référence introuvable")
    eng = load_reference_engine(rd)
    fs = fixture_frames(cache)
    fp = engine_fingerprint(run_engine(eng, fs))
    fp["meta"] = {"synth": {k: v for k, v in FIXTURE_SYNTH.items()}, "engine": "référence (moteur 1.1, avant optimisation)",
                  "frames": fs.n, "cv2": cv2.__version__, "numpy": np.__version__}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp), encoding="utf-8")
    return fp


# --------------------------------------------------------------------------- #
# Mémoire (Windows : pic de mémoire privée engagée du processus)
# --------------------------------------------------------------------------- #
class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def process_memory() -> dict:
    if os.name != "nt":
        return {}
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    fn = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
    fn.restype = ctypes.c_int
    if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return {}
    return {"private_mb": pmc.PrivateUsage / 2 ** 20, "peak_private_mb": pmc.PeakPagefileUsage / 2 ** 20,
            "ws_mb": pmc.WorkingSetSize / 2 ** 20}


def mem_child(engine: str, inp: str, cache: Path, ref_dir: str | None, max_frames: int) -> dict:
    eng = live_engine() if engine == "live" else load_reference_engine(find_ref_dir(ref_dir))
    fs = get_input(inp, cache, max_frames)
    # Préchauffage des imports / caches (IA de référence, scipy…) avant la mesure
    warm = run_engine(eng, FrameSet(fs.name, fs.fps, fs.hw, 0, getter=lambda i: None), collect=False)
    del warm
    m0 = process_memory()
    r = run_engine(eng, fs, collect=False)
    m1 = process_memory()
    det = r["det"]
    return {"engine": engine, "input": inp, "frames": r["n"], "private_before_mb": m0.get("private_mb"),
            "peak_private_mb": m1.get("peak_private_mb"), "private_after_mb": m1.get("private_mb"),
            "events": len(det.events), "fps_median": 1.0 / statistics.median(r["times"])}


# --------------------------------------------------------------------------- #
# Session longue (audit des structures qui grandissent)
# --------------------------------------------------------------------------- #
def long_run(minutes: float, width: int = 640, height: int = 360, seed: int = 11, report_every_s: float = 60.0,
             spill_dir: Path | None = None, noise_sigma: float = 2.5, rain_per_frame: int = 250):
    """Session longue simulée (moteur actuel, réglages par défaut : rejets bornés à 2000) ;
    vignettes délestées chaque minute dans ``spill_dir`` (300 gardées en mémoire)."""
    import shutil
    import tempfile
    from wilsontrack.synth import SynthConfig, SyntheticChamber
    sc = SynthConfig(width=width, height=height, duration_s=minutes * 60.0, seed=seed,
                     rates_per_min={"alpha": 30.0, "beta": 60.0, "muon": 60.0, "proton": 1.0},
                     noise_sigma=noise_sigma, rain_per_frame=rain_per_frame)
    sim = SyntheticChamber(sc)
    cfg = AppConfig()
    det = live_engine().TrackDetector(cfg, sc.fps, (height, width))
    tmp = Path(tempfile.mkdtemp(prefix="wt_long_")) if spill_dir is None else Path(spill_dir)
    rows = []
    every = int(report_every_s * sc.fps)
    t0 = time.perf_counter()
    spent = 0.0
    for i in range(sim.n_frames):
        fr = sim.render_next()
        t1 = time.perf_counter()
        det.process(fr, i / sc.fps)
        spent += time.perf_counter() - t1
        if (i + 1) % every == 0:
            spilled = det.spill_thumbnails(tmp, keep_recent=300) if hasattr(det, "spill_thumbnails") else 0
            rep = det.memory_report() if hasattr(det, "memory_report") else {}
            rows.append({"min": round((i + 1) / sc.fps / 60.0, 2), "events": len(det.events),
                         "valid": sum(1 for e in det.events if e.valid), "rejected": det.rejected,
                         "active": len(det.active), "ghosts": len(det.ghosts),
                         "timeline": len(getattr(det, "timeline", [])), "spilled": spilled,
                         "thumbs_in_memory": rep.get("thumbs_in_memory"),
                         "workspace_mb": round(rep.get("workspace_mb", 0.0), 2),
                         "proc_ms_per_img": round(1000 * spent / (i + 1), 2),
                         **{k: round(v, 1) for k, v in process_memory().items()}})
            print(rows[-1], flush=True)
    det.finalize_run()
    rows.append({"wall_s": round(time.perf_counter() - t0, 1), "timeline_entries": len(det.timeline),
                 "invalid_counts": dict(det.invalid_counts)})
    print(rows[-1], flush=True)
    if spill_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


# --------------------------------------------------------------------------- #
def _fmt_stage_table(res_by_engine: dict, n: int) -> str:
    lines = [f"    {'étape':14s}" + "".join(f"{e:>12s}" for e in res_by_engine)]
    for st in STAGES + ("total",):
        vals = []
        for e, r in res_by_engine.items():
            v = r.get(st)
            vals.append(f"{1000 * v / max(n, 1):10.2f}ms" if v is not None else f"{'—':>12s}")
        lines.append(f"    {st:14s}" + "".join(vals))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="video,s720,s1080")
    ap.add_argument("--engines", default="ref,live")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--cache", default=str(SCRATCH / "bench_cache"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="vérifie l'égalité des résultats (réf. ↔ actuel)")
    ap.add_argument("--render", action="store_true", help="chronomètre aussi draw_overlay")
    ap.add_argument("--profile", action="store_true", help="cProfile du moteur actuel")
    ap.add_argument("--profile-engine", default="live")
    ap.add_argument("--mem", action="store_true", help="pic de mémoire (un sous-processus par moteur)")
    ap.add_argument("--long", type=float, default=0.0, help="session longue simulée (minutes)")
    ap.add_argument("--long-noise", type=float, default=2.5, help="bruit du capteur de la session longue")
    ap.add_argument("--long-rain", type=int, default=250, help="gouttelettes par image (session longue)")
    ap.add_argument("--threads", type=int, default=-1, help="cv2.setNumThreads (défaut : celui du paquet)")
    ap.add_argument("--json", default="")
    ap.add_argument("--make-fixture", default="", help="écrit l'empreinte JSON du moteur de référence")
    ap.add_argument("--child-mem", nargs=2, metavar=("ENGINE", "INPUT"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    if a.threads >= 0:
        cv2.setNumThreads(a.threads)
    if a.make_fixture:
        fp = make_fixture(Path(a.make_fixture), a.ref_dir, cache)
        live = engine_fingerprint(run_engine(live_engine(), fixture_frames(cache)))
        diffs = compare_fingerprints(fp, live)
        print(f"Fixture : {a.make_fixture} ({len(fp['events'])} événements) ; moteur actuel : "
              + ("IDENTIQUE" if not diffs else f"{len(diffs)} DIFFÉRENCES"))
        for d in diffs[:20]:
            print("   ", d)
        return 0
    if a.child_mem:
        print(json.dumps(mem_child(a.child_mem[0], a.child_mem[1], cache, a.ref_dir, a.max_frames)))
        return 0
    if a.long:
        rows = long_run(a.long, noise_sigma=a.long_noise, rain_per_frame=a.long_rain)
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return 0
    engines = {}
    for e in a.engines.split(","):
        if e == "live":
            engines["live"] = live_engine()
        elif e == "ref":
            rd = find_ref_dir(a.ref_dir)
            if rd is None:
                print("Moteur de référence introuvable (--ref-dir).")
                return 2
            engines["ref"] = load_reference_engine(rd)
            print(f"Référence : {rd}")
    report: dict = {"cv2_threads": cv2.getNumThreads(), "inputs": {}}
    for key in a.inputs.split(","):
        fs = get_input(key, cache, a.max_frames)
        print(f"\n=== {key} : {fs.n} images {fs.hw[1]}×{fs.hw[0]} à {fs.fps:g} img/s", flush=True)
        if a.profile:
            eng = engines.get(a.profile_engine) or live_engine()
            pr = cProfile.Profile()
            pr.enable()
            run_engine(eng, fs, collect=False, render=a.render)
            pr.disable()
            s = io.StringIO()
            st = pstats.Stats(pr, stream=s)
            st.sort_stats("tottime").print_stats(28)
            st.sort_stats("cumulative").print_stats(30)
            print(s.getvalue())
            continue
        if a.reps <= 0:
            continue
        per: dict[str, list] = {e: [] for e in engines}
        last: dict[str, dict] = {}
        for rep in range(a.reps):
            for name, eng in engines.items():
                r = run_engine(eng, fs, collect=a.check and rep == 0, timers=True, render=a.render,
                               render_check=a.check and rep == 0)
                st = dict(r["stages"])
                st["total"] = sum(r["times"])
                st["finalize_run"] = r["t_finalize"]
                st["render"] = sum(r["render_times"]) if r["render_times"] else None
                st["fps"] = r["n"] / st["total"]
                st["fps_median_frame"] = 1.0 / statistics.median(r["times"])
                st["p95_ms"] = 1000 * float(np.percentile(r["times"], 95))
                st["max_ms"] = 1000 * max(r["times"])
                st["events"] = len(r["det"].events)
                st["valid"] = len(r["det"].valid_events)
                per[name].append(st)
                if rep == 0:
                    last[name] = r
                print(f"  rép. {rep + 1} {name:4s}: {st['fps']:6.1f} img/s (médiane/image {st['fps_median_frame']:6.1f}, "
                      f"p95 {st['p95_ms']:.1f} ms)  traces {st['valid']}/{st['events']}"
                      + (f"  rendu {1000 * st['render'] / r['n']:.2f} ms/img" if st['render'] else ""), flush=True)
        med = {}
        for name, lst in per.items():
            med[name] = {k: (statistics.median([x[k] for x in lst]) if lst[0].get(k) is not None else None)
                         for k in lst[0]}
        n = fs.n
        print("  Temps médians par image (inclusifs ; mesures ⊂ association + finalisation) :")
        print(_fmt_stage_table(med, n))
        entry = {"frames": n, "hw": fs.hw, "median": med, "reps": per}
        if "ref" in med and "live" in med:
            sp = med["live"]["fps"] / med["ref"]["fps"]
            entry["speedup"] = sp
            print(f"  Accélération (médiane des débits) : ×{sp:.2f}")
        if a.check and "ref" in last and "live" in last:
            diffs = compare_runs(last["ref"], last["live"])
            entry["regression_diffs"] = diffs
            print("  Non-régression : " + ("IDENTIQUE" if not diffs else f"{len(diffs)} DIFFÉRENCES"))
            for d in diffs[:30]:
                print("    " + d)
        report["inputs"][key] = entry
        last.clear()
    if a.mem:
        report["memory"] = []
        for key in a.inputs.split(","):
            for name in engines:
                cmd = [sys.executable, str(Path(__file__).resolve()), "--child-mem", name, key, "--cache", str(cache)]
                if a.ref_dir:
                    cmd += ["--ref-dir", a.ref_dir]
                if a.max_frames:
                    cmd += ["--max-frames", str(a.max_frames)]
                out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
                try:
                    m = json.loads(out.stdout.strip().splitlines()[-1])
                except Exception:
                    m = {"engine": name, "input": key, "error": out.stderr[-2000:]}
                report["memory"].append(m)
                print("  mémoire", m, flush=True)
    if a.json:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, float) and not math.isfinite(o):
                return None
            return o
        Path(a.json).write_text(json.dumps(clean(report), indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
