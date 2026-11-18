"""Moteur de détection 2.0 : non-régression, outils exacts et sessions longues.

1. Non-régression — les optimisations ne changent AUCUN résultat :
   - empreinte JSON durable (tests/data/engine_reference_v1_1.json, produite par le moteur 1.1
     avant optimisation : ``python tools/bench_engine.py --make-fixture …``) : événements valides
     ET rejetés, images, identifiants, toutes les mesures de compute_features, chemins, vignettes ;
   - comparaison directe avec le moteur de référence figé (copie des fichiers 1.1, chargée par
     tools/bench_engine.py ; test ignoré si la copie est absente) : en plus, classes, confiances,
     raisons, contours affichés en direct, images de diagnostic et rendu (draw_overlay).
2. Outils exacts : médiane, MAD, centile, seuils équivalents, étiquetage par remplissage, plus
   long chemin du squelette, moments — comparés aux calculs NumPy / OpenCV / SciPy d'origine.
3. Sessions longues : rejets bornés, vignettes délestées sur disque (chemin accentué),
   chronologie (contrat partagé), structures internes bornées.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from wilsontrack.config import AppConfig
from wilsontrack.detection import TrackDetector
from wilsontrack.detection import features as F
from wilsontrack.detection import preprocess as P
from wilsontrack.detection import segment as S

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "engine_reference_v1_1.json"
VIDEO = ROOT / "exemples" / "exemple fond (rayons cosmiques).mp4"
TIMELINE_KEYS = {"t_s", "wall_s", "frames", "disturbed_frames", "noise", "lit", "counts", "rejected"}


def _load_bench():
    if "bench_engine" in sys.modules:
        return sys.modules["bench_engine"]
    spec = importlib.util.spec_from_file_location("bench_engine", ROOT / "tools" / "bench_engine.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def be():
    return _load_bench()


@pytest.fixture(scope="module")
def ref_engine(be):
    rd = be.find_ref_dir()
    if rd is None:
        pytest.skip("Copie du moteur de référence 1.1 introuvable (WILSONTRACK_ENGINE_REF)")
    return be.load_reference_engine(rd)


def _synth(be, w=640, h=360, dur=6.0, seed=31, **kw):
    rates = kw.pop("rates_per_min", {"alpha": 40.0, "beta": 70.0, "muon": 70.0, "proton": 2.0})
    return be.synth_frames(w, h, dur, seed, None, stream=True, rates_per_min=rates, **kw)


# =========================================================================== #
# 1. Non-régression
# =========================================================================== #
def test_fixture_engine_identical(be):
    """Moteur actuel ≡ empreinte du moteur 1.1 (1280×720 : réduction 4/3, traces nombreuses)."""
    ref = json.loads(FIXTURE.read_text(encoding="utf-8"))
    run = be.run_engine(be.live_engine(), be.fixture_frames(None))
    live = be.engine_fingerprint(run)
    assert len(ref["events"]) >= 10 and sum(e["valid"] for e in ref["events"]) >= 8
    diffs = be.compare_fingerprints(ref, live)
    assert not diffs, "\n".join(diffs[:20])


def test_live_equals_reference_engine(be, ref_engine):
    """Comparaison complète avec le moteur figé : événements (classes, confiances, raisons…),
    images (contours affichés en direct), diagnostic (want_debug) et rendu."""
    fs = _synth(be, 960, 540, 7.0, seed=33)
    r_ref = be.run_engine(ref_engine, fs, render=True, render_check=True, debug_every=17)
    r_live = be.run_engine(be.live_engine(), fs, render=True, render_check=True, debug_every=17)
    assert len(r_ref["det"].valid_events) >= 5
    diffs = be.compare_runs(r_ref, r_live)
    assert not diffs, "\n".join(diffs[:20])


@pytest.mark.skipif(not VIDEO.exists(), reason="vidéo d'exemple absente")
def test_live_equals_reference_noisy_video(be, ref_engine):
    """Vidéo d'exemple « fond » : très nombreuses détections (rejets, croisements, fantômes,
    carte d'association par étiquettes, évidence complète)."""
    fs = be.video_frames(VIDEO, 75)
    r_ref = be.run_engine(ref_engine, fs, debug_every=29)
    r_live = be.run_engine(be.live_engine(), fs, debug_every=29)
    assert len(r_ref["det"].events) >= 20
    diffs = be.compare_runs(r_ref, r_live)
    assert not diffs, "\n".join(diffs[:20])


# =========================================================================== #
# 2. Outils exacts
# =========================================================================== #
def _random_arrays(rng, n_cases=300):
    for it in range(n_cases):
        n = int(rng.integers(1, 600)) if it % 25 else int(rng.integers(20000, 34000))
        dt = np.float32 if it % 3 else np.float64
        v = rng.normal(0, rng.uniform(0.01, 50), n).astype(dt)
        if it % 4 == 0:
            v = np.round(v)                # nombreux ex aequo
        if it % 5 == 0:
            v[::3] = -0.0
        yield v


def test_median_mad_percentile_exact():
    rng = np.random.default_rng(1)
    for v in _random_arrays(rng):
        med, srt = P.median_sorted(v)
        ref = np.median(v)
        assert type(med) is type(ref) and med == ref
        if v.dtype == np.float32:
            m = float(med)
            mad = P.mad_median(v, srt, m, np.empty(v.size, np.float32))
            ref_mad = np.median(np.abs(v - m))
            assert type(mad) is type(ref_mad) and mad == ref_mad
            assert float(max(1.4826 * mad, 1e-3)) == float(max(1.4826 * ref_mad, 1e-3))
        for q in (60, 70, 84.0, 90, 12.5):
            a, b = P.percentile(v, q), np.percentile(v, q)
            assert type(a) is type(b) and a == b


def test_gt_threshold_equivalent_to_multiply_then_compare():
    """(cv2.multiply(x, c) > k) ⟺ (x > seuil) pour toute image float32 x."""
    rng = np.random.default_rng(2)
    for _ in range(60):
        c = float(10 ** rng.uniform(-3, 3))
        k = float(rng.choice([2.2, 4.5, 3.7, 1.1, rng.uniform(0.5, 9.0)]))
        t = P.gt_threshold(c, k)
        assert t is not None
        x = np.concatenate([rng.normal(k / c, 3 * k / c, 4000),
                            np.float32(t) + np.arange(-200, 200) * np.spacing(np.float32(t))]).astype(np.float32)
        x = x.reshape(40, -1)
        want = cv2.multiply(x, c) > k                        # NumPy : comparaison en float32
        want_cv = cv2.compare(cv2.multiply(x, c), k, cv2.CMP_GT) > 0
        got = cv2.compare(x, t, cv2.CMP_GT) > 0
        assert np.array_equal(want, got) and np.array_equal(want_cv, got)


def test_fill_labeling_matches_opencv_component_order():
    """Composantes gardées (contenant une graine) : mêmes cadres, aires, masques et ORDRE que
    cv2.connectedComponentsWithStats (ordre par blocs 2×2), y compris avec plusieurs fils."""
    rng = np.random.default_rng(3)
    ws = P.Workspace()
    for it in range(150):
        h, w = int(rng.integers(10, 160)), int(rng.integers(10, 240))
        low = ((rng.uniform(0, 1, (h, w)) < rng.uniform(0.05, 0.6)) * 255).astype(np.uint8)
        if it % 2:
            low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, P.ellipse_kernel(3))
        seeds = ((rng.uniform(0, 1, (h, w)) < 0.03) & (low > 0)).astype(np.uint8)
        pts = cv2.findNonZero(seeds)
        if pts is None:
            continue
        a = S._kept_by_fill(low, pts, ws)
        if a is None:
            continue                       # > 254 composantes : repli (étiquetage complet)
        b = S._kept_by_labels(low, seeds > 0, ws)
        assert [t[:5] for t in a] == [t[:5] for t in b]
        assert all(np.array_equal(p[5], q[5]) for p, q in zip(a, b))
    fm = ws.zeros("seg_fill", (h + 2, w + 2), np.uint8)
    assert not fm[1:-1, 1:-1].any()        # masque de remplissage remis à zéro


def test_longest_skeleton_path_matches_scipy(monkeypatch):
    rng = np.random.default_rng(4)
    cases = []
    for it in range(400):
        h, w = int(rng.integers(3, 50)), int(rng.integers(3, 50))
        if it % 3 == 0:
            img = np.zeros((h, w), np.uint8)
            for _ in range(int(rng.integers(1, 4))):
                cv2.line(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                         (int(rng.integers(0, w)), int(rng.integers(0, h))), 1, int(rng.integers(1, 4)))
            from skimage.morphology import skeletonize
            cases.append(skeletonize(img > 0))
        else:
            cases.append(rng.uniform(0, 1, (h, w)) < rng.uniform(0.05, 0.6))
    fast = [F.longest_skeleton_path(sk) for sk in cases]
    monkeypatch.setattr(F, "_longest_path_exact", lambda *a: None)      # force la version SciPy
    for sk, p in zip(cases, fast):
        q = F.longest_skeleton_path(sk)
        assert p.shape == q.shape and np.array_equal(p, q)


def _moments_v1(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    sxx = float((dx * dx).mean()) + 1.0 / 12.0
    syy = float((dy * dy).mean()) + 1.0 / 12.0
    sxy = float((dx * dy).mean())
    tr = (sxx + syy) / 2
    disc = math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    l1, l2 = tr + disc, max(tr - disc, 1e-6)
    return float(cx), float(cy), 0.5 * math.atan2(2 * sxy, sxx - syy), l1, l2


def test_moments_exact():
    rng = np.random.default_rng(5)
    for _ in range(300):
        m = rng.uniform(0, 1, (int(rng.integers(1, 80)), int(rng.integers(1, 80)))) < rng.uniform(0.02, 0.9)
        assert S._moments(m) == _moments_v1(m)


def test_compat_apis_unchanged():
    """Preprocessor / EvidenceBuilder / segment sans tampons partagés : copies indépendantes,
    mêmes valeurs que l'évidence évaluée à la demande."""
    rng = np.random.default_rng(6)
    pre = P.Preprocessor((360, 640), 480, 1.0, 2, [], 0.0, False)
    frames = [rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(3)]
    outs = [pre(f) for f in frames]
    assert outs[0] is not outs[1] and not np.shares_memory(outs[1], outs[2])
    g = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32)
         for f in frames]
    assert np.array_equal(outs[2], cv2.GaussianBlur((g[1] + g[2]) / 2, (0, 0), 1.0))
    diff = (outs[2] - outs[0]).astype(np.float32)
    evb = P.EvidenceBuilder(True, (1.5, 3.0))
    seed, ext, sig, hp = evb(diff, pre)
    evb2 = P.EvidenceBuilder(True, (1.5, 3.0))
    ev = evb2.compute(diff, pre)
    assert sig == ev.sig and np.array_equal(ext, ev.ext_full()) and np.array_equal(seed, ev.seed_full())
    for k in (2.2, 4.5, 1.0):
        assert np.array_equal(ev.ext_mask(k, np.empty(ext.shape, np.uint8)) > 0, ext > k)
        assert np.array_equal(ev.seed_mask(k, np.empty(ext.shape, np.uint8)) > 0, seed > k)
    assert np.array_equal(ev[10:50, 20:90], ext[10:50, 20:90])
    a = S.segment(seed, ext, pre.roi_bool, 2.2, 4.5, 3, 12, 30.0, 14.0)
    b = S.segment_evidence(ev, pre.roi_bool, 2.2, 4.5, 3, 12, 30.0, 14.0)
    assert a[2] == b[2] and np.array_equal(a[1], b[1]) and len(a[0]) == len(b[0])
    for d1, d2 in zip(a[0], b[0]):
        assert (d1.x0, d1.y0, d1.x1, d1.y1, d1.area, d1.cx, d1.cy, d1.sum_e, d1.mean_e) == \
               (d2.x0, d2.y0, d2.x1, d2.y1, d2.area, d2.cx, d2.cy, d2.sum_e, d2.mean_e)


def test_draw_overlay_buffers_identical(be):
    from wilsontrack.render import draw_overlay
    fs = _synth(be, 640, 360, 3.0, seed=7)
    det = TrackDetector(AppConfig(), fs.fps, fs.hw)
    out = np.empty((360, 640, 3), np.uint8)
    n_drawn = 0
    for i, fr in enumerate(fs):
        res = det.process(fr, i / fs.fps)
        recent = [e for e in det.events[-60:] if e.valid]
        a = draw_overlay(fr, res, det, recent)
        b = draw_overlay(fr, res, det, recent, out=out)
        c = draw_overlay(fr.copy(), res, det, recent, inplace=True)
        assert b is out and np.array_equal(a, b) and np.array_equal(a, c)
        n_drawn += bool(res.active)
    assert n_drawn > 0


# =========================================================================== #
# 3. Sessions longues
# =========================================================================== #
def _run(fs, cfg=None, **attrs):
    det = TrackDetector(cfg or AppConfig(), fs.fps, fs.hw)
    for k, v in attrs.items():
        setattr(det, k, v)
    finished = []
    for i, fr in enumerate(fs):
        res = det.process(fr, i / fs.fps)
        finished += res.finished
    det.finalize_run()
    return det, finished


@pytest.fixture(scope="module")
def noisy_frames(be):
    """Entrée riche en détections rejetées (vidéo d'exemple, sinon simulation très bruitée)."""
    if VIDEO.exists():
        return be.video_frames(VIDEO, 150)
    return _synth(be, 640, 360, 5.0, seed=8, noise_sigma=7.0, rain_per_frame=2500)


def _sig(e):
    return (e.id, e.valid, e.first_frame, e.last_frame, e.best_frame, e.n_seen, e.cls, e.confidence,
            tuple(sorted((k, v) for k, v in e.features.items() if isinstance(v, (int, float)))))


def test_invalid_events_cap_and_counts(noisy_frames):
    full, _ = _run(noisy_frames, max_invalid_kept=None)
    capped, _ = _run(noisy_frames, max_invalid_kept=10)
    off, _ = _run(noisy_frames, keep_invalid_events=False)
    inv_full = [e for e in full.events if not e.valid]
    assert len(inv_full) > 20, "entrée trop pauvre en rejets pour ce test"
    for det in (full, capped, off):
        assert det.rejected == full.rejected
        assert det.invalid_counts == full.invalid_counts
        assert sum(det.invalid_counts.values()) == det.rejected
        assert [_sig(e) for e in det.valid_events] == [_sig(e) for e in full.valid_events]
    kept = [e for e in capped.events if not e.valid]
    assert 0 < len(kept) <= 10
    assert [_sig(e) for e in kept] == [_sig(e) for e in inv_full[-len(kept):]]   # les plus récentes
    assert not any(not e.valid for e in off.events)
    assert set(full.invalid_counts) <= {"trop courte", "pas assez allongée (tache/gouttelette)",
                                        "contraste trop faible (bruit / brouillard)",
                                        "tache floue courte (bouffée de brume / amas de gouttelettes)",
                                        "vue sur trop peu d'images"}


def test_spill_thumbnails_accented_folder(be, tmp_path):
    det, _ = _run(_synth(be, 640, 360, 6.0, seed=9))
    valid = det.valid_events
    assert len(valid) >= 5
    keep = 2
    folder = tmp_path / "vignettes délestées été"
    n = det.spill_thumbnails(folder, keep_recent=keep)
    assert n == len(valid) - keep
    for e in valid[:-keep]:
        p = Path(e.extra["thumb_path"])
        assert e.thumb is None and e.thumb_mask is None
        assert p.parent == folder and p.name == f"trace_{e.id:05d}_{e.final_cls}.png" and p.exists()
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.ndim == 3
    assert all(e.thumb is not None and "thumb_path" not in e.extra for e in valid[-keep:])
    assert det.spill_thumbnails(folder, keep_recent=keep) == 0       # rien de plus à délester
    assert det.spill_thumbnails(folder, keep_recent=0) == keep


def test_timeline_contract(be):
    fs = _synth(be, 640, 360, 8.0, seed=10)
    det = TrackDetector(AppConfig(), fs.fps, fs.hw)
    det.timeline_period_s = 2.0
    for i, fr in enumerate(fs):
        res = det.process(fr, i / fs.fps)
        if det.timeline and det.timeline[-1]["frames"] == det.n_frames_processed and not res.warming:
            e = det.timeline[-1]
            assert e["t_s"] == det.analyzed_s and e["wall_s"] == res.t
    periodic = list(det.timeline)
    assert len(periodic) == int(det.analyzed_s // 2.0)
    det.finalize_run()
    tl = det.timeline
    assert len(tl) == len(periodic) + 1 and tl[-1]["final"] and not any(e["final"] for e in tl[:-1])
    for a, b in zip(tl, tl[1:]):
        assert b["t_s"] >= a["t_s"] and b["frames"] >= a["frames"] and b["rejected"] >= a["rejected"]
        if not b.get("final"):   # la reclassification finale peut déplacer des traces d'une classe à l'autre
            assert all(b["counts"].get(k, 0) >= v for k, v in a["counts"].items())
    for k, e in enumerate(periodic):
        assert TIMELINE_KEYS <= set(e)
        assert e["t_s"] >= 2.0 * (k + 1) - 1e-9 and e["t_s"] < 2.0 * (k + 1) + 1.0 / fs.fps + 1e-9
        assert isinstance(e["frames"], int) and isinstance(e["counts"], dict)
        assert e["noise"] > 0 and 0.0 <= e["lit"] < 1.0
        json.dumps(e)
    last = tl[-1]
    assert last["frames"] == det.n_frames_processed and last["t_s"] == det.analyzed_s
    assert last["rejected"] == det.rejected and sum(last["counts"].values()) == len(det.valid_events)
    det.finalize_run()                                               # 2e appel : pas de doublon
    assert len(det.timeline) == len(tl)


def test_long_session_structures_bounded(be, tmp_path):
    """Simulation d'une minute à cadence élevée (≈ 270 traces) : aucune structure interne ne
    grandit sans borne (rejets, fantômes, tampons, moyenne glissante, chronologie)."""
    fs = _synth(be, 320, 180, 60.0, seed=11,
                rates_per_min={"alpha": 60.0, "beta": 100.0, "muon": 100.0, "proton": 3.0})
    det = TrackDetector(AppConfig(), fs.fps, fs.hw)
    det.max_invalid_kept = 15
    det.timeline_period_s = 10.0
    max_active = max_ghosts = 0
    ws_sizes = []
    for i, fr in enumerate(fs):
        det.process(fr, i / fs.fps)
        max_active = max(max_active, len(det.active))
        max_ghosts = max(max_ghosts, len(det.ghosts))
        if i % 300 == 299:
            ws_sizes.append(det.memory_report()["workspace_mb"])
    frame_mb = 180 * 320 * 4 / 2 ** 20
    assert max(ws_sizes) < 60 * frame_mb                         # tampons de travail de taille fixe
    assert sum(1 for e in det.events if not e.valid) <= 15
    assert max_active < 60 and max_ghosts < 60
    assert len(det._ghost_counted) == len(det.ghosts)
    assert len(det.timeline) == int(det.analyzed_s // 10.0)
    assert len(det.pre.buf) <= det.cfg.detection.temporal_avg and len(det.pre._free) <= 2
    n_valid = len(det.valid_events)
    assert n_valid >= 20
    assert det.n_valid == n_valid and sum(det.valid_counts.values()) == n_valid
    det.spill_thumbnails(tmp_path / "vignettes", keep_recent=10)
    assert sum(1 for e in det.events if e.thumb is not None) <= 10 + 15


def test_expansion_history_bounded(be):
    cfg = AppConfig()
    cfg.expansion.enabled = True
    fs = _synth(be, 320, 180, 6.0, seed=12)
    det = TrackDetector(cfg, fs.fps, fs.hw)
    for i, fr in enumerate(fs):
        det.process(fr, i / fs.fps)
    assert len(det._mean_hist) <= 90
