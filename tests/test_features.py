"""Mesures morphologiques d'une trace (detection.features) sur des traces de géométrie connue.

Les traces sont rendues comme une ligne de densité uniforme convoluée par une gaussienne
isotrope (σ) : profil transversal gaussien (FWHM = 2.355 σ) et intensité qui tombe à
mi-hauteur exactement aux extrémités → la longueur « à mi-hauteur » vaut la longueur vraie.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from wilsontrack.detection.features import (compute_features, fit_circle_taubin,
                                            longest_skeleton_path, resample_path)

PX_PER_MM = 4.0
AMP = 100.0
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


# --------------------------------------------------------------------------- #
# Rendu de traces synthétiques
# --------------------------------------------------------------------------- #
def render_segment(shape, p0, p1, sigma, amp=AMP) -> np.ndarray:
    """Segment rectiligne (x, y) → (x, y), rendu analytique exact (erf × gaussienne)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.hypot(*(p1 - p0)))
    u = (p1 - p0) / length
    rx, ry = xx - p0[0], yy - p0[1]
    s = rx * u[0] + ry * u[1]
    d = rx * u[1] - ry * u[0]
    k = math.sqrt(2.0) * sigma
    img = amp * np.exp(-d * d / (2 * sigma ** 2)) * 0.5 * (erf((length - s) / k) + erf(s / k))
    return img.astype(np.float32)


def render_curve(shape, pts_xy, sigma, amp=AMP, ds=0.25) -> np.ndarray:
    """Polyligne quelconque : somme de gaussiennes le long du chemin (pas ds)."""
    h, w = shape
    pts = np.asarray(pts_xy, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cs[-1], ds) + ds / 2
    cx, cy = np.interp(s, cs, pts[:, 0]), np.interp(s, cs, pts[:, 1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w))
    for i in range(0, len(s), 64):
        dx = xx[None] - cx[i:i + 64, None, None]
        dy = yy[None] - cy[i:i + 64, None, None]
        img += np.exp(-(dx * dx + dy * dy) / (2 * sigma ** 2)).sum(0)
    return (img * ds * amp / (math.sqrt(2 * math.pi) * sigma)).astype(np.float32)


def features_of(img: np.ndarray, thr: float = 0.25) -> dict:
    return compute_features(img > thr * AMP, img, PX_PER_MM, noise_sigma=1.0)


def _polyline_length(p: np.ndarray) -> float:
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def _random_walk(seed: int, n_steps: int = 70, step: float = 2.0, turn_sd: float = 0.35):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * math.pi)
    p = [np.zeros(2)]
    for _ in range(n_steps):
        a += rng.normal(0, turn_sd)
        p.append(p[-1] + step * np.array([math.cos(a), math.sin(a)]))
    p = np.array(p)
    p -= p.min(0)
    p += 15.0
    return p


# --------------------------------------------------------------------------- #
# Trace rectiligne
# --------------------------------------------------------------------------- #
LINE_LEN_PX = 120.0
LINE_ANGLE_DEG = 30.0


def _line(sigma):
    a = math.radians(LINE_ANGLE_DEG)
    p0 = (20.0, 90.0)
    p1 = (p0[0] + LINE_LEN_PX * math.cos(a), p0[1] - LINE_LEN_PX * math.sin(a))  # y vers le bas
    return render_segment((120, 160), p0, p1, sigma)


@pytest.mark.parametrize("sigma", [1.5, 2.0, 3.0])
def test_straight_line_length_width_straightness(sigma):
    f = features_of(_line(sigma))
    true_len_mm = LINE_LEN_PX / PX_PER_MM
    assert f["length_mm"] == pytest.approx(true_len_mm, rel=0.05)
    assert f["straightness"] > 0.97
    # Largeur à mi-hauteur du profil transversal
    fwhm = FWHM_PER_SIGMA * sigma
    assert f["width_fwhm_px"] == pytest.approx(fwhm, rel=0.10)
    assert f["width_mm"] == pytest.approx(fwhm / PX_PER_MM, rel=0.10)
    # Orientation de la corde (0–180°, sens trigonométrique, y image vers le bas)
    assert f["orientation_deg"] == pytest.approx(LINE_ANGLE_DEG, abs=1.5)
    # Une droite n'a pas de courbure mesurable
    assert f["radius_mm"] > 200.0 or math.isinf(f["radius_mm"])
    assert f["branches"] == 0
    assert f["sagitta_ratio"] < 0.01
    assert f["elongation"] > 10
    assert f["peak_snr"] > 50


def test_length_independent_of_threshold():
    """La longueur est mesurée sur le profil d'intensité, pas sur le masque binaire."""
    img = _line(2.0)
    lens = [features_of(img, thr)["length_mm"] for thr in (0.15, 0.25, 0.4)]
    assert max(lens) - min(lens) < 0.03 * LINE_LEN_PX / PX_PER_MM


def test_length_scales_with_calibration():
    img = _line(2.0)
    m = img > 25
    f4 = compute_features(m, img, 4.0)
    f8 = compute_features(m, img, 8.0)
    assert f4["length_px"] == pytest.approx(f8["length_px"])
    assert f4["length_mm"] == pytest.approx(2.0 * f8["length_mm"])


def test_straight_line_with_noise():
    rng = np.random.default_rng(0)
    img = _line(2.0) + rng.normal(0, 3.0, (120, 160)).astype(np.float32)
    f = compute_features(_line(2.0) > 25, img, PX_PER_MM, noise_sigma=3.0)
    assert f["length_mm"] == pytest.approx(LINE_LEN_PX / PX_PER_MM, rel=0.05)
    assert f["straightness"] > 0.97
    assert f["peak_snr"] == pytest.approx(AMP / 3.0, rel=0.25)


# --------------------------------------------------------------------------- #
# Arc de cercle
# --------------------------------------------------------------------------- #
def test_arc_radius_and_length():
    r_px = 60.0
    th = np.radians(np.linspace(200, 320, 400))
    pts = np.column_stack([100 + r_px * np.cos(th), 30 - r_px * np.sin(th)])
    f = features_of(render_curve((140, 200), pts, 1.5))
    assert f["radius_mm"] == pytest.approx(r_px / PX_PER_MM, rel=0.05)
    assert f["length_mm"] == pytest.approx(r_px * math.radians(120) / PX_PER_MM, rel=0.05)
    # corde / arc = 2 sin(60°) / (2π/3) ≈ 0.827
    assert f["straightness"] == pytest.approx(2 * math.sin(math.radians(60)) / math.radians(120), abs=0.03)
    assert f["circle_rms_mm"] < 0.1
    assert f["curl_consistency"] > 0.9       # courbure régulière, toujours du même côté


def test_fit_circle_taubin_points():
    rng = np.random.default_rng(1)
    th = np.radians(np.linspace(10, 100, 80))
    x = 100 + 50 * np.cos(th) + rng.normal(0, 0.3, th.size)
    y = 80 + 50 * np.sin(th) + rng.normal(0, 0.3, th.size)
    xc, yc, r, rms = fit_circle_taubin(x, y)
    assert r == pytest.approx(50.0, rel=0.05)
    assert xc == pytest.approx(100.0, abs=2.5)
    assert yc == pytest.approx(80.0, abs=2.5)
    assert rms < 0.6


def test_fit_circle_taubin_degenerate():
    x = np.linspace(0, 50, 30)
    assert math.isinf(fit_circle_taubin(x, 2 * x + 1)[2])       # points alignés
    assert math.isinf(fit_circle_taubin(x[:4], x[:4] ** 2)[2])   # trop peu de points
    assert math.isinf(fit_circle_taubin(np.ones(10), np.ones(10))[2])


# --------------------------------------------------------------------------- #
# Trace sinueuse (électron : marche aléatoire)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_curly_random_walk(seed):
    p = _random_walk(seed)
    shape = (int(p[:, 1].max() + 16), int(p[:, 0].max() + 16))
    f = features_of(render_curve(shape, p, 1.2))
    straight = features_of(_line(1.5))
    assert f["straightness"] < 0.9
    assert f["length_mm"] == pytest.approx(_polyline_length(p) / PX_PER_MM, rel=0.08)
    assert f["tortuosity"] > 3 * straight["tortuosity"]
    assert f["tortuosity"] > 0.1


# --------------------------------------------------------------------------- #
# Squelette : plus long chemin
# --------------------------------------------------------------------------- #
def _is_8_connected(path: np.ndarray) -> bool:
    d = np.abs(np.diff(path, axis=0))
    return bool(np.all(d.max(1) == 1))


def test_longest_skeleton_path_L_shape():
    sk = np.zeros((30, 40), bool)
    sk[5, 5:35] = True          # branche horizontale (30 px)
    sk[5:25, 34] = True         # branche verticale (20 px), coin commun (5, 34)
    path = longest_skeleton_path(sk)
    ends = {tuple(map(int, path[0])), tuple(map(int, path[-1]))}
    assert ends == {(5, 5), (24, 34)}
    assert _is_8_connected(path)
    # 29 + 19 pas de 1 px (48) — ou 47.41 si le coin est coupé en diagonale (√2)
    length = float(np.hypot(*np.diff(path, axis=0).T).sum())
    assert 28 + math.sqrt(2) + 18 - 1e-6 <= length <= 48 + 1e-6
    assert len(path) in (48, 49)


def test_longest_skeleton_path_ignores_short_spur():
    sk = np.zeros((20, 60), bool)
    sk[10, 5:55] = True         # trace principale (50 px)
    sk[4:10, 30] = True         # petite branche latérale (6 px)
    path = longest_skeleton_path(sk)
    ends = {tuple(map(int, path[0])), tuple(map(int, path[-1]))}
    assert ends == {(10, 5), (10, 54)}
    assert _is_8_connected(path)


def test_longest_skeleton_path_trivial():
    assert longest_skeleton_path(np.zeros((5, 5), bool)).shape == (0, 2)
    one = np.zeros((5, 5), bool)
    one[2, 3] = True
    assert longest_skeleton_path(one).tolist() == [[2.0, 3.0]]


def test_resample_path_unit_steps():
    p = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    r = resample_path(p, 1.0)
    assert len(r) == 21
    steps = np.hypot(*np.diff(r, axis=0).T)
    assert np.allclose(steps, 1.0)
