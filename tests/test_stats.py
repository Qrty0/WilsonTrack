"""Statistiques de comptage (Poisson, Garwood, test binomial, dispersion, Rayleigh)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wilsontrack.analysis import stats


def _poisson_times(rate_per_s: float, duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate_per_s * duration_s)
    return np.sort(rng.uniform(0.0, duration_s, n))


# --------------------------------------------------------------------------- #
# rate()
# --------------------------------------------------------------------------- #
def test_rate_values():
    r = stats.rate(100, 120.0)          # 100 traces en 2 min
    assert r.n == 100
    assert r.per_min == pytest.approx(50.0)
    assert r.err == pytest.approx(math.sqrt(100) / 2.0)
    assert r.lo < r.per_min < r.hi
    # Garwood exact à 68.27 % pour N = 100 : [90.02 ; 111.03] événements (en 2 min)
    assert r.lo == pytest.approx(90.02 / 2.0, abs=0.02)
    assert r.hi == pytest.approx(111.03 / 2.0, abs=0.02)
    assert "±" in r.fmt() and "/min" in r.fmt()


def test_rate_garwood_tends_to_sqrt_n():
    """Pour N grand, l'intervalle de Garwood tend vers N ± √N (asymétrie ≈ +1/2 en haut)."""
    for n in (1000, 10000):
        r = stats.rate(n, 60.0)
        assert (r.hi - r.lo) / 2 == pytest.approx(math.sqrt(n), rel=0.02)
        assert r.hi - n > n - r.lo                  # asymétrie de la loi de Poisson


def test_rate_zero_counts_garwood_upper_bound():
    # N = 0, IC 68 % (bilatéral) : borne supérieure = −ln(0.1587) ≈ 1.84 événements
    r = stats.rate(0, 60.0)
    assert r.per_min == 0.0
    assert r.lo == 0.0
    assert r.hi == pytest.approx(1.841, abs=0.005)
    r2 = stats.rate(0, 300.0)            # 5 min → 1.84 / 5
    assert r2.hi == pytest.approx(1.841 / 5.0, abs=0.002)
    # incertitude non nulle même sans trace
    assert r.err > 0


def test_rate_garwood_small_counts():
    # Valeurs tabulées (Garwood 1936, 68.27 %) pour N = 1 et N = 5 (T = 1 min)
    r1 = stats.rate(1, 60.0)
    assert r1.lo == pytest.approx(0.173, abs=0.005)
    assert r1.hi == pytest.approx(3.300, abs=0.01)
    r5 = stats.rate(5, 60.0)
    assert r5.lo == pytest.approx(2.84, abs=0.02)
    assert r5.hi == pytest.approx(8.38, abs=0.03)


# --------------------------------------------------------------------------- #
# compare()
# --------------------------------------------------------------------------- #
def test_compare_detects_excess():
    c = stats.compare(100, 300.0, 50, 300.0, "beta")
    assert c.cls == "beta"
    assert c.p_value < 1e-3
    assert c.sigma > 3.0
    assert c.significant
    assert c.diff == pytest.approx(10.0)         # 20/min − 10/min
    assert c.ratio == pytest.approx(2.0)
    assert c.diff_err == pytest.approx(math.hypot(10 / 5, math.sqrt(50) / 5))


def test_compare_equal_counts_not_significant():
    c = stats.compare(50, 300.0, 50, 300.0)
    assert c.p_value == pytest.approx(0.5, abs=0.06)
    assert not c.significant
    assert c.diff == pytest.approx(0.0)


def test_compare_area_ratio_and_two_sided():
    # Même taux par unité de surface : échantillon sur 2× la surface → 2× plus de traces
    c = stats.compare(100, 300.0, 50, 300.0, "total", area_ratio=2.0)
    assert c.b.per_min == pytest.approx(20.0)            # fond ramené à la surface de l'échantillon
    assert c.diff == pytest.approx(0.0)
    assert not c.significant and c.p_value > 0.3
    # Test bilatéral : une BAISSE nette est détectée (p petit) mais n'est pas un « excès »
    d = stats.compare(20, 300.0, 80, 300.0, "alpha", alternative="two-sided")
    assert d.p_value < 1e-6 and d.sigma > 4.0
    assert not d.significant                             # significant = excès uniquement
    e = stats.compare(50, 300.0, 50, 300.0, alternative="two-sided")
    assert e.p_value == pytest.approx(1.0) and e.sigma == 0.0


def test_compare_deficit_and_edge_cases():
    # Déficit : test unilatéral (excès de l'échantillon) → pas significatif
    c = stats.compare(50, 300.0, 100, 300.0)
    assert c.p_value > 0.9 and not c.significant
    # Aucun comptage
    c0 = stats.compare(0, 60.0, 0, 60.0)
    assert c0.p_value == 1.0 and not c0.significant
    # Durées différentes : même taux → pas d'excès
    c2 = stats.compare(40, 120.0, 20, 60.0)
    assert c2.diff == pytest.approx(0.0)
    assert not c2.significant


# --------------------------------------------------------------------------- #
# dispersion_test()
# --------------------------------------------------------------------------- #
def test_dispersion_poisson_passes():
    for seed in (1, 2, 3):
        t = _poisson_times(0.5, 600.0, seed)
        d = stats.dispersion_test(t, 600.0)
        assert d is not None
        assert d.poisson_ok, (seed, d)
        assert 0.6 < d.index < 1.5


def test_dispersion_clustered_fails():
    rng = np.random.default_rng(7)
    centers = rng.uniform(0, 590, 10)
    t = np.sort(np.concatenate([c + rng.uniform(0, 3.0, 30) for c in centers]))
    d = stats.dispersion_test(t, 600.0)
    assert d is not None
    assert not d.poisson_ok
    assert d.p_value < 0.01
    assert d.index > 3.0


def test_dispersion_too_few_returns_none():
    assert stats.dispersion_test(np.array([1.0, 2.0, 3.0]), 60.0) is None
    assert stats.dispersion_test(np.arange(20, dtype=float), 0.0) is None


# --------------------------------------------------------------------------- #
# interarrival_test()
# --------------------------------------------------------------------------- #
def test_interarrival_poisson_compatible():
    t = _poisson_times(0.5, 600.0, 11)
    res = stats.interarrival_test(t)
    assert res is not None
    mu, p = res
    assert mu == pytest.approx(2.0, rel=0.2)
    assert p > 0.01


def test_interarrival_regular_rejected():
    t = np.arange(0.0, 300.0, 2.0)        # une trace toutes les 2 s exactement
    mu, p = stats.interarrival_test(t)
    assert mu == pytest.approx(2.0)
    # p-valeur de Monte-Carlo (≈ 2000 tirages) : la plus petite valeur possible est ≈ 5e-4
    assert p <= 1e-3


def test_interarrival_is_reproducible():
    t = _poisson_times(0.5, 300.0, 5)
    assert stats.interarrival_test(t) == stats.interarrival_test(t.copy())


def test_interarrival_too_few():
    assert stats.interarrival_test(np.arange(5, dtype=float)) is None


# --------------------------------------------------------------------------- #
# rayleigh_axial()
# --------------------------------------------------------------------------- #
def test_rayleigh_uniform_is_isotropic():
    rng = np.random.default_rng(3)
    R, p = stats.rayleigh_axial(rng.uniform(0, 180, 200))
    assert R < 0.2
    assert p > 0.05


def test_rayleigh_concentrated_is_not_isotropic():
    rng = np.random.default_rng(4)
    R, p = stats.rayleigh_axial((45 + rng.normal(0, 8, 60)) % 180)
    assert R > 0.8
    assert p < 1e-6


def test_rayleigh_is_axial():
    """2° et 178° sont presque la même orientation (données axiales, mod 180°)."""
    ang = np.array([1, 2, 3, 177, 178, 179, 0.5, 179.5, 2.5, 177.5] * 3, float)
    R, p = stats.rayleigh_axial(ang)
    assert R > 0.95
    assert p < 1e-6
    assert stats.rayleigh_axial(np.array([10.0, 20.0])) is None


# --------------------------------------------------------------------------- #
# time_series() et summarize()
# --------------------------------------------------------------------------- #
def test_time_series_sums():
    times = np.array([1, 5, 12, 15, 25, 29.9])
    classes = ["alpha", "beta", "beta", "muon", "muon", "muon"]
    ts = stats.time_series(times, classes, 30.0, bin_s=10.0)
    assert list(ts["edges"]) == [0, 10, 20, 30]
    assert list(ts["total"]) == [2, 2, 2]
    assert list(ts["muon"]) == [0, 1, 2]
    per_class = sum(ts[c] for c in stats.CLASSES)
    assert list(per_class) == list(ts["total"])


def test_summarize_sample_run(bg_run):
    rows = bg_run.table()
    s = stats.summarize(rows, bg_run.duration_s, bg_run.roi_area_mm2)
    assert s["total"].n == len(rows) > 0
    assert s["area_cm2"] == pytest.approx(bg_run.roi_area_mm2 / 100.0)
    assert sum(s["classes"][c]["rate"].n for c in stats.CLASSES) == len(rows)
    assert sum(s["classes"][c]["fraction"] for c in stats.CLASSES) == pytest.approx(1.0)
    for c, counts in bg_run.counts().items():
        cs = s["classes"][c]
        assert cs["rate"].n == counts
        assert cs["rate"].per_min == pytest.approx(counts / (bg_run.duration_s / 60.0))
        L, Le, n = cs["length"]
        assert n == counts and L > 0
    # 46 traces en 90 s : la dispersion et les temps d'arrivée sont calculables
    assert s["dispersion"] is not None and s["dispersion"].poisson_ok
    assert s["interarrival"] is not None
    # Les muons du fond : 30 orientations → test de Rayleigh
    assert s["muon_isotropy"] is not None
    em, ee, ne = s["alpha_energy"]
    assert ne == len(s["alpha_energies"]) >= 1
    assert 3.0 < em < 10.0


def test_event_times_analysis_clock_and_fallback():
    rows = [{"t_debut_s": 12.0, "t_analyse_s": 11.6}, {"t_debut_s": 20.0, "t_analyse_s": 19.1}]
    assert list(stats.event_times(rows)) == [11.6, 19.1]
    # Anciens résultats sans horloge d'analyse : t_debut_s ramené à 0
    old = [{"t_debut_s": 12.0}, {"t_debut_s": 20.0}]
    assert list(stats.event_times(old)) == [0.0, 8.0]
    assert stats.event_times([]).size == 0


def test_summarize_empty():
    s = stats.summarize([], 60.0, 10000.0)
    assert s["total"].n == 0
    assert s["dispersion"] is None and s["interarrival"] is None
    assert all(s["classes"][c]["fraction"] == 0.0 for c in stats.CLASSES)
