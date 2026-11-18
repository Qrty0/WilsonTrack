"""Constantes et relations physiques (K-40, alphas, impulsion en champ B)."""
from __future__ import annotations

import math
import re

import pytest

from wilsontrack import physics


# --------------------------------------------------------------------------- #
# Potassium-40
# --------------------------------------------------------------------------- #
def test_specific_activity_k_is_about_31_7_bq_per_g():
    a = physics.specific_activity_k()
    assert a == pytest.approx(31.7, rel=0.02)


def test_kcl_mass_fraction():
    frac = physics.SAMPLES["KCl (chlorure de potassium)"]
    assert frac == pytest.approx(39.0983 / 74.551, rel=1e-9)
    assert frac == pytest.approx(0.5245, abs=5e-4)
    # Substitut de sel = moitié de KCl ; pas d'échantillon = 0
    assert physics.SAMPLES["Substitut de sel (≈ 50 % KCl)"] == pytest.approx(0.5 * frac)
    assert physics.SAMPLES["Aucun"] == 0.0
    for name, f in physics.SAMPLES.items():
        assert 0.0 <= f <= 1.0, name


def test_sample_activity_100g_kcl_is_about_1_66_kbq():
    frac = physics.SAMPLES["KCl (chlorure de potassium)"]
    s = physics.sample_activity(100.0, frac)
    assert s.k_mass_g == pytest.approx(52.45, rel=1e-3)
    assert s.activity_bq == pytest.approx(1660.0, rel=0.02)
    assert s.beta_per_s == pytest.approx(s.activity_bq * physics.K40_BETA_BRANCH)
    assert s.gamma_per_s == pytest.approx(s.activity_bq * physics.K40_GAMMA_YIELD)
    # Les branches β⁻ et CE couvrent toutes les désintégrations
    assert physics.K40_BETA_BRANCH + physics.K40_EC_BRANCH == pytest.approx(1.0, abs=1e-3)
    txt = s.as_text()
    assert "activité" in txt and "Bq" in txt


def test_sample_activity_is_linear_in_mass():
    frac = physics.SAMPLES["Banane (chair, ≈ 3.6 g K/kg)"]
    a1 = physics.sample_activity(150.0, frac).activity_bq
    a2 = physics.sample_activity(300.0, frac).activity_bq
    assert a2 == pytest.approx(2 * a1)
    # Une banane (~150 g de chair) : quelques becquerels à ~20 Bq
    assert 5.0 < a1 < 30.0


# --------------------------------------------------------------------------- #
# Alphas : relation portée-énergie
# --------------------------------------------------------------------------- #
def test_air_density_factor():
    assert physics.air_density_factor(15.0) == pytest.approx(1.0)
    assert physics.air_density_factor(-25.0) == pytest.approx(288.15 / 248.15)
    assert physics.air_density_factor(15.0, 101.325 / 2) == pytest.approx(0.5)
    # Air froid → plus dense
    assert physics.air_density_factor(-25.0) > physics.air_density_factor(20.0)


def test_alpha_range_rn222_at_15c_is_about_4_1_cm():
    r = physics.alpha_range_cm(5.49, 15.0)
    assert r == pytest.approx(4.1, abs=0.1)
    # Air froid de la zone sensible (−25 °C) : portée plus courte du facteur de densité
    r_cold = physics.alpha_range_cm(5.49, -25.0)
    assert r_cold == pytest.approx(r / physics.air_density_factor(-25.0))
    assert r_cold < r


@pytest.mark.parametrize("energy", [4.0, 5.304, 5.49, 6.0, 7.687, 8.785])
@pytest.mark.parametrize("temp_c", [-25.0, 15.0, 22.0])
def test_alpha_range_energy_round_trip(energy, temp_c):
    r_mm = physics.alpha_range_cm(energy, temp_c) * 10.0
    assert physics.alpha_energy_from_range(r_mm, temp_c) == pytest.approx(energy, rel=1e-9)


def test_alpha_energy_from_range_monotonic_and_invalid():
    assert math.isnan(physics.alpha_energy_from_range(0.0))
    assert math.isnan(physics.alpha_energy_from_range(-3.0))
    es = [physics.alpha_energy_from_range(r) for r in (20, 30, 40, 50, 60)]
    assert all(b > a for a, b in zip(es, es[1:]))


def test_closest_alpha_emitter():
    assert physics.closest_alpha_emitter(5.5)[0] == "Rn-222"
    assert physics.closest_alpha_emitter(7.6)[0] == "Po-214"
    assert physics.closest_alpha_emitter(5.3)[0] == "Po-210"


# --------------------------------------------------------------------------- #
# Champ magnétique : p = 0.3 B R
# --------------------------------------------------------------------------- #
def test_momentum_from_curvature():
    p = physics.momentum_mev(300.0, 20.0)
    assert p == pytest.approx(1.8, rel=0.01)
    assert p == pytest.approx(299.792458 * 0.3 * 0.02)
    assert physics.momentum_mev(300.0, 20.0, charge=2) == pytest.approx(2 * p)
    assert physics.momentum_mev(300.0, 20.0, charge=-1) == pytest.approx(p)
    assert physics.momentum_mev(0.0, 20.0) == 0.0


def test_kinetic_energy_electron():
    me = physics.ELECTRON_MASS_MEV
    assert physics.kinetic_energy_mev(0.0, me) == 0.0
    # p = 1.8 MeV/c → T = √(p² + m²) − m ≈ 1.36 MeV
    t = physics.kinetic_energy_mev(1.7988, me)
    assert t == pytest.approx(1.359, abs=2e-3)
    assert (t + me) ** 2 == pytest.approx(1.7988 ** 2 + me ** 2)
    # Limite non relativiste : T ≈ p²/2m
    p = 0.005
    assert physics.kinetic_energy_mev(p, me) == pytest.approx(p * p / (2 * me), rel=1e-3)
    # Limite ultra-relativiste : T ≈ p − m
    p = 1000.0
    assert physics.kinetic_energy_mev(p, me) == pytest.approx(p - me, rel=1e-6)


def test_kinetic_energy_k40_endpoint_consistency():
    """Un électron de 1.311 MeV (Emax du K-40) a p = √((T+m)² − m²) ≈ 1.75 MeV/c ;
    l'aller-retour p → T est cohérent. Dans B = 100 mT, son rayon de courbure ≈ 58 mm."""
    me = physics.ELECTRON_MASS_MEV
    t = physics.K40_BETA_EMAX_MEV
    p = math.sqrt((t + me) ** 2 - me ** 2)
    assert p == pytest.approx(1.749, abs=0.002)
    assert physics.kinetic_energy_mev(p, me) == pytest.approx(t)
    r_mm = p / physics.momentum_mev(100.0, 1.0)          # p ∝ R : rayon pour p donné
    assert r_mm == pytest.approx(58.3, abs=0.2)
    assert physics.momentum_mev(100.0, r_mm) == pytest.approx(p)


# --------------------------------------------------------------------------- #
# Descriptions des classes
# --------------------------------------------------------------------------- #
def test_class_info_complete():
    for c in physics.COUNTED_CLASSES:
        info = physics.CLASS_INFO[c]
        for key in ("nom", "symbole", "couleur", "hex", "description"):
            assert key in info, (c, key)
        assert len(info["couleur"]) == 3 and all(0 <= v <= 255 for v in info["couleur"])
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", info["hex"])
        assert len(info["description"]) > 20
