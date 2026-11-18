"""Conclusions scientifiques automatiques (en français)."""
from __future__ import annotations

import copy

import pytest

from wilsontrack.analysis.conclusions import (Conclusion, conclude_comparison, conclude_run,
                                              conclusions_text, expected_sample)
from wilsontrack.analysis.stats import CLASSES

LEVELS = {"info", "succes", "attention"}
# Mots français attendus dans les textes (aucun texte ne doit être en anglais)
FRENCH_MARKERS = ("traces", "taux", "fond", "échantillon", "alphas", "enregistrement", "loi", "min")


def _check_french(items: list[Conclusion]):
    assert items, "aucune conclusion"
    for c in items:
        assert isinstance(c, Conclusion)
        assert c.title.strip() and c.text.strip()
        assert c.level in LEVELS
        low = (c.title + " " + c.text).lower()
        assert any(w in low for w in FRENCH_MARKERS), c
        for en in (" the ", " rate ", " background ", " significant "):
            assert en not in f" {low} ", (en, c)


def _titles(items):
    return [c.title for c in items]


# --------------------------------------------------------------------------- #
# Un enregistrement seul
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("which", ["bg_run", "kcl_run"])
def test_conclude_run_sample(which, request):
    run = request.getfixturevalue(which)
    summary, items = conclude_run(run)
    _check_french(items)
    titles = _titles(items)
    assert titles[0] == "Bilan des observations"
    assert f"{run.total} traces" in items[0].text
    # 90 s calibrées, ≥ 20 traces : pas d'avertissement de calibration ni de faible statistique
    assert "Échelle non calibrée" not in titles
    assert "Statistique limitée" not in titles
    assert "Caractère aléatoire (loi de Poisson)" in titles
    assert "Particules alpha" in titles and "Muons cosmiques" in titles
    assert summary["total"].n == run.total
    txt = conclusions_text(items)
    assert txt.count("**") == 2 * len(items)


def test_conclude_run_warnings_for_small_uncalibrated_run(bg_run):
    run = copy.copy(bg_run)
    run.rows = bg_run.rows[:5]
    run.calibrated = False
    _summary, items = conclude_run(run)
    _check_french(items)
    titles = _titles(items)
    assert "Échelle non calibrée" in titles
    assert "Statistique limitée" in titles
    lvl = {c.title: c.level for c in items}
    assert lvl["Échelle non calibrée"] == "attention"


# --------------------------------------------------------------------------- #
# Échantillon contre fond
# --------------------------------------------------------------------------- #
def test_comparison_kcl_vs_background(kcl_run, bg_run):
    comps, items = conclude_comparison(kcl_run, bg_run)
    _check_french(items)
    assert set(CLASSES) | {"fines", "total"} <= set(comps)
    assert comps["beta"].significant or comps["fines"].significant
    assert comps["total"].diff > 0
    # Le K-40 n'émet pas d'alphas : pas d'excès significatif attendu (ni de baisse : test bilatéral)
    assert not comps["alpha"].significant
    if "alpha_bilateral" in comps:
        assert comps["alpha_bilateral"].p_value > 0.01
    titles = _titles(items)
    assert any(t.startswith("Radioactivité") and "KCl" in t for t in titles)
    assert "Contrôle : alphas inchangées" in titles
    detected = [c for c in items if c.title.startswith("Radioactivité")]
    assert all(c.level == "succes" for c in detected)
    # Activité attendue : 100 g de KCl ≈ 1.66 kBq
    act = [c for c in items if c.title == "Activité attendue (K-40)"]
    assert act and "Bq" in act[0].text and "efficacité" in act[0].text
    exp = expected_sample(kcl_run)
    assert exp is not None and exp.activity_bq == pytest.approx(1660, rel=0.02)


def test_comparison_with_itself_not_significant(bg_run, kcl_run):
    for run in (bg_run, kcl_run):
        comps, items = conclude_comparison(run, run)
        _check_french(items)
        assert not any(c.significant for c in comps.values())
        for c in comps.values():
            assert c.diff == pytest.approx(0.0)
        titles = _titles(items)
        assert not any(t.startswith("Radioactivité") for t in titles)
        assert not any(t == "Excès global" for t in titles)
        assert any(t.startswith("Pas d'excès β significatif") for t in titles)


def test_comparison_background_vs_kcl_not_significant(kcl_run, bg_run):
    """Inverser les rôles (fond pris comme « échantillon ») ne doit rien détecter (test unilatéral)."""
    comps, items = conclude_comparison(bg_run, kcl_run)
    assert not comps["beta"].significant and not comps["fines"].significant
    assert expected_sample(bg_run) is None     # pas d'échantillon → pas d'activité attendue
    assert "Activité attendue (K-40)" not in _titles(items)


def test_comparison_normalises_different_areas(bg_run):
    """Même enregistrement, surface analysée déclarée 2× plus petite : le taux du fond est ramené
    à la surface de l'échantillon (×0.5) → excès apparent, et un avertissement est émis."""
    small = copy.copy(bg_run)
    small.roi_area_mm2 = bg_run.roi_area_mm2 / 2.0
    comps, items = conclude_comparison(small, bg_run)
    _check_french(items)
    assert comps["total"].b.per_min == pytest.approx(0.5 * comps["total"].a.per_min)
    assert comps["total"].diff > 0
    assert "Surfaces analysées différentes" in _titles(items)


def test_zero_duration_is_handled(bg_run):
    empty = copy.copy(bg_run)
    empty.rows = []
    empty.duration_s = 0.0
    _summary, items = conclude_run(empty)
    _check_french(items)
    assert all(c.level == "attention" for c in items)
    comps, items = conclude_comparison(empty, bg_run)
    _check_french(items)
    assert not any(c.significant for c in comps.values())


def test_conclusion_markdown():
    c = Conclusion("Titre", "Texte.", "succes")
    md = c.as_markdown()
    assert md.startswith("**") and "Titre" in md and md.endswith("Texte.")
    assert conclusions_text([c, Conclusion("A", "B")]).count("\n\n") == 1
