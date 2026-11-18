"""Explication « pourquoi cette classe ? » (wilsontrack/detection/explain.py) et portées ASTAR."""
from __future__ import annotations

import copy
import math
import re
import time

import numpy as np
import pytest

from wilsontrack import physics
from wilsontrack.config import ClassificationParams, PhysicsParams
from wilsontrack.detection import explain
from wilsontrack.detection.classifier import References, classify_rules
from wilsontrack.detection.events import TrackEvent
from wilsontrack.detection.explain import (BASE_CLASSES, COMPLEMENTARY_LABEL, VERDICTS, Criterion, Explanation,
                                           explain_event, explain_features)

PPM = 3.84                                   # px/mm (échelle de traitement des vidéos d'exemple)
REFS = References(width_px=3.85, density=200.0, n=30, method="test")
CP = ClassificationParams()
PHP = PhysicsParams()
HYBRID = "Hybride (règles + IA de référence)"
FORBIDDEN = re.compile(r"\b(nan|NaN|inf|Inf|Infinity|None)\b")


def _track(length_mm, width_mm, density, straightness, tortuosity=0.0, sagitta=0.0, line_rms=0.0, lateral=0.0,
           peak_snr=40.0, asymmetry=1.1, branches=0):
    w_px = width_mm * PPM
    return {"length_mm": length_mm, "length_px": length_mm * PPM, "chord_mm": length_mm * straightness,
            "width_mm": width_mm, "width_px": w_px, "density": density, "straightness": straightness,
            "tortuosity": tortuosity, "sagitta_ratio": sagitta, "line_rms_mm": line_rms, "lateral_mm": lateral,
            "elongation": length_mm / max(width_mm, 0.1), "peak_snr": peak_snr, "asymmetry": asymmetry,
            "profile_cv": 0.3, "branches": branches, "curl_consistency": 0.5, "radius_mm": float("inf")}


ARCHETYPES = {
    # alpha : épaisse (×2,4), très dense (×4), droite, 38 mm
    "alpha": _track(38.0, 2.4, 800.0, 0.995, 0.01, 0.01, 0.05, 0.10, peak_snr=150.0, asymmetry=1.6),
    # électron lent : fine, sinueuse
    "beta": _track(30.0, 0.9, 180.0, 0.80, 0.15, 0.12, 1.5, 1.6, peak_snr=30.0),
    # muon : fine, longue et parfaitement droite
    "muon": _track(110.0, 0.85, 170.0, 0.999, 0.005, 0.003, 0.06, 0.2, peak_snr=35.0),
    # proton : épaisseur intermédiaire, droite, plus longue que toute alpha naturelle
    "proton": _track(120.0, 1.9, 520.0, 0.995, 0.01, 0.01, 0.10, 0.15, peak_snr=90.0),
    # courte et droite : muon presque vertical OU électron énergétique (ambiguïté réelle)
    "short_straight": _track(14.0, 0.95, 190.0, 0.999, 0.0, 0.005, 0.02, 0.10, peak_snr=40.0),
    # tache pâle et peu allongée (artefact selon l'IA)
    "blob": _track(7.0, 2.3, 260.0, 0.95, 0.05, 0.03, 0.2, 0.5, peak_snr=9.0),
}


def classified(name, calibrated=True):
    f = copy.deepcopy(ARCHETYPES[name])
    cls, conf, scores, reason = classify_rules(f, REFS, CP, calibrated, PPM)
    return f, cls, conf, scores


def _no_url(s: str) -> str:
    return re.sub(r"https?://\S+", "", s)


def check_common(ex: Explanation):
    """Invariants communs à toutes les explications."""
    assert isinstance(ex, Explanation)
    html, text, short = ex.to_html(), ex.to_text(), ex.short()
    assert "Pourquoi" in html
    for s in (html, text, short, ex.title, ex.summary, ex.physics):
        assert not FORBIDDEN.search(s), FORBIDDEN.search(s).group(0)
        assert not re.search(r"\d\.\d", _no_url(s)), "point décimal au lieu d'une virgule"
    for d in (ex.why_not, ex.scores):
        for k, v in d.items():
            assert not FORBIDDEN.search(str(v))
    assert 0 < len(short) <= 160
    assert "\n" not in short
    assert ex.cls_name == physics.CLASS_INFO[ex.cls]["nom"]
    assert ex.criteria, "au moins un critère"
    for c in ex.criteria:
        assert isinstance(c, Criterion)
        assert c.verdict in VERDICTS
        assert 0.0 <= c.strength <= 1.0
        assert c.label and c.measured and c.expected and c.explanation
        assert isinstance(c.used_for_decision, bool)
    for k in ex.scores:
        assert k in BASE_CLASSES or k == "artefact"
        assert 0.0 <= ex.scores[k] <= 1.0
    if ex.scores:
        assert sum(ex.scores.values()) == pytest.approx(1.0, abs=1e-3)
    assert ex.sources and all(isinstance(s, str) for s in ex.sources)
    assert 0.0 <= ex.confidence <= 1.0
    return html, text, short


# --------------------------------------------------------------------------- #
# Archétypes passés par les règles physiques
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [("alpha", "alpha"), ("beta", "beta"), ("muon", "muon"),
                                           ("proton", "proton")])
def test_archetypes_rules_explained(name, expected):
    f, cls, conf, scores = classified(name)
    assert cls == expected
    before = copy.deepcopy(f)
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP, duration_s=0.8, calibrated=True)
    assert f == before or all((f[k] == before[k]) or (isinstance(f[k], float) and math.isnan(f[k]))
                              for k in before), "explain_features ne doit pas modifier les mesures"
    html, text, short = check_common(ex)
    assert ex.cls == expected
    assert physics.CLASS_INFO[expected]["nom"] in ex.title
    assert physics.CLASS_INFO[expected]["hex"] in html
    assert set(ex.why_not) == set(BASE_CLASSES) - {expected}
    for k, s in ex.why_not.items():
        assert re.search(r"\d", s), f"« pourquoi pas {k} » sans nombre : {s}"
    keys = [c.key for c in ex.criteria]
    for k in ("thickness", "density", "shape"):
        assert k in keys
    # Règles seules : la décision a utilisé les critères des règles, pas la durée
    dec = {c.key: c for c in ex.criteria}
    assert dec["thickness"].used_for_decision and dec["density"].used_for_decision
    assert not dec["duration"].used_for_decision
    assert COMPLEMENTARY_LABEL in html and COMPLEMENTARY_LABEL in text
    assert "Scores relatifs des règles" in text
    assert ex.method == "Règles physiques"


def test_alpha_details():
    f, cls, conf, scores = classified("alpha")
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP, duration_s=1.4)
    dec = {c.key: c for c in ex.criteria}
    assert dec["thickness"].verdict == "pour" and dec["density"].verdict == "pour"
    assert dec["length"].verdict == "pour"
    assert dec["duration"].verdict == "pour"
    assert "ASTAR" in ex.physics and "Bethe" in ex.physics and "Po-212" in ex.physics
    assert "trop dense" in ex.why_not["muon"] and "trop dense" in ex.why_not["beta"]
    assert any("ASTAR" in s for s in ex.sources)
    assert any("température" in c for c in ex.caveats)
    assert any("projection 2D" in c for c in ex.caveats)
    assert ex.short().startswith("α alpha")


def test_beta_and_muon_details():
    f, cls, conf, scores = classified("beta")
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP)
    dec = {c.key: c for c in ex.criteria}
    assert dec["shape"].verdict == "pour"
    assert "Highland" in ex.physics and "K-40" in ex.physics
    assert "pas assez droite" in ex.why_not["muon"]
    f, cls, conf, scores = classified("muon")
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP)
    dec = {c.key: c for c in ex.criteria}
    assert dec["shape"].verdict == "pour" and dec["length"].verdict == "pour"
    assert "Highland" in ex.why_not["beta"]           # un électron dévierait de quelques mm
    assert "110 mm" in ex.why_not["alpha"].replace(" ", " ")
    assert "cos²θ" in ex.physics and "muon probable" in ex.physics


def test_proton_why_not_alpha_uses_astar_range():
    f, cls, conf, scores = classified("proton")
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP)
    wn = ex.why_not["alpha"].replace(" ", " ")
    max_mm = physics.alpha_range_astar_cm(8.785, PHP.chamber_temp_c, PHP.pressure_kpa) * 10
    assert f"{max_mm:.0f} mm" in wn and "Po-212" in wn and "120 mm" in wn
    assert "PSTAR" in ex.physics


def test_short_straight_rules_cap_is_explained():
    f, cls, conf, scores = classified("short_straight")
    assert cls == "muon" and conf == pytest.approx(0.45)
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP)
    check_common(ex)
    assert "ambiguity" in [c.key for c in ex.criteria]
    assert any("plafonnée" in c for c in ex.caveats)
    assert "électron énergétique" in ex.summary
    assert "ambiguë" in ex.short()


# --------------------------------------------------------------------------- #
# Abstention (« incertaine »), correction manuelle, hybride, artefact
# --------------------------------------------------------------------------- #
def test_abstained_short_straight():
    f, _cls, _conf, rules_scores = classified("short_straight")
    probs = {"alpha": 0.02, "beta": 0.40, "muon": 0.57, "proton": 0.01}
    ml = {"alpha": 0.01, "beta": 0.45, "muon": 0.53, "proton": 0.01}
    ex = explain_features(f, "incertain", 0.57, probs, cp=CP, php=PHP, duration_s=0.3, method=HYBRID,
                          ml_proba=ml, rules_cls="muon", rules_scores=rules_scores, abstained=True,
                          threshold=0.85, family="fine")
    html, text, short = check_common(ex)
    assert ex.cls == "incertain"
    assert "Incertaine" in ex.title and "muon" in ex.title and "électron" in ex.title
    assert "57" in ex.title and "40" in ex.title
    summ = ex.summary.replace(" ", " ")
    assert "85 %" in summ                                  # seuil d'honnêteté
    assert "Highland" in summ and "ambiguïté physique réelle" in summ
    assert "sûre à 97 %" in summ                            # la famille « fine » est sûre
    assert set(ex.why_not) == set(BASE_CLASSES)
    assert "hypothèse la plus probable" in ex.why_not["muon"]
    assert "deuxième hypothèse" in ex.why_not["beta"]
    assert ex.criteria[0].key == "ambiguity" and ex.criteria[0].verdict == "contre"
    assert "voulue" in ex.physics
    assert short.startswith("? muon")


def test_rules_uncertain_without_abstention():
    f, _cls, _conf, scores = classified("short_straight")
    ex = explain_features(f, "incertain", 0.2, scores, cp=CP, php=PHP)
    check_common(ex)
    assert "Incertaine" in ex.title
    assert "seuil minimal" in ex.summary


def test_manual_label_disagreement():
    f, cls, conf, scores = classified("muon")
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP, manual_cls="alpha")
    html, text, short = check_common(ex)
    assert ex.cls == "alpha"
    assert "utilisateur" in ex.title and "Alpha" in ex.title
    assert "Désaccord" in ex.summary and "muon" in ex.summary
    assert ex.method.startswith("Correction manuelle")
    assert "Confiance de l'analyse automatique" in html
    assert any("étiquette" in c for c in ex.caveats)
    assert set(ex.why_not) == {"beta", "muon", "proton"}
    assert "selon les règles" in ex.why_not["muon"] and "vous avez choisi" in ex.why_not["muon"]
    # même chose si l'appelant passe la classe finale au lieu de la classe automatique
    ex2 = explain_features(f, "alpha", conf, scores, cp=CP, php=PHP, manual_cls="alpha")
    assert "Désaccord" in ex2.summary


def test_manual_label_agreement():
    f, cls, conf, scores = classified("alpha")
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP, manual_cls="alpha")
    check_common(ex)
    assert "d'accord avec votre choix" in ex.summary
    assert not any("étiquette" in c for c in ex.caveats)


def test_hybrid_disagreement_rules_muon_ai_beta():
    f, rules_cls, rules_conf, rules_scores = classified("muon")
    assert rules_cls == "muon"
    ml = {"alpha": 0.003, "beta": 0.95, "muon": 0.04, "proton": 0.007}
    blended = {"alpha": 0.01, "beta": 0.70, "muon": 0.28, "proton": 0.01}
    ex = explain_features(f, "beta", 0.70, blended, cp=CP, php=PHP, duration_s=0.4, method=HYBRID,
                          ml_proba=ml, rules_cls="muon", rules_scores=rules_scores)
    html, text, short = check_common(ex)
    dec = {c.key: c for c in ex.criteria}
    ai = dec["reference_ai"]
    assert "simulées" in ai.label and ai.verdict == "pour" and ai.used_for_decision
    assert dec["shape"].verdict == "contre"                 # les règles la voyaient droite
    assert dec["duration"].used_for_decision                # mesure de l'IA de référence
    assert "désaccord" in ex.summary
    assert "selon les règles" in ex.why_not["muon"]
    assert any("simulées" in c and "jamais vu votre chambre" in c for c in ex.caveats)
    assert any("en désaccord" in c for c in ex.caveats)
    assert "Probabilités (règles + IA de référence)" in text
    assert "IA" in short


def test_hybrid_agreement_and_model_info_cached():
    f, cls, conf, rules_scores = classified("alpha")
    ml = {"alpha": 0.97, "beta": 0.01, "muon": 0.01, "proton": 0.01}
    explain.clear_cache()
    for _ in range(3):
        ex = explain_features(f, "alpha", 0.96, ml, cp=CP, php=PHP, method=HYBRID, ml_proba=ml,
                              rules_cls="alpha", rules_scores=rules_scores)
    assert explain._model_info.cache_info().hits >= 2
    check_common(ex)
    assert "d'accord" in ex.summary
    info = explain._model_info()
    n = info.get("n_tracks_total")
    label = {c.key: c for c in ex.criteria}["reference_ai"].label.replace(" ", " ")
    if n:
        assert f"{int(n):,d}".replace(",", " ") in label


def test_artefact_from_reference_ai():
    f, _cls, _conf, rules_scores = classified("blob")
    ml = {"alpha": 0.05, "beta": 0.08, "muon": 0.05, "proton": 0.02, "artefact": 0.80}
    ex = explain_features(f, "artefact", 0.80, ml, cp=CP, php=PHP, duration_s=0.2, method=HYBRID,
                          ml_proba=ml, rules_scores=rules_scores)
    html, text, short = check_common(ex)
    assert ex.cls == "artefact"
    assert "Artefact" in ex.title
    assert "exclue des comptages" in ex.summary
    assert set(ex.why_not) == set(BASE_CLASSES)
    assert "brume" in ex.physics
    keys = [c.key for c in ex.criteria]
    assert "elongation" in keys and "contrast" in keys and "reference_ai" in keys
    assert {c.key: c for c in ex.criteria}["contrast"].verdict == "pour"     # 9 σ : pâle
    assert "artefact" in ex.scores
    assert physics.CLASS_INFO["artefact"]["hex"] in html
    assert short.startswith("×")


def test_artefact_considered_by_ai_adds_why_not_artefact():
    f, cls, conf, rules_scores = classified("alpha")
    ml = {"alpha": 0.90, "beta": 0.02, "muon": 0.02, "proton": 0.02, "artefact": 0.04}
    ex = explain_features(f, "alpha", 0.92, ml, cp=CP, php=PHP, method=HYBRID, ml_proba=ml,
                          rules_scores=rules_scores)
    check_common(ex)
    assert set(ex.why_not) == {"beta", "muon", "proton", "artefact"}
    assert ex.why_not["artefact"].startswith("Pas un artefact")


def test_local_ai_mode():
    f, cls, conf, rules_scores = classified("beta")
    probs = {"alpha": 0.02, "beta": 0.90, "muon": 0.06, "proton": 0.02}
    ex = explain_features(f, "beta", 0.90, probs, cp=CP, php=PHP, method="IA locale (vos corrections)")
    check_common(ex)
    assert "local_ai" in [c.key for c in ex.criteria]
    assert any("IA locale" in c for c in ex.caveats)
    assert "IA locale" in ex.summary


# --------------------------------------------------------------------------- #
# Robustesse : anciennes mesures, valeurs manquantes ou aberrantes
# --------------------------------------------------------------------------- #
def test_features_without_ev_keys_are_rebuilt():
    f, cls, conf, scores = classified("alpha")
    old = {k: v for k, v in f.items() if not k.startswith("ev_")}       # ancienne version : pas de clés ev_*
    snapshot = copy.deepcopy(old)
    ex_old = explain_features(old, cls, conf, scores, cp=CP, php=PHP)
    ex_new = explain_features(f, cls, conf, scores, cp=CP, php=PHP)
    assert old.keys() == snapshot.keys()                                  # entrée non modifiée
    check_common(ex_old)
    assert [c.verdict for c in ex_old.criteria] == [c.verdict for c in ex_new.criteria]
    assert any("recalculées" in c for c in ex_old.caveats)
    # mesures brutes seules (ni ev_*, ni width_rel / density_rel) : dégradation propre
    raw = copy.deepcopy(ARCHETYPES["muon"])
    ex_raw = explain_features(raw, "muon", 0.8, {"muon": 0.8, "beta": 0.1}, cp=CP, php=PHP)
    check_common(ex_raw)
    assert any("référence" in c.lower() for c in ex_raw.caveats)


def test_minimal_and_garbage_inputs():
    for f in ({}, {"length_mm": None, "width_mm": "abc"},
              {"length_mm": float("nan"), "width_mm": float("inf"), "straightness": None, "density": -1}):
        for cls in ("alpha", "beta", "muon", "proton", "incertain", "artefact", "inconnue"):
            ex = explain_features(f, cls, float("nan"), {"alpha": float("inf"), "beta": None}, cp=None, php=None)
            text = ex.to_text() + ex.to_html() + ex.short()
            assert not FORBIDDEN.search(text), (cls, FORBIDDEN.search(text).group(0))
            assert len(ex.short()) <= 160


def test_params_as_dicts_and_temperature():
    f, cls, conf, scores = classified("proton")
    ex_cold = explain_features(f, cls, conf, scores, cp=CP.__dict__.copy(), php={"chamber_temp_c": -25.0})
    ex_warm = explain_features(f, cls, conf, scores, cp=CP, php={"chamber_temp_c": 15.0})
    cold = physics.alpha_range_astar_cm(8.785, -25.0, PHP.pressure_kpa) * 10
    warm = physics.alpha_range_astar_cm(8.785, 15.0, PHP.pressure_kpa) * 10
    assert f"{cold:.0f}" in ex_cold.why_not["alpha"] and f"{warm:.0f}" in ex_warm.why_not["alpha"]


def test_uncalibrated_and_border_caveats():
    f, cls, conf, scores = classified("alpha", calibrated=False)
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP, calibrated=False, touches_border=True)
    check_common(ex)
    assert any("non calibrée" in c for c in ex.caveats)
    assert any("touche le bord" in c for c in ex.caveats)
    assert "minimum" in {c.key: c for c in ex.criteria}["length"].measured


def test_refs_not_ready_caveat():
    f = copy.deepcopy(ARCHETYPES["alpha"])
    cls, conf, scores, _ = classify_rules(f, References(), CP, True, PPM)
    ex = explain_features(f, cls, conf, scores, cp=CP, php=PHP)
    assert any("pas encore établie" in c for c in ex.caveats)


# --------------------------------------------------------------------------- #
# TrackEvent, rendu HTML, performances
# --------------------------------------------------------------------------- #
def _event(name, **extra):
    f, cls, conf, scores = classified(name)
    e = TrackEvent(id=7, first_frame=0, last_frame=30, best_frame=10, t_first=1.0, t_best=1.3, t_last=2.2,
                   n_seen=30, bbox_proc=(0, 0, 10, 10), centroid_proc=(5.0, 5.0), features=f,
                   path_full=np.zeros((2, 2)), cls=cls, confidence=conf, scores=scores,
                   extra={"calibrated": True, "method": "Règles physiques", **extra})
    return e


def test_explain_event_matches_explain_features():
    e = _event("alpha")
    ex = explain_event(e)
    ex2 = explain_features(e.features, e.cls, e.confidence, e.scores, cp=CP, php=PHP, duration_s=e.duration_s)
    assert ex.to_text() == ex2.to_text()
    e.manual_cls = "proton"
    exm = explain_event(e, CP, PHP)
    assert exm.cls == "proton" and "utilisateur" in exm.title
    # paramètres enregistrés dans extra (objets ou dicts)
    e2 = _event("proton", cls_params=CP.__dict__.copy(), phys_params={"chamber_temp_c": 15.0})
    warm = physics.alpha_range_astar_cm(8.785, 15.0, PHP.pressure_kpa) * 10
    assert f"{warm:.0f}" in explain_event(e2).why_not["alpha"]
    # la colonne « pourquoi » de l'export utilise short()
    assert e2.why_short() == explain_event(e2).short()


def test_html_sections_and_compact():
    f, _cls, _conf, rules_scores = classified("muon")
    ml = {"alpha": 0.01, "beta": 0.02, "muon": 0.96, "proton": 0.01}
    ex = explain_features(f, "muon", 0.95, ml, cp=CP, php=PHP, duration_s=0.5, method=HYBRID, ml_proba=ml,
                          rules_cls="muon", rules_scores=rules_scores)
    html = ex.to_html()
    for sec in ("Verdict", "Ce qui a été mesuré", "Pourquoi pas les autres ?", "Physique", "Limites", "Sources"):
        assert sec in html, sec
    assert "bgcolor" in html and 'href="https://' in html
    assert "#30a46c" in html                                    # ✔ vert
    assert html.count("<table") >= 2
    compact = ex.to_html(compact=True)
    assert "Pourquoi" in compact and "Sources" not in compact and "Physique" not in compact
    assert len(compact) < len(html)


def test_formatting_helpers():
    nb = " "
    assert explain._num(1234.5, 1) == f"1{nb}234,5"
    assert explain._num(-0.04, 1) == "0,0"
    assert explain._num(-3.25, 2) == "−3,25"
    assert explain._num(float("nan")) == "—"
    assert explain._pct(0.004) == f"<{nb}1{nb}%"
    assert explain._pct(0.9996) == f">{nb}99{nb}%"
    assert explain._pct(0.5) == f"50{nb}%"
    assert explain._rel(2.44) == "×2,4"


def test_performance_hundreds_of_tracks():
    items = [classified(n) for n in ("alpha", "beta", "muon", "proton", "short_straight")]
    explain_features(*items[0], cp=CP, php=PHP)                 # préchauffage
    t0 = time.perf_counter()
    for i in range(300):
        f, cls, conf, scores = items[i % len(items)]
        explain_features(f, cls, conf, scores, cp=CP, php=PHP, duration_s=0.5).short()
    assert time.perf_counter() - t0 < 6.0


# --------------------------------------------------------------------------- #
# physics.py : portées ASTAR et classe « artefact »
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("energy,temp,expected_cm", [
    (5.304, 20.0, 3.96), (5.49, 20.0, 4.18), (6.002, 20.0, 4.79), (7.687, 20.0, 7.08), (8.785, 20.0, 8.79),
    (5.304, -25.0, 3.36), (5.49, -25.0, 3.54), (6.002, -25.0, 4.06), (7.687, -25.0, 6.00), (8.785, -25.0, 7.44),
])
def test_alpha_range_astar_matches_nist(energy, temp, expected_cm):
    # Valeurs NIST ASTAR (air sec ; densité ramenée à T par la loi des gaz parfaits), ±1 %
    assert physics.alpha_range_astar_cm(energy, temp) == pytest.approx(expected_cm, rel=0.01)


def test_alpha_range_astar_properties():
    assert math.isnan(physics.alpha_range_astar_cm(0.0))
    assert math.isnan(physics.alpha_range_astar_cm(-1.0))
    assert math.isnan(physics.alpha_range_astar_cm(float("nan")))
    rs = [physics.alpha_range_astar_cm(e) for e in np.linspace(3.0, 10.0, 30)]
    assert all(b > a for a, b in zip(rs, rs[1:]))
    # air froid → portée plus courte, dans le rapport des densités
    assert physics.alpha_range_astar_cm(5.49, -25.0) == pytest.approx(
        physics.alpha_range_astar_cm(5.49, 15.0) / physics.air_density_factor(-25.0))
    # Geiger sous-estime la portée à haute énergie (≈ 4 % à 8,8 MeV, ASTAR)
    assert physics.alpha_range_astar_cm(8.785) > physics.alpha_range_cm(8.785)
    # la règle de Geiger existante est inchangée
    assert physics.alpha_range_cm(5.49, 15.0) == pytest.approx(0.318 * 5.49 ** 1.5)


def test_class_info_artefact_and_texts():
    info = physics.CLASS_INFO["artefact"]
    assert info["nom"] == "Artefact (brume, gouttes…)"
    assert info["symbole"] == "×" and info["hex"] == "#a0785a"
    assert tuple(info["couleur"]) == (0x5a, 0x78, 0xa0)          # BGR
    assert "comptages" in info["description"]
    assert "artefact" not in physics.COUNTED_CLASSES
    assert "Bragg" in physics.CLASS_INFO["alpha"]["description"]
    assert "traverse souvent tout le champ" not in physics.CLASS_INFO["muon"]["description"]
    assert "1 MeV" in physics.CLASS_INFO["beta"]["description"]
    assert "intermédiaire" in physics.CLASS_INFO["proton"]["description"]
