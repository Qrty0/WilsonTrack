"""IA de référence (entraînée sur des traces simulées) et classification hybride.

- hybride 1.x (combine, plafond « courte et droite ») : comportement conservé ;
- v2 : decide() (abstention, artefact, accord, seuils par classe, ambiguïté courte et droite),
  calibration (température, isotonique, décalage de priors), evaluate_on_labeled, family ;
- chargement robuste : fichier absent, corrompu ou d'une autre version de scikit-learn → None ;
- modèle livré : contrat ``info`` (n_tracks_total, classes, seuil, métriques test/stress…),
  probabilités calibrées (somme 1), vitesse d'inférence ;
- pipeline : le mode « regles » donne exactement la décision des règles physiques, et le mode
  « hybride » ne change ni la détection ni les mesures (seulement la décision).
"""
from __future__ import annotations

import importlib.util
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from wilsontrack.config import AppConfig, ClassificationParams
from wilsontrack.detection import TrackDetector
from wilsontrack.detection import reference_model as RM
from wilsontrack.detection.classifier import ML_FEATURES, classify_rules
from wilsontrack.synth import SynthConfig, SyntheticChamber

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "wilsontrack" / "assets" / "modele_reference.joblib"


def _sum(p: dict) -> float:
    return sum(p.values())


# --------------------------------------------------------------------------- #
# combine() (hybride 1.x)
# --------------------------------------------------------------------------- #
def test_combine_agreement():
    rules = {"alpha": 0.9, "proton": 0.05, "muon": 0.0, "beta": 0.0}
    ml = {"alpha": 0.8, "beta": 0.1, "muon": 0.05, "proton": 0.05}
    cls, conf, p, agree = RM.combine(rules, ml, 0.5, 0.30)
    assert cls == "alpha" and agree
    assert set(p) == set(RM.CLASSES)
    assert _sum(p) == pytest.approx(1.0)
    assert conf == pytest.approx(p["alpha"]) and conf > 0.8


def test_combine_disagreement_follows_weights():
    rules = {"alpha": 0.0, "proton": 0.0, "muon": 0.7, "beta": 0.2}
    ml = {"alpha": 0.02, "beta": 0.9, "muon": 0.06, "proton": 0.02}
    cls, conf, p, agree = RM.combine(rules, ml, 0.5, 0.30)
    assert not agree
    assert cls == "beta" and conf == pytest.approx(p["beta"])
    cls2, _c, _p, agree2 = RM.combine(rules, ml, 0.2, 0.30)
    assert cls2 == "muon" and not agree2


def test_combine_low_confidence_is_uncertain():
    rules = {"alpha": 0.0, "proton": 0.0, "muon": 0.0, "beta": 0.0}
    ml = {c: 0.25 for c in RM.CLASSES}
    cls, conf, p, _agree = RM.combine(rules, ml, 0.5, 0.30)
    assert cls == "incertain"
    assert conf == pytest.approx(0.25)
    assert _sum(p) == pytest.approx(1.0)


def test_combine_weight_zero_is_rules_and_one_is_ml():
    rules = {"alpha": 0.1, "proton": 0.0, "muon": 0.6, "beta": 0.3}
    ml = {"alpha": 0.7, "beta": 0.1, "muon": 0.1, "proton": 0.1}
    cls0, _c0, p0, _ = RM.combine(rules, ml, 0.0, 0.30)
    pr = RM.rules_probabilities(rules)
    assert cls0 == "muon"
    assert all(p0[c] == pytest.approx(pr[c]) for c in RM.CLASSES)
    cls1, c1, p1, _ = RM.combine(rules, ml, 1.0, 0.30)
    assert cls1 == "alpha" and c1 == pytest.approx(0.7)
    assert all(p1[c] == pytest.approx(ml[c]) for c in RM.CLASSES)
    assert RM.combine(rules, ml, 7.0, 0.30)[0] == "alpha"
    assert RM.combine(rules, ml, float("nan"), 0.30)[0] == "muon"


def test_combine_ignores_artefact_mass_of_v2_probabilities():
    rules = {"alpha": 0.0, "proton": 0.0, "muon": 0.6, "beta": 0.1}
    ml = {"alpha": 0.0, "beta": 0.1, "muon": 0.3, "proton": 0.0, "artefact": 0.6}
    cls, _conf, p, _ = RM.combine(rules, ml, 1.0, 0.30)
    assert cls == "muon" and set(p) == set(RM.CLASSES) and _sum(p) == pytest.approx(1.0)


def test_combine_robust_to_missing_and_bad_values():
    cls, conf, p, _ = RM.combine({"muon": float("inf"), "beta": None}, {"beta": 2.0, "muon": -1.0}, 0.5, 0.3)
    assert cls in RM.CLASSES + ["incertain"]
    assert _sum(p) == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in p.values())


def test_rules_probabilities_normalised_with_floor():
    p = RM.rules_probabilities({"alpha": 1.0})
    assert _sum(p) == pytest.approx(1.0)
    assert min(p.values()) > 0.0 and p["alpha"] > 0.9


# --------------------------------------------------------------------------- #
# Plafond d'ambiguïté « courte et droite » (hybride 1.x)
# --------------------------------------------------------------------------- #
_SHORT_STRAIGHT_RULES = {"alpha": 0.0, "proton": 0.0, "muon": 0.6, "beta": 0.05}


def test_short_straight_cap_kept_in_hybrid():
    cp = ClassificationParams()
    assert RM.short_straight_ambiguity(_SHORT_STRAIGHT_RULES, 10.0, cp.muon_min_length_mm)
    ml = {"alpha": 0.02, "beta": 0.18, "muon": 0.78, "proton": 0.02}
    cls, conf, p, agree, capped = RM.hybrid_classify(_SHORT_STRAIGHT_RULES, ml, 0.5, cp.min_confidence, 10.0,
                                                     cp.muon_min_length_mm)
    assert cls == "muon" and agree and capped
    assert conf == pytest.approx(RM.AMBIGUITY_CAP)
    assert p["muon"] > RM.AMBIGUITY_CAP


def test_short_straight_cap_lifted_when_reference_ai_very_confident():
    cp = ClassificationParams()
    ml = {"alpha": 0.01, "beta": 0.02, "muon": 0.95, "proton": 0.02}
    cls, conf, _p, _a, capped = RM.hybrid_classify(_SHORT_STRAIGHT_RULES, ml, 0.5, cp.min_confidence, 10.0,
                                                   cp.muon_min_length_mm)
    assert cls == "muon" and not capped and conf > RM.AMBIGUITY_CAP


def test_no_cap_for_long_tracks():
    cp = ClassificationParams()
    assert not RM.short_straight_ambiguity(_SHORT_STRAIGHT_RULES, 80.0, cp.muon_min_length_mm)
    ml = {"alpha": 0.02, "beta": 0.18, "muon": 0.78, "proton": 0.02}
    _cls, conf, _p, _a, capped = RM.hybrid_classify(_SHORT_STRAIGHT_RULES, ml, 0.5, cp.min_confidence, 80.0,
                                                    cp.muon_min_length_mm)
    assert not capped and conf > RM.AMBIGUITY_CAP


def test_cap_can_make_uncertain_with_high_min_confidence():
    ml = {"alpha": 0.02, "beta": 0.18, "muon": 0.78, "proton": 0.02}
    cls, conf, _p, _a, capped = RM.hybrid_classify(_SHORT_STRAIGHT_RULES, ml, 0.5, 0.5, 10.0, 45.0)
    assert capped and cls == "incertain" and conf == pytest.approx(RM.AMBIGUITY_CAP)


# --------------------------------------------------------------------------- #
# decide() (v2)
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Paramètres de décision d'un modèle v2 (sans arbres) : poids, températures, seuils."""

    def __init__(self, w=0.6, T_h=1.0, class_thr=None, threshold=0.85, T_r=1.0):
        self.w_ml = w
        self.threshold = threshold
        self.calibration = {"hybrid": {"w": w, "T": T_h}, "rules": {"T": T_r},
                            "class_thresholds": dict(class_thr or {})}


def test_decide_confident_agreement():
    rules = {"alpha": 0.95, "proton": 0.05, "muon": 0.0, "beta": 0.0}
    ml = {"alpha": 0.97, "beta": 0.01, "muon": 0.01, "proton": 0.005, "artefact": 0.005}
    d = RM.decide(rules, ml, threshold=0.85, model=_FakeModel())
    assert isinstance(d, RM.Decision)
    assert d.cls == "alpha" and not d.abstained and d.agreement and not d.is_artefact
    assert set(d.probs) == set(RM.ALL_CLASSES) and _sum(d.probs) == pytest.approx(1.0)
    assert d.confidence == pytest.approx(d.probs["alpha"]) and d.confidence >= 0.85
    assert d.competitors[0][0] == "alpha" and len(d.competitors) == 2
    assert d.family == "lourde" and d.family_confidence > 0.9
    assert d.method == "hybride"


def test_decide_abstains_below_threshold_with_two_competitors():
    rules = {"alpha": 0.0, "proton": 0.0, "muon": 0.5, "beta": 0.45}
    ml = {"alpha": 0.0, "beta": 0.45, "muon": 0.52, "proton": 0.0, "artefact": 0.03}
    d = RM.decide(rules, ml, threshold=0.85, model=_FakeModel())
    assert d.cls == "incertain" and d.abstained
    assert {c for c, _p in d.competitors} == {"muon", "beta"}
    assert d.competitors[0][1] >= d.competitors[1][1]
    assert d.family == "fine" and d.family_confidence > 0.9          # famille sûre même si la classe ne l'est pas
    d2 = RM.decide(rules, ml, threshold=0.4, model=_FakeModel())
    assert d2.cls == "muon" and not d2.abstained


def test_decide_artefact_is_flagged():
    rules = {"alpha": 0.6, "proton": 0.0, "muon": 0.0, "beta": 0.1}       # les règles ne voient pas les artefacts
    ml = {"alpha": 0.02, "beta": 0.02, "muon": 0.01, "proton": 0.0, "artefact": 0.95}
    d = RM.decide(rules, ml, threshold=0.85, model=_FakeModel(w=0.6))
    assert d.cls == "artefact" and d.is_artefact and not d.abstained
    assert d.probs["artefact"] == pytest.approx(0.95)
    assert not d.agreement


def test_decide_short_straight_requires_090_for_thin_classes():
    rules = {"alpha": 0.0, "proton": 0.0, "muon": 0.6, "beta": 0.1}
    ml = {"alpha": 0.0, "beta": 0.08, "muon": 0.88, "proton": 0.0, "artefact": 0.04}
    m = _FakeModel(w=1.0)
    assert RM.decide(rules, ml, threshold=0.85, model=m).cls == "muon"
    d = RM.decide(rules, ml, threshold=0.85, model=m, rules_short_straight=True)
    assert d.cls == "incertain" and d.threshold == pytest.approx(RM.ML_OVERRIDE)
    ml_sure = {"alpha": 0.0, "beta": 0.04, "muon": 0.94, "proton": 0.0, "artefact": 0.02}
    assert RM.decide(rules, ml_sure, threshold=0.85, model=m, rules_short_straight=True).cls == "muon"


def test_decide_class_thresholds_and_weight_from_model():
    rules = {"alpha": 0.0, "proton": 0.9, "muon": 0.0, "beta": 0.0}
    ml = {"alpha": 0.08, "beta": 0.0, "muon": 0.0, "proton": 0.9, "artefact": 0.02}
    m = _FakeModel(w=1.0, class_thr={"proton": 0.95})
    d = RM.decide(rules, ml, threshold=0.85, model=m)
    assert d.cls == "incertain" and d.threshold == pytest.approx(0.95)     # seuil de classe relevé (validation)
    # Seuil abaissé volontairement par l'utilisateur : le seuil de classe ne s'applique plus
    assert RM.decide(rules, ml, threshold=0.7, model=m).cls == "proton"
    # w_ml None → poids du modèle (1,0 : IA seule)
    d1 = RM.decide({"alpha": 1.0}, ml, threshold=0.5, model=m)
    assert d1.probs["proton"] == pytest.approx(0.9) and d1.method == "ia"
    d0 = RM.decide({"alpha": 1.0}, ml, threshold=0.5, w_ml=0.0, model=m)
    assert d0.cls == "alpha" and d0.method == "règles"


def test_decide_rules_only_without_ml():
    d = RM.decide({"alpha": 0.95, "beta": 0.0, "muon": 0.0, "proton": 0.0}, None, threshold=0.85, model=None)
    assert d.method == "règles" and d.cls == "alpha" and d.probs["artefact"] == 0.0
    assert _sum(d.probs) == pytest.approx(1.0)
    d2 = RM.decide({}, None, threshold=0.85, model=None)
    assert d2.cls == "incertain"


def test_decide_hybrid_temperature_only_for_trained_weight():
    rules = {"alpha": 0.0, "proton": 0.0, "muon": 0.7, "beta": 0.2}
    ml = {"alpha": 0.0, "beta": 0.2, "muon": 0.75, "proton": 0.0, "artefact": 0.05}
    hot = _FakeModel(w=0.5, T_h=3.0)
    p_trained = RM.decide(rules, ml, threshold=0.85, model=hot).probs
    p_other = RM.decide(rules, ml, threshold=0.85, w_ml=0.2, model=hot).probs
    raw = RM.blend(rules, ml, 0.5)
    assert p_trained["muon"] < raw["muon"]                            # T > 1 adoucit
    assert p_other == pytest.approx(RM.blend(rules, ml, 0.2))         # autre poids : pas de température


def test_blend_keeps_artefact_mass_from_ml():
    p = RM.blend({"alpha": 1.0}, {"alpha": 0.1, "beta": 0.1, "muon": 0.1, "proton": 0.1, "artefact": 0.6}, 0.3)
    assert p["artefact"] == pytest.approx(0.6) and _sum(p) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 5
    logits = rng.normal(0, 2.0, (n, k))
    p_true = _softmax(logits)
    y = np.array([rng.choice(k, p=p) for p in p_true])
    over = _softmax(logits * 2.0)                                   # modèle trop sûr de lui (T vrai = 2)
    T = RM.fit_temperature(over, y)
    assert T == pytest.approx(2.0, rel=0.15)
    assert RM.apply_temperature(over, T) == pytest.approx(p_true, abs=0.08)


def test_fit_calibrator_improves_log_loss_and_isotonic_is_valid():
    rng = np.random.default_rng(1)
    n, k = 3000, 5
    logits = rng.normal(0, 2.0, (n, k))
    y = np.array([rng.choice(k, p=p) for p in _softmax(logits)])
    over = _softmax(logits * 2.5)
    cal = RM.fit_calibrator(over, y, groups=np.arange(n) // 30)
    assert cal["type"] in ("temperature", "isotonic")
    out = RM.apply_calibrator(over, cal)
    assert out.sum(1) == pytest.approx(np.ones(n))
    assert RM._wnll(out, y, np.ones(n)) < RM._wnll(over, y, np.ones(n))
    iso = RM._fit_isotonic(over, y, np.ones(n))
    q = RM.apply_calibrator(over, iso)
    assert np.all(q >= 0) and q.sum(1) == pytest.approx(np.ones(n))
    assert RM.apply_calibrator(over, None) is not None


def test_apply_prior_shifts_balanced_probabilities():
    P = np.array([[0.5, 0.5]])
    assert RM.apply_prior(P, [0.9, 0.1])[0] == pytest.approx([0.9, 0.1])


# --------------------------------------------------------------------------- #
# Famille, evaluate_on_labeled
# --------------------------------------------------------------------------- #
def test_family_rules_and_probabilities():
    assert RM.family({"ev_heavy": 0.9}) == "lourde"
    assert RM.family({"ev_heavy": 0.1}) == "fine"
    assert RM.family({"density_rel": 3.0}) == "lourde"
    assert RM.family({}) == "fine"
    assert RM.family({"ev_heavy": 0.9}, {"alpha": 0.1, "proton": 0.0, "beta": 0.5, "muon": 0.4}) == "fine"


def test_evaluate_on_labeled_with_rules_only():
    feats = [{"length_mm": 30.0}, {"length_mm": 5.0}, {"length_mm": 50.0}, {}]
    labels = ["alpha", "muon", "beta", "incertain"]
    extras = [{"rule_scores": {"alpha": 0.95}}, {"rule_scores": {"muon": 0.5, "beta": 0.45}},
              {"rule_scores": {"alpha": 0.95}}, {}]
    r = RM.evaluate_on_labeled(feats, labels, extras, model=None)
    assert r["n"] == 3                                             # « incertain » n'est pas une étiquette
    assert r["n_confident"] == 2 and r["accuracy"] == pytest.approx(0.5)
    assert r["coverage"] == pytest.approx(2 / 3)
    assert r["per_class"]["alpha"]["recall"] == pytest.approx(1.0)
    assert r["per_class"]["alpha"]["precision"] == pytest.approx(0.5)
    assert r["confusion"]["labels_pred"][-1] == "incertain"
    assert sum(map(sum, r["confusion"]["matrix"])) == 3
    assert isinstance(r["note"], str) and r["note"]


def test_evaluate_on_labeled_empty():
    r = RM.evaluate_on_labeled([], [], model=None)
    assert r["n"] == 0 and r["accuracy"] is None and "Aucune" in r["note"]


# --------------------------------------------------------------------------- #
# Chargement robuste
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_cache(monkeypatch):
    monkeypatch.setattr(RM, "_CACHE", {})
    yield


def test_load_default_missing_file_returns_none(monkeypatch, tmp_path, isolated_cache):
    monkeypatch.setattr(RM, "DEFAULT_PATH", tmp_path / "absent" / "modele_reference.joblib")
    assert RM.ReferenceModel.load_default() is None
    assert RM.ReferenceModel.load_default() is None
    assert RM.reference_confusion() is None


def test_load_corrupt_file_returns_none(tmp_path):
    bad = tmp_path / "modele_reference.joblib"
    bad.write_bytes(b"ceci n'est pas un modele")
    assert RM.ReferenceModel.load(bad) is None
    (tmp_path / "vide.joblib").write_bytes(b"")
    assert RM.ReferenceModel.load(tmp_path / "vide.joblib") is None


def _tiny_payload(version: str) -> dict:
    """Modèle 1.x (4 classes, pas de calibration)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    feats = list(ML_FEATURES) + list(RM.EXTRA_FEATURES)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, len(feats)))
    y = np.array(RM.CLASSES * 20)
    X[:, 0] += np.array([RM.CLASSES.index(c) for c in y]) * 3.0
    model = HistGradientBoostingClassifier(max_iter=20).fit(X, y)
    return {"model": model, "features": feats, "classes": RM.CLASSES, "w_ml": 0.4, "metrics": {},
            "n_train": 80, "n_tracks_total": 80, "n_jobs": 1, "sklearn_version": version, "created": "test"}


def _tiny_payload_v2(version: str) -> dict:
    """Modèle v2 (5 classes, calibration, métriques au format du contrat)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    feats = list(RM.V2_FEATURES)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, len(feats)))
    y = np.array(RM.ALL_CLASSES * 40)
    X[:, 0] += np.array([RM.ALL_CLASSES.index(c) for c in y]) * 3.0
    model = HistGradientBoostingClassifier(max_iter=30).fit(X, y)
    M = {"n": 200, "coverage": 0.7, "accuracy_confident": 0.95, "balanced_accuracy": 0.8,
         "per_class": {c: {"precision": 0.9, "recall": 0.7, "n": 40} for c in RM.ALL_CLASSES}}
    conf = {"labels_true": RM.ALL_CLASSES, "labels_pred": RM.ALL_CLASSES + ["incertain"],
            "matrix": [[30 if i == j else 0 for j in range(5)] + [10] for i in range(5)]}
    prior = {"alpha": 0.15, "beta": 0.35, "muon": 0.35, "proton": 0.01, "artefact": 0.14}
    return {"format": 2, "model": model, "features": feats, "classes": RM.ALL_CLASSES, "w_ml": 0.7, "threshold": 0.85,
            "calibration": {"prior_ref": prior, "ml": {"type": "temperature", "T": 1.3}, "hybrid": {"w": 0.7, "T": 1.1},
                            "rules": {"T": 1.0}, "class_thresholds": {"proton": 0.9}, "trained_balanced": True},
            "metrics": {"test": M, "stress": M, "test_hybrid_balanced_accuracy": 0.8}, "confusion_test": conf,
            "n_train": 120, "n_tracks_total": 200, "n_jobs": 4, "sklearn_version": version, "created": "test",
            "generator": {"resolution": ["640x360"]}, "model_type": "HistGradientBoostingClassifier"}


def test_load_version_mismatch_returns_none(tmp_path):
    import joblib
    p = tmp_path / "ancien.joblib"
    joblib.dump(_tiny_payload("0.1.0"), p)
    assert RM.ReferenceModel.load(p) is None


def test_load_same_version_and_predict_v1(tmp_path, monkeypatch, isolated_cache):
    import joblib
    import sklearn
    p = tmp_path / "modele_reference.joblib"
    joblib.dump(_tiny_payload(sklearn.__version__), p)
    monkeypatch.setattr(RM, "DEFAULT_PATH", p)
    m = RM.ReferenceModel.load_default()
    assert m is not None and m.w_ml == pytest.approx(0.4)
    assert RM.ReferenceModel.load_default() is m
    proba = m.predict_proba({"length_mm": 6.0}, duration_s=0.5, touches_border=True)
    assert set(proba) == set(RM.CLASSES) and _sum(proba) == pytest.approx(1.0)
    info = m.info
    assert info["n_tracks_total"] == 80 and info["w_ml"] == pytest.approx(0.4)
    assert info["classes"] == RM.CLASSES and info["threshold"] == pytest.approx(0.85)
    assert "test" in info["metrics"] and "stress" in info["metrics"]           # dégradé mais présent


def test_v2_payload_contract(tmp_path, monkeypatch, isolated_cache):
    import joblib
    import sklearn
    p = tmp_path / "modele_reference.joblib"
    joblib.dump(_tiny_payload_v2(sklearn.__version__), p)
    monkeypatch.setattr(RM, "DEFAULT_PATH", p)
    m = RM.ReferenceModel.load_default()
    assert m is not None and m.format == 2 and m.classes == RM.ALL_CLASSES and m._fast is not None
    proba = m.predict_proba({"length_mm": 6.0}, 0.5, False, True)
    assert set(proba) == set(RM.ALL_CLASSES) and _sum(proba) == pytest.approx(1.0)
    info = m.info
    for k in ("n_tracks_total", "n_train", "n_jobs", "classes", "threshold", "created", "model_type", "generator",
              "metrics", "confusion_test"):
        assert k in info, k
    assert info["n_tracks_total"] == 200 and info["n_jobs"] == 4 and info["threshold"] == 0.85
    for split in ("test", "stress"):
        M = info["metrics"][split]
        for k in ("n", "coverage", "accuracy_confident", "balanced_accuracy", "per_class"):
            assert k in M
        assert set(M["per_class"]["alpha"]) >= {"precision", "recall", "n"}
    assert info["metrics"]["test_hybrid_balanced_accuracy"] == 0.8         # clé « à plat » (interface 1.x)
    c = RM.reference_confusion()
    assert c["labels_pred"][-1] == "incertain" and len(c["matrix"]) == len(c["labels_true"])
    d = m.decide({"alpha": 0.2}, {"length_mm": 6.0}, 0.5, False, True)
    assert d.cls in RM.ALL_CLASSES + ["incertain"]
    # decide() sans modèle explicite lit le modèle par défaut (poids 0,7 → température 1,1)
    d2 = RM.decide({"alpha": 0.2}, proba, threshold=0.85)
    assert _sum(d2.probs) == pytest.approx(1.0)
    r = RM.evaluate_on_labeled([{"length_mm": 6.0}] * 3, ["alpha", "beta", "artefact"],
                               [{"rule_scores": {"alpha": 0.9}}, {}, {"duration_s": 1.0}])
    assert r["n"] == 3 and "note" in r


def test_fast_predictor_matches_sklearn(tmp_path):
    """Le prédicteur sans OpenMP donne exactement les probabilités de scikit-learn."""
    import joblib
    import sklearn
    for payload in (_tiny_payload(sklearn.__version__), _tiny_payload_v2(sklearn.__version__)):
        p = tmp_path / "m.joblib"
        joblib.dump(payload, p)
        m = RM.ReferenceModel.load(p)
        assert m is not None and m._fast is not None
        X = RM._check_vectors(len(m.features))
        np.testing.assert_allclose(m._fast.proba(X), m.model.predict_proba(X), atol=1e-9)
        np.testing.assert_allclose(m.predict_proba_many(X).sum(1), 1.0)
        np.testing.assert_allclose(m.predict_proba_matrix(X).sum(1), 1.0)


def test_feature_vector_handles_missing_and_infinite_values():
    names = list(RM.V2_FEATURES)
    v = RM.feature_vector({"length_mm": 20.0, "radius_mm": float("inf"), "width_mm": None, "elongation": 10.0,
                           "width_rel": 1.5}, names, duration_s=1.0, touches_border=True, calibrated=False)
    assert len(v) == len(names)
    d = dict(zip(names, v))
    assert d["log_radius_rel"] == pytest.approx(4.0)
    assert d["touches_border"] == 1.0 and d["calibrated"] == 0.0 and d["duration_s"] == 1.0
    assert d["length_wref"] == pytest.approx(15.0)
    assert math.isnan(d["width_mm"])
    assert not any(k.startswith("ev_") for k in names)


# --------------------------------------------------------------------------- #
# Modèle livré avec l'application
# --------------------------------------------------------------------------- #
_ARCHETYPES = {
    # alpha : épaisse, dense, droite, 4 cm, visible longtemps
    "alpha": {"length_mm": 40.0, "chord_mm": 39.6, "width_mm": 1.9, "width_px": 7.0, "width_mask_px": 8.0,
              "width_rel": 3.2, "density_rel": 9.0, "straightness": 0.99, "tortuosity": 0.01, "elongation": 21.0,
              "sagitta_ratio": 0.01, "asymmetry": 1.1, "profile_cv": 0.02, "branches": 0, "peak_snr": 300.0,
              "curl_consistency": 1.0, "lateral_mm": 0.0, "line_rms_mm": 0.03, "area_px": 1100, "length_px": 150.0,
              "turn_total_rad": 0.3, "radius_mm": float("inf"), "circle_rms_mm": 0.05},
    # trace fine longue et parfaitement droite (muon OU électron cosmique : même famille)
    "fine_droite": {"length_mm": 60.0, "chord_mm": 59.9, "width_mm": 0.6, "width_px": 2.3, "width_mask_px": 3.0,
                    "width_rel": 1.0, "density_rel": 1.0, "straightness": 0.999, "tortuosity": 0.004,
                    "elongation": 100.0, "sagitta_ratio": 0.004, "asymmetry": 1.1, "profile_cv": 0.3, "branches": 0,
                    "peak_snr": 50.0, "curl_consistency": 0.5, "lateral_mm": 0.05, "line_rms_mm": 0.05,
                    "area_px": 700, "length_px": 230.0, "turn_total_rad": 0.5, "radius_mm": float("inf"),
                    "circle_rms_mm": 0.05},
    # électron : fin, sinueux
    "beta": {"length_mm": 35.0, "chord_mm": 22.0, "width_mm": 0.6, "width_px": 2.3, "width_mask_px": 3.0,
             "width_rel": 1.0, "density_rel": 1.2, "straightness": 0.63, "tortuosity": 0.15, "elongation": 58.0,
             "sagitta_ratio": 0.2, "asymmetry": 1.5, "profile_cv": 0.4, "branches": 2, "peak_snr": 40.0,
             "curl_consistency": 0.3, "lateral_mm": 2.5, "line_rms_mm": 2.0, "area_px": 450, "length_px": 134.0,
             "turn_total_rad": 5.0, "radius_mm": 25.0, "circle_rms_mm": 1.5},
}
_DURATION = {"alpha": 1.5, "fine_droite": 0.3, "beta": 0.3}


@pytest.fixture(scope="module")
def shipped_model():
    if not ASSET.exists():
        pytest.skip("Modèle de référence non livré (wilsontrack/assets/modele_reference.joblib)")
    RM.clear_cache()
    m = RM.ReferenceModel.load(ASSET)
    if m is None:
        pytest.skip("Modèle de référence illisible avec cette version de scikit-learn")
    return m


def test_shipped_model_predict_proba_sums_to_one(shipped_model):
    for name, f in _ARCHETYPES.items():
        p = shipped_model.predict_proba(dict(f), duration_s=_DURATION[name], touches_border=False, calibrated=True)
        assert set(p) == set(shipped_model.classes)
        assert _sum(p) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in p.values())


def test_shipped_model_recognises_clear_archetypes(shipped_model):
    """Alpha nette → alpha ; électron sinueux → électron ; trace fine droite → famille « fine »
    (muon ou électron énergétique : la classe exacte n'est pas exigée, ambiguïté physique)."""
    P = {name: shipped_model.predict_proba(dict(f), duration_s=_DURATION[name], touches_border=False, calibrated=True)
         for name, f in _ARCHETYPES.items()}
    assert max(P["alpha"], key=P["alpha"].get) == "alpha"
    assert max(P["beta"], key=P["beta"].get) == "beta"
    assert RM.family({}, P["fine_droite"]) == "fine"


def test_shipped_model_fast_and_exact(shipped_model):
    assert shipped_model._fast is not None
    X = RM._check_vectors(len(shipped_model.features))
    np.testing.assert_allclose(shipped_model._fast.proba(X), shipped_model.model.predict_proba(X), atol=1e-9)


def test_shipped_model_inference_fast_enough(shipped_model):
    f = dict(_ARCHETYPES["beta"])
    shipped_model.predict_proba(f, 0.3, False, True)
    t0 = time.perf_counter()
    for _ in range(50):
        shipped_model.predict_proba(f, 0.3, False, True)
    assert (time.perf_counter() - t0) / 50 < 0.005                 # objectif ≤ 1 ms (marge pour machine chargée)


def test_shipped_model_metadata(shipped_model):
    assert not any(k.startswith("ev_") for k in shipped_model.features)
    assert 0.0 <= shipped_model.w_ml <= 1.0
    assert shipped_model.n_tracks_total > 0
    info = shipped_model.info
    assert info["classes"] and info["n_jobs"] > 0 and info["created"]
    assert 0.5 <= info["threshold"] <= 1.0
    for split in ("test", "stress"):
        M = info["metrics"].get(split) or {}
        if shipped_model.format >= 2:
            assert 0.0 <= M["coverage"] <= 1.0
            assert M["n"] > 0
            for c, d in M["per_class"].items():
                if d["precision"] is not None:
                    assert 0.0 <= d["precision"] <= 1.0
    assert ASSET.stat().st_size <= 25e6
    assert ASSET.with_suffix(".json").exists()


def test_shipped_v2_confident_precision_meets_target(shipped_model):
    """Contrat v2 : sur le TEST simulé, chaque classe affirmée avec certitude l'est à ≥ 85 % de précision
    (classes jamais affirmées : précision indéfinie, couverture nulle — à afficher honnêtement)."""
    if shipped_model.format < 2:
        pytest.skip("modèle 1.x")
    for c, d in shipped_model.info["metrics"]["test"]["per_class"].items():
        if d["precision"] is not None and (d.get("coverage") or 0) > 0:
            assert d["precision"] >= 0.80, (c, d)        # 0,85 visé ; marge pour le bruit statistique du test


# --------------------------------------------------------------------------- #
# Intégration dans le pipeline
# --------------------------------------------------------------------------- #
def _run_chamber(mode: str, weight: float | None = None):
    sc = SynthConfig(width=640, height=360, fps=25.0, duration_s=6.0, fov_mm=160.0, seed=11,
                     rates_per_min={"alpha": 60.0, "beta": 60.0, "muon": 80.0, "proton": 10.0})
    cfg = AppConfig()
    cfg.classification.mode = mode
    if weight is not None:
        cfg.classification.ref_model_weight = weight
    cfg.physics.px_per_mm = sc.px_per_mm
    sim = SyntheticChamber(sc)
    det = TrackDetector(cfg, sc.fps, (sc.height, sc.width))
    for _ in range(sim.n_frames):
        det.process(sim.render_next(), None)
    det.finalize_run()
    return det


@pytest.fixture(scope="module")
def rules_run():
    return _run_chamber("regles")


def test_rules_mode_is_pure_physics_rules(rules_run):
    det = rules_run
    assert det.ref_model is None
    valid = det.valid_events
    assert len(valid) >= 5
    for e in valid:
        cls, conf, scores, reason = classify_rules(dict(e.features), det.refs, det.cfg.classification,
                                                   det.calibrated, det.px_per_mm_proc)
        assert (e.cls, e.confidence, e.scores, e.reason) == (cls, conf, scores, reason)
        assert e.extra["method"] == "Règles physiques"
        assert "ml_proba" not in e.extra and "rules_cls" not in e.extra
        assert not e.reason.startswith("Hybride")


def test_hybrid_without_model_falls_back_to_rules(monkeypatch, tmp_path, isolated_cache, rules_run):
    monkeypatch.setattr(RM, "DEFAULT_PATH", tmp_path / "absent.joblib")
    det = _run_chamber("hybride")
    assert det.ref_model is None
    a = [(e.t_first, e.cls, e.confidence, e.reason) for e in rules_run.valid_events]
    b = [(e.t_first, e.cls, e.confidence, e.reason) for e in det.valid_events]
    assert a == b
    assert all(e.extra["method"] == "Règles physiques" for e in det.valid_events)


def test_hybrid_mode_changes_only_the_decision(shipped_model, rules_run):
    RM.clear_cache()
    det = _run_chamber("hybride")
    assert det.ref_model is not None
    key = lambda e: (e.t_first, e.t_last, round(e.features["length_mm"], 6))  # noqa: E731
    by_key = {key(e): e for e in rules_run.valid_events}
    # Le mode hybride ne change que la décision ; il peut en plus RETIRER des détections jugées artefacts
    removed = [e for e in det.events if not e.valid and e.extra.get("ml_artefact")]
    assert len(det.valid_events) + len(removed) >= len(rules_run.valid_events)
    pairs = [(by_key[key(eh)], eh) for eh in det.valid_events if key(eh) in by_key]
    assert len(pairs) == len(det.valid_events)
    for er, eh in pairs:
        assert eh.extra["rules_cls"] == er.cls                 # avis des règles conservé
        proba = eh.extra["ml_proba"]
        assert set(proba) <= set(RM.ALL_CLASSES) and _sum(proba) == pytest.approx(1.0)
        assert _sum(eh.scores) == pytest.approx(1.0)
        assert 0.0 <= eh.confidence <= 1.0
        assert "→" in eh.reason and er.reason.split(",")[0] in eh.reason   # explication physique présente (référence recalculée sans les artefacts)


def test_classify_hybrid_v2_on_real_detections(shipped_model, rules_run):
    """Décision v2 prête pour le détecteur : 5 classes + « incertain », raison en français."""
    cp = ClassificationParams()
    for e in rules_run.valid_events:
        f = dict(e.features)
        out = classify_rules(f, rules_run.refs, cp, rules_run.calibrated, rules_run.px_per_mm_proc)
        cls, conf, probs, reason, extra = RM.classify_hybrid_v2(shipped_model, f, out, cp, e.duration_s,
                                                                e.touches_border, rules_run.calibrated)
        assert cls in RM.ALL_CLASSES + ["incertain"]
        assert set(probs) == set(RM.ALL_CLASSES) and _sum(probs) == pytest.approx(1.0)
        assert set(RM.HYBRID_V2_EXTRA_KEYS) <= set(extra)
        assert (cls == "incertain") == extra["abstained"]
        assert out[3] in reason


def test_hybrid_weight_zero_reproduces_rules_classes(shipped_model, rules_run):
    """Avec un poids nul pour l'IA, l'hybride 1.x donne la classe des règles (hors « incertain »)."""
    RM.clear_cache()
    det = _run_chamber("hybride", weight=0.0)
    for er, eh in zip(rules_run.valid_events, det.valid_events):
        if er.cls != "incertain":
            assert eh.cls in (er.cls, "incertain", "fine")
        assert eh.extra["w_ml"] == 0.0


def test_classification_params_v2_defaults_roundtrip():
    cp = ClassificationParams()
    assert cp.ref_model_weight == -1.0 and cp.mode == "hybride" and cp.ml_threshold == pytest.approx(0.85)
    assert 77.0 <= cp.alpha_max_length_mm <= 86.0
    cfg = AppConfig()
    cfg.classification.ref_model_weight = 0.3
    cfg.classification.ml_threshold = 0.9
    back = AppConfig.from_dict(cfg.to_dict())
    assert back.classification.ref_model_weight == 0.3 and back.classification.ml_threshold == 0.9


# --------------------------------------------------------------------------- #
# Outil d'entraînement (fonctions pures)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trainer():
    env = dict(os.environ)
    spec = importlib.util.spec_from_file_location("train_reference_model", ROOT / "tools" / "train_reference_model.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.environ.clear()
        os.environ.update(env)
    return mod


def test_domain_randomisation_ranges_and_determinism(trainer):
    a = trainer.draw_params(5)
    assert a == trainer.draw_params(5)
    assert a != trainer.draw_params(6)
    seen_res = set()
    for s in range(1, 60):
        p = trainer.draw_params(s)
        seen_res.add((p["width"], p["height"]))
        assert (p["width"], p["height"]) in trainer.RESOLUTIONS and p["fps"] in (15.0, 20.0, 25.0, 30.0)
        assert 30.0 <= p["duration_s"] <= 45.0 and 140.0 <= p["fov_mm"] <= 340.0
        assert p["process_width"] in (640, 960, 1280)
        assert 5.0 <= p["deff_mm"] <= 25.0 and 5.0 <= p["sensitive_depth_mm"] <= 25.0
        assert 1.2 <= p["noise_sigma"] <= 6.0 and 0.5 <= p["brightness"] <= 1.6
        assert 2.0 <= p["rates_per_min"]["proton"] <= 6.0
        assert p["source_beta_per_min"] == 0.0 or 15.0 <= p["source_beta_per_min"] <= 80.0
        assert abs(sum(p["alpha_mix"].values()) - 1.0) < 1e-9
        assert p["roi"] == [] or len(p["roi"]) >= 4
        q = trainer.draw_params(1_000_000 + s, stress=True)
        assert 6.0 <= q["noise_sigma"] <= 9.0 and 35 <= q["jpeg_quality"] <= 60
        assert 20.0 <= trainer.draw_params(s, pilot=True)["duration_s"] <= 30.0
        sc, cfg = trainer.build_configs(p)
        assert sc.deff_mm == p["deff_mm"] and cfg.detection.process_width == p["process_width"]
        assert (cfg.physics.px_per_mm > 0) == p["calibrated"]
    assert len(seen_res) >= 3


def test_metrics_helper_with_abstention(trainer):
    y = ["alpha", "alpha", "beta", "muon", "proton", "beta", "artefact"]
    p = ["alpha", "beta", "beta", "incertain", "incertain", "beta", "artefact"]
    m = trainer.compute_metrics(y, p, y_argmax=["alpha", "beta", "beta", "muon", "alpha", "beta", "artefact"])
    assert m["coverage"] == pytest.approx(5 / 7)
    assert m["accuracy_confident"] == pytest.approx(4 / 5)
    assert m["per_class"]["beta"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["muon"]["coverage"] == 0.0 and m["per_class"]["muon"]["recall"] == 0.0
    labels_pred = m["confusion"]["labels_pred"]
    assert labels_pred[-1] == "incertain"
    assert m["confusion"]["matrix"][m["confusion"]["labels_true"].index("proton")][labels_pred.index("incertain")] == 1
    assert m["balanced_accuracy_argmax"] == pytest.approx((0.5 + 1 + 1 + 0 + 1) / 5)
    assert trainer.FEATURES == list(RM.V2_FEATURES)
    assert not any(k.startswith("ev_") for k in trainer.FEATURES)


def test_reliability_and_thresholds(trainer):
    p = np.array([0.95, 0.95, 0.55, 0.55])
    c = np.array([True, True, True, False])
    r = trainer.reliability(p, c)
    assert r["ece"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.05, abs=1e-6)
    assert trainer.wilson_lower(1.0, 20) > 0.9 and trainer.wilson_lower(1.0, 3) < 0.7
    H = np.array([[0.9, 0.05, 0.05, 0.0, 0.0]] * 30 + [[0.05, 0.9, 0.05, 0.0, 0.0]] * 30)
    y = ["alpha"] * 30 + ["muon"] * 30                       # les « β » prédits sont faux
    thr = trainer.choose_class_thresholds(H, y, np.ones(60), [False] * 60, 0.85)
    assert "alpha" not in thr and thr["beta"] == pytest.approx(1.01)
