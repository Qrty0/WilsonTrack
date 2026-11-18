"""IA de référence : classificateur entraîné sur un très grand nombre de traces SIMULÉES.

Il n'existe pas de grande collection publique de vidéos de chambre de Wilson dont chaque
trace serait étiquetée (alpha, bêta, muon, proton) : on ne peut donc pas « vérifier des
millions de vidéos en ligne ». La démarche honnête et réalisable est la suivante :

1. le simulateur physique (``wilsontrack.synth``, v2 : transport 3 D, tables NIST ASTAR /
   ESTAR / PSTAR, diffusion de Highland, muons en cos²θ, artefacts) produit des
   enregistrements complets dont on connaît la vérité (nature de chaque trace) ;
2. les conditions sont tirées au hasard pour chaque enregistrement (« randomisation de
   domaine ») : résolution, cadence, champ filmé, bruit, compression, brouillard, éclairage,
   épaisseur de la couche, taux de chaque source, calibration présente ou absente… ;
3. chaque vidéo simulée passe par le VRAI détecteur de l'application, puis un modèle de
   « gradient boosting » apprend à reconnaître 5 classes (alpha, bêta, muon, proton et
   ARTEFACT) à partir des mesures (outil : ``tools/train_reference_model.py``) ;
4. ses probabilités sont CALIBRÉES sur des enregistrements de validation (une « probabilité
   de 90 % » doit être juste ≈ 9 fois sur 10), puis combinées aux règles physiques :
   p = (1 − w)·(1 − p_art)·p_règles + w·p_IA ; enfin, ABSTENTION : sous le seuil
   (0,85 par défaut) la trace est déclarée « incertaine » avec ses deux classes possibles.
   Une trace « artefact » est exclue des comptages.

Les performances publiées (``wilsontrack/assets/modele_reference.json``) sont mesurées sur
des enregistrements simulés jamais vus et sur un jeu « conditions extrêmes » ; elles ne
garantissent rien sur de vraies vidéos : ``evaluate_on_labeled`` mesure l'exactitude sur les
traces réelles corrigées par l'utilisateur.

Ce module ne plante jamais l'application : si le fichier du modèle manque, est corrompu
ou a été produit par une autre version de scikit-learn, ``load_default()`` renvoie None
et l'application revient aux règles physiques seules.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSES = ["alpha", "beta", "muon", "proton"]          # particules (règles, hybride 1.x)
ALL_CLASSES = CLASSES + ["artefact"]                    # classes du modèle v2
RULES_EPS = 0.02          # plancher ajouté aux scores des règles avant normalisation
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "assets" / "modele_reference.joblib"
DEFAULT_THRESHOLD = 0.85

# Mesures supplémentaires (en plus de classifier.ML_FEATURES), toutes disponibles à
# l'inférence et peu sensibles à l'échelle (rapports sans dimension, durée, bord, calibration).
EXTRA_FEATURES = [
    "duration_s",          # durée de visibilité de la trace (s)
    "touches_border",      # 1 si la trace touche le bord (longueur tronquée)
    "calibrated",          # 1 si l'échelle (px/mm) a été calibrée
    "chord_ratio",         # corde / longueur
    "fill_ratio",          # aire du masque / (longueur × épaisseur) : trace continue ou pointillée
    "turn_total_rad",      # somme des changements de direction (rad, sans dimension)
    "lateral_rel",         # étalement latéral / longueur
    "wiggle_rel",          # écart RMS à la droite / longueur
    "log_radius_rel",      # log10(rayon de courbure / longueur)
    "circle_rms_rel",      # écart à l'arc de cercle / longueur
    "mask_width_ratio",    # épaisseur du masque / épaisseur d'intensité
]
# v2 : longueur exprimée en « largeurs de trace fine » de l'enregistrement (≈ longueur en unités de
# 0,5–0,8 mm, même sans calibration) = élongation × épaisseur relative.
V2_EXTRA = ["length_wref"]
V2_FEATURES = ["length_mm", "width_mm", "width_rel", "density_rel", "straightness", "tortuosity",
               "elongation", "sagitta_ratio", "asymmetry", "profile_cv", "branches", "peak_snr",
               "curl_consistency", "lateral_mm", "line_rms_mm"] + EXTRA_FEATURES + V2_EXTRA

_CACHE: dict[str, "ReferenceModel | None"] = {}
_NAN = float("nan")


def _num(x) -> float:
    """Valeur numérique finie, sinon NaN (le gradient boosting gère les valeurs manquantes)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NAN
    return v if math.isfinite(v) else _NAN


def _pos(x) -> float:
    """Valeur finie ≥ 0, sinon 0."""
    v = _num(x)
    return max(0.0, v) if math.isfinite(v) else 0.0


def derived_features(f: dict, duration_s=None, touches_border=False, calibrated=None) -> dict:
    """Calcule les mesures supplémentaires (EXTRA_FEATURES + V2_EXTRA) à partir des mesures d'une trace."""
    g = lambda k: _num(f.get(k, _NAN))          # noqa: E731
    L = g("length_mm")
    L = max(L, 1e-3) if math.isfinite(L) else 1e-3
    Lr = max(L, 1.0)
    length_px, width_px, area, w_mask = g("length_px"), g("width_px"), g("area_px"), g("width_mask_px")
    fill = area / max(length_px * max(width_px, 1.0), 1.0)          # NaN si une mesure manque
    try:
        r = float(f.get("radius_mm", float("inf")))
    except (TypeError, ValueError):
        r = float("inf")
    if not math.isfinite(r) or r <= 0:
        log_r = 4.0                              # droite parfaite : rayon infini
    else:
        log_r = float(np.clip(math.log10(max(r / L, 1e-2)), -2.0, 4.0))
    return {
        "duration_s": _num(duration_s) if duration_s is not None else _NAN,
        "touches_border": 1.0 if touches_border else 0.0,
        "calibrated": _NAN if calibrated is None else (1.0 if calibrated else 0.0),
        "chord_ratio": g("chord_mm") / L,
        "fill_ratio": fill,
        "turn_total_rad": g("turn_total_rad"),
        "lateral_rel": g("lateral_mm") / Lr,
        "wiggle_rel": g("line_rms_mm") / Lr,
        "log_radius_rel": log_r,
        "circle_rms_rel": g("circle_rms_mm") / Lr,
        "mask_width_ratio": w_mask / max(width_px, 0.5),
        "length_wref": g("elongation") * g("width_rel"),
    }


def feature_vector(f: dict, names: list[str], duration_s=None, touches_border=False,
                   calibrated=None) -> list[float]:
    """Vecteur de mesures dans l'ordre ``names`` (valeurs manquantes → NaN)."""
    extra = derived_features(f, duration_s, touches_border, calibrated)
    out = []
    for k in names:
        out.append(extra[k] if k in extra else _num(f.get(k, float("nan"))))
    return out


def rules_probabilities(rule_scores: dict, eps: float = RULES_EPS) -> dict:
    """Scores flous des règles (0..1) → vecteur de probabilités sur les 4 classes de particules."""
    v = np.array([_pos((rule_scores or {}).get(c, 0.0)) + eps for c in CLASSES])
    v = v / v.sum()
    return {c: float(p) for c, p in zip(CLASSES, v)}


def _normalise(p: dict, classes: list[str] = CLASSES) -> dict:
    v = np.array([_pos((p or {}).get(c, 0.0)) for c in classes])
    s = float(v.sum())
    if s <= 0:
        v = np.full(len(classes), 1.0 / len(classes))
    else:
        v = v / s
    return {c: float(x) for c, x in zip(classes, v)}


# --------------------------------------------------------------------------- #
# Hybride 1.x (conservé : utilisé par le détecteur tant que decide() n'est pas intégré)
# --------------------------------------------------------------------------- #
def combine(rule_scores: dict, ml_proba: dict, w_ml: float, min_confidence: float
            ) -> tuple[str, float, dict, bool]:
    """Combine l'avis des règles physiques et celui de l'IA de référence (4 classes, version 1.x).

    p = (1 − w)·p_règles + w·p_IA ; classe = argmax, ou « incertain » si la probabilité
    la plus forte est sous ``min_confidence``. La masse « artefact » éventuelle de l'IA v2
    est ignorée (renormalisation sur les 4 particules).
    Renvoie (classe, confiance, probabilités combinées, accord règles/IA).
    """
    w = min(1.0, _pos(w_ml))
    pr = rules_probabilities(rule_scores)
    pm = _normalise(ml_proba)
    blended = {c: (1.0 - w) * pr[c] + w * pm[c] for c in CLASSES}
    best = max(CLASSES, key=blended.get)
    conf = float(blended[best])
    agree = max(CLASSES, key=pr.get) == max(CLASSES, key=pm.get)
    cls = best if conf >= min_confidence else "incertain"
    return cls, conf, blended, bool(agree)


AMBIGUITY_CAP = 0.45      # plafond de confiance des règles pour une trace fine, courte et droite
ML_OVERRIDE = 0.90        # … levé seulement si l'IA de référence est très sûre d'elle (et d'accord)


def short_straight_ambiguity(rule_scores: dict, length_mm: float, muon_min_length_mm: float) -> bool:
    """Même critère que les règles : meilleur score « muon » mais trace courte (long_ < 0,5).

    Vue de dessus, une trace fine, courte et droite peut être un muon presque vertical OU un
    électron énergétique : ambiguïté physique réelle, que les règles signalent en plafonnant
    leur confiance à 0,45.
    """
    from .classifier import ramp
    best = max(["alpha", "proton", "muon", "beta"], key=lambda c: _pos((rule_scores or {}).get(c, 0.0)))
    L = _num(length_mm)
    L = L if math.isfinite(L) else 0.0
    return best == "muon" and ramp(L, 0.65 * muon_min_length_mm, muon_min_length_mm) < 0.5


def hybrid_classify(rule_scores: dict, ml_proba: dict, w_ml: float, min_confidence: float,
                    length_mm: float, muon_min_length_mm: float) -> tuple[str, float, dict, bool, bool]:
    """Décision hybride 1.x : ``combine`` + plafond d'ambiguïté « courte et droite »."""
    cls, conf, blended, agree = combine(rule_scores, ml_proba, w_ml, min_confidence)
    best = max(CLASSES, key=blended.get)
    pm = _normalise(ml_proba)
    ml_best = max(CLASSES, key=pm.get)
    capped = False
    if (best in ("muon", "beta") and conf > AMBIGUITY_CAP
            and short_straight_ambiguity(rule_scores, length_mm, muon_min_length_mm)
            and not (ml_best == best and pm[ml_best] >= ML_OVERRIDE)):
        conf = AMBIGUITY_CAP
        capped = True
        cls = best if conf >= min_confidence else "incertain"
    return cls, conf, blended, agree, capped


METHOD_HYBRID = "Hybride (règles + IA de référence)"
METHOD_HYBRID_V2 = "Hybride v2 (règles + IA de référence calibrée)"
HYBRID_EXTRA_KEYS = ("rules_cls", "rules_conf", "rules_scores", "ml_proba", "w_ml", "agree", "capped")
HYBRID_V2_EXTRA_KEYS = ("rules_cls", "rules_conf", "rules_scores", "ml_proba", "w_ml", "agree", "abstained",
                        "competitors", "is_artefact", "threshold", "decision_method", "probs", "family",
                        "family_confidence")
_SYM = {"alpha": "α", "beta": "β", "muon": "μ", "proton": "p", "artefact": "artefact", "fine": "β/μ", "incertain": "?"}
_NOM = {"alpha": "alpha", "beta": "électron (β)", "muon": "muon", "proton": "proton", "artefact": "artefact",
        "fine": "trace fine (β ou μ)", "incertain": "incertaine"}


def _pct(p: float) -> str:
    return f"{int(round(100 * p))} %"


def classify_hybrid(model: "ReferenceModel", f: dict, rules_out: tuple, cp, duration_s=None,
                    touches_border=False, calibrated=None) -> tuple[str, float, dict, str, dict]:
    """Classe une trace en mode hybride 1.x à partir de la sortie de ``classify_rules``.

    Conservé pour compatibilité (4 classes, seuil cp.min_confidence). La version 2 est
    ``classify_hybrid_v2`` (5 classes, probabilités calibrées, abstention à cp.ml_threshold).
    """
    rules_cls, rules_conf, rules_scores, rules_reason = rules_out
    proba = model.predict_proba(f, duration_s, touches_border, calibrated)
    w = _num(getattr(cp, "ref_model_weight", -1.0))
    if not math.isfinite(w) or w < 0:
        w = model.w_ml
    w = min(1.0, max(0.0, float(w)))
    cls, conf, blended, agree, capped = hybrid_classify(
        rules_scores, proba, w, cp.min_confidence, f.get("length_mm", 0.0), cp.muon_min_length_mm)
    r_best = max(["alpha", "proton", "muon", "beta"], key=lambda c: _pos(rules_scores.get(c, 0.0)))
    pm4 = _normalise(proba)
    m_best = max(CLASSES, key=pm4.get)
    if agree:
        tag = f"Hybride : règles et IA d'accord (IA {_pct(pm4[m_best])}) — "
    else:
        tag = (f"Hybride : désaccord règles ({_SYM[r_best]}) / IA ({_SYM[m_best]}, {_pct(pm4[m_best])}) "
               f"→ {_SYM.get(cls, '?')} — ")
    extra = {"rules_cls": rules_cls, "rules_conf": float(rules_conf), "rules_scores": dict(rules_scores),
             "ml_proba": proba, "w_ml": w, "agree": agree, "capped": capped, "method": METHOD_HYBRID}
    return cls, conf, blended, tag + rules_reason, extra


# --------------------------------------------------------------------------- #
# Calibration des probabilités (fonctions pures, numpy ; isotonique ajustée avec scikit-learn)
# --------------------------------------------------------------------------- #
def apply_prior(P: np.ndarray, prior) -> np.ndarray:
    """Décalage de priors : modèle entraîné à classes équilibrées → fréquences de référence π."""
    Q = np.asarray(P, float) * np.asarray(prior, float)[None, :]
    s = Q.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    return Q / s


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    """p ∝ p^(1/T) (T > 1 : probabilités adoucies ; T < 1 : accentuées)."""
    P = np.clip(np.asarray(P, float), 0.0, 1.0)
    T = float(T) if T and math.isfinite(T) and T > 0 else 1.0
    with np.errstate(divide="ignore"):
        L = np.where(P > 0, np.log(np.maximum(P, 1e-300)) / T, -np.inf)
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def _wnll(P: np.ndarray, y_idx: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(P[np.arange(len(P)), y_idx], 1e-4, 1.0)
    return float(-(w * np.log(p)).sum() / max(w.sum(), 1e-12))


def fit_temperature(P: np.ndarray, y_idx: np.ndarray, w: np.ndarray | None = None) -> float:
    """Température minimisant la log-vraisemblance négative pondérée (recherche dorée sur ln T)."""
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    y_idx = np.asarray(y_idx, int)
    a, b = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc = _wnll(apply_temperature(P, math.exp(c)), y_idx, w)
    fd = _wnll(apply_temperature(P, math.exp(d)), y_idx, w)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _wnll(apply_temperature(P, math.exp(c)), y_idx, w)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _wnll(apply_temperature(P, math.exp(d)), y_idx, w)
    return float(math.exp(0.5 * (a + b)))


def _fit_isotonic(P: np.ndarray, y_idx: np.ndarray, w: np.ndarray) -> dict:
    from sklearn.isotonic import IsotonicRegression
    xs, ys = [], []
    for j in range(P.shape[1]):
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(P[:, j], (y_idx == j).astype(float), sample_weight=w)
        xs.append([float(v) for v in ir.X_thresholds_])
        ys.append([float(v) for v in ir.y_thresholds_])
    return {"type": "isotonic", "x": xs, "y": ys}


def apply_calibrator(P: np.ndarray, cal: dict | None) -> np.ndarray:
    """Applique un calibrateur (``none`` / ``temperature`` / ``isotonic``) → probabilités (somme 1)."""
    P = np.asarray(P, float)
    if not cal or cal.get("type") in (None, "none"):
        return P
    if cal["type"] == "temperature":
        return apply_temperature(P, cal.get("T", 1.0))
    if cal["type"] == "isotonic":
        Q = np.empty_like(P)
        for j in range(P.shape[1]):
            x, y = cal["x"][j], cal["y"][j]
            Q[:, j] = np.interp(P[:, j], x, y) if len(x) else P[:, j]
        Q = np.clip(Q, 1e-4, 1.0) * (P > 0)
        s = Q.sum(1, keepdims=True)
        s[s <= 0] = 1.0
        return Q / s
    return P


def fit_calibrator(P: np.ndarray, y_idx: np.ndarray, w: np.ndarray | None = None, groups=None) -> dict:
    """Choisit et ajuste la calibration (aucune, température ou isotonique par classe).

    Le choix se fait par validation croisée en 2 plis (par enregistrement si ``groups`` est
    donné) sur la log-vraisemblance pondérée ; le calibrateur retenu est ensuite ajusté sur
    toutes les données fournies.
    """
    P = np.asarray(P, float)
    y_idx = np.asarray(y_idx, int)
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    n = len(P)
    if groups is not None:
        ug = np.unique(np.asarray(groups))
        rng = np.random.default_rng(0)
        half = set(rng.permutation(ug)[: len(ug) // 2].tolist())
        fold = np.array([g in half for g in np.asarray(groups)])
    else:
        fold = np.arange(n) % 2 == 0
    scores = {"none": 0.0, "temperature": 0.0, "isotonic": 0.0}
    for f in (fold, ~fold):
        tr, te = f, ~f
        if tr.sum() < 20 or te.sum() < 20:
            continue
        scores["none"] += _wnll(P[te], y_idx[te], w[te])
        T = fit_temperature(P[tr], y_idx[tr], w[tr])
        scores["temperature"] += _wnll(apply_temperature(P[te], T), y_idx[te], w[te])
        try:
            iso = _fit_isotonic(P[tr], y_idx[tr], w[tr])
            scores["isotonic"] += _wnll(apply_calibrator(P[te], iso), y_idx[te], w[te])
        except Exception:
            scores["isotonic"] = float("inf")
    best = min(scores, key=scores.get)
    if best == "temperature":
        cal = {"type": "temperature", "T": fit_temperature(P, y_idx, w)}
    elif best == "isotonic":
        cal = _fit_isotonic(P, y_idx, w)
    else:
        cal = {"type": "none"}
    cal["cv_logloss"] = {k: round(v / 2.0, 5) for k, v in scores.items()}
    return cal


# --------------------------------------------------------------------------- #
# Décision v2 : hybride calibré + abstention
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    """Décision de classification d'une trace (v2).

    cls : "alpha" | "beta" | "muon" | "proton" | "artefact" | "incertain" ;
    confidence : probabilité calibrée de la classe la plus probable (même si abstention) ;
    probs : probabilités sur les 5 classes (somme 1) ; agreement : règles et IA d'accord ;
    abstained : sous le seuil → « incertain » ; competitors : [(classe, p), (classe, p)] les deux
    plus probables ; is_artefact : trace jugée artefact (à exclure des comptages) ;
    method : "hybride" | "ia" | "règles" ; threshold : seuil effectivement appliqué ;
    family : "fine" (β, μ) ou "lourde" (α, p) selon p(β)+p(μ) contre p(α)+p(p) ;
    family_confidence : probabilité de cette famille (utile quand la classe est « incertaine »
    entre μ et β : la famille, elle, est presque toujours sûre).
    """
    cls: str
    confidence: float
    probs: dict = field(default_factory=dict)
    agreement: bool = True
    abstained: bool = False
    competitors: list = field(default_factory=list)
    is_artefact: bool = False
    method: str = "hybride"
    threshold: float = DEFAULT_THRESHOLD
    family: str = "fine"
    family_confidence: float = 0.0


def blend(rule_scores: dict, ml_proba: dict, w_ml: float) -> dict:
    """p(c) = (1 − w)·(1 − p_art)·p_règles(c) + w·p_IA(c) pour les particules ; p(artefact) = p_IA(artefact).

    Les règles physiques ne jugent pas les artefacts : la masse « artefact » vient de l'IA seule.
    """
    w = min(1.0, _pos(w_ml))
    pr = rules_probabilities(rule_scores)
    pm = _normalise(ml_proba, ALL_CLASSES)
    pa = pm["artefact"]
    out = {c: (1.0 - w) * (1.0 - pa) * pr[c] + w * pm[c] for c in CLASSES}
    out["artefact"] = pa
    return _normalise(out, ALL_CLASSES)


_DEFAULT = object()


def _default_model():
    try:
        return ReferenceModel.load_default()
    except Exception:
        return None


def _temper(p: dict, T: float) -> dict:
    keys = list(p)
    P = np.array([[max(0.0, p[k]) for k in keys]])
    Q = apply_temperature(P, T)[0]
    return {k: float(v) for k, v in zip(keys, Q)}


def decide(rule_scores: dict, ml_proba: dict | None, *, threshold: float = DEFAULT_THRESHOLD,
           w_ml: float | None = None, rules_short_straight: bool = False, model=_DEFAULT) -> Decision:
    """Décision finale v2 (fonction pure ; lit seulement les paramètres de calibration du modèle).

    - ``ml_proba`` : probabilités CALIBRÉES de l'IA (``ReferenceModel.predict_proba``) ou None
      (règles seules : probabilités normalisées des règles, température des règles si connue) ;
    - ``w_ml`` : poids de l'IA ; None (ou < 0) → poids choisi sur la validation (modèle), 0,5 sinon ;
    - température de l'hybride (validation) appliquée si le poids est celui de l'entraînement ;
    - abstention : probabilité maximale < seuil → « incertain » ; au seuil par défaut, les seuils
      par classe relevés sur la validation s'appliquent aussi ; si ``rules_short_straight``
      (trace fine, courte et droite : muon raide OU électron énergétique), μ/β exigent ≥ 0,90 ;
    - « artefact » : is_artefact = True (trace à exclure des comptages).
    """
    if model is _DEFAULT:
        model = _default_model()
    cal = getattr(model, "calibration", None) or {}
    thr0 = _num(threshold)
    thr0 = thr0 if math.isfinite(thr0) else DEFAULT_THRESHOLD
    pr = rules_probabilities(rule_scores)
    if ml_proba is None:
        probs = {c: pr[c] for c in CLASSES}
        probs["artefact"] = 0.0
        T_r = _num((cal.get("rules") or {}).get("T", 1.0))
        if math.isfinite(T_r) and T_r > 0 and abs(T_r - 1.0) > 1e-9:
            probs = _temper(probs, T_r)
        method, agree = "règles", True
    else:
        w = _num(w_ml) if w_ml is not None else _NAN
        if not math.isfinite(w) or w < 0:
            w = float(getattr(model, "w_ml", 0.5)) if model is not None else 0.5
        w = min(1.0, max(0.0, w))
        probs = blend(rule_scores, ml_proba, w)
        hyb = cal.get("hybrid") or {}
        T_h = _num(hyb.get("T", 1.0))
        if math.isfinite(T_h) and T_h > 0 and abs(_num(hyb.get("w", -9)) - w) < 0.051:
            probs = _temper(probs, T_h)
        pm = _normalise(ml_proba, ALL_CLASSES)
        agree = max(CLASSES, key=pr.get) == max(ALL_CLASSES, key=pm.get)
        method = "ia" if w >= 1.0 else ("règles" if w <= 0.0 else "hybride")
    order = sorted(probs, key=lambda c: -probs[c])
    top = order[0]
    p_top = float(probs[top])
    thr = thr0
    class_thr = cal.get("class_thresholds") or {}
    model_thr = _num(getattr(model, "threshold", DEFAULT_THRESHOLD)) if model is not None else DEFAULT_THRESHOLD
    if class_thr and thr0 >= (model_thr if math.isfinite(model_thr) else DEFAULT_THRESHOLD) - 1e-9:
        thr = max(thr, _num(class_thr.get(top, 0.0)) if math.isfinite(_num(class_thr.get(top, 0.0))) else 0.0)
    if rules_short_straight and top in ("muon", "beta"):
        thr = max(thr, ML_OVERRIDE)
    abstained = p_top < thr
    cls = "incertain" if abstained else top
    comps = [(c, float(probs[c])) for c in order[:2]]
    p_thin = float(probs.get("beta", 0.0) + probs.get("muon", 0.0))
    p_heavy = float(probs.get("alpha", 0.0) + probs.get("proton", 0.0))
    fam = "lourde" if p_heavy > p_thin else "fine"
    return Decision(cls, p_top, probs, bool(agree), bool(abstained), comps, cls == "artefact", method, float(thr),
                    fam, max(p_thin, p_heavy))


def _model_proba(model, f: dict, duration_s, touches_border, calibrated):
    if model is None:
        return None
    try:
        return model.predict_proba(f, duration_s, touches_border, calibrated)
    except Exception:
        return None


def classify_hybrid_v2(model: "ReferenceModel | None", f: dict, rules_out: tuple, cp, duration_s=None,
                       touches_border=False, calibrated=None) -> tuple[str, float, dict, str, dict]:
    """Décision v2 prête à l'emploi pour ``TrackDetector._classify_event`` et la reclassification.

    rules_out = (classe, confiance, scores, raison) de ``classify_rules`` (qui a enregistré
    f["ev_short_straight"]) ; cp = ClassificationParams (ml_threshold, ref_model_weight).
    Renvoie (classe, confiance, probabilités sur 5 classes, raison, infos pour TrackEvent.extra).
    La classe peut être "artefact" (à exclure des comptages) ou "incertain".
    """
    rules_cls, rules_conf, rules_scores, rules_reason = rules_out
    proba = _model_proba(model, f, duration_s, touches_border, True if calibrated is None else calibrated)
    thr = _num(getattr(cp, "ml_threshold", DEFAULT_THRESHOLD))
    thr = thr if math.isfinite(thr) else DEFAULT_THRESHOLD
    w = _num(getattr(cp, "ref_model_weight", -1.0))
    ss = bool(_pos(f.get("ev_short_straight", 0.0)) > 0.5)
    dec = decide(rules_scores, proba, threshold=thr, w_ml=w if (math.isfinite(w) and w >= 0) else None,
                 rules_short_straight=ss, model=model)
    if dec.abstained:
        (c1, p1), (c2, p2) = dec.competitors[0], dec.competitors[1]
        tag = (f"Incertaine (< {_pct(min(dec.threshold, thr) if dec.threshold > 1.0 else dec.threshold)}) : {_NOM[c1]} {_pct(p1)} ou {_NOM[c2]} {_pct(p2)}"
               + (" — courte et droite : muon raide ou électron énergétique" if ss and {c1, c2} <= {"muon", "beta"}
                  else "")
               + (f" ; trace {dec.family} sûre à {_pct(dec.family_confidence)}" if dec.family_confidence >= thr else "")
               + " — ")
    elif dec.is_artefact:
        tag = f"Artefact probable ({_pct(dec.confidence)}) : exclue des comptages — "
    else:
        src = {"hybride": "règles + IA", "ia": "IA", "règles": "règles"}[dec.method]
        tag = (f"{_NOM[dec.cls].capitalize()} à {_pct(dec.confidence)} ({src}"
               + (", d'accord" if dec.agreement and dec.method == "hybride" else
                  (", désaccord règles/IA" if dec.method == "hybride" else "")) + ") — ")
    if math.isfinite(w) and w >= 0:
        w_used = min(1.0, float(w))
    else:
        w_used = float(getattr(model, "w_ml", 0.5)) if model is not None else 0.5
    extra = {"rules_cls": rules_cls, "rules_conf": float(_pos(rules_conf)), "rules_scores": dict(rules_scores or {}),
             "ml_proba": proba, "w_ml": w_used,
             "agree": dec.agreement, "abstained": dec.abstained, "competitors": [list(c) for c in dec.competitors],
             "is_artefact": dec.is_artefact, "threshold": dec.threshold, "decision_method": dec.method,
             "probs": dict(dec.probs), "family": dec.family, "family_confidence": dec.family_confidence,
             "method": METHOD_HYBRID_V2 if proba is not None else "Règles physiques"}
    return dec.cls, dec.confidence, dec.probs, tag + rules_reason, extra


def family(f: dict, proba: dict | None = None) -> str:
    """« fine » (minimum d'ionisation : électron rapide, muon) ou « lourde » (alpha, proton).

    Avec ``proba`` (probabilités du modèle) : comparaison p(α) + p(p) contre p(β) + p(μ) ;
    sinon la « lourdeur » des règles (épaisseur et densité relatives, f["ev_heavy"]) ; à défaut,
    la densité / l'épaisseur relatives.
    """
    if proba:
        heavy = _pos(proba.get("alpha")) + _pos(proba.get("proton"))
        light = _pos(proba.get("beta")) + _pos(proba.get("muon"))
        if heavy + light > 0:
            return "lourde" if heavy > light else "fine"
    h = _num((f or {}).get("ev_heavy"))
    if math.isfinite(h):
        return "lourde" if h >= 0.5 else "fine"
    d, wr = _num((f or {}).get("density_rel")), _num((f or {}).get("width_rel"))
    return "lourde" if ((math.isfinite(d) and d >= 1.9) or (math.isfinite(wr) and wr >= 1.9)) else "fine"


def _wilson(k: int, n: int, z: float = 1.96) -> list:
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def _fr(x: float, nd: int = 0) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def evaluate_on_labeled(features_list: list[dict], labels: list[str], extras: list[dict] | None = None,
                        *, threshold: float | None = None, model=_DEFAULT) -> dict:
    """Mesure l'exactitude de la décision v2 sur des traces étiquetées (ex. : vraies traces corrigées
    à la main par l'utilisateur).

    extras[i] (facultatif) : {"duration_s", "touches_border", "calibrated", "rule_scores" (ou
    "rules_scores"), "rules_short_straight", "w_ml"}. Sans scores des règles : IA seule.
    Seules les étiquettes alpha / beta / muon / proton / artefact sont prises en compte.
    Renvoie {"n", "accuracy" (sur les traces classées avec certitude), "coverage", "accuracy_all",
    "accuracy_ci95", "per_class": {cls: {"precision", "recall", "n"}}, "confusion": {...}, "note"}.
    """
    if model is _DEFAULT:
        model = _default_model()
    thr = _num(threshold) if threshold is not None else _num(getattr(model, "threshold", DEFAULT_THRESHOLD))
    thr = thr if math.isfinite(thr) else DEFAULT_THRESHOLD
    extras = extras or [{} for _ in features_list]
    labels_true = list(ALL_CLASSES)
    labels_pred = list(ALL_CLASSES) + ["incertain"]
    M = np.zeros((len(labels_true), len(labels_pred)), int)
    n = 0
    for f, lab, ex in zip(features_list, labels, extras):
        if lab not in labels_true:
            continue
        ex = ex or {}
        f = f or {}
        proba = _model_proba(model, f, ex.get("duration_s"), bool(ex.get("touches_border", False)),
                             ex.get("calibrated", True))
        rs = ex.get("rule_scores", ex.get("rules_scores"))
        ss = bool(ex.get("rules_short_straight", _pos(f.get("ev_short_straight", 0.0)) > 0.5))
        if rs is None and proba is None:
            pred = "incertain"
        elif rs is None:
            pred = decide({}, proba, threshold=thr, w_ml=1.0, rules_short_straight=ss, model=model).cls
        else:
            pred = decide(rs, proba, threshold=thr, w_ml=ex.get("w_ml"), rules_short_straight=ss, model=model).cls
        M[labels_true.index(lab), labels_pred.index(pred)] += 1
        n += 1
    conf_cols = [j for j, c in enumerate(labels_pred) if c != "incertain"]
    n_conf = int(M[:, conf_cols].sum())
    correct = int(sum(M[i, labels_pred.index(c)] for i, c in enumerate(labels_true)))
    per = {}
    for i, c in enumerate(labels_true):
        nt = int(M[i].sum())
        j = labels_pred.index(c)
        npred = int(M[:, j].sum())
        if nt or npred:
            per[c] = {"precision": (M[i, j] / npred) if npred else None, "recall": (M[i, j] / nt) if nt else None,
                      "n": nt}
    acc = correct / n_conf if n_conf else None
    cov = n_conf / n if n else None
    ci = _wilson(correct, n_conf)
    if n == 0:
        note = "Aucune trace étiquetée (alpha, bêta, muon, proton ou artefact) : rien à mesurer."
    elif model is None:
        note = ("Modèle de référence indisponible : décision des règles physiques seules "
                f"(seuil {_fr(100 * thr)} %).")
    else:
        note = (f"Mesuré sur {n} traces étiquetées : {n_conf} classées avec certitude (couverture "
                f"{_fr(100 * cov)} %)" + (f", dont {correct} justes (exactitude {_fr(100 * acc)} %, intervalle à 95 % : "
                                          f"{_fr(100 * ci[0])}–{_fr(100 * ci[1])} %)" if n_conf else "")
                + ". « Incertain » n'est pas compté comme une erreur, mais réduit la couverture.")
        if n_conf < 30:
            note += " Échantillon petit : résultat encore très incertain."
    return {"n": n, "accuracy": acc, "coverage": cov, "accuracy_all": correct / n if n else None,
            "accuracy_ci95": ci, "n_confident": n_conf, "threshold": thr, "per_class": per,
            "confusion": {"labels_true": labels_true, "labels_pred": labels_pred, "matrix": M.tolist()},
            "note": note}


def reference_confusion() -> dict | None:
    """Matrice de confusion du modèle livré sur le TEST simulé (seuil d'abstention appliqué), ou None."""
    m = _default_model()
    if m is None:
        return None
    c = getattr(m, "confusion_test", None)
    return dict(c) if isinstance(c, dict) and c.get("matrix") else None


# --------------------------------------------------------------------------- #
# Prédicteur rapide
# --------------------------------------------------------------------------- #
class _FastForest:
    """Prédiction d'UNE trace sans OpenMP : parcours vectorisé (numpy) de tous les arbres.

    ``HistGradientBoostingClassifier.predict_proba`` ouvre une région parallèle OpenMP par
    arbre (des centaines par appel) : pour une seule trace, sur un processeur déjà occupé
    par l'analyse vidéo, un appel peut prendre plus de 0,2 s. Ici, tous les arbres avancent
    d'un niveau à la fois (≈ 0,3 ms par trace). Le résultat est vérifié au chargement
    contre scikit-learn ; en cas d'écart, on revient à la méthode de scikit-learn.
    """

    def __init__(self, model):
        K = int(model.n_trees_per_iteration_)
        if K < 3 or K != len(model.classes_):
            raise ValueError("modèle non multiclasse")
        feat, thr, mleft, left, right, leaf, value, roots, tree_cls = [], [], [], [], [], [], [], [], []
        off, depth = 0, 0
        for it in model._predictors:
            for k, p in enumerate(it):
                n = p.nodes
                if n["is_categorical"].any():
                    raise ValueError("variables catégorielles non prises en charge")
                roots.append(off)
                tree_cls.append(k)
                feat.append(n["feature_idx"].astype(np.int64))
                thr.append(n["num_threshold"].astype(np.float64))
                mleft.append(n["missing_go_to_left"].astype(bool))
                left.append(n["left"].astype(np.int64) + off)
                right.append(n["right"].astype(np.int64) + off)
                leaf.append(n["is_leaf"].astype(bool))
                value.append(n["value"].astype(np.float64))
                depth = max(depth, int(n["depth"].max()))
                off += len(n)
        cat = np.concatenate
        self.feat, self.thr, self.mleft = cat(feat), cat(thr), cat(mleft)
        self.left, self.right, self.leaf, self.value = cat(left), cat(right), cat(leaf), cat(value)
        self.roots = np.array(roots, np.int64)
        self.tree_cls = np.array(tree_cls, np.int64)
        self.depth = depth
        self.K = K
        self.baseline = np.asarray(model._baseline_prediction, np.float64).reshape(-1)
        if self.baseline.size != K:
            raise ValueError("prédiction de base inattendue")

    def proba(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((len(X), self.K))
        for i, x in enumerate(np.asarray(X, np.float64)):
            idx = self.roots.copy()
            for _ in range(self.depth + 1):
                lf = self.leaf[idx]
                if lf.all():
                    break
                v = x[self.feat[idx]]
                go_left = np.where(np.isnan(v), self.mleft[idx], v <= self.thr[idx])
                idx = np.where(lf, idx, np.where(go_left, self.left[idx], self.right[idx]))
            raw = self.baseline + np.bincount(self.tree_cls, weights=self.value[idx], minlength=self.K)
            raw -= raw.max()
            e = np.exp(raw)
            out[i] = e / e.sum()
        return out


def _sk_proba(model, X: np.ndarray) -> np.ndarray:
    """predict_proba de scikit-learn limité à un fil OpenMP (évite la sursouscription)."""
    try:
        from threadpoolctl import threadpool_limits
    except Exception:                       # pragma: no cover
        return model.predict_proba(X)
    with threadpool_limits(limits=1, user_api="openmp"):
        return model.predict_proba(X)


def _check_vectors(n_features: int) -> np.ndarray:
    """Vecteurs de contrôle (échelles variées, valeurs manquantes) pour valider _FastForest."""
    rng = np.random.default_rng(12345)
    X = np.concatenate([rng.normal(0, 1, (8, n_features)) * s for s in (0.1, 1.0, 5.0, 30.0)])
    X = np.abs(X) * np.where(rng.uniform(size=X.shape) < 0.2, -1.0, 1.0)
    X[rng.uniform(size=X.shape) < 0.1] = np.nan
    return np.vstack([X, np.zeros((1, n_features)), np.full((1, n_features), np.nan)])


# --------------------------------------------------------------------------- #
# Modèle
# --------------------------------------------------------------------------- #
class ReferenceModel:
    """Modèle de référence chargé depuis ``wilsontrack/assets/modele_reference.joblib``."""

    def __init__(self, payload: dict):
        self.model = payload["model"]
        self.features: list[str] = list(payload["features"])
        self.format: int = int(payload.get("format", 1))
        model_classes = [str(c) for c in getattr(self.model, "classes_", payload.get("classes", CLASSES))]
        default_classes = ALL_CLASSES if "artefact" in model_classes else CLASSES
        self.classes: list[str] = [c for c in payload.get("classes", default_classes) if c in model_classes] \
            or [c for c in default_classes if c in model_classes]
        self.w_ml: float = float(payload.get("w_ml", 0.5))
        self.threshold: float = float(payload.get("threshold", DEFAULT_THRESHOLD))
        self.calibration: dict = dict(payload.get("calibration") or {})
        self.metrics: dict = dict(payload.get("metrics", {}))
        self.confusion_test: dict | None = payload.get("confusion_test")
        self.n_train: int = int(payload.get("n_train", 0))
        self.n_tracks_total: int = int(payload.get("n_tracks_total", 0))
        self.n_jobs: int = int(payload.get("n_jobs", 0))
        self.created: str = str(payload.get("created", ""))
        self.sklearn_version: str = str(payload.get("sklearn_version", ""))
        self.generator: dict = dict(payload.get("generator", {}))
        self.model_type: str = str(payload.get("model_type", type(self.model).__name__))
        self._col = [model_classes.index(c) for c in self.classes]
        prior = (self.calibration.get("prior_ref") or {}) if self.calibration.get("trained_balanced") else {}
        self._prior = np.array([float(prior.get(c, 1.0)) for c in self.classes]) if prior else None
        self._fast: _FastForest | None = None

    def _enable_fast_path(self) -> bool:
        """Active le prédicteur sans OpenMP s'il reproduit scikit-learn (écart < 1e-7)."""
        try:
            Xc = _check_vectors(len(self.features))
            ref = _sk_proba(self.model, Xc)
            fast = _FastForest(self.model)
            if np.allclose(fast.proba(Xc), ref, rtol=0.0, atol=1e-7):
                self._fast = fast
        except Exception:
            self._fast = None
        return self._fast is not None

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        if self._fast is not None:
            return self._fast.proba(X)
        return _sk_proba(self.model, X)

    # ------------------------------------------------------------------ chargement --
    @classmethod
    def load(cls, path: str | Path) -> "ReferenceModel | None":
        """Charge un modèle ; None en cas de problème (jamais d'exception)."""
        try:
            path = Path(path)
            if not path.is_file():
                return None
            import joblib
            import sklearn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                payload = joblib.load(path)
            if not isinstance(payload, dict) or "model" not in payload or "features" not in payload:
                return None
            ver = str(payload.get("sklearn_version", ""))
            if ver.split(".")[:2] != str(sklearn.__version__).split(".")[:2]:
                return None
            m = cls(payload)
            # Vérification : une prédiction sur un vecteur vide doit fonctionner ; puis
            # activation (si identique) du prédicteur rapide sans OpenMP.
            _sk_proba(m.model, np.full((1, len(m.features)), np.nan))
            m._enable_fast_path()
            return m
        except Exception:
            return None

    @classmethod
    def load_default(cls) -> "ReferenceModel | None":
        """Modèle livré avec l'application (mis en cache : chargé une seule fois ; jamais d'exception)."""
        key = str(DEFAULT_PATH)
        if key not in _CACHE:
            try:
                _CACHE[key] = cls.load(DEFAULT_PATH)
            except Exception:
                _CACHE[key] = None
        return _CACHE[key]

    # ----------------------------------------------------------------- prédiction --
    def vectorize(self, f: dict, duration_s=None, touches_border=False, calibrated=None) -> np.ndarray:
        return np.array([feature_vector(f, self.features, duration_s, touches_border, calibrated)],
                        dtype=np.float64)

    def _calibrate(self, P: np.ndarray) -> np.ndarray:
        """Colonnes du modèle → self.classes, décalage de priors et calibration (modèle v2)."""
        Q = P[:, self._col]
        s = Q.sum(1, keepdims=True)
        s[s <= 0] = 1.0
        Q = Q / s
        if self._prior is not None:
            Q = apply_prior(Q, self._prior)
        cal = self.calibration.get("ml")
        if cal:
            Q = apply_calibrator(Q, cal)
        return Q

    def predict_proba(self, f: dict, duration_s=None, touches_border=False, calibrated=True) -> dict:
        """Probabilités CALIBRÉES (somme = 1) sur les classes du modèle (5 pour la v2, 4 pour la v1)."""
        q = self._calibrate(self._raw_proba(self.vectorize(f, duration_s, touches_border, calibrated)))[0]
        return {c: float(v) for c, v in zip(self.classes, q)}

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Probabilités calibrées pour une matrice déjà vectorisée (colonnes = self.classes)."""
        X = np.asarray(X, np.float64)
        p = _sk_proba(self.model, X) if len(X) > 64 or self._fast is None else self._fast.proba(X)
        return self._calibrate(p)

    def predict_proba_many(self, X: np.ndarray) -> np.ndarray:
        """Probabilités sur les 4 particules (colonnes = CLASSES, renormalisées) — compatibilité 1.x."""
        P = self.predict_proba_matrix(X)
        out = np.zeros((len(X), len(CLASSES)))
        for j, c in enumerate(CLASSES):
            if c in self.classes:
                out[:, j] = P[:, self.classes.index(c)]
        s = out.sum(1, keepdims=True)
        s[s <= 0] = 1.0
        return out / s

    def decide(self, rule_scores: dict, f: dict, duration_s=None, touches_border=False, calibrated=True, *,
               threshold: float | None = None, w_ml: float | None = None,
               rules_short_straight: bool | None = None) -> Decision:
        """Raccourci : probabilités de ce modèle + ``decide``."""
        ss = bool(_pos(f.get("ev_short_straight", 0.0)) > 0.5) if rules_short_straight is None else rules_short_straight
        return decide(rule_scores, self.predict_proba(f, duration_s, touches_border, calibrated),
                      threshold=self.threshold if threshold is None else threshold, w_ml=w_ml,
                      rules_short_straight=ss, model=self)

    # --------------------------------------------------------------------- infos --
    @property
    def info(self) -> dict:
        """Résumé lisible (interface, rapport). Contrat : voir le docstring du module et l'outil."""
        m = self.metrics or {}
        summary: dict = {}
        if self.format >= 2:
            for split in ("test", "stress"):
                summary[split] = dict(m.get(split) or {})
            for k, v in m.items():
                if isinstance(k, str) and k.count("_") >= 2 and k.split("_")[0] in ("test", "stress") \
                        and not isinstance(v, (dict, list)):
                    summary[k] = v
        else:
            for split in ("test", "stress"):
                d = m.get(split) or {}
                hy = d.get("hybrid") or {}
                summary[split] = {"n": hy.get("n"), "coverage": None, "accuracy_confident": hy.get("accuracy"),
                                  "balanced_accuracy": hy.get("balanced_accuracy"), "per_class": {},
                                  "rules": (d.get("rules") or {}).get("balanced_accuracy"),
                                  "ml": (d.get("ml") or {}).get("balanced_accuracy"),
                                  "hybrid": hy.get("balanced_accuracy")}
                for k in ("rules", "ml", "hybrid"):
                    dk = d.get(k) or {}
                    summary[f"{split}_{k}_balanced_accuracy"] = dk.get("balanced_accuracy")
                    summary[f"{split}_{k}_accuracy"] = dk.get("accuracy")
        details = (m.get("details") or {}).get("test") or {}
        ece = (details.get("reliability_v2") or {}).get("ece")
        return {"n_tracks_total": self.n_tracks_total, "n_train": self.n_train, "n_jobs": self.n_jobs,
                "classes": list(self.classes), "threshold": self.threshold, "created": self.created,
                "model_type": self.model_type, "generator": self.generator, "metrics": summary,
                "confusion_test": self.confusion_test, "w_ml": self.w_ml, "format": self.format,
                "class_thresholds": dict(self.calibration.get("class_thresholds") or {}),
                "prior_ref": dict(self.calibration.get("prior_ref") or {}),
                "calibration_type": (self.calibration.get("ml") or {}).get("type"), "ece_test": ece,
                "sklearn_version": self.sklearn_version, "fast_predictor": self._fast is not None}


def clear_cache() -> None:
    _CACHE.clear()
