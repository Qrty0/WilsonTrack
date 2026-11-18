"""Classification des traces : règles physiques explicables + IA optionnelle.

Principe physique (pouvoir d'arrêt, formule de Bethe) : plus une particule ionise
fortement (particule lourde et lente, charge 2 pour l'alpha), plus sa trace est
épaisse et dense en gouttelettes. Une particule au minimum d'ionisation (muon
cosmique, électron rapide) laisse une trace fine. La rectitude distingue les
particules massives/énergétiques (droites) des électrons légers (diffusion
multiple → trajectoire sinueuse).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import physics
from ..config import ClassificationParams, PhysicsParams

CLASSES = ["alpha", "beta", "muon", "proton"]

ML_FEATURES = ["length_mm", "width_mm", "width_rel", "density_rel", "straightness", "tortuosity",
               "elongation", "sagitta_ratio", "asymmetry", "profile_cv", "branches", "peak_snr",
               "curl_consistency", "lateral_mm", "line_rms_mm"]


def ramp(x: float, a: float, b: float) -> float:
    """0 si x ≤ a, 1 si x ≥ b, linéaire entre les deux."""
    if b == a:
        return 1.0 if x >= b else 0.0
    return float(min(1.0, max(0.0, (x - a) / (b - a))))


def alpha_max_length_for(temp_c: float, pressure_kpa: float = 101.325) -> float:
    """Longueur maximale (mm) d'une alpha naturelle à la température de la couche sensible :
    1,05 × portée NIST ASTAR du Po-212 (8,785 MeV, la plus énergétique des alphas naturelles)
    dans l'air à (T, P). ≈ 78 mm à −25 °C, ≈ 86 mm à 0 °C, ≈ 91 mm à 15 °C."""
    from ..synth import alpha_range_mm
    return 1.05 * alpha_range_mm(8.785, temp_c, pressure_kpa)


@dataclass
class References:
    """Niveau de référence des traces « fines » (particules au minimum d'ionisation)."""
    width_px: float = 0.0
    density: float = 0.0
    n: int = 0
    method: str = "défaut"

    @property
    def ready(self) -> bool:
        return self.n > 0 and self.width_px > 0 and self.density > 0


def compute_references(feature_list: list[dict], percentile: float = 35.0) -> References:
    """Estime la référence « trace fine » à partir d'un ensemble de traces.

    Si la distribution de log(densité) est bimodale (mélange traces fines / traces
    épaisses), on prend la composante basse d'un mélange gaussien ; sinon un percentile.
    """
    feats = [f for f in feature_list if f.get("length_px", 0) > 0 and f.get("density", 0) > 0]
    # Seules les traces BIEN MESURÉES servent de référence : les petites détections pâles,
    # proches du seuil de bruit, tireraient la référence vers le bas (toutes les vraies traces
    # paraîtraient alors « denses »).
    good = [f for f in feats if f.get("peak_snr", 0) >= 12.0 and f.get("length_mm", 0) >= 6.0]
    if len(good) >= 6:
        feats = good
    if len(feats) < 6:
        return References()
    dens = np.array([f["density"] for f in feats])
    wid = np.array([max(f.get("width_px", f.get("width_eff_px", 1.0)), 0.3) for f in feats])
    ld = np.log(dens)
    method = f"percentile {percentile:g}"
    ref_d = float(np.exp(np.percentile(ld, percentile)))
    ref_w = float(np.percentile(wid, percentile))
    if len(feats) >= 20:
        try:
            from sklearn.mixture import GaussianMixture
            X = np.column_stack([ld, np.log(wid)])
            gm = GaussianMixture(2, random_state=0, n_init=2).fit(X)
            lo, hi = np.argsort(gm.means_[:, 0])
            sep = gm.means_[hi, 0] - gm.means_[lo, 0]
            spread = math.sqrt(gm.covariances_[lo][0, 0]) + math.sqrt(gm.covariances_[hi][0, 0])
            if sep > 0.8 and sep > 1.2 * spread and gm.weights_[lo] > 0.1:
                ref_d = float(np.exp(gm.means_[lo, 0]))
                ref_w = float(np.exp(gm.means_[lo, 1]))
                method = "mélange gaussien (bimodal)"
        except Exception:
            pass
    return References(ref_w, ref_d, len(feats), method)


def classify_rules(f: dict, refs: References, cp: ClassificationParams, calibrated: bool,
                   px_per_mm_proc: float) -> tuple[str, float, dict, str]:
    L = f["length_mm"]
    W = f["width_mm"]
    S = f["straightness"]
    T = f["tortuosity"]
    if refs.ready:
        w_rel = f["width_px"] / refs.width_px
        d_rel = f["density"] / refs.density
    else:
        # Sans référence : estimation à partir d'une largeur typique de trace fine (0.6 mm)
        w_rel = W / 0.6 if calibrated else f["width_px"] / max(2.0, 0.6 * px_per_mm_proc)
        d_rel = w_rel * 0.9
    f["width_rel"] = w_rel
    f["density_rel"] = d_rel

    thick_rel = ramp(w_rel, 0.8 * cp.heavy_width_rel, 1.2 * cp.heavy_width_rel)
    thick_abs = ramp(W, 0.75 * cp.alpha_width_mm, 1.15 * cp.alpha_width_mm) if calibrated else 0.0
    thick = max(thick_rel, thick_abs)
    dense = ramp(d_rel, 0.75 * cp.heavy_density_rel, 1.3 * cp.heavy_density_rel)
    heavy = 0.35 * thick + 0.65 * dense
    light = 1.0 - heavy

    sag = f.get("sagitta_ratio", 0.0)                       # flèche max / longueur
    wig = f.get("line_rms_mm", 0.0) / max(L, 1.0)           # écart RMS à la droite / longueur
    lat = f.get("lateral_mm", 0.0)                          # étalement latéral excédentaire (mm)
    # Pour une trace courte (< ~2 cm), le squelette est trop imprécis aux extrémités : la forme
    # est jugée sur l'étalement latéral (ACP pondérée), robuste ; pour une trace longue, sur le
    # chemin central (rectitude, flèche, écart RMS, tortuosité).
    sh = ramp(L, 12.0, 25.0)
    straight_long = (ramp(S, cp.straight_min - 0.04, cp.straight_min + 0.01)
                     * (1.0 - ramp(sag, 0.035, 0.07)) * (1.0 - ramp(wig, 0.012, 0.022)))
    straight_short = 1.0 - ramp(lat, 0.20, 0.45)
    straight_m = sh * straight_long + (1.0 - sh) * straight_short
    # Critère facultatif (cp.muon_max_rms_mm > 0) : écart RMS à la droite. Un muon de 4 GeV ne s'écarte
    # que de ≈ 0,001 mm ; un électron du K-40, de 0,5 à 7 mm sur 2–5 cm (Highland). Appliqué aux
    # traces assez longues pour que l'écart soit mesurable (poids sh).
    rms_ok = 1.0
    max_rms = float(getattr(cp, "muon_max_rms_mm", 0.0) or 0.0)
    if max_rms > 0:
        rms_ok = 1.0 - sh * ramp(f.get("line_rms_mm", 0.0), max_rms, 1.6 * max_rms)
        straight_m *= rms_ok
    straight_a = ramp(S, 0.80, 0.90) if sh > 0.5 else max(ramp(S, 0.80, 0.90), 1.0 - ramp(lat, 0.3, 0.6))
    long_ = ramp(L, 0.65 * cp.muon_min_length_mm, cp.muon_min_length_mm)
    short_a = 1.0 - ramp(L, cp.alpha_max_length_mm, 1.3 * cp.alpha_max_length_mm)
    curly_long = max(1.0 - ramp(S, cp.curly_max, cp.straight_min), ramp(sag, 0.05, 0.10),
                     ramp(wig, 0.015, 0.03), ramp(T, 0.10, 0.20))
    curly_short = ramp(lat, 0.30, 0.60)
    curly = sh * curly_long + (1.0 - sh) * curly_short

    scores = {
        "alpha": heavy * straight_a * short_a,
        "proton": heavy * straight_a * (1.0 - short_a),
        "muon": light * straight_m * (0.3 + 0.7 * long_),
        # Plancher « court → électron » seulement pour les traces qui ne sont PAS nettement droites :
        # une trace fine, courte et parfaitement droite est le plus souvent un muon quasi vertical
        # (longueur projetée ≈ épaisseur de la couche × tan θ).
        "beta": light * max(curly, 0.45 * (1.0 - long_) * (1.0 - straight_m)),
    }
    order = sorted(scores, key=scores.get, reverse=True)
    best, second = order[0], order[1]
    b, s2 = scores[best], scores[second]
    conf = b * (b / (b + s2 + 1e-9)) * 2.0 if b > 0 else 0.0
    conf = float(min(1.0, conf))
    short_straight = best == "muon" and long_ < 0.5
    if short_straight:
        conf = min(conf, 0.45)   # ambiguïté réelle : muon raide ou électron énergétique
    cls = best if conf >= cp.min_confidence else "incertain"

    # Valeurs floues intermédiaires (0..1) réellement utilisées pour décider : enregistrées pour
    # l'explication « pourquoi cette classe ? » (detection/explain.py). Clés plates, flottants
    # Python (export CSV / JSON). N'influencent pas la décision ci-dessus.
    f["ev_thick"] = float(thick)
    f["ev_dense"] = float(dense)
    f["ev_heavy"] = float(heavy)
    f["ev_straight_m"] = float(straight_m)
    f["ev_straight_a"] = float(straight_a)
    f["ev_curly"] = float(curly)
    f["ev_long"] = float(long_)
    f["ev_short_a"] = float(short_a)
    f["ev_sh"] = float(sh)
    f["ev_sag"] = float(sag)
    f["ev_wig"] = float(wig)
    f["ev_lat"] = float(lat)
    f["ev_refs_ready"] = 1.0 if refs.ready else 0.0
    f["ev_short_straight"] = 1.0 if short_straight else 0.0
    f["ev_rms_ok"] = float(rms_ok)            # critère d'écart RMS des muons (1 = respecté ou désactivé)

    reason = (f"L={L:.1f} mm, épaisseur={W:.2f} mm (×{w_rel:.1f} réf.), densité ×{d_rel:.1f}, "
              f"rectitude={S:.3f}, tortuosité={T:.3f} rad/mm → "
              f"{'lourde' if heavy >= 0.5 else 'fine'}, "
              f"{'droite' if S >= cp.straight_min else ('sinueuse' if S < cp.curly_max else 'peu courbée')}, "
              f"{'longue' if long_ >= 0.5 else 'courte'}"
              + (" (courte et droite : muon raide ou électron énergétique)" if short_straight else ""))
    return cls, conf, scores, reason


def physics_annotations(f: dict, cls: str, php: PhysicsParams, touches_border: bool) -> None:
    """Ajoute l'énergie estimée (alphas : portée) ou l'impulsion (champ B : courbure)."""
    f.pop("momentum_mev", None)
    f.pop("energy_mev", None)
    f.pop("emitter", None)
    if cls == "alpha" and not touches_border and f["length_mm"] > 3:
        # Longueur projetée (vue de dessus) et tronquée par la couche sensible → énergie MINIMALE.
        # L'émetteur ne s'identifie qu'à partir du bord supérieur de la distribution (voir stats).
        e = physics.alpha_energy_from_range(f["length_mm"], php.chamber_temp_c, php.pressure_kpa)
        if 0.5 < e < 12.0:
            f["energy_mev"] = e
    if php.b_field_mt > 0 and math.isfinite(f.get("radius_mm", float("inf"))):
        r = f["radius_mm"]
        # N'utiliser la courbure que si elle est cohérente (arc régulier, pas un zigzag)
        if f.get("curl_consistency", 0) > 0.6 and f.get("sagitta_ratio", 0) > 0.02 and r < 5000:
            p = physics.momentum_mev(php.b_field_mt, r)
            f["momentum_mev"] = p
            mass = {"alpha": physics.ALPHA_MASS_MEV, "proton": physics.PROTON_MASS_MEV,
                    "muon": physics.MUON_MASS_MEV}.get(cls, physics.ELECTRON_MASS_MEV)
            charge = 2 if cls == "alpha" else 1
            p_real = p * charge
            f["momentum_mev"] = p_real
            if cls != "alpha":
                f["energy_mev"] = physics.kinetic_energy_mev(p_real, mass)


# --------------------------------------------------------------------------- #
# IA : apprentissage supervisé à partir des étiquettes de l'utilisateur
# --------------------------------------------------------------------------- #
class MLClassifier:
    def __init__(self):
        self.model = None
        self.classes_: list[str] = []
        self.cv_accuracy: float | None = None
        self.n_train = 0

    @staticmethod
    def vectorize(f: dict) -> list[float]:
        v = []
        for k in ML_FEATURES:
            x = f.get(k, 0.0)
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                x = 0.0
            v.append(float(x))
        return v

    def train(self, feature_list: list[dict], labels: list[str]) -> str:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        X = np.array([self.vectorize(f) for f in feature_list])
        y = np.array(labels)
        if len(set(labels)) < 2 or len(labels) < 6:
            raise ValueError("Il faut au moins 6 traces étiquetées et 2 classes différentes.")
        self.model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            min_samples_leaf=1, random_state=0)
        counts = {c: int((y == c).sum()) for c in set(labels)}
        k = min(5, min(counts.values()))
        msg = ""
        if k >= 2:
            sc = cross_val_score(self.model, X, y, cv=k)
            self.cv_accuracy = float(sc.mean())
            msg = f"Exactitude en validation croisée ({k} plis) : {100 * self.cv_accuracy:.1f} %"
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        self.n_train = len(labels)
        imp = sorted(zip(ML_FEATURES, self.model.feature_importances_), key=lambda t: -t[1])[:5]
        msg += "\nVariables les plus utiles : " + ", ".join(f"{k} ({v:.2f})" for k, v in imp)
        return msg

    def predict(self, f: dict) -> tuple[str, float, dict]:
        proba = self.model.predict_proba(np.array([self.vectorize(f)]))[0]
        scores = {c: float(p) for c, p in zip(self.classes_, proba)}
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"model": self.model, "classes": self.classes_, "features": ML_FEATURES,
                     "cv_accuracy": self.cv_accuracy, "n_train": self.n_train}, path)

    @classmethod
    def load(cls, path: str) -> "MLClassifier":
        import joblib
        d = joblib.load(path)
        m = cls()
        m.model = d["model"]
        m.classes_ = d["classes"]
        m.cv_accuracy = d.get("cv_accuracy")
        m.n_train = d.get("n_train", 0)
        return m


def cluster_tracks(feature_list: list[dict], k: int = 3) -> np.ndarray:
    """Regroupement non supervisé (mélange gaussien) selon l'apparence des traces."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    X = np.array([[math.log(max(f["length_mm"], 0.1)), math.log(max(f.get("width_rel", 1.0), 0.05)),
                   math.log(max(f.get("density_rel", 1.0), 0.05)), f["straightness"]]
                  for f in feature_list])
    if len(X) < k:
        return np.zeros(len(X), int)
    Xs = StandardScaler().fit_transform(X)
    return GaussianMixture(k, random_state=0, n_init=3).fit_predict(Xs)
