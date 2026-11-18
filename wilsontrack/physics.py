"""Constantes et relations physiques utilisées pour interpréter les traces.

Sources principales (voir docs/METHODE.md pour la bibliographie complète) :
- NNDC / Laboratoire National Henri Becquerel : données de désintégration du K-40.
- Relation portée-énergie de Geiger pour les alphas dans l'air (R ≈ 0.318·E^1.5) ;
  NIST ASTAR (portées CSDA dans l'air sec) pour alpha_range_astar_cm.
- Particle Data Group (PDG) : flux de muons au niveau de la mer, p = 0.3·B·R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Constantes fondamentales
# --------------------------------------------------------------------------- #
AVOGADRO = 6.02214076e23
YEAR_S = 365.25 * 24 * 3600
ELECTRON_MASS_MEV = 0.51099895
ALPHA_MASS_MEV = 3727.379
PROTON_MASS_MEV = 938.272
MUON_MASS_MEV = 105.658
C_MEV_PER_T_M = 299.792458  # p[MeV/c] = 299.79 · B[T] · R[m] (charge unitaire)

# --------------------------------------------------------------------------- #
# Potassium-40
# --------------------------------------------------------------------------- #
K_MOLAR_MASS = 39.0983          # g/mol
KCL_MOLAR_MASS = 74.551         # g/mol
K40_ABUNDANCE = 1.17e-4         # fraction isotopique naturelle (0.0117 %)
K40_HALF_LIFE_Y = 1.248e9       # années
K40_BETA_BRANCH = 0.8928        # β⁻ vers Ca-40
K40_EC_BRANCH = 0.1072          # capture électronique vers Ar-40 (γ 1460.8 keV à 10.66 %)
K40_GAMMA_YIELD = 0.1066
K40_BETA_EMAX_MEV = 1.311
K40_BETA_MEAN_MEV = 0.560


def specific_activity_k() -> float:
    """Activité massique du potassium naturel (Bq par gramme de K) ≈ 31.7 Bq/g."""
    n_k40 = AVOGADRO / K_MOLAR_MASS * K40_ABUNDANCE
    lam = math.log(2) / (K40_HALF_LIFE_Y * YEAR_S)
    return n_k40 * lam


# Fraction massique de potassium de quelques échantillons usuels
SAMPLES = {
    "Aucun": 0.0,
    "KCl (chlorure de potassium)": K_MOLAR_MASS / KCL_MOLAR_MASS,         # 0.5245
    "Substitut de sel (≈ 50 % KCl)": 0.5 * K_MOLAR_MASS / KCL_MOLAR_MASS,
    "Banane (chair, ≈ 3.6 g K/kg)": 0.0036,
    "Engrais potassique (60 % K₂O)": 0.60 * 2 * K_MOLAR_MASS / (2 * K_MOLAR_MASS + 15.999),
    "Potassium métallique": 1.0,
}


@dataclass
class SampleActivity:
    mass_g: float
    k_fraction: float
    k_mass_g: float
    activity_bq: float
    beta_per_s: float
    gamma_per_s: float

    def as_text(self) -> str:
        return (f"{self.mass_g:g} g d'échantillon → {self.k_mass_g:.3g} g de K → "
                f"activité ≈ {self.activity_bq:.3g} Bq "
                f"({self.beta_per_s:.3g} β⁻/s, {self.gamma_per_s:.3g} γ/s)")


def sample_activity(mass_g: float, k_fraction: float) -> SampleActivity:
    k_mass = mass_g * k_fraction
    a = k_mass * specific_activity_k()
    return SampleActivity(mass_g, k_fraction, k_mass, a, a * K40_BETA_BRANCH, a * K40_GAMMA_YIELD)


# --------------------------------------------------------------------------- #
# Alphas : relation portée-énergie
# --------------------------------------------------------------------------- #
# Émetteurs alpha naturels susceptibles d'être vus dans une chambre de Wilson
# (chaînes du radon-222 et du thoron-220 présents dans l'air ambiant).
ALPHA_EMITTERS = [
    ("Po-210", 5.304),
    ("Rn-222", 5.490),
    ("Po-218", 6.002),
    ("Bi-212", 6.051),
    ("Rn-220", 6.288),
    ("Po-216", 6.778),
    ("Po-214", 7.687),
    ("Po-212", 8.785),
]


def air_density_factor(temp_c: float, pressure_kpa: float = 101.325) -> float:
    """Rapport densité(T,P) / densité(15 °C, 1 atm)."""
    return (288.15 / (temp_c + 273.15)) * (pressure_kpa / 101.325)


def alpha_range_cm(energy_mev: float, temp_c: float = 15.0, pressure_kpa: float = 101.325) -> float:
    """Portée d'une alpha dans l'air (cm) — règle de Geiger, valable ≈ 4–8 MeV."""
    r_std = 0.318 * energy_mev ** 1.5
    return r_std / air_density_factor(temp_c, pressure_kpa)


def alpha_energy_from_range(range_mm: float, temp_c: float = 15.0, pressure_kpa: float = 101.325) -> float:
    """Énergie estimée (MeV) d'une alpha à partir de sa portée mesurée (mm)."""
    if range_mm <= 0:
        return float("nan")
    r_std_cm = range_mm / 10.0 * air_density_factor(temp_c, pressure_kpa)
    return (r_std_cm / 0.318) ** (2.0 / 3.0)


def closest_alpha_emitter(energy_mev: float) -> tuple[str, float]:
    return min(ALPHA_EMITTERS, key=lambda e: abs(e[1] - energy_mev))


# Portée CSDA des alphas dans l'air sec (g/cm²), NIST ASTAR (Berger et al., SRD 124) :
# énergie (MeV) → portée. Plus précise que la règle de Geiger au-delà de 7 MeV
# (Geiger sous-estime de 2,7 % à 7,69 MeV et de 4 % à 8,8 MeV).
ASTAR_AIR_CSDA_G_CM2 = (
    (4.0, 3.147e-3), (4.5, 3.734e-3), (5.0, 4.368e-3), (5.5, 5.048e-3), (6.0, 5.772e-3),
    (6.5, 6.540e-3), (7.0, 7.351e-3), (7.5, 8.205e-3), (8.0, 9.100e-3), (8.5, 1.004e-2),
    (9.0, 1.102e-2),
)
AIR_DENSITY_15C_G_CM3 = 1.2250e-3     # air sec à 15 °C et 101,325 kPa


def alpha_range_astar_cm(energy_mev: float, temp_c: float = 15.0, pressure_kpa: float = 101.325) -> float:
    """Portée CSDA (cm) d'une alpha dans l'air à (T, P), par interpolation log-log de NIST ASTAR.

    Table valable de 4 à 9 MeV (extrapolation log-log en dehors) ; portée = R(g/cm²) / ρ(T, P),
    avec ρ = 1,2250e-3 · (288,15 / T) · (P / 101,325) g/cm³ (gaz parfait). Renvoie NaN si
    l'énergie n'est pas un nombre positif. Ex. : Rn-222 (5,49 MeV) à −25 °C ≈ 3,54 cm ;
    Po-212 (8,785 MeV) à −25 °C ≈ 7,45 cm.
    """
    try:
        e = float(energy_mev)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(e) or e <= 0:
        return float("nan")
    tab = ASTAR_AIR_CSDA_G_CM2
    if e <= tab[0][0]:
        (e0, r0), (e1, r1) = tab[0], tab[1]
    elif e >= tab[-1][0]:
        (e0, r0), (e1, r1) = tab[-2], tab[-1]
    else:
        i = next(k for k in range(1, len(tab)) if tab[k][0] >= e)
        (e0, r0), (e1, r1) = tab[i - 1], tab[i]
    slope = math.log(r1 / r0) / math.log(e1 / e0)
    r_g_cm2 = r0 * (e / e0) ** slope
    rho = AIR_DENSITY_15C_G_CM3 * air_density_factor(temp_c, pressure_kpa)
    return r_g_cm2 / rho


# --------------------------------------------------------------------------- #
# Muons cosmiques
# --------------------------------------------------------------------------- #
MUON_FLUX_PER_CM2_MIN = 1.0      # flux à travers une surface horizontale au niveau de la mer
MUON_VERTICAL_INTENSITY = 70.0   # m⁻² s⁻¹ sr⁻¹ (PDG)


# --------------------------------------------------------------------------- #
# Champ magnétique : quantité de mouvement à partir du rayon de courbure
# --------------------------------------------------------------------------- #
def momentum_mev(b_field_mt: float, radius_mm: float, charge: int = 1) -> float:
    """p (MeV/c) = 299.79 · B(T) · R(m) · |q|   (équivalent : p (GeV/c) = 0.2998 · B(T) · R(m) · |q|).

    Ici B est donné en mT et R en mm : p (MeV/c) = 2.998e-4 · B(mT) · R(mm) · |q|
    (ex. B = 100 mT, R = 60 mm → p ≈ 1.80 MeV/c).
    """
    return C_MEV_PER_T_M * (b_field_mt / 1000.0) * (radius_mm / 1000.0) * abs(charge)


def kinetic_energy_mev(p_mev: float, mass_mev: float) -> float:
    return math.sqrt(p_mev ** 2 + mass_mev ** 2) - mass_mev


# --------------------------------------------------------------------------- #
# Descriptions pédagogiques des classes de traces
# --------------------------------------------------------------------------- #
CLASS_INFO = {
    "alpha": {
        "nom": "Alpha (α)",
        "symbole": "α",
        "couleur": (60, 60, 255),      # BGR (rouge)
        "hex": "#e5484d",
        "description": "Noyau d'hélium (2p + 2n). Ionisation très forte : trace épaisse, brillante et "
                       "rectiligne ; portée ≈ 3,5 à 7,5 cm dans l'air froid selon l'émetteur, souvent plus "
                       "courte (trace inclinée ou sortant de la couche) ; plus épaisse vers la fin (pic de "
                       "Bragg). Un V partant d'un même point = thoron Rn-220 → Po-216 (0,14 s). Source "
                       "typique : radon de l'air et ses descendants, Po-210.",
    },
    "beta": {
        "nom": "Bêta / électron (β⁻)",
        "symbole": "β",
        "couleur": (255, 190, 40),     # BGR (bleu clair)
        "hex": "#2f9bff",
        "description": "Électron léger : trace fine, souvent sinueuse ou en zigzag (diffusion multiple). "
                       "Électrons lents (< 0,1 MeV) : traces courtes très tortueuses ; électrons d'environ "
                       "1 MeV : traces longues presque droites qui sortent de la couche. Source typique : "
                       "désintégration β⁻ du K-40 (KCl, banane ; Emax 1,311 MeV), électrons Compton des gammas.",
    },
    "muon": {
        "nom": "Muon cosmique (μ)",
        "symbole": "μ",
        "couleur": (80, 220, 80),      # BGR (vert)
        "hex": "#30a46c",
        "description": "Particule cosmique très énergétique (≈ 4 GeV) : trace fine, parfois pointillée "
                       "(ionisation minimale), parfaitement droite ; longue seulement si le muon est très "
                       "incliné (la plupart arrivent presque verticalement et laissent des traces courtes). "
                       "Flux ≈ 1 /cm²/min au niveau de la mer.",
    },
    "proton": {
        "nom": "Proton / particule lourde",
        "symbole": "p",
        "couleur": (40, 160, 255),     # BGR (orange)
        "hex": "#f76b15",
        "description": "Épaisseur intermédiaire (plus fine qu'une alpha, bien plus épaisse qu'un muon), trace "
                       "droite pouvant dépasser la portée d'une alpha naturelle : protons lents de quelques MeV "
                       "à quelques dizaines de MeV (un proton du GeV ressemble à un muon). Rare (protons de "
                       "recul, secondaires cosmiques ≈ 1 % des muons).",
    },
    "fine": {
        "nom": "Trace fine (β ou μ)",
        "symbole": "f",
        "couleur": (214, 184, 94),     # BGR de #5eb8d6 (bleu-vert)
        "hex": "#5eb8d6",
        "description": "Trace fine et droite d'une particule au minimum d'ionisation : électron rapide ou muon "
                       "cosmique. La famille est sûre, mais sur quelques centimètres ces deux particules laissent "
                       "la même trace : la physique ne permet pas de les départager sans aimant. Compte dans le "
                       "groupe « traces fines » (comparaison KCl / fond).",
    },
    "incertain": {
        "nom": "Incertaine",
        "symbole": "?",
        "couleur": (200, 200, 200),
        "hex": "#8b8d98",
        "description": "Trace détectée dont la forme ne correspond clairement à aucune catégorie.",
    },
    "artefact": {
        "nom": "Artefact (brume, gouttes…)",
        "symbole": "×",
        "couleur": (90, 120, 160),     # BGR de #a0785a (brun)
        "hex": "#a0785a",
        "description": "Détection qui n'est probablement pas une particule : bouffée de brume, amas de "
                       "gouttelettes qui tombent, reflet ou poussière (tache peu allongée, peu contrastée ou "
                       "immobile). Exclue des comptages de particules.",
    },
}

COUNTED_CLASSES = ["alpha", "beta", "muon", "proton", "fine", "incertain"]
