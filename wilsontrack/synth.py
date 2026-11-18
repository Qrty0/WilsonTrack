"""Générateur de vidéos synthétiques de chambre de Wilson (avec vérité terrain) — modèle v2.

Sert à :
1. valider objectivement l'algorithme (rappel, précision, matrice de confusion) ;
2. faire des démonstrations sans chambre fonctionnelle ;
3. entraîner / tester l'IA de référence (classificateur entraîné sur des traces simulées).

Règle suivie : toute différence d'apparence entre les classes vient d'un fait physique cité
(voir ``JUSTIFICATIONS`` en fin de module). Le modèle :

- transport de chaque particule chargée en 3 D, pas à pas, dans l'air froid de la couche
  sensible (masse volumique ρ(T, P) des gaz parfaits) :
    * perte d'énergie : tables NIST ESTAR (électrons, pouvoir d'arrêt par collision),
      ASTAR (alphas, portée CSDA) et PSTAR (protons), interpolées en log-log ;
    * diffusion multiple : formule de Highland (PDG, éq. 34.16) avec le terme logarithmique
      évalué une fois pour une épaisseur de référence (≈ 1 cm) puis appliqué à chaque pas ;
    * rayons delta (électrons arrachés, PDG §34.2.7) pour les particules rapides ;
- sources : radon et descendants (Rn-222, Po-218, Po-214), Po-210 des surfaces, paires en V
  du thoron (Rn-220 → Po-216, 0,14 s), Po-212 / Bi-212 rares ; électrons Compton (γ de
  1,461 MeV du K-40) sortant de la plaque et des parois ; β⁻ du K-40 d'un échantillon ;
  muons cosmiques (cos²θ, longueur projetée L = d·tanθ), électrons cosmiques énergétiques,
  rares gerbes, désintégrations de muons (coude) ; protons lents (recul / secondaires) ;
- RENDU COMMUN à toutes les particules : le nombre de gouttelettes par mm est proportionnel
  à l'ionisation locale (dE/dx), la diffusion latérale est la même pour toutes, et la réponse
  optique sature (1 − e^(−densité/densité_sat)). L'épaisseur apparente plus grande des traces
  très ionisantes (alphas, fin de parcours) en découle, sans paramètre propre à une classe ;
- durées de visibilité : alphas 1,5 ± 0,4 s ; TOUTES les traces fines (électrons et muons)
  suivent la même loi log-normale (Sussman : médiane ≈ 0,27–0,30 s) ;
- artefacts avec vérité terrain (classe « artefact ») : bouffées de brume, amas de
  gouttelettes, scintillements, reflets lents près des parois, éblouissement au bord ;
- image : brouillard mouvant, « pluie » de gouttelettes, éclairage latéral (gradient
  réglable), bruit du capteur (+ bruit de grenaille et compression JPEG optionnels).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import physics

SYNTH_VERSION = 2

# --------------------------------------------------------------------------- #
# Données physiques (sources : voir JUSTIFICATIONS)
# --------------------------------------------------------------------------- #
RHO_AIR_15C_G_CM3 = 1.2250e-3        # air sec, 15 °C, 1 atm (PDG : 1,205e-3 à 20 °C)
X0_AIR_G_CM2 = 36.62                 # longueur de radiation de l'air (PDG)
S_MIP = 1.815                        # MeV·cm²/g : minimum d'ionisation dans l'air (PDG)
S_MIN_ELECTRON = 1.655               # MeV·cm²/g : minimum ESTAR (collision) d'un électron
HALF_K_ZA = 0.5 * 0.307075 * 0.49919  # ½·K·Z/A de l'air (MeV·cm²/g), production de rayons delta
ME = physics.ELECTRON_MASS_MEV
M_ALPHA = physics.ALPHA_MASS_MEV
M_PROTON = physics.PROTON_MASS_MEV
M_MUON = physics.MUON_MASS_MEV

# NIST ESTAR : pouvoir d'arrêt par collision des électrons dans l'air sec (MeV, MeV·cm²/g)
ESTAR_COLLISION = np.array([
    (0.01, 19.75), (0.02, 11.57), (0.03, 8.491), (0.05, 5.818), (0.07, 4.593), (0.1, 3.633),
    (0.15, 2.861), (0.2, 2.469), (0.3, 2.084), (0.5, 1.802), (0.7, 1.706), (1.0, 1.661),
    (1.25, 1.655), (1.5, 1.661)])
# NIST ESTAR : portée CSDA des électrons (g/cm²) — sert aux tests de cohérence
ESTAR_CSDA = np.array([(0.05, 4.91e-3), (0.1, 1.62e-2), (0.2, 5.08e-2), (0.5, 0.1995), (1.0, 0.491)])
# NIST ASTAR : portée CSDA des alphas dans l'air sec (MeV, g/cm²)
ASTAR_CSDA = np.array([
    (4.0, 3.147e-3), (4.5, 3.734e-3), (5.0, 4.368e-3), (5.5, 5.048e-3), (6.0, 5.772e-3),
    (6.5, 6.540e-3), (7.0, 7.351e-3), (7.5, 8.205e-3), (8.0, 9.100e-3), (8.5, 1.004e-2),
    (9.0, 1.102e-2)])
ASTAR_S_MAX = 2021.0                 # MeV·cm²/g, maximum du pic de Bragg (≈ 0,7 MeV)
ALPHA_DETOUR = 0.9925                # portée projetée / CSDA (ASTAR, 5,49 MeV)
# NIST PSTAR : pouvoir d'arrêt des protons (MeV, MeV·cm²/g). 3, 10, 50 et 1 000 MeV : valeurs
# PSTAR ; 1 et 2 MeV : dérivées des portées PSTAR (1 → 2,87e-3 ; 2 → 8,79e-3 ; 3 → 1,74e-2 g/cm²),
# ce qui redonne exactement S(3 MeV) = 102.
PSTAR_STOPPING = np.array([(1.0, 216.0), (2.0, 138.0), (3.0, 102.0), (10.0, 40.0), (50.0, 11.0),
                           (1000.0, 1.96)])
PSTAR_CSDA = np.array([(1.0, 2.87e-3), (2.0, 8.79e-3), (3.0, 1.74e-2), (10.0, 0.141)])
PSTAR_S_CAP = 760.0                  # plafond approximatif du pic de Bragg du proton (≈ 0,1 MeV)
# PDG : intensité verticale intégrale des électrons cosmiques au sol (m⁻² s⁻¹ sr⁻¹)
COSMIC_E_INTEGRAL = np.array([(10.0, 30.0), (100.0, 6.0), (1000.0, 0.2)])
K40_GAMMA_MEV = 1.46082              # γ du K-40 (NNDC)
MICHEL_EMAX_MEV = 52.83              # énergie maximale de l'électron de désintégration du muon

# Émetteurs alpha (NNDC / NuDat 3) et origine : air (« air ») ou surfaces (« surface »)
ALPHA_LINES = {"Rn-222": 5.490, "Po-218": 6.003, "Po-214": 7.687, "Po-210": 5.304,
               "Rn-220": 6.288, "Po-216": 6.778, "Po-212": 8.785, "Bi-212": 6.051}
ALPHA_ORIGIN = {"Rn-222": 1.0, "Po-218": 1.0, "Po-214": 0.5, "Po-210": 0.0, "Rn-220": 1.0,
                "Po-216": 1.0, "Po-212": 0.5, "Bi-212": 0.5}     # fraction émise dans l'air
PO216_MEAN_LIFE_S = 0.145 / math.log(2.0)                          # T½ = 0,145 s (NNDC)

ARTEFACT_KINDS = ("brume", "amas", "scintillement", "reflet", "reflet_bord")
PARTICLE_CLASSES = ("alpha", "beta", "muon", "proton")


def air_density_g_cm3(temp_c: float, pressure_kpa: float = 101.325) -> float:
    """Masse volumique de l'air sec (g/cm³), gaz parfait : 1,2250e-3·(288,15/T)·(P/101,325)."""
    return RHO_AIR_15C_G_CM3 * (288.15 / (temp_c + 273.15)) * (pressure_kpa / 101.325)


def _loglog(x, table: np.ndarray):
    """Interpolation log-log (extrapolation linéaire en log-log aux deux bouts)."""
    lx, ly = np.log(table[:, 0]), np.log(table[:, 1])
    v = np.log(np.asarray(x, dtype=np.float64))
    out = np.interp(v, lx, ly)
    lo = v < lx[0]
    hi = v > lx[-1]
    if np.any(lo):
        s = (ly[1] - ly[0]) / (lx[1] - lx[0])
        out = np.where(lo, ly[0] + s * (v - lx[0]), out)
    if np.any(hi):
        s = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
        out = np.where(hi, ly[-1] + s * (v - lx[-1]), out)
    return np.exp(out)


# --- Tables fines (grille logarithmique) pour un transport rapide en Python pur ---------------
_LN_E0, _LN_E1, _NG = math.log(1e-3), math.log(2e4), 700
_LN_GRID = np.linspace(_LN_E0, _LN_E1, _NG)
_E_GRID = np.exp(_LN_GRID)
_DLN = (_LN_E1 - _LN_E0) / (_NG - 1)


def _alpha_stopping_table() -> np.ndarray:
    """S(E) des alphas = dE/dR à partir des portées ASTAR (plafonné au maximum de Bragg)."""
    e = _E_GRID
    r = _loglog(e, ASTAR_CSDA)
    s = np.gradient(e, r)
    return np.clip(s, 1.0, ASTAR_S_MAX)


def _cumrange(e: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Portée CSDA R(E) = ∫ dE / S (g/cm²) sur la grille (trapèzes, R(E_min) ≈ E_min / S)."""
    inv = 1.0 / s
    r = np.concatenate([[e[0] * inv[0]], 0.5 * (inv[1:] + inv[:-1]) * np.diff(e)])
    return np.cumsum(r)


_S_E = _loglog(_E_GRID, ESTAR_COLLISION)
_S_E[_E_GRID > 1.25] = np.maximum(_S_E[_E_GRID > 1.25], S_MIN_ELECTRON)
_S_A = _alpha_stopping_table()
_S_P = np.minimum(_loglog(_E_GRID, PSTAR_STOPPING), PSTAR_S_CAP)
_TABLES = {
    "e": (np.log(_S_E), np.log(_cumrange(_E_GRID, _S_E))),
    "a": (np.log(_S_A), np.log(_cumrange(_E_GRID, _S_A))),
    "p": (np.log(_S_P), np.log(_cumrange(_E_GRID, _S_P))),
}
_LN_S = {k: v[0].tolist() for k, v in _TABLES.items()}
_LN_R = {k: v[1].tolist() for k, v in _TABLES.items()}


def _grid_interp(arr: list, e: float) -> float:
    """exp(interpolation linéaire en ln E) dans une table de la grille fine."""
    t = (math.log(max(e, 1.0001e-3)) - _LN_E0) / _DLN
    i = int(t)
    if i >= _NG - 1:
        return math.exp(arr[-1])
    f = t - i
    return math.exp(arr[i] * (1.0 - f) + arr[i + 1] * f)


def stopping_power(kind: str, e_mev: float) -> float:
    """Pouvoir d'arrêt (MeV·cm²/g) : « e » électron (collision, ESTAR), « a » alpha (ASTAR),
    « p » proton (PSTAR)."""
    return _grid_interp(_LN_S[kind], e_mev)


def csda_range_g_cm2(kind: str, e_mev: float) -> float:
    """Portée CSDA (g/cm²) obtenue en intégrant 1/S (e, a, p)."""
    return _grid_interp(_LN_R[kind], e_mev)


def energy_from_range(kind: str, r_g_cm2: float) -> float:
    """Énergie (MeV) dont la portée CSDA vaut r (inverse de csda_range_g_cm2)."""
    lr = _TABLES[kind][1]
    return float(math.exp(np.interp(math.log(max(r_g_cm2, 1e-12)), lr, _LN_GRID)))


def alpha_range_mm(e_mev: float, temp_c: float, pressure_kpa: float = 101.325) -> float:
    """Portée (mm) d'une alpha dans l'air à la température donnée (ASTAR / ρ(T, P))."""
    return float(_loglog(e_mev, ASTAR_CSDA)) / air_density_g_cm3(temp_c, pressure_kpa) * 10.0


def proton_range_mm(e_mev: float, temp_c: float, pressure_kpa: float = 101.325) -> float:
    return csda_range_g_cm2("p", e_mev) / air_density_g_cm3(temp_c, pressure_kpa) * 10.0


def ionisation_ratio(kind: str, e_mev: float) -> float:
    """Ionisation locale rapportée à celle d'une particule au minimum d'ionisation (≥ 1).

    Électrons : S_col/1,655 (minimum ESTAR), plafonné à 1 au-dessus de 1,25 MeV (un électron
    rapide ionise comme un muon) ; alphas et protons : S/1,815 (minimum PDG)."""
    if kind == "e":
        return 1.0 if e_mev >= 1.25 else max(1.0, stopping_power("e", e_mev) / S_MIN_ELECTRON)
    if kind == "mu":
        return 1.0
    return max(1.0, stopping_power(kind, e_mev) / S_MIP)


def muon_projected_length(rng: np.random.Generator, d_mm: float, l_min: float = 0.0) -> float:
    """Longueur projetée L = d·tanθ d'une particule traversant une couche horizontale d'épaisseur
    d, flux ∝ cos²θ à travers une surface horizontale (cos θ = U^(1/4)) :
    P(L > x) = (1 + (x/d)²)⁻². Tirage exact, conditionné à L ≥ l_min."""
    p_min = (1.0 + (l_min / d_mm) ** 2) ** -2 if l_min > 0 else 1.0
    v = max(rng.uniform() * p_min, 1e-12)
    return d_mm * math.sqrt(max(0.0, v ** -0.5 - 1.0))


def _klein_nishina_electron(rng: np.random.Generator, e_gamma: float) -> float:
    """Énergie cinétique d'un électron Compton (section efficace de Klein-Nishina)."""
    k = e_gamma / ME
    t_max = e_gamma * 2 * k / (1 + 2 * k)

    def f(t):
        s = t / e_gamma
        return 2.0 + s * s / (k * k * (1 - s) ** 2) + s / (1 - s) * (s - 2.0 / k)
    f_max = max(f(0.0), f(t_max * 0.999))
    for _ in range(200):
        t = rng.uniform(0.0, t_max)
        if rng.uniform(0.0, f_max) <= f(t):
            return float(t)
    return float(t_max * rng.uniform())


def _escaping_energy(rng: np.random.Generator, e0: float) -> float:
    """Énergie d'un électron créé dans un matériau épais (paroi, plaque, échantillon) qui en
    sort : la profondeur de création étant uniforme, la portée résiduelle est uniforme entre
    0 et R(E0) → spectre dominé par les basses énergies."""
    r = csda_range_g_cm2("e", e0) * rng.uniform()
    return energy_from_range("e", r)


def _cosmic_electron_energy(rng: np.random.Generator) -> float:
    """Énergie (MeV) d'un électron cosmique > 10 MeV (spectre intégral du PDG, log-log)."""
    i10 = COSMIC_E_INTEGRAL[0, 1]
    target = i10 * max(rng.uniform(), COSMIC_E_INTEGRAL[-1, 1] / i10)
    le, li = np.log(COSMIC_E_INTEGRAL[:, 0]), np.log(COSMIC_E_INTEGRAL[:, 1])
    return float(math.exp(np.interp(math.log(target), li[::-1], le[::-1])))


def _michel_energy(rng: np.random.Generator) -> float:
    """Énergie de l'électron de désintégration du muon (spectre de Michel x²(3 − 2x))."""
    for _ in range(100):
        x = rng.uniform()
        if rng.uniform() <= x * x * (3 - 2 * x):
            return float(x * MICHEL_EMAX_MEV)
    return 0.7 * MICHEL_EMAX_MEV


# --------------------------------------------------------------------------- #
# Spectre β⁻ du K-40
# --------------------------------------------------------------------------- #
def _k40_shape(e: float) -> float:
    """Densité (non normalisée) du spectre β⁻ du K-40 : transition 4⁻ → 0⁺ « troisième interdite
    unique ». Facteur de forme C ∝ p⁶ + 7p⁴q² + 7p²q⁴ + q⁶ (p : électron, q : antineutrino),
    fonction de Fermi non relativiste pour Z = 20 (Ca)."""
    q_end = physics.K40_BETA_EMAX_MEV
    if not (0.0 < e < q_end):
        return 0.0
    w = e + ME
    p = math.sqrt(w * w - ME * ME)
    q = q_end - e
    beta = p / w
    eta = 20 / 137.036 / beta
    fermi = 2 * math.pi * eta / (1 - math.exp(-2 * math.pi * eta))
    c = p ** 6 + 7 * p ** 4 * q ** 2 + 7 * p ** 2 * q ** 4 + q ** 6
    return fermi * p * w * q * q * c


def _k40_table():
    """Table de tirage. Un facteur de forme empirique (1 + a·E) est ajusté pour retrouver
    l'énergie moyenne de l'évaluation NNDC/ENSDF 2017 (J. Chen, Nucl. Data Sheets 140) :
    0,560 MeV (le LNHB 2025 donne 0,583 MeV ; effet < 1 % sur l'expérience)."""
    grid = np.linspace(0.002, physics.K40_BETA_EMAX_MEV - 1e-4, 800)
    base = np.array([_k40_shape(float(e)) for e in grid])
    lo, hi = -0.95, 0.5
    a = 0.0
    for _ in range(60):
        a = 0.5 * (lo + hi)
        w = base * (1 + a * grid)
        mean = float((w * grid).sum() / w.sum())
        if mean > physics.K40_BETA_MEAN_MEV:
            hi = a
        else:
            lo = a
    cdf = np.cumsum(base * (1 + a * grid))
    return grid, cdf / cdf[-1]


_K40_GRID, _K40_CDF = _k40_table()


def _k40_beta_energy(rng: np.random.Generator) -> float:
    """Tire une énergie β⁻ du K-40 (moyenne ≈ 0.56 MeV, Emax 1.311 MeV)."""
    return float(np.interp(rng.uniform(), _K40_CDF, _K40_GRID))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _default_alpha_mix() -> dict:
    # Fractions des désintégrations alpha visibles (hypothèse « pièce typique », radon dominant ;
    # randomisées pendant l'entraînement). « thoron » = paire en V Rn-220 → Po-216.
    return {"Rn-222": 0.38, "Po-218": 0.24, "Po-214": 0.18, "Po-210": 0.08, "thoron": 0.08,
            "Po-212": 0.025, "Bi-212": 0.015}


def _default_artefacts() -> dict:
    return {"brume": 1.5, "amas": 2.0, "scintillement": 1.5, "reflet": 0.5, "reflet_bord": 0.3}


@dataclass
class SynthConfig:
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    duration_s: float = 60.0
    fov_mm: float = 250.0                  # largeur du champ filmé
    # Taux (par minute) d'événements VISIBLES : alpha = désintégrations alpha (une paire du thoron
    # compte pour un événement), beta = électrons de basse énergie (Compton) venus de la plaque
    # et des parois, muon = muons cosmiques (longueur projetée ≥ crossing_min_mm), proton = protons lents.
    rates_per_min: dict = field(default_factory=lambda: {"alpha": 5.0, "beta": 8.0, "muon": 12.0,
                                                         "proton": 0.2})
    source_beta_per_min: float = 0.0       # traces β supplémentaires issues d'une source (ex. KCl)
    source_xy_mm: tuple = (125.0, 70.0)
    source_radius_mm: float = 15.0
    chamber_temp_c: float = -25.0
    sensitive_depth_mm: float = 8.0        # épaisseur de la couche sensible (traces lourdes)
    b_field_mt: float = 0.0
    noise_sigma: float = 2.5
    fog_amp: float = 7.0
    rain_per_frame: int = 250
    base_level: float = 16.0
    brightness: float = 1.0                # facteur global sur l'intensité des traces
    seed: int = 1
    disturbance_at_s: float = -1.0         # simule une perturbation (lumière allumée) à cet instant
    # Facteurs de « randomisation de domaine » (communs à TOUTES les particules)
    width_scale: float = 1.0               # multiplie la diffusion latérale (épaisseur des traces)
    droplet_scale: float = 1.0             # multiplie le nombre de gouttelettes par unité d'ionisation
    life_scale: float = 1.0                # multiplie la durée de visibilité des traces
    # ---- Réalisme (v2) ----------------------------------------------------------------
    pressure_kpa: float = 101.325
    deff_mm: float = 10.0                  # épaisseur effective où les traces FINES sont visibles
    crossing_min_mm: float = 3.0           # longueur projetée minimale des muons simulés (taux = au-delà)
    cosmic_e_frac: float = 0.10            # électrons cosmiques énergétiques / muons (hypothèse intérieure)
    shower_per_min: float = 0.05           # gerbes cosmiques (2–5 traces parallèles simultanées)
    muon_decay_frac: float = 0.002         # muons qui s'arrêtent et se désintègrent (coude) — exagéré
    delta_tmin_kev: float = 10.0           # énergie minimale d'un rayon delta rendu (≈ 2 mm de portée)
    alpha_mix: dict = field(default_factory=_default_alpha_mix)
    alpha_kink_frac: float = 0.02          # alphas déviées en fin de parcours (diffusion nucléaire)
    proton_energy_mev: tuple = (2.0, 40.0)  # protons lents : énergie log-uniforme
    source_self_absorption: float = 0.5    # fraction des β de la source issus de la profondeur (spectre dégradé)
    wall_electron_frac: float = 0.5        # électrons Compton venus des parois (les autres : de la plaque)
    supersat_sigma: float = 0.30           # inhomogénéité de la sursaturation le long des traces (σ de ln)
    ms_ref_mm: float = 10.0                # épaisseur de référence du terme logarithmique de Highland
    # Durées de visibilité (vérité terrain) calibrées pour que les durées MESURÉES par le détecteur
    # reproduisent Sussman (traces fines : médiane 0,27 s, moyenne 0,38 s ; alphas 1,49 ± 0,43 s).
    thin_life_median_s: float = 0.24       # traces fines (électrons ET muons), médiane
    thin_life_sigma: float = 0.75          # … écart-type de ln(durée)
    alpha_life_mean_s: float = 1.5
    alpha_life_sd_s: float = 0.45
    droplets_per_mm: tuple = (2.5, 5.0)    # gouttelettes par mm pour une particule au minimum d'ionisation
    sat_density_mm2: float = 24.0          # densité de gouttelettes (mm⁻²) de saturation optique
    diffusion_mm: float = 0.40             # étalement latéral des gouttelettes après 1 s (∝ √t)
    artefact_rates_per_min: dict = field(default_factory=_default_artefacts)
    light_gradient: float = 0.0            # gradient d'éclairage (fraction de base_level d'un bord à l'autre)
    light_angle_deg: float = 0.0
    shot_noise: float = 0.0                # bruit de grenaille : variance ajoutée = shot_noise × niveau
    jpeg_quality: int = 0                  # > 0 : compression JPEG de chaque image (qualité 1–100)

    @property
    def px_per_mm(self) -> float:
        return self.width / self.fov_mm


@dataclass
class _Track:
    gid: int
    cls: str
    frame_birth: int
    life_frames: int
    rise_frames: int
    pts_px: np.ndarray        # polyligne (x, y) px (vérité terrain)
    width_px: float
    amp: float
    droplets_per_px: float
    drift_px: tuple
    sprite: np.ndarray | None = None
    ox: int = 0
    oy: int = 0
    length_mm: float = 0.0
    energy_mev: float = 0.0
    kind: str = ""
    density: bool = True      # sprite = densité de gouttelettes (mm⁻²) ; sinon intensité directe
    flicker: float = 0.0      # probabilité d'être éteint à une image (scintillement)
    mod_hz: float = 0.0       # modulation sinusoïdale de l'intensité (éblouissement)
    mod_phase: float = 0.0
    sigma_px: float = 1.0
    q_mean: float = 1.0
    group: int = -1
    n_delta: int = 0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Transport 3 D d'une particule chargée dans la couche sensible
# --------------------------------------------------------------------------- #
_MASS = {"e": ME, "p": M_PROTON, "a": M_ALPHA, "mu": M_MUON}
_CHARGE = {"e": 1, "p": 1, "a": 2, "mu": 1}


def _perp(dx, dy, dz):
    """Deux vecteurs unitaires orthogonaux à la direction (dx, dy, dz)."""
    if abs(dx) < 0.9:
        ax, ay, az = 1.0, 0.0, 0.0
    else:
        ax, ay, az = 0.0, 1.0, 0.0
    ux, uy, uz = dy * az - dz * ay, dz * ax - dx * az, dx * ay - dy * ax
    n = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
    ux, uy, uz = ux / n, uy / n, uz / n
    vx, vy, vz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
    return (ux, uy, uz), (vx, vy, vz)


def _rotate(d, theta, rng_u):
    """Tourne la direction d d'un angle theta autour d'un axe perpendiculaire aléatoire."""
    (ux, uy, uz), (vx, vy, vz) = _perp(*d)
    phi = 2 * math.pi * rng_u
    c, s = math.cos(theta), math.sin(theta)
    px, py, pz = math.cos(phi) * ux + math.sin(phi) * vx, math.cos(phi) * uy + math.sin(phi) * vy, \
        math.cos(phi) * uz + math.sin(phi) * vz
    nx, ny, nz = c * d[0] + s * px, c * d[1] + s * py, c * d[2] + s * pz
    n = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / n, ny / n, nz / n


def transport(rng: np.random.Generator, kind: str, e0: float, pos, direction, rho: float, x0_mm: float,
              zlo: float, zhi: float, max_path_mm: float = 400.0, b_mt: float = 0.0, ms_ref_mm: float = 10.0,
              kink_frac: float = 0.0, box=None):
    """Transporte une particule (« e », « p », « a ») pas à pas en 3 D.

    Perte d'énergie continue (tables NIST, ρ en g/cm³), diffusion multiple de Highland (PDG) :
    θ0 = 13,6 MeV/(βcp)·z·√(pas/X0)·[1 + 0,038·ln(x_réf·z²/(X0·β²))], le terme logarithmique
    étant évalué pour l'épaisseur de référence x_réf (≈ 1 cm) : appliquer la formule sans ce
    terme à chaque pas surestimerait la déviation d'un facteur ≈ 1,5 (PDG : formule à appliquer
    sur l'épaisseur totale). Champ B vertical : rotation de la direction autour de z.

    Renvoie (points (N, 3) en mm, ionisation relative q (N,), énergie finale, longueur parcourue).
    Le transport s'arrête quand la particule s'arrête, sort de la couche [zlo, zhi] (les électrons
    peuvent y revenir s'ils ne s'en éloignent pas de plus de 3 mm) ou du domaine ``box``.
    """
    m, z = _MASS[kind], _CHARGE[kind]
    x, y, zz = float(pos[0]), float(pos[1]), float(pos[2])
    dx, dy, dz = direction
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dx, dy, dz = dx / n, dy / n, dz / n
    e = float(e0)
    pts = [(x, y, zz)]
    qs = [ionisation_ratio(kind, e)]
    s_tot = 0.0
    rho_mm = rho * 0.1                       # g/cm² par mm
    e_stop = 0.003 if kind == "e" else 0.05
    is_e = kind == "e"
    margin = 3.0 if is_e else 0.0
    kink_at = -1.0
    if kink_frac > 0 and rng.uniform() < kink_frac:
        kink_at = rng.uniform(0.80, 0.95)
    r0 = csda_range_g_cm2(kind, e) / rho_mm  # portée (mm)
    gauss = rng.standard_normal(4096)
    gi = 0
    lnx = math.log(ms_ref_mm * z * z / x0_mm)
    while e > e_stop and s_tot < max_path_mm:
        r_res = csda_range_g_cm2(kind, e) / rho_mm
        if is_e:
            step = min(0.6, max(0.06, r_res / 12.0))
        else:
            step = min(1.0, max(0.04, r_res / 20.0))
        w = e + m
        p = math.sqrt(max(w * w - m * m, 1e-12))
        beta = p / w
        f_log = max(0.3, 1.0 + 0.038 * (lnx - 2.0 * math.log(beta)))
        theta0 = 13.6 / (beta * p) * z * math.sqrt(step / x0_mm) * f_log
        if gi + 2 >= len(gauss):
            gauss = rng.standard_normal(4096)
            gi = 0
        tx, ty = gauss[gi] * theta0, gauss[gi + 1] * theta0
        gi += 2
        (ux, uy, uz), (vx, vy, vz) = _perp(dx, dy, dz)
        dx, dy, dz = dx + tx * ux + ty * vx, dy + tx * uy + ty * vy, dz + tx * uz + ty * vz
        if b_mt > 0:
            r_mm = p / (physics.C_MEV_PER_T_M * b_mt / 1000.0 * z) * 1000.0
            a = -step / r_mm
            ca, sa = math.cos(a), math.sin(a)
            dx, dy = ca * dx - sa * dy, sa * dx + ca * dy
        n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        dx, dy, dz = dx / n, dy / n, dz / n
        if kink_at > 0 and s_tot >= kink_at * r0:
            dx, dy, dz = _rotate((dx, dy, dz), math.radians(rng.uniform(10.0, 40.0)), rng.uniform())
            kink_at = -1.0
        x += step * dx
        y += step * dy
        zz += step * dz
        s_tot += step
        e -= stopping_power(kind, e) * rho_mm * step
        pts.append((x, y, zz))
        qs.append(ionisation_ratio(kind, max(e, e_stop)))
        if zz < zlo or zz > zhi + margin:     # sous zlo : absorbée par la plaque
            break
        if box is not None and not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
            break
    return np.array(pts, np.float64), np.array(qs, np.float64), max(e, 0.0), s_tot


def visible_segments(pts: np.ndarray, q: np.ndarray, zlo: float, zhi: float) -> list:
    """Parties du trajet situées dans la couche sensible [zlo, zhi] → [(xy (N, 2), q (N,)), …]."""
    inside = (pts[:, 2] >= zlo) & (pts[:, 2] <= zhi)
    segs = []
    i, n = 0, len(pts)
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j < n and inside[j]:
            j += 1
        if j - i >= 2:
            segs.append((pts[i:j, :2].copy(), q[i:j].copy()))
        i = j
    return segs


def _straight(start_xyz, d, length: float, q: float = 1.0, step: float = 0.5):
    """Segment rectiligne (traces de muons : déviation invisible à 4 GeV)."""
    n = max(2, int(length / step) + 1)
    s = np.linspace(0.0, length, n)
    pts = np.column_stack([start_xyz[0] + s * d[0], start_xyz[1] + s * d[1], start_xyz[2] + s * d[2]])
    return pts, np.full(n, q)


# --------------------------------------------------------------------------- #
# Chambre simulée
# --------------------------------------------------------------------------- #
class SyntheticChamber:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        W, H = cfg.width, cfg.height
        rng = self.rng
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        wv = 0.18 * W
        vign = np.exp(-xx / wv) + np.exp(-(W - xx) / wv)
        tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 3) * 6.0
        static = cfg.base_level + 14.0 * vign + tex
        if cfg.light_gradient:
            a = math.radians(cfg.light_angle_deg)
            ramp = ((xx - W / 2) * math.cos(a) + (yy - H / 2) * math.sin(a)) / max(W, H)
            static = static + cfg.light_gradient * cfg.base_level * ramp
        self.static = static.astype(np.float32)
        del yy, xx
        self.fog_keys = [self._fog_field() for _ in range(2)]
        self._fog_k = 0
        self.fog_period = int(2.5 * cfg.fps)
        self.rain = np.zeros((0, 4), np.float32)  # x, y, vie restante, amplitude
        self.tracks: list[_Track] = []
        self.gt: list[dict] = []
        self.frame = 0
        self.next_gid = 1
        self.next_group = 1
        self.rho = air_density_g_cm3(cfg.chamber_temp_c, cfg.pressure_kpa)
        self.x0_mm = X0_AIR_G_CM2 / self.rho * 10.0
        self.ppm = cfg.px_per_mm
        self.w_mm, self.h_mm = W / self.ppm, H / self.ppm
        self.amp_chamber = 150.0 * cfg.brightness
        n_frames = int(cfg.duration_s * cfg.fps)
        self.n_frames = n_frames
        self.pending: dict[int, list] = {}
        self.schedule = self._schedule(n_frames)

    # ----------------------------------------------------------- génération --
    def _fog_field(self):
        c = self.cfg
        small = self.rng.normal(0, 1, (c.height // 40 + 2, c.width // 40 + 2)).astype(np.float32)
        small = cv2.GaussianBlur(small, (0, 0), 1.0)
        return cv2.resize(small, (c.width, c.height), interpolation=cv2.INTER_CUBIC)

    def _schedule(self, n_frames):
        c = self.cfg
        events = []
        dur_min = n_frames / c.fps / 60.0

        def add(kind, rate, src=False):
            if rate <= 0:
                return
            n = self.rng.poisson(rate * dur_min)
            for fb in self.rng.integers(0, max(1, n_frames - 1), n):
                events.append((int(fb), kind, src))
        rates = dict(c.rates_per_min)
        for cls in PARTICLE_CLASSES:
            add(cls, float(rates.get(cls, 0.0)))
        add("cosmic_e", c.cosmic_e_frac * float(rates.get("muon", 0.0)))
        add("shower", c.shower_per_min)
        add("beta", c.source_beta_per_min, True)
        for kind, r in dict(c.artefact_rates_per_min or {}).items():
            if kind in ARTEFACT_KINDS:
                add("art:" + kind, float(r))
        events.sort(key=lambda t: (t[0], t[1], t[2]))
        sched: dict[int, list] = {}
        for fb, kind, src in events:
            sched.setdefault(fb, []).append((kind, src))
        return sched

    # ------------------------------------------------------------ helpers --
    def _w_mm(self) -> float:
        """Largeur (FWHM) de la diffusion latérale d'une trace : MÊME loi pour toutes les particules."""
        return float(self.rng.uniform(0.45, 0.8) * self.cfg.width_scale)

    def _n0(self) -> float:
        lo, hi = self.cfg.droplets_per_mm
        return float(self.rng.uniform(lo, hi) * self.cfg.droplet_scale)

    def _thin_life(self) -> float:
        c = self.cfg
        return float(np.clip(self.rng.lognormal(math.log(c.thin_life_median_s), c.thin_life_sigma), 0.08, 2.5))

    def _start_xy(self, margin_mm: float = 0.0):
        r = self.rng
        return (r.uniform(-margin_mm, self.w_mm + margin_mm), r.uniform(-margin_mm, self.h_mm + margin_mm))

    def _box(self, margin=40.0):
        return (-margin, -margin, self.w_mm + margin, self.h_mm + margin)

    def _cos_law_up(self):
        """Direction vers le haut, flux sortant d'une surface (loi en cosinus : cos θ = √U)."""
        r = self.rng
        cz = math.sqrt(max(r.uniform(), 1e-9))
        sz = math.sqrt(max(0.0, 1 - cz * cz))
        ph = r.uniform(0, 2 * math.pi)
        return (sz * math.cos(ph), sz * math.sin(ph), cz)

    def _wall_start(self):
        """Électron sortant d'une paroi latérale (juste hors du champ filmé), dans la couche, direction
        en loi du cosinus autour de la normale (horizontale) rentrante."""
        r = self.rng
        W, H = self.w_mm, self.h_mm
        per = 2 * (W + H)
        u = r.uniform(0, per)
        out = r.uniform(0.0, 15.0)
        if u < W:
            x, y, n = u, -out, (0.0, 1.0)
        elif u < 2 * W:
            x, y, n = u - W, H + out, (0.0, -1.0)
        elif u < 2 * W + H:
            x, y, n = -out, u - 2 * W, (1.0, 0.0)
        else:
            x, y, n = W + out, u - 2 * W - H, (-1.0, 0.0)
        z = r.uniform(0.0, self.cfg.deff_mm)
        ca = math.sqrt(max(r.uniform(), 1e-9))
        sa = math.sqrt(max(0.0, 1 - ca * ca))
        ph = r.uniform(0, 2 * math.pi)
        t1 = (-n[1], n[0])
        d = (ca * n[0] + sa * math.cos(ph) * t1[0], ca * n[1] + sa * math.cos(ph) * t1[1], sa * math.sin(ph))
        return (x, y, z), d

    def _isotropic(self, up_only: bool = False):
        r = self.rng
        cz = r.uniform(0.0, 1.0) if up_only else r.uniform(-1.0, 1.0)
        sz = math.sqrt(max(0.0, 1 - cz * cz))
        ph = r.uniform(0, 2 * math.pi)
        return (sz * math.cos(ph), sz * math.sin(ph), cz)

    def _delta_rays(self, kind: str, energy: float, path_pts: np.ndarray, d, zlo, zhi) -> list:
        """Rayons delta (PDG §34.2.7) : dN/dx = ½K·(Z/A)·z²/β²·(1/Tmin − 1/Tmax)·ρ, spectre en 1/T²,
        émis presque perpendiculairement (cos θ ≈ √(T/(T + 2mₑ)))."""
        c = self.cfg
        tmin = c.delta_tmin_kev / 1000.0
        if tmin <= 0 or len(path_pts) < 2:
            return []
        m, z = _MASS[kind], _CHARGE[kind]
        g = (energy + m) / m
        b2 = max(1e-9, 1 - 1 / (g * g))
        bg2 = b2 * g * g
        tmax = 2 * ME * bg2 / (1 + 2 * g * ME / m + (ME / m) ** 2)
        tmax = min(tmax, 2.0)
        if tmax <= tmin:
            return []
        seg = np.sqrt((np.diff(path_pts, axis=0) ** 2).sum(1))
        length_cm = float(seg.sum()) / 10.0
        lam = HALF_K_ZA * z * z / b2 * (1.0 / tmin - 1.0 / tmax) * self.rho * length_cm
        n = self.rng.poisson(lam)
        out = []
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        for _ in range(n):
            t = 1.0 / (1.0 / tmin - self.rng.uniform() * (1.0 / tmin - 1.0 / tmax))
            sp = self.rng.uniform(0.0, cs[-1])
            p0 = np.array([np.interp(sp, cs, path_pts[:, k]) for k in range(3)])
            ct = min(1.0, math.sqrt(t / (t + 2 * ME)) * math.sqrt((tmax + 2 * ME) / tmax))
            dd = _rotate(d, math.acos(ct), self.rng.uniform())
            pts, q, _e, _s = transport(self.rng, "e", t, p0, dd, self.rho, self.x0_mm, zlo, zhi,
                                       box=self._box(), ms_ref_mm=c.ms_ref_mm)
            out.extend(visible_segments(pts, q, zlo, zhi))
        return out

    # ----------------------------------------------------------- particules --
    def _alpha(self, emitter: str, start=None):
        """Alpha : portée ASTAR à la température de la couche, émission isotrope (air) ou depuis la
        plaque (surfaces), trajet tronqué par la couche sensible puis projeté (vue de dessus)."""
        c, r = self.cfg, self.rng
        e0 = ALPHA_LINES[emitter]
        depth = c.sensitive_depth_mm
        airborne = r.uniform() < ALPHA_ORIGIN.get(emitter, 1.0)
        best = None
        for _ in range(30):
            if start is not None:
                p0 = start
            else:
                x, y = self._start_xy()
                p0 = (x, y, r.uniform(0.0, depth) if airborne else 0.0)
            d = self._isotropic(up_only=not airborne and start is None)
            pts, q, _e, _s = transport(r, "a", e0, p0, d, self.rho, self.x0_mm, 0.0, depth, box=self._box(),
                                       ms_ref_mm=c.ms_ref_mm, kink_frac=c.alpha_kink_frac)
            segs = visible_segments(pts, q, 0.0, depth)
            L = sum(float(np.sqrt((np.diff(s[0], axis=0) ** 2).sum(1)).sum()) for s in segs)
            best = (segs, p0, L)
            if L >= 1.5:
                break
        segs, p0, _L = best
        return segs, e0, p0

    def _electron(self, source: bool):
        """Électron de basse énergie : Compton (γ 1,461 MeV, Klein-Nishina) créé dans la plaque, les
        parois ou l'échantillon, spectre dégradé par la sortie du matériau ; ou β⁻ du K-40 (source)."""
        c, r = self.cfg, self.rng
        best = None
        for _ in range(12):
            if source:
                e0 = _k40_beta_energy(r)
                if r.uniform() < c.source_self_absorption:
                    e0 = _escaping_energy(r, e0)
                rr = c.source_radius_mm * math.sqrt(r.uniform())
                a = r.uniform(0, 2 * math.pi)
                xy = (c.source_xy_mm[0] + rr * math.cos(a), c.source_xy_mm[1] + rr * math.sin(a))
            else:
                e0 = _escaping_energy(r, _klein_nishina_electron(r, K40_GAMMA_MEV))
                xy = self._start_xy(10.0)
            if e0 < 0.004:
                continue
            if not source and r.uniform() < c.wall_electron_frac:
                p0, d0 = self._wall_start()
            else:
                p0, d0 = (xy[0], xy[1], 0.0), self._cos_law_up()
            pts, q, _e, _s = transport(r, "e", e0, p0, d0, self.rho, self.x0_mm, 0.0,
                                       c.deff_mm, b_mt=c.b_field_mt, box=self._box(), ms_ref_mm=c.ms_ref_mm)
            segs = visible_segments(pts, q, 0.0, c.deff_mm)
            L = sum(float(np.sqrt((np.diff(s[0], axis=0) ** 2).sum(1)).sum()) for s in segs)
            best = (segs, e0)
            if L >= 1.5:
                break
        if best is None:
            return [], 0.0
        return best

    def _crossing(self, kind: str, d_mm: float, centre=None, direction=None):
        """Particule cosmique qui traverse la couche (muon : droite ; électron cosmique : diffusion
        multiple). Longueur projetée L = d·tanθ, flux ∝ cos²θ (u = U^(1/4))."""
        c, r = self.cfg, self.rng
        if direction is None:
            L = muon_projected_length(r, d_mm, c.crossing_min_mm)
            ct = 1.0 / math.sqrt(1.0 + (L / d_mm) ** 2)
            ph = r.uniform(0, 2 * math.pi)
            st = math.sqrt(max(0.0, 1 - ct * ct))
            direction = (st * math.cos(ph), st * math.sin(ph), -ct)
        cx, cy = centre if centre is not None else (r.uniform(0.05, 0.95) * self.w_mm, r.uniform(0.05, 0.95) * self.h_mm)
        dz = direction[2]
        path3d = d_mm / max(abs(dz), 1e-3)
        path3d = min(path3d, 1.5 * math.hypot(self.w_mm, self.h_mm) / max(math.hypot(direction[0], direction[1]), 1e-3))
        s0 = path3d / 2.0
        start = (cx - s0 * direction[0], cy - s0 * direction[1], d_mm / 2.0 - s0 * dz)
        if kind == "mu":
            pts, q = _straight(start, direction, path3d)
            energy = 4000.0
        else:
            energy = _cosmic_electron_energy(r)
            pts, q, _e, _s = transport(r, "e", energy, start, direction, self.rho, self.x0_mm, -1.0,
                                       d_mm + 1.0, max_path_mm=path3d, ms_ref_mm=c.ms_ref_mm)
        segs = visible_segments(pts, q, 0.0, d_mm) if kind != "mu" else [(pts[:, :2], q)]
        deltas = self._delta_rays("mu" if kind == "mu" else "e", energy, pts, direction, 0.0, d_mm)
        return segs, deltas, energy, direction

    def _muon_decay(self):
        """Muon qui s'arrête dans la couche (fin très ionisante : PSTAR mis à l'échelle de la masse)
        puis se désintègre ; l'électron de Michel part du point d'arrêt → trace coudée."""
        c, r = self.cfg, self.rng
        d = c.deff_mm
        z_stop = r.uniform(0.15, 0.85) * d
        L = muon_projected_length(r, d - z_stop, max(1.0, c.crossing_min_mm))
        ct = 1.0 / math.sqrt(1.0 + (L / (d - z_stop)) ** 2)
        ph = r.uniform(0, 2 * math.pi)
        st = math.sqrt(1 - ct * ct)
        direc = (st * math.cos(ph), st * math.sin(ph), -ct)
        path = (d - z_stop) / ct
        end = (r.uniform(0.15, 0.85) * self.w_mm, r.uniform(0.15, 0.85) * self.h_mm, z_stop)
        start = (end[0] - path * direc[0], end[1] - path * direc[1], d)
        pts, _q = _straight(start, direc, path, step=0.25)
        res_mm = path - np.linspace(0, path, len(pts))
        q = np.array([max(1.0, stopping_power("p", energy_from_range("p", max(rm, 1e-3) * 0.1 * self.rho * M_PROTON / M_MUON)) / S_MIP)
                      for rm in res_mm])
        segs = [(pts[:, :2], q)]
        e_m = _michel_energy(r)
        pe, qe, _e, _s = transport(r, "e", e_m, end, self._isotropic(), self.rho, self.x0_mm, 0.0, d,
                                   box=self._box(), ms_ref_mm=c.ms_ref_mm)
        segs += visible_segments(pe, qe, 0.0, d)
        return segs, e_m

    def _proton(self):
        """Proton lent (recul ou secondaire cosmique) : PSTAR, émis dans la couche, isotrope."""
        c, r = self.cfg, self.rng
        lo, hi = c.proton_energy_mev
        depth = c.sensitive_depth_mm
        best = None
        for _ in range(20):
            e0 = float(math.exp(r.uniform(math.log(lo), math.log(hi))))
            x, y = self._start_xy()
            p0 = (x, y, r.uniform(0.0, depth))
            d = self._isotropic()
            pts, q, _e, _s = transport(r, "p", e0, p0, d, self.rho, self.x0_mm, 0.0, depth, box=self._box(),
                                       ms_ref_mm=c.ms_ref_mm)
            segs = visible_segments(pts, q, 0.0, depth)
            L = sum(float(np.sqrt((np.diff(s[0], axis=0) ** 2).sum(1)).sum()) for s in segs)
            best = (segs, e0, pts, d)
            if L >= 3.0:
                break
        segs, e0, pts, d = best
        deltas = self._delta_rays("p", e0, pts, d, 0.0, depth)
        return segs, deltas, e0

    # ------------------------------------------------------------ traces --
    def _new_track(self, cls: str, kind: str, segs: list, energy: float, life_s: float,
                   source: bool = False, group: int = -1, n_delta: int = 0, main_segs: list | None = None):
        """Crée la trace (sprite de densité de gouttelettes) et son entrée de vérité terrain."""
        c, r = self.cfg, self.rng
        ppm = self.ppm
        segs = [s for s in segs if len(s[0]) >= 2]
        if not segs:
            return None
        main = main_segs if main_segs else segs
        pts_px = np.vstack([s[0] for s in main]) * ppm
        inside = (pts_px[:, 0] >= 0) & (pts_px[:, 0] < c.width) & (pts_px[:, 1] >= 0) & (pts_px[:, 1] < c.height)
        if not inside.any():
            return None
        L_vis = 0.0
        for s in main:
            p = s[0] * ppm
            ok = (p[:, 0] >= 0) & (p[:, 0] < c.width) & (p[:, 1] >= 0) & (p[:, 1] < c.height)
            seg_len = np.sqrt((np.diff(p, axis=0) ** 2).sum(1))
            L_vis += float((seg_len * (ok[1:] & ok[:-1])).sum()) / ppm
        w_mm = self._w_mm()
        n0 = self._n0()
        sprite, ox, oy, sig_px = self._render_density(segs, w_mm, n0)
        if sprite is None:
            return None
        allq = np.concatenate([s[1] for s in main])
        life = int(max(1, round(life_s * c.life_scale * c.fps)))
        rise = int(max(1, r.uniform(0.05, 0.15) * c.fps))
        drift = (r.normal(0, 1.5) * ppm / c.fps, r.normal(0, 1.5) * ppm / c.fps)
        t = _Track(self.next_gid, cls, self.frame, life, rise, pts_px, w_mm * ppm,
                   self.amp_chamber * r.uniform(0.8, 1.2), n0 / ppm, drift)
        t.sprite, t.ox, t.oy, t.sigma_px = sprite, ox, oy, sig_px
        t.length_mm, t.energy_mev, t.kind = L_vis, energy, kind
        t.q_mean = float(np.mean(allq))
        t.group, t.n_delta = group, n_delta
        t.extra["source"] = source
        self.next_gid += 1
        return t

    def _render_density(self, segs: list, w_mm: float, n0: float):
        """Densité de gouttelettes (mm⁻²) : n0·q gouttelettes par mm le long du trajet (q = ionisation
        relative), dispersion latérale gaussienne commune (FWHM w_mm), flou optique."""
        c, r = self.cfg, self.rng
        ppm = self.ppm
        sig_mm = w_mm / 2.355
        sig_b_px = max(0.7, 0.8 * sig_mm * ppm)
        sig_lat_px = math.sqrt(max(0.0, (sig_mm * ppm) ** 2 - sig_b_px ** 2))
        allpx = np.vstack([s[0] for s in segs]) * ppm
        margin = int(math.ceil(3 * max(sig_b_px, sig_mm * ppm) + 8))
        x0 = int(math.floor(max(allpx[:, 0].min(), -margin))) - margin
        y0 = int(math.floor(max(allpx[:, 1].min(), -margin))) - margin
        x1 = int(math.ceil(min(allpx[:, 0].max(), c.width + margin))) + margin
        y1 = int(math.ceil(min(allpx[:, 1].max(), c.height + margin))) + margin
        if x1 <= x0 or y1 <= y0:
            return None, 0, 0, 1.0
        h, w = y1 - y0 + 1, x1 - x0 + 1
        canvas = np.zeros((h, w), np.float32)
        for xy, q in segs:
            p = xy * ppm - [x0, y0]
            seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(1))
            lam = n0 * 0.5 * (q[1:] + q[:-1]) * seg / ppm
            if c.supersat_sigma > 0 and len(lam) >= 2:
                # Sursaturation inégale le long de la trace (corrélée sur ≈ 2 mm) : MÊME loi pour toutes
                # les particules ; moyenne conservée (facteur log-normal centré).
                s_mm = np.concatenate([[0.0], np.cumsum(seg)]) / ppm
                knots = np.arange(0.0, s_mm[-1] + 2.0, 2.0)
                g = r.standard_normal(len(knots))
                mod = np.interp(0.5 * (s_mm[1:] + s_mm[:-1]), knots, g)
                sg = c.supersat_sigma
                lam = lam * np.exp(sg * mod - 0.5 * sg * sg)
            cum = np.concatenate([[0.0], np.cumsum(lam)])
            tot = float(cum[-1])
            if tot <= 0:
                continue
            n = int(r.poisson(tot))
            if n == 0:
                continue
            wgt = 1.0
            if n > 120_000:
                wgt, n = n / 120_000, 120_000
            u = r.uniform(0.0, tot, n)
            k = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(seg) - 1)
            f = (u - cum[k]) / np.maximum(lam[k], 1e-12)
            a, b = p[k], p[k + 1]
            pos = a + (b - a) * f[:, None]
            t = b - a
            tn = np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)
            nx, ny = -t[:, 1] / tn, t[:, 0] / tn
            off = r.standard_normal(n) * sig_lat_px
            xs = pos[:, 0] + nx * off
            ys = pos[:, 1] + ny * off
            xi = np.round(xs).astype(np.int64)
            yi = np.round(ys).astype(np.int64)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            wts = r.uniform(0.6, 1.4, n) * wgt
            np.add.at(canvas, (yi[ok], xi[ok]), wts[ok].astype(np.float32))
        canvas = cv2.GaussianBlur(canvas, (0, 0), sig_b_px)
        canvas *= np.float32(ppm * ppm)          # gouttelettes par px² → par mm²
        return canvas, x0, y0, sig_mm * ppm

    def _make_event(self, kind: str, src: bool) -> list:
        """Crée la ou les traces d'un événement programmé."""
        c, r = self.cfg, self.rng
        out = []
        if kind == "alpha":
            mix = dict(c.alpha_mix or _default_alpha_mix())
            names = list(mix)
            p = np.array([max(0.0, float(mix[k])) for k in names])
            p = p / p.sum() if p.sum() > 0 else np.full(len(names), 1.0 / len(names))
            em = names[int(r.choice(len(names), p=p))]
            if em == "thoron":
                segs, e0, p0 = self._alpha("Rn-220")
                life = float(np.clip(r.normal(c.alpha_life_mean_s, c.alpha_life_sd_s), 0.7, 2.6))
                g = self.next_group
                self.next_group += 1
                t = self._new_track("alpha", "Rn-220", segs, e0, life, group=g)
                if t is not None:
                    out.append(t)
                # Po-216 : même point (recul ≈ 0,1 mm), après une durée exponentielle (T½ = 0,145 s)
                delay = int(round(r.exponential(PO216_MEAN_LIFE_S) * c.fps))
                p1 = (p0[0] + r.normal(0, 0.1), p0[1] + r.normal(0, 0.1), p0[2])
                self.pending.setdefault(self.frame + max(0, delay), []).append(("po216", p1, g))
            else:
                segs, e0, _p0 = self._alpha(em)
                life = float(np.clip(r.normal(c.alpha_life_mean_s, c.alpha_life_sd_s), 0.7, 2.6))
                t = self._new_track("alpha", em, segs, e0, life)
                if t is not None:
                    out.append(t)
        elif kind == "beta":
            segs, e0 = self._electron(src)
            t = self._new_track("beta", "k40" if src else "compton", segs, e0, self._thin_life(), source=src)
            if t is not None:
                out.append(t)
        elif kind in ("muon", "cosmic_e"):
            if kind == "muon" and r.uniform() < c.muon_decay_frac:
                segs, e_m = self._muon_decay()
                t = self._new_track("muon", "muon_decay", segs, e_m, self._thin_life(), main_segs=segs[:1])
            else:
                pk = "mu" if kind == "muon" else "e"
                segs, deltas, energy, _d = self._crossing(pk, c.deff_mm)
                t = self._new_track("muon" if kind == "muon" else "beta", "muon" if kind == "muon" else "cosmic_e",
                                    segs + deltas, energy, self._thin_life(), n_delta=len(deltas), main_segs=segs)
            if t is not None:
                out.append(t)
        elif kind == "proton":
            segs, deltas, e0 = self._proton()
            qbar = float(np.mean(np.concatenate([s[1] for s in segs]))) if segs else 10.0
            med = c.thin_life_median_s * max(1.0, qbar / 4.0) ** 0.35
            life = float(np.clip(r.lognormal(math.log(med), 0.5), 0.15, 2.6))
            t = self._new_track("proton", "proton", segs + deltas, e0, life, n_delta=len(deltas), main_segs=segs)
            if t is not None:
                out.append(t)
        elif kind == "shower":
            n = int(r.integers(2, 6))
            L = muon_projected_length(r, c.deff_mm, max(c.crossing_min_mm, 4.0))
            ct = 1.0 / math.sqrt(1.0 + (L / c.deff_mm) ** 2)
            ph = r.uniform(0, 2 * math.pi)
            g = self.next_group
            self.next_group += 1
            for _ in range(n):
                th = math.acos(ct) + math.radians(r.normal(0, 1.5))
                st, ctt = math.sin(th), math.cos(th)
                phj = ph + math.radians(r.normal(0, 1.5))
                d = (st * math.cos(phj), st * math.sin(phj), -abs(ctt))
                is_mu = r.uniform() < 0.5
                segs, deltas, energy, _d = self._crossing("mu" if is_mu else "e", c.deff_mm, direction=d)
                t = self._new_track("muon" if is_mu else "beta", "gerbe_mu" if is_mu else "gerbe_e", segs + deltas,
                                    energy, self._thin_life(), group=g, n_delta=len(deltas), main_segs=segs)
                if t is not None:
                    out.append(t)
        elif kind.startswith("art:"):
            t = self._artefact(kind[4:])
            if t is not None:
                out.append(t)
        return out

    def _make_po216(self, p1, group) -> list:
        c, r = self.cfg, self.rng
        segs, e0, _p = self._alpha("Po-216", start=p1)
        life = float(np.clip(r.normal(c.alpha_life_mean_s, c.alpha_life_sd_s), 0.7, 2.6))
        t = self._new_track("alpha", "Po-216", segs, e0, life, group=group)
        return [t] if t is not None else []

    # ------------------------------------------------------------ artefacts --
    def _artefact(self, kind: str) -> _Track | None:
        """Artefacts qui trompent le détecteur (vérité terrain : classe « artefact »)."""
        c, r = self.cfg, self.rng
        ppm = self.ppm
        W, H = c.width, c.height
        fps = c.fps
        A = self.amp_chamber
        density = False
        flicker = 0.0
        mod_hz = 0.0
        if kind == "brume":
            a_mm, b_mm = r.uniform(4.0, 20.0), r.uniform(1.2, 5.0)
            ang = r.uniform(0, math.pi)
            cx, cy = r.uniform(0.05, 0.95) * W, r.uniform(0.05, 0.95) * H
            sa, sb = a_mm * ppm / 2.0, b_mm * ppm / 2.0
            half = int(math.ceil(3 * max(sa, sb))) + 4
            yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
            u = xx * math.cos(ang) + yy * math.sin(ang)
            v = -xx * math.sin(ang) + yy * math.cos(ang)
            spr = np.exp(-0.5 * ((u / sa) ** 2 + (v / sb) ** 2))
            tex = cv2.GaussianBlur(r.normal(0, 1, spr.shape).astype(np.float32), (0, 0), max(1.0, sb / 2))
            spr = np.clip(spr * (1.0 + 0.6 * tex / max(float(tex.std()), 1e-6)), 0, None).astype(np.float32)
            amp = r.uniform(0.06, 0.20) * A
            life_s, rise_s = r.uniform(0.8, 3.5), r.uniform(0.25, 0.8)
            v_mm = r.uniform(1.5, 7.0)
            dang = r.uniform(0, 2 * math.pi)
            ox, oy = int(cx) - half, int(cy) - half
            pts = np.array([[cx - sa * math.cos(ang), cy - sa * math.sin(ang)],
                            [cx + sa * math.cos(ang), cy + sa * math.sin(ang)]])
        elif kind == "amas":
            L_mm, w_mm = r.uniform(2.5, 9.0), r.uniform(0.4, 1.4)
            ang = r.uniform(0, math.pi)
            cx, cy = r.uniform(0.03, 0.97) * self.w_mm, r.uniform(0.03, 0.97) * self.h_mm
            n = int(r.integers(8, 41))
            s = r.normal(0, L_mm / 4.0, n)
            t = r.normal(0, w_mm / 2.355, n)
            xs = cx + s * math.cos(ang) - t * math.sin(ang)
            ys = cy + s * math.sin(ang) + t * math.cos(ang)
            # Gouttelettes plus grosses : poids ×1–3, rendu comme une trace (densité → réponse optique)
            segs = []
            for i in range(n):
                segs.append((np.array([[xs[i], ys[i]], [xs[i] + 0.05, ys[i]]]), np.full(2, r.uniform(15.0, 45.0))))
            spr, ox, oy, _sig = self._render_density(segs, r.uniform(0.35, 0.6), 1.0)
            if spr is None:
                return None
            density = True
            amp = A * r.uniform(0.8, 1.2)
            life_s, rise_s = r.uniform(0.15, 1.0), r.uniform(0.03, 0.12)
            v_mm = r.uniform(0.0, 3.0)
            dang = r.uniform(0, 2 * math.pi)
            pts = np.array([[cx - L_mm / 2 * math.cos(ang), cy - L_mm / 2 * math.sin(ang)],
                            [cx + L_mm / 2 * math.cos(ang), cy + L_mm / 2 * math.sin(ang)]]) * ppm
        elif kind == "scintillement":
            n = int(r.integers(2, 8))
            rad = r.uniform(0.5, 3.0) * ppm
            cx, cy = r.uniform(0.03, 0.97) * W, r.uniform(0.03, 0.97) * H
            half = int(math.ceil(rad + 6))
            spr = np.zeros((2 * half + 1, 2 * half + 1), np.float32)
            xs = r.uniform(-rad, rad, n)
            ys = r.uniform(-rad, rad, n)
            for i in range(n):
                cv2.circle(spr, (int(round(xs[i] * 4 + half * 4)), int(round(ys[i] * 4 + half * 4))), 2,
                           float(r.uniform(0.8, 1.6)), -1, cv2.LINE_AA, shift=2)
            spr = cv2.GaussianBlur(spr, (0, 0), r.uniform(0.6, 1.2))
            spr /= max(float(spr.max()), 1e-6)
            amp = A * r.uniform(0.6, 1.6)
            life_s, rise_s = r.uniform(0.1, 0.7), 1.0 / fps
            v_mm = r.uniform(0.0, 1.0)
            dang = r.uniform(0, 2 * math.pi)
            flicker = r.uniform(0.15, 0.45)
            ox, oy = int(cx) - half, int(cy) - half
            i0, i1 = int(np.argmin(xs)), int(np.argmax(xs))
            pts = np.array([[cx + xs[i0] - 1, cy + ys[i0]], [cx + xs[i1] + 1, cy + ys[i1]]])
        elif kind in ("reflet", "reflet_bord"):
            side = int(r.integers(4))                       # 0 haut, 1 bas, 2 gauche, 3 droite
            horiz = side in (0, 1)
            span = W if horiz else H
            if kind == "reflet":
                L = r.uniform(8.0, 40.0) * ppm
                wid = r.uniform(0.6, 2.5) * ppm / 2.355
                dist = r.uniform(0.01, 0.10) * (H if horiz else W)
                ang = math.radians(r.uniform(-15, 15))
                amp = r.uniform(0.07, 0.27) * A
                life_s, rise_s = r.uniform(1.0, 5.0), r.uniform(0.3, 1.2)
                v_mm = r.uniform(0.5, 4.0)
            else:
                L = r.uniform(20.0, 100.0) * ppm
                wid = r.uniform(4.0, 15.0) * ppm / 2.0
                dist = 0.0
                ang = 0.0
                amp = r.uniform(0.07, 0.30) * A
                life_s, rise_s = r.uniform(0.5, 3.0), r.uniform(0.1, 0.5)
                v_mm = r.uniform(0.0, 1.0)
                mod_hz = r.uniform(0.5, 3.0)
            pos = r.uniform(0.1, 0.9) * span
            half_l = L / 2
            hw = int(math.ceil(3 * wid + 4))
            nl = int(math.ceil(half_l)) + hw
            yy, xx = np.mgrid[-hw:hw + 1, -nl:nl + 1].astype(np.float32)
            u = xx * math.cos(ang) + yy * math.sin(ang)
            vv = -xx * math.sin(ang) + yy * math.cos(ang)
            along = np.clip(1.0 - np.maximum(np.abs(u) - half_l * 0.7, 0) / max(half_l * 0.3, 1.0), 0, 1)
            if kind == "reflet":
                prof = np.exp(-0.5 * (vv / wid) ** 2)
            else:
                prof = np.exp(-np.abs(vv - (-hw if side in (0, 2) else hw)) / max(wid, 1.0))
            mod = 1.0 + 0.25 * np.sin(u / max(L / 6, 1.0) + r.uniform(0, 6.28))
            spr = (prof * along * mod).astype(np.float32)
            if not horiz:
                spr = np.ascontiguousarray(spr.T)
            sh, sw = spr.shape
            # Centre du sprite (cx, cy) ; ligne de vérité terrain : axe du reflet ou milieu de la bande
            glare = kind == "reflet_bord"          # profil maximal sur le bord du sprite
            line = min(wid, hw) if glare else dist
            if side == 0:
                cx, cy, ly = pos, (sh / 2 if glare else line), line
            elif side == 1:
                cx, cy, ly = pos, H - 1 - (sh / 2 if glare else line), H - 1 - line
            elif side == 2:
                cx, cy, ly = (sw / 2 if glare else line), pos, line
            else:
                cx, cy, ly = W - 1 - (sw / 2 if glare else line), pos, W - 1 - line
            ox, oy = int(round(cx - sw / 2)), int(round(cy - sh / 2))
            dang = 0.0 if horiz else math.pi / 2
            if r.uniform() < 0.5:
                dang += math.pi
            if horiz:
                pts = np.array([[cx - half_l, ly], [cx + half_l, ly]])
            else:
                pts = np.array([[ly, cy - half_l], [ly, cy + half_l]])
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        else:
            return None
        L_vis = float(np.hypot(*(pts[1] - pts[0]))) / ppm
        life = int(max(1, round(life_s * fps)))
        rise = int(max(1, round(rise_s * fps)))
        drift = (v_mm * math.cos(dang) * ppm / fps, v_mm * math.sin(dang) * ppm / fps)
        t = _Track(self.next_gid, "artefact", self.frame, life, rise, pts, 0.0, amp, 0.0, drift)
        t.sprite, t.ox, t.oy = spr, int(ox), int(oy)
        t.length_mm, t.kind, t.density, t.flicker = L_vis, kind, density, flicker
        t.mod_hz, t.mod_phase = mod_hz, float(r.uniform(0, 2 * math.pi))
        t.sigma_px = 1.0
        self.next_gid += 1
        return t

    def _gt_entry(self, t: _Track, fi: int) -> dict:
        c = self.cfg
        pts = t.pts_px
        poly = np.round(pts[:: max(1, len(pts) // 60)], 1).tolist() + [pts[-1].round(1).tolist()]
        return {"id": t.gid, "cls": t.cls, "kind": t.kind, "source": bool(t.extra.get("source", False)),
                "frame_birth": fi, "t_birth": fi / c.fps, "life_frames": t.life_frames,
                "length_mm": round(t.length_mm, 2), "width_mm": round(t.width_px / c.px_per_mm, 3),
                "energy_mev": round(t.energy_mev, 4), "q_mean": round(t.q_mean, 2), "group": t.group,
                "n_delta": t.n_delta, "polyline": poly}

    # ----------------------------------------------------------------- rendu --
    def render_next(self) -> np.ndarray:
        c, rng = self.cfg, self.rng
        W, H = c.width, c.height
        fi = self.frame
        new = []
        for kind, src in self.schedule.get(fi, []):
            new.extend(self._make_event(kind, src))
        for _tag, p1, g in self.pending.pop(fi, []):
            new.extend(self._make_po216(p1, g))
        for t in new:
            t.frame_birth = fi
            self.tracks.append(t)
            self.gt.append(self._gt_entry(t, fi))
        # Brouillard mouvant
        k = fi // self.fog_period
        a = (fi % self.fog_period) / self.fog_period
        while self._fog_k < k:
            self.fog_keys = [self.fog_keys[1], self._fog_field()]
            self._fog_k += 1
        img = cv2.addWeighted(self.fog_keys[0], (1 - a) * 0.5 * c.fog_amp, self.fog_keys[1], a * 0.5 * c.fog_amp,
                              0.6 * c.fog_amp)
        img += self.static
        # « Pluie » de gouttelettes
        nnew = rng.poisson(c.rain_per_frame / 4.0)
        newr = np.column_stack([rng.uniform(0, W, nnew), rng.uniform(0, H, nnew),
                                rng.integers(2, 7, nnew), rng.uniform(6, 26, nnew)]).astype(np.float32)
        self.rain = np.vstack([self.rain, newr]) if len(self.rain) else newr
        if len(self.rain):
            rain_canvas = np.zeros((H, W), np.float32)
            xi = np.clip(self.rain[:, 0].astype(int), 0, W - 1)
            yi = np.clip(self.rain[:, 1].astype(int), 0, H - 1)
            np.add.at(rain_canvas, (yi, xi), self.rain[:, 3] * 2.0)
            self.rain[:, 2] -= 1
            self.rain = self.rain[self.rain[:, 2] > 0]
            img += cv2.GaussianBlur(rain_canvas, (0, 0), 0.8)
        # Traces et artefacts
        alive = []
        dsat = np.float32(-1.0 / max(c.sat_density_mm2, 1e-6))
        for t in self.tracks:
            age = fi - t.frame_birth
            if age >= t.life_frames:
                continue
            alive.append(t)
            if t.flicker > 0 and rng.uniform() < t.flicker:
                continue
            inten = min(1.0, (age + 1) / t.rise_frames) * math.exp(-max(0, age - t.rise_frames) / (0.45 * t.life_frames))
            if t.mod_hz > 0:
                inten *= 1.0 + 0.4 * math.sin(2 * math.pi * t.mod_hz * age / c.fps + t.mod_phase)
            spr = t.sprite
            if t.density:
                extra = c.diffusion_mm * math.sqrt(age / c.fps) * self.ppm
                if extra > 0.35:
                    spr = cv2.GaussianBlur(spr, (0, 0), extra)
                spr = 1.0 - cv2.exp(spr * dsat)
            dx = int(round(t.drift_px[0] * age))
            dy = int(round(t.drift_px[1] * age))
            x0, y0 = t.ox + dx, t.oy + dy
            h, w = spr.shape
            X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
            if X1 <= X0 or Y1 <= Y0:
                continue
            img[Y0:Y1, X0:X1] += spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * np.float32(t.amp * inten)
        self.tracks = alive
        if c.disturbance_at_s >= 0 and abs(fi / c.fps - c.disturbance_at_s) < 0.35:
            img += 90.0
        # Bruit du capteur tiré avec le générateur de la simulation (graine → vidéo reproductible)
        noise = rng.standard_normal((H, W), dtype=np.float32)
        if c.shot_noise > 0:
            noise *= np.sqrt(np.float32(c.noise_sigma ** 2) + np.float32(c.shot_noise) * np.maximum(img, 0))
            img += noise
        else:
            img += noise * np.float32(c.noise_sigma)
        self.frame += 1
        g = np.clip(img, 0, 255).astype(np.uint8)
        # Légère teinte bleutée de l'éclairage LED (addition / soustraction saturées)
        out = cv2.merge([cv2.add(g, 6), g, cv2.subtract(g, 3)])
        if c.jpeg_quality and c.jpeg_quality > 0:
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(c.jpeg_quality)])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return out

    def frames(self):
        for _ in range(self.n_frames):
            yield self.render_next()


def generate_video(path: str | Path, cfg: SynthConfig, progress=None) -> dict:
    """Écrit une vidéo MP4 et sa vérité terrain (<nom>.gt.json). Renvoie la vérité terrain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sim = SyntheticChamber(cfg)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (cfg.width, cfg.height))
    if not wr.isOpened():
        raise RuntimeError(f"Impossible d'écrire la vidéo {path}")
    for i, fr in enumerate(sim.frames()):
        wr.write(fr)
        if progress and i % 30 == 0:
            progress(i / sim.n_frames)
    wr.release()
    gt = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
          "px_per_mm": cfg.px_per_mm, "n_frames": sim.n_frames, "synth_version": SYNTH_VERSION,
          "tracks": sim.gt}
    Path(str(path) + ".gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    return gt


# --------------------------------------------------------------------------- #
# Justification de chaque différence simulée entre les classes
# --------------------------------------------------------------------------- #
JUSTIFICATIONS = [
    ("Densité de gouttelettes ∝ ionisation locale (alpha ≈ 390×, proton 3 MeV ≈ 56×, muon/électron "
     "rapide 1×, électron lent 2–12×)", "NIST ASTAR / PSTAR / ESTAR ; PDG (dE/dx)min = 1,815 MeV·cm²/g"),
    ("Épaisseur apparente plus grande des traces très ionisantes (même diffusion latérale pour toutes, "
     "saturation optique commune)", "conséquence du point précédent ; S'Cool LAB (alpha épaisse, muon fine)"),
    ("Pic de Bragg : alpha plus dense en fin de parcours (×2,8)", "NIST ASTAR (S max ≈ 2 021 MeV·cm²/g)"),
    ("Portée des alphas 3,4–7,4 cm à −25 °C selon l'émetteur, ∝ 1/ρ(T)", "NIST ASTAR ; NNDC (énergies) ; gaz parfait"),
    ("Alphas presque rectilignes", "ASTAR, facteur de détour 0,9925 ; Highland z = 2"),
    ("Paires en V du thoron (Rn-220 → Po-216, 0,145 s)", "NNDC (T½ Po-216) ; Nuledo"),
    ("Électrons sinueux (diffusion multiple) ; électrons lents courts et tortueux, rapides presque droits",
     "PDG, formule de Highland ; NIST ESTAR (portées)"),
    ("Électrons Compton de la plaque/des parois à spectre dégradé (front Compton 1,243 MeV)",
     "NIST XCOM (γ 1,461 MeV peu absorbé dans l'air) ; Klein-Nishina"),
    ("β⁻ du K-40 (Emax 1,311 MeV, moyenne 0,560 MeV), visibles sur ≈ 1 cm de couche",
     "NNDC/ENSDF 2017 ; ESTAR (portée ≫ couche)"),
    ("Muons droits, longueur projetée d·tanθ avec flux ∝ cos²θ : P(L > x) = (1 + (x/d)²)⁻²",
     "PDG (cosmic rays, cos²θ) ; Cambridge HEP (traînées courtes)"),
    ("Électrons cosmiques énergétiques (≈ 10 % des muons à l'intérieur, hypothèse) : fins et presque droits",
     "PDG (I_v(e) ≈ 30 m⁻²s⁻¹sr⁻¹ > 10 MeV, en plein air)"),
    ("Rayons delta (embranchements) sur les traces rapides ; aucun sur les alphas (Tmax ≈ 3 keV)",
     "PDG §34.2.7 ; S'Cool LAB (traces en Y)"),
    ("Protons lents : épaisseur intermédiaire, longs et droits", "NIST PSTAR ; Müller et al. 2023 ; Nuledo"),
    ("Durée de visibilité : alphas 1,5 ± 0,4 s ; traces fines (électrons ET muons) même loi log-normale",
     "Sussman (731 traces, 30 im/s)"),
    ("Protons : durée interpolée en fonction de l'ionisation (hypothèse, aucune mesure publiée)", "—"),
    ("Coude de désintégration du muon (fréquence exagérée pour l'apprentissage)", "S'Cool LAB (coudes)"),
    ("Artefacts : bouffées de brume, amas de gouttelettes, scintillements, reflets près des parois, "
     "éblouissement au bord", "Toronto PHY326 (gouttelettes qui tombent) ; observation courante"),
]
