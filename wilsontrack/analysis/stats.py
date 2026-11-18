"""Statistiques de comptage (loi de Poisson) et tests d'hypothèses.

La désintégration radioactive et l'arrivée des rayons cosmiques sont des
processus aléatoires indépendants : le nombre N de traces observées pendant une
durée T suit une loi de Poisson. On en déduit :
- taux  r = N / T,  incertitude  σ_r = √N / T ;
- intervalle de confiance exact (Garwood) pour les petits comptages ;
- comparaison de deux taux : test binomial conditionnel exact ;
- test de dispersion (variance/moyenne ≈ 1 pour un processus de Poisson) ;
- temps entre traces : loi exponentielle (test de Kolmogorov-Smirnov / Lilliefors,
  échelle estimée, p-valeur par Monte-Carlo).

Analyse approfondie (longs enregistrements, nuits entières) :
- taux par tranche de temps (heure, 10 min…) et par type, avec incertitudes de Poisson ;
- test de stabilité du taux (χ² contre un taux constant, p-valeur asymptotique ou
  Monte-Carlo exacte pour les petits comptages) et estimation de la dérive (%/h) ;
- recherche des périodes où la chambre était inactive ou dégradée (taux quasi nul,
  bruit de l'image en hausse) : exclues des taux « sur le temps actif », jamais des
  comptages bruts ;
- familles de traces (fines : électrons et muons, au minimum d'ionisation ; lourdes :
  alphas et protons) ;
- estimation des comptages corrigés des erreurs de classification connues
  (« déconvolution » bayésienne itérative à partir de la matrice de confusion mesurée
  sur simulation) avec incertitudes par rééchantillonnage (bootstrap de Poisson) ;
- résumé des confiances de classification.

Tous les nombres destinés à l'affichage passent par ``fr_num`` / ``fr_p`` (virgule décimale).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

CLASSES = ["alpha", "beta", "muon", "proton", "fine", "incertain"]
REAL_CLASSES = ["alpha", "beta", "muon", "proton"]
THIN_CLASSES = ("beta", "muon", "fine")          # particules au minimum d'ionisation
HEAVY_CLASSES = ("alpha", "proton")
FAMILIES = ("fine", "lourde")

NAN = float("nan")


# =========================================================================== #
# Formats d'affichage (français : virgule décimale)
# =========================================================================== #
_SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def _fnum(v) -> float:
    """Conversion tolérante en float (``""``, ``None``, texte → NaN)."""
    if v is None or isinstance(v, bool):
        return float(v) if isinstance(v, bool) else NAN
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", ".")
        if not s:
            return NAN
        try:
            return float(s)
        except ValueError:
            return NAN
    return NAN


def _finite(v) -> bool:
    return math.isfinite(_fnum(v))


def fr_num(x, d: int = 2, dash: str = "—", sign: bool = False) -> str:
    """Nombre avec ``d`` décimales et virgule décimale (« 3,14 ») ; ``dash`` si non fini."""
    v = _fnum(x)
    if not math.isfinite(v):
        return dash
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if s.strip("+-0.") == "":                  # « -0,00 » → « 0,00 » (ou « +0,00 »)
        s = s.replace("-", "+") if sign else s.lstrip("-")
    return s.replace(".", ",")


def fr_int(n) -> str:
    """Entier avec espace insécable comme séparateur de milliers (« 12 345 »)."""
    v = _fnum(n)
    if not math.isfinite(v):
        return "—"
    return f"{int(round(v)):,}".replace(",", " ")


def fr_sci(x, d: int = 1) -> str:
    """Notation scientifique lisible : « 1,2 × 10⁻⁵ »."""
    v = _fnum(x)
    if not math.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    if round(abs(m), d) >= 10:
        m /= 10
        e += 1
    return f"{fr_num(m, d)} × 10{str(e).translate(_SUP)}"


def fr_g(x, digits: int = 3) -> str:
    """Équivalent de ``:.3g`` avec virgule décimale (sans notation « e+03 »)."""
    v = _fnum(x)
    if not math.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 10 ** digits:                       # « 1 663 » → « 1 660 » (chiffres significatifs)
        return fr_int(round(v, -(int(math.floor(math.log10(a))) - digits + 1)))
    if a < 1e-3:
        return fr_sci(v, max(digits - 1, 0))
    d = max(digits - 1 - int(math.floor(math.log10(a))), 0)
    s = f"{v:.{d}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",")


def fr_p(p) -> str:
    """p-valeur : « 0,034 », « 0,0012 », « 2,1 × 10⁻⁵ »."""
    v = _fnum(p)
    if not math.isfinite(v):
        return "—"
    if v < 1e-16:                              # au-delà de la précision du calcul
        return "< 10⁻¹⁶"
    if v < 1e-4:
        return fr_sci(v, 1)
    return fr_num(v, 4) if v < 0.01 else fr_num(v, 3)


def fr_p_eq(p) -> str:
    """« p = 0,034 », ou « p < 10⁻¹⁶ » pour une p-valeur hors de la précision du calcul."""
    txt = fr_p(p)
    return f"p {txt}" if txt.startswith("<") else f"p = {txt}"


def fr_duration(s) -> str:
    """Durée lisible : « 45 s », « 12,5 min », « 7 h 32 min »."""
    v = _fnum(s)
    if not math.isfinite(v):
        return "—"
    v = max(v, 0.0)
    if v < 120:
        return f"{fr_num(v, 0)} s"
    if v < 3600:
        return f"{fr_num(v / 60, 1)} min"
    h = int(v // 3600)
    m = int(round((v - 3600 * h) / 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h} h" if m == 0 else f"{h} h {m:02d} min"


def fr_clock(s) -> str:
    """Instant depuis le début (s) sous la forme « 2 h 05 » ou « 12 min 30 s »."""
    v = _fnum(s)
    if not math.isfinite(v):
        return "—"
    v = max(v, 0.0)
    if v < 3600:
        m = int(v // 60)
        return f"{m} min {int(round(v - 60 * m)):02d} s" if m else f"{fr_num(v, 0)} s"
    h = int(v // 3600)
    m = int(round((v - 3600 * h) / 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h} h {m:02d}"


# =========================================================================== #
# Taux et comparaisons
# =========================================================================== #
@dataclass
class Rate:
    n: int
    t_s: float
    per_min: float
    err: float
    lo: float
    hi: float

    def fmt(self, unit: str = "/min", digits: int = 2) -> str:
        return f"{fr_num(self.per_min, digits)} ± {fr_num(self.err, digits)} {unit}"


def rate(n: int, t_s: float, cl: float = 0.6827) -> Rate:
    """Taux par minute, incertitude poissonnienne et intervalle de Garwood.

    Durée nulle (image fixe, perturbation permanente…) : pas de taux défini → NaN.
    """
    if not (t_s > 0):
        nan = float("nan")
        return Rate(n, 0.0, nan, nan, nan, nan)
    t_min = t_s / 60.0
    a = 1 - cl
    lo = 0.0 if n == 0 else stats.chi2.ppf(a / 2, 2 * n) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * (n + 1)) / 2
    err = math.sqrt(max(n, 1)) / t_min if n > 0 else hi / t_min
    return Rate(n, t_s, n / t_min, err, lo / t_min, hi / t_min)


@dataclass
class Comparison:
    cls: str
    a: Rate
    b: Rate
    diff: float
    diff_err: float
    ratio: float
    p_value: float
    sigma: float
    label: str = ""              # description du groupe (ex. définition des « traces fines »)

    @property
    def significant(self) -> bool:
        return self.p_value < 0.01 and self.diff > 0


def compare(n_src: int, t_src: float, n_bg: int, t_bg: float, cls: str = "",
            area_ratio: float = 1.0, alternative: str = "greater") -> Comparison:
    """Compare un enregistrement « échantillon » à un enregistrement « fond ».

    Test exact : sachant N = n_src + n_bg, sous H0 (même taux par unité de surface)
    n_src suit une loi binomiale B(N, T_src·A_src / (T_src·A_src + T_bg·A_bg)).
    ``area_ratio`` = A_src / A_bg : le taux du fond est ramené à la surface analysée de
    l'échantillon (valable car le fond — rayons cosmiques, radon — est uniforme).
    Par défaut, test unilatéral (excès de l'échantillon) ; ``alternative="two-sided"``
    pour un contrôle (variation dans un sens ou dans l'autre).
    """
    k = area_ratio if (area_ratio > 0 and math.isfinite(area_ratio)) else 1.0
    a = rate(n_src, t_src)
    b = rate(n_bg, t_bg)
    if k != 1.0:
        b = Rate(n_bg, t_bg, b.per_min * k, b.err * k, b.lo * k, b.hi * k)
    diff = a.per_min - b.per_min
    diff_err = (math.hypot(math.sqrt(n_src) / (t_src / 60), k * math.sqrt(n_bg) / (t_bg / 60))
                if (t_src > 0 and t_bg > 0) else float("nan"))
    ratio = a.per_min / b.per_min if b.per_min > 0 else float("inf")
    n = n_src + n_bg
    if n == 0 or t_src <= 0 or t_bg <= 0:
        return Comparison(cls, a, b, diff, diff_err, ratio, 1.0, 0.0)
    p0 = (t_src * k) / (t_src * k + t_bg)
    p = float(stats.binomtest(n_src, n, p0, alternative=alternative).pvalue)
    p = min(max(p, 1e-300), 1.0)
    if alternative == "two-sided":
        sigma = float(stats.norm.isf(p / 2)) if p < 1.0 else 0.0
    else:
        sigma = float(stats.norm.isf(p)) if p < 0.5 else 0.0
    return Comparison(cls, a, b, diff, diff_err, ratio, p, sigma)


@dataclass
class DispersionTest:
    bin_s: float
    n_bins: int
    mean: float
    var: float
    index: float
    chi2: float
    p_value: float

    @property
    def poisson_ok(self) -> bool:
        return self.p_value > 0.01


def dispersion_test(times: np.ndarray, duration_s: float, bin_s: float | None = None) -> DispersionTest | None:
    """Test de l'indice de dispersion (variance / moyenne) des comptages par intervalle."""
    times = np.asarray(times, float)
    if duration_s <= 0 or times.size < 8:
        return None
    if bin_s is None:
        # environ 5 traces par intervalle en moyenne, au moins 8 intervalles
        bin_s = max(duration_s / max(8, times.size / 5), 1.0)
    nb = int(duration_s // bin_s)
    if nb < 5:
        return None
    counts = np.histogram(times, bins=nb, range=(0, nb * bin_s))[0]
    m = counts.mean()
    v = counts.var(ddof=1)
    if m <= 0:
        return None
    chi2 = (nb - 1) * v / m
    cdf = stats.chi2.cdf(chi2, nb - 1)
    p = 2 * min(cdf, 1 - cdf)
    return DispersionTest(bin_s, nb, float(m), float(v), float(v / m), float(chi2), float(p))


_IA_MC_SAMPLES = 1999
_IA_SEED = 20260929
_IA_MAX_N = 4000                 # au-delà : sous-échantillon aléatoire (graine fixe) d'intervalles


def _ks_expon(x: np.ndarray) -> np.ndarray:
    """Statistique de Kolmogorov-Smirnov contre une loi exponentielle d'échelle estimée (moyenne),
    pour chaque ligne de ``x`` (forme (k, n))."""
    x = np.sort(np.asarray(x, float), axis=-1)
    n = x.shape[-1]
    m = x.mean(axis=-1, keepdims=True)
    F = 1.0 - np.exp(-x / np.where(m > 0, m, 1.0))
    i = np.arange(1, n + 1, dtype=float)
    return np.maximum(np.max(i / n - F, axis=-1), np.max(F - (i - 1) / n, axis=-1))


_LILLIEFORS_CACHE: dict[int, np.ndarray] = {}


def _lilliefors_null(n: int) -> np.ndarray:
    """Distribution (Monte-Carlo, graine fixe) de la statistique de Lilliefors exponentielle pour n
    intervalles. Elle ne dépend que de n (l'échelle est estimée) : mise en cache."""
    null = _LILLIEFORS_CACHE.get(n)
    if null is None:
        rng = np.random.default_rng(_IA_SEED)
        null = np.empty(_IA_MC_SAMPLES)
        batch = max(1, int(2_000_000 // max(n, 1)))
        for i in range(0, _IA_MC_SAMPLES, batch):
            k = min(batch, _IA_MC_SAMPLES - i)
            null[i:i + k] = _ks_expon(rng.exponential(size=(k, n)))
        if len(_LILLIEFORS_CACHE) > 32:
            _LILLIEFORS_CACHE.clear()
        _LILLIEFORS_CACHE[n] = null
    return null


def interarrival_test(times: np.ndarray) -> tuple[float, float] | None:
    """Temps entre traces successives vs loi exponentielle.

    Test de Kolmogorov-Smirnov « de Lilliefors » : l'échelle (temps moyen) est estimée
    sur les données, donc la p-valeur est obtenue par Monte-Carlo paramétrique (le test
    KS classique serait trop indulgent) : p = (1 + #{D_sim ≥ D_obs}) / (1 + 1999).
    Graine fixe : résultat reproductible. La distribution simulée ne dépend que du nombre
    d'intervalles (mise en cache) ; au-delà de 4 000 intervalles (longs enregistrements), le
    test porte sur un sous-échantillon aléatoire de 4 000 intervalles (graine fixe), ce qui
    garde une puissance très élevée pour un coût constant.
    Renvoie (temps moyen en s, p-valeur).
    """
    t = np.sort(np.asarray(times, float))
    t = t[np.isfinite(t)]
    if t.size < 10:
        return None
    dt = np.diff(t)                    # on garde les intervalles nuls (même image)
    if dt.size < 8:
        return None
    mu = float(dt.mean())
    if mu <= 0:
        return None
    x = dt
    if x.size > _IA_MAX_N:
        x = np.random.default_rng(_IA_SEED).choice(x, _IA_MAX_N, replace=False)
    if not (x.mean() > 0):
        return mu, 1.0
    d = float(_ks_expon(x[None, :])[0])
    null = _lilliefors_null(int(x.size))
    p = float((1 + np.sum(null >= d - 1e-12)) / (1 + null.size))
    return mu, p


def rayleigh_axial(angles_deg: np.ndarray) -> tuple[float, float] | None:
    """Test de Rayleigh pour des orientations (données axiales, 0–180°) : isotropie ?"""
    a = np.radians(np.asarray(angles_deg, float)) * 2.0
    n = a.size
    if n < 8:
        return None
    R = abs(np.mean(np.exp(1j * a)))
    Z = n * R * R
    p = math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n) - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n * n))
    return float(R), float(min(max(p, 0.0), 1.0))


def time_series(times: np.ndarray, classes: list[str], duration_s: float, bin_s: float = 10.0) -> dict:
    nb = max(1, int(math.ceil(duration_s / bin_s)))
    edges = np.arange(nb + 1) * bin_s
    out = {"edges": edges}
    times = np.asarray(times, float)
    cl = np.asarray(classes)
    for c in CLASSES:
        sel = times[cl == c] if cl.size else np.array([])
        out[c] = np.histogram(sel, bins=edges)[0]
    out["total"] = np.histogram(times, bins=edges)[0]
    return out


def mean_err(values) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan"), int(v.size)


def summarize(rows: list[dict], duration_s: float, area_mm2: float) -> dict:
    """Résumé d'un enregistrement : taux par classe, par unité de surface, mesures moyennes."""
    out = {"duration_s": duration_s, "area_cm2": area_mm2 / 100.0, "classes": {}}
    tot = len(rows)
    out["total"] = rate(tot, duration_s)
    for c in CLASSES:
        rc = [r for r in rows if r.get("classe") == c]
        r = rate(len(rc), duration_s)
        L = mean_err(r_["longueur_mm"] for r_ in rc)
        W = mean_err(r_["epaisseur_mm"] for r_ in rc)
        D = mean_err(r_["duree_visible_s"] for r_ in rc)
        out["classes"][c] = {
            "rate": r,
            "per_cm2_min": r.per_min / max(area_mm2 / 100.0, 1e-9),
            "length": L, "width": W, "lifetime": D,
            "fraction": len(rc) / tot if tot else 0.0,
        }
    times = event_times(rows)
    out["dispersion"] = dispersion_test(times, duration_s)
    out["interarrival"] = interarrival_test(times)
    mu = [r["orientation_deg"] for r in rows if r.get("classe") == "muon"]
    out["muon_isotropy"] = rayleigh_axial(np.array(mu)) if len(mu) >= 8 else None
    energies = [float(r["energie_MeV"]) for r in rows
                if r.get("classe") == "alpha" and isinstance(r.get("energie_MeV"), (int, float))
                and math.isfinite(float(r["energie_MeV"]))]
    out["alpha_energy"] = mean_err(energies)
    out["alpha_energies"] = energies
    # Vue de dessus, les alphas sont projetées et tronquées par la couche sensible :
    # l'énergie déduite de la longueur est une énergie MINIMALE. Les traces les plus
    # longues (entièrement contenues) estiment le mieux l'énergie réelle → bord supérieur.
    E = np.asarray(energies, float)
    edge = float(np.percentile(E, 90)) if E.size >= 10 else (float(E.max()) if E.size else float("nan"))
    out["alpha_energy_edge"] = (edge, int(E.size))
    fam = family_counts(rows)
    fam["rate"] = rate(fam["fines"], duration_s)
    fam["rate_lourdes"] = rate(fam["lourdes"], duration_s)
    out["families"] = fam
    return out


def event_times(rows: list[dict]) -> np.ndarray:
    """Instants d'apparition sur l'horloge d'analyse (0 = début de l'analyse effective).

    Utilise « t_analyse_s » (temps réellement analysé, hors perturbations et pauses) ;
    pour d'anciens résultats qui ne l'ont pas, se rabat sur t_debut_s décalé à 0.
    """
    ta = [r.get("t_analyse_s") for r in rows]
    if rows and all(isinstance(v, (int, float)) for v in ta):
        return np.asarray(ta, float)
    t = np.asarray([float(r.get("t_debut_s") or 0.0) for r in rows], float)
    return t - t.min() if t.size else t


def times_and_clock(rows: Sequence[dict], duration_s: float | None = None) -> tuple[np.ndarray, str]:
    """(instants, horloge) : « analyse » (t_analyse_s, hors perturbations) ou « video »
    (t_debut_s, anciens résultats).

    Horloge « video » : t_debut_s tel quel s'il tient dans la durée analysée (vidéo lue depuis le
    début), sinon ramené à 0 (vidéo lue à partir d'un instant donné, ou durée inconnue).
    """
    rows = list(rows)
    ta = [r.get("t_analyse_s") for r in rows]
    if rows and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in ta):
        return np.asarray(ta, float), "analyse"
    t = np.asarray([_fnum(r.get("t_debut_s")) for r in rows], float)
    t = np.where(np.isfinite(t), t, 0.0)
    d = _fnum(duration_s)
    if t.size and (not (math.isfinite(d) and d > 0) or t.max() > 1.05 * d or t.min() < 0):
        t = t - t.min()
    return t, "video"


# =========================================================================== #
# Familles de traces : fines (électrons, muons) / lourdes (alphas, protons)
# =========================================================================== #
def row_family(r: dict) -> str | None:
    """Famille d'une trace (« fine » / « lourde ») si la colonne existe et est valide."""
    f = r.get("famille") if isinstance(r, dict) else None
    if isinstance(f, str):
        f = f.strip().lower()
        if f in FAMILIES:
            return f
    return None


def has_family(rows: Iterable[dict]) -> bool:
    """True si TOUTES les traces ont une famille valide (nouveaux enregistrements)."""
    rows = list(rows)
    return bool(rows) and all(row_family(r) is not None for r in rows)


def common_family_mode(*row_lists: Sequence[dict]) -> bool:
    """Définition commune des « traces fines » pour comparer plusieurs enregistrements.

    La famille n'est utilisée que si chaque enregistrement non vide l'a pour toutes ses
    traces (sinon on comparerait deux définitions différentes) ; sinon β + μ partout.
    """
    nonempty = [list(r) for r in row_lists if r]
    return bool(nonempty) and all(has_family(r) for r in nonempty)


def is_fine(r: dict, use_family: bool) -> bool:
    """Trace fine : β ou μ (classe finale), plus — si la colonne « famille » existe — les traces
    INCERTAINES de famille « fine ». La classe finale prime quand elle est définie (une trace classée
    alpha n'est jamais « fine », une trace classée muon l'est toujours) : ainsi fines = β + μ +
    incertaines fines, toujours ≥ β + μ."""
    c = r.get("classe")
    if c in THIN_CLASSES:
        return True
    if c in HEAVY_CLASSES:
        return False
    return use_family and row_family(r) == "fine"


def is_heavy(r: dict, use_family: bool) -> bool:
    """Trace lourde : α ou proton (classe finale), plus les incertaines de famille « lourde »."""
    c = r.get("classe")
    if c in HEAVY_CLASSES:
        return True
    if c in THIN_CLASSES:
        return False
    return use_family and row_family(r) == "lourde"


def fines_label(use_family: bool) -> str:
    return ("Traces fines (β, μ et fines incertaines)" if use_family else "Traces fines (β + μ)")


def family_counts(rows: Sequence[dict], use_family: bool | None = None) -> dict:
    """Comptages par famille.

    Mode « famille » (colonne ``famille`` présente) : les traces fines incluent celles
    classées « incertaines » de famille « fine » (ambiguïté β/μ), ce qui conserve toute la
    statistique pour la recherche de radioactivité β. Mode « classes » (anciens résultats) :
    fines = β + μ. ``famille_desaccord`` : traces dont la famille mesurée par les règles contredit
    la classe finale (ex. muon de famille « lourde ») — la classe finale est retenue.
    """
    rows = list(rows)
    if use_family is None:
        use_family = has_family(rows)
    fines = [r for r in rows if is_fine(r, use_family)]
    lourdes = [r for r in rows if is_heavy(r, use_family)]
    return {
        "mode": "famille" if use_family else "classes",
        "label": fines_label(use_family),
        "total": len(rows),
        "fines": len(fines),
        "lourdes": len(lourdes),
        "fines_incertaines": sum(1 for r in fines if r.get("classe") == "incertain"),
        "lourdes_incertaines": sum(1 for r in lourdes if r.get("classe") == "incertain"),
        "fines_par_classe": dict(Counter(str(r.get("classe") or "incertain") for r in fines)),
        "lourdes_par_classe": dict(Counter(str(r.get("classe") or "incertain") for r in lourdes)),
        "famille_desaccord": sum(1 for r in rows if use_family and (
            (r.get("classe") in THIN_CLASSES and row_family(r) == "lourde")
            or (r.get("classe") in HEAVY_CLASSES and row_family(r) == "fine"))),
    }


# =========================================================================== #
# Journal temporel (« timeline ») du détecteur
# =========================================================================== #
def timeline_intervals(timeline: Sequence[dict] | None) -> list[dict]:
    """Transforme le journal cumulatif du détecteur en intervalles successifs.

    Entrées attendues (contrat du détecteur) : t_s (horloge d'analyse), wall_s (horloge de
    la source), frames (images analysées, cumul), disturbed_frames (cumul), noise, lit,
    counts (cumul par classe), rejected (cumul). Les entrées invalides sont ignorées.

    Chaque intervalle : t0, t1, dt (s analysées), wall0, wall1, dwall (s de vidéo/caméra),
    frames, disturbed (images de l'intervalle), disturbed_frac, unanalysed_frac
    (1 − dt/dwall : perturbations, pauses, apprentissage du fond), noise, lit, counts
    (par classe), total, rejected.
    """
    ents = []
    for e in timeline or []:
        if isinstance(e, dict) and _finite(e.get("t_s")):
            ents.append(e)
    ents.sort(key=lambda e: _fnum(e.get("t_s")))
    out: list[dict] = []
    prev_t, prev_f, prev_d, prev_rej = 0.0, 0, 0, 0
    prev_c: dict = {}
    prev_w = None

    def _int(v) -> int:
        x = _fnum(v)
        return int(x) if math.isfinite(x) else 0

    for e in ents:
        t = _fnum(e.get("t_s"))
        f, d, rej = _int(e.get("frames")), _int(e.get("disturbed_frames")), _int(e.get("rejected"))
        counts = e.get("counts") if isinstance(e.get("counts"), dict) else {}
        dc = {str(k): max(_int(v) - _int(prev_c.get(k, 0)), 0) for k, v in counts.items()}
        df, dd, drej = max(f - prev_f, 0), max(d - prev_d, 0), max(rej - prev_rej, 0)
        w = _fnum(e.get("wall_s"))
        dt = t - prev_t
        if dt <= 1e-6:
            if out:                                   # entrée finale confondue avec la précédente
                last = out[-1]
                for k, v in dc.items():
                    last["counts"][k] = last["counts"].get(k, 0) + v
                last["total"] = int(sum(last["counts"].values()))
                last["frames"] += df
                last["disturbed"] += dd
                last["rejected"] += drej
                tot = last["frames"] + last["disturbed"]
                last["disturbed_frac"] = last["disturbed"] / tot if tot else NAN
            prev_f, prev_d, prev_rej, prev_c = f, d, rej, dict(counts)
            if math.isfinite(w):
                prev_w = w
            continue
        tot_f = df + dd
        dist_frac = dd / tot_f if tot_f > 0 else NAN
        if prev_w is None and math.isfinite(w):
            # Début de la source inconnu (vidéo commencée en cours, apprentissage du fond) :
            # on suppose que le seul temps non analysé du premier intervalle est perturbé.
            frac = dist_frac if math.isfinite(dist_frac) else 0.0
            prev_w = w - dt / max(1.0 - frac, 0.05)
        dwall = (w - prev_w) if (math.isfinite(w) and prev_w is not None and math.isfinite(prev_w)) else NAN
        unan = float(np.clip(1.0 - dt / dwall, 0.0, 1.0)) if (math.isfinite(dwall) and dwall > 0) else NAN
        noise = _fnum(e.get("noise"))
        out.append({
            "t0": prev_t, "t1": t, "dt": dt,
            "wall0": prev_w if prev_w is not None else NAN, "wall1": w, "dwall": dwall,
            "frames": df, "disturbed": dd, "disturbed_frac": dist_frac, "unanalysed_frac": unan,
            "noise": noise if (math.isfinite(noise) and noise > 0) else NAN,
            "lit": _fnum(e.get("lit")),
            "counts": dc, "total": int(sum(dc.values())), "rejected": drej,
        })
        prev_t, prev_f, prev_d, prev_rej, prev_c = t, f, d, rej, dict(counts)
        if math.isfinite(w):
            prev_w = w
    return out


class ClockMap:
    """Correspondance horloge d'analyse → horloge de la source (vidéo / caméra), en s."""

    def __init__(self, t: Sequence[float], w: Sequence[float], source: str):
        t = np.asarray(t, float)
        w = np.asarray(w, float)
        order = np.argsort(t, kind="stable")
        self.t, self.w = t[order], np.maximum.accumulate(w[order])
        self.source = source            # « journal » ou « traces »

    def __call__(self, x):
        x = np.asarray(x, float)
        y = np.interp(x, self.t, self.w)
        beyond = x > self.t[-1]
        if np.any(beyond):
            y = np.where(beyond, self.w[-1] + (x - self.t[-1]), y)
        before = x < self.t[0]
        if np.any(before):
            y = np.where(before, self.w[0] - (self.t[0] - x), y)
        return y if y.ndim else float(y)


def clock_map(intervals: Sequence[dict] | None, rows: Sequence[dict] | None = None) -> ClockMap | None:
    """Horloge d'analyse → horloge de la source : d'après le journal, sinon d'après les traces."""
    iv = [i for i in (intervals or []) if math.isfinite(i.get("wall1", NAN)) and math.isfinite(i.get("wall0", NAN))]
    if iv:
        t = [iv[0]["t0"]] + [i["t1"] for i in iv]
        w = [iv[0]["wall0"]] + [i["wall1"] for i in iv]
        return ClockMap(t, w, "journal")
    pairs = [(_fnum(r.get("t_analyse_s")), _fnum(r.get("t_debut_s"))) for r in (rows or [])]
    pairs = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return None
    pairs.sort()
    t = [p[0] for p in pairs]
    w = [p[1] for p in pairs]
    if t[0] > 0:
        t.insert(0, 0.0)
        w.insert(0, w[0] - pairs[0][0])
    return ClockMap(t, w, "traces")


def noise_summary(intervals: Sequence[dict] | None, spike_factor: float = 2.0) -> dict | None:
    """Bruit de l'image et perturbations au cours du temps (d'après le journal du détecteur).

    Renvoie None sans journal. Clés : n_intervals, analysed_s, wall_s, median, first, last,
    drift_ratio (médiane du dernier quart / premier quart), n_spikes, spike_frac,
    spikes [(t0, t1)], disturbed_frac (images perturbées / images), unanalysed_frac
    (1 − temps analysé / temps de la source), lit_median.
    """
    iv = list(intervals or [])
    if not iv:
        return None
    noise = np.array([i["noise"] for i in iv], float)
    ok = np.isfinite(noise)
    frames = sum(i["frames"] for i in iv)
    dist = sum(i["disturbed"] for i in iv)
    dt = sum(i["dt"] for i in iv)
    dwall = sum(i["dwall"] for i in iv if math.isfinite(i["dwall"]))
    out = {
        "n_intervals": len(iv), "analysed_s": float(dt), "wall_s": float(dwall) if dwall > 0 else NAN,
        "disturbed_frac": dist / (frames + dist) if (frames + dist) > 0 else NAN,
        "unanalysed_frac": float(np.clip(1.0 - dt / dwall, 0.0, 1.0)) if dwall > 0 else NAN,
        "lit_median": float(np.nanmedian([i["lit"] for i in iv])) if any(math.isfinite(i["lit"]) for i in iv) else NAN,
        "median": NAN, "first": NAN, "last": NAN, "drift_ratio": NAN,
        "n_spikes": 0, "spike_frac": 0.0, "spikes": [],
    }
    if ok.sum() >= 2:
        nz = noise[ok]
        med = float(np.median(nz))
        q = max(1, nz.size // 4)
        first, last = float(np.median(nz[:q])), float(np.median(nz[-q:]))
        spikes = [(i["t0"], i["t1"]) for i in iv if math.isfinite(i["noise"]) and med > 0
                  and i["noise"] > spike_factor * med]
        out.update(median=med, first=first, last=last,
                   drift_ratio=last / first if first > 0 else NAN,
                   n_spikes=len(spikes), spike_frac=len(spikes) / int(ok.sum()), spikes=spikes)
    return out


# =========================================================================== #
# Périodes mortes ou dégradées
# =========================================================================== #
@dataclass
class DeadPeriod:
    t0: float                    # horloge d'analyse (s)
    t1: float
    n_obs: int
    n_expected: float
    p_value: float
    kind: str = "morte"          # « morte » (taux quasi nul) ou « dégradée » (bruit élevé + taux bas)
    wall0: float = NAN           # horloge de la source (s)
    wall1: float = NAN
    noise_ratio: float = NAN     # bruit médian pendant la période / bruit médian global

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t1 - self.t0)

    def contains(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return (t >= self.t0) & (t < self.t1)


def _interval_noise_by_bin(intervals: Sequence[dict], edges: np.ndarray) -> np.ndarray:
    """Bruit médian des intervalles du journal dont le milieu tombe dans chaque tranche."""
    out = np.full(len(edges) - 1, NAN)
    if not intervals:
        return out
    mids = np.array([(i["t0"] + i["t1"]) / 2 for i in intervals])
    nz = np.array([i["noise"] for i in intervals], float)
    idx = np.searchsorted(edges, mids, side="right") - 1
    for b in range(len(out)):
        v = nz[(idx == b) & np.isfinite(nz)]
        if v.size:
            out[b] = float(np.median(v))
    return out


DEAD_EXPECTED_PER_BIN = 30.0


def detect_dead_periods(times, duration_s: float, intervals: Sequence[dict] | None = None,
                        cmap: ClockMap | None = None, dead_frac: float = 0.25, degraded_frac: float = 0.6,
                        noise_factor: float = 2.0, p_max: float = 1e-3, min_bins: int = 6) -> dict:
    """Recherche des périodes où la chambre ne détectait (presque) plus rien.

    Méthode : tranches d'environ 30 traces attendues (multiples de 60 s, au plus 30 min) ;
    taux de référence = médiane des taux par tranche (robuste aux périodes mortes).
    - « morte » : au moins 2 tranches consécutives dont le taux est ≤ 25 % de la référence ;
    - « dégradée » : suite de tranches consécutives où le bruit de l'image est au moins 2× plus
      élevé que d'habitude (journal du détecteur) et dont le taux global est ≤ 60 % de la référence.
    Une période n'est retenue que si un si petit nombre de traces est très improbable par hasard :
    p de Poisson < 0,001 / (nombre de tranches) (correction de Bonferroni pour les tests multiples).
    Les périodes retenues ne sont retirées que des taux « sur le temps actif ».

    Renvoie {possible, reason, bin_s, ref_rate_per_min, live_rate_per_min, periods, dead_s, live_s}.
    """
    dur = _fnum(duration_s)
    res = {"possible": False, "reason": "", "bin_s": NAN, "ref_rate_per_min": NAN,
           "live_rate_per_min": NAN, "periods": [], "dead_s": 0.0,
           "live_s": dur if math.isfinite(dur) else 0.0}
    t = np.asarray(times, float)
    t = t[np.isfinite(t)]
    if not (math.isfinite(dur) and dur > 0):
        res["reason"] = "durée analysée nulle"
        return res
    t = np.clip(t, 0.0, dur)
    N = int(t.size)
    if dur < 600:
        res["reason"] = "enregistrement trop court (moins de 10 min) pour rechercher des périodes inactives"
        return res
    if N < 30:
        res["reason"] = "trop peu de traces (moins de 30) pour rechercher des périodes inactives"
        return res
    r = N / dur
    w = float(min(max(60.0, math.ceil(DEAD_EXPECTED_PER_BIN / r / 60.0) * 60.0), 1800.0))
    nb_full = int(dur // w)
    rem = dur - nb_full * w
    edges = list(np.arange(nb_full + 1, dtype=float) * w)
    if rem >= 0.5 * w:
        edges.append(dur)                       # dernière tranche partielle gardée
    else:
        edges[-1] = dur                         # reste court : fusionné dans la dernière tranche
    edges = np.asarray(edges, float)
    nb = len(edges) - 1
    if nb < min_bins:
        res["reason"] = (f"pas assez de tranches de {fr_duration(w)} (au moins {min_bins}) pour rechercher "
                         "des périodes inactives")
        return res
    counts = np.histogram(t, bins=edges)[0].astype(float)
    expo = np.diff(edges)
    rates = counts / expo
    full = expo >= 0.5 * w
    r_ref = float(np.median(rates[full])) if full.any() else float(np.median(rates))
    if r_ref <= 0:
        r_ref = float(np.percentile(rates, 75))
    if r_ref <= 0:
        res["reason"] = "presque aucune trace pendant l'enregistrement"
        return res
    noise_bin = _interval_noise_by_bin(intervals or [], edges)
    gl = [i["noise"] for i in (intervals or []) if math.isfinite(i["noise"])]
    noise_med = float(np.median(gl)) if gl else NAN
    noisy = (np.isfinite(noise_bin) & (noise_bin >= noise_factor * noise_med)) if math.isfinite(noise_med) \
        else np.zeros(nb, bool)
    exp_b = r_ref * expo
    p_lim = p_max / nb                                   # correction de Bonferroni

    def runs(mask: np.ndarray) -> list[tuple[int, int]]:
        out, b = [], 0
        while b < nb:
            if not mask[b]:
                b += 1
                continue
            e = b
            while e + 1 < nb and mask[e + 1]:
                e += 1
            out.append((b, e))
            b = e + 1
        return out

    cand: list[tuple[int, int, str]] = []
    claimed = np.zeros(nb, bool)
    for b, e in runs(noisy):                             # 1) périodes bruitées (cause : image dégradée)
        n_seg, mu = float(counts[b:e + 1].sum()), float(exp_b[b:e + 1].sum())
        if n_seg <= degraded_frac * mu:
            cand.append((b, e, "morte" if n_seg <= dead_frac * mu else "dégradée"))
            claimed[b:e + 1] = True
    low = (counts <= dead_frac * exp_b) & ~claimed      # 2) taux quasi nul (bruit normal)
    # au moins 2 tranches consécutives : une tranche isolée très basse est plus souvent une
    # fluctuation (ou une baisse partielle) qu'un arrêt de la chambre
    cand += [(b, e, "morte") for b, e in runs(low) if e > b]
    periods: list[DeadPeriod] = []
    for b, e, kind in sorted(cand):
        n_seg = int(counts[b:e + 1].sum())
        mu = float(exp_b[b:e + 1].sum())
        p = float(stats.poisson.cdf(n_seg, mu))
        if p >= p_lim:
            continue
        t0, t1 = float(edges[b]), float(edges[e + 1])
        nr = NAN
        if math.isfinite(noise_med) and noise_med > 0:
            seg = noise_bin[b:e + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                nr = float(np.median(seg) / noise_med)
        w0 = float(cmap(t0)) if cmap is not None else NAN
        w1 = float(cmap(t1)) if cmap is not None else NAN
        periods.append(DeadPeriod(t0, t1, n_seg, mu, p, kind, w0, w1, nr))
    dead_s = float(sum(p.duration_s for p in periods))
    n_dead = int(sum(p.n_obs for p in periods))
    live_s = max(dur - dead_s, 0.0)
    res.update(possible=True, bin_s=w, ref_rate_per_min=r_ref * 60.0, periods=periods, dead_s=dead_s,
               live_s=live_s, live_rate_per_min=(N - n_dead) / live_s * 60.0 if live_s > 0 else NAN)
    return res


def in_periods(t, periods: Sequence[DeadPeriod]) -> np.ndarray:
    t = np.asarray(t, float)
    m = np.zeros(t.shape, bool)
    for p in periods or []:
        m |= p.contains(t)
    return m


def live_selection(rows: Sequence[dict], duration_s: float, periods: Sequence[DeadPeriod],
                   times: np.ndarray | None = None) -> tuple[list[dict], float]:
    """Traces et durée « sur le temps actif » (périodes mortes retirées)."""
    rows = list(rows)
    if not periods:
        return rows, float(duration_s)
    if times is None:
        times, _clock = times_and_clock(rows, duration_s)
    mask = in_periods(times, periods)
    live = [r for r, m in zip(rows, mask) if not m]
    dead_s = sum(p.duration_s for p in periods)
    return live, max(float(duration_s) - dead_s, 0.0)


# =========================================================================== #
# Taux par tranche de temps
# =========================================================================== #
@dataclass
class TimeBin:
    index: int
    t0: float                    # horloge d'analyse (s)
    t1: float
    exposure_s: float            # durée analysée de la tranche
    counts: dict                 # par classe
    fines: int
    live_s: float = NAN          # hors périodes mortes
    live_counts: dict = field(default_factory=dict)
    live_fines: int = 0
    wall0: float = NAN           # horloge de la source (s)
    wall1: float = NAN
    noise: float = NAN           # bruit médian (journal)
    disturbed_frac: float = NAN  # fraction d'images perturbées (journal)
    dead_s: float = 0.0
    partial: bool = False        # tranche incomplète (fin de l'enregistrement)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    @property
    def live_total(self) -> int:
        return int(sum(self.live_counts.values()))

    def n(self, key: str | None = None, live: bool = False) -> int:
        if key in (None, "total"):
            return self.live_total if live else self.total
        if key == "fines":
            return self.live_fines if live else self.fines
        return int((self.live_counts if live else self.counts).get(key, 0))

    def rate(self, key: str | None = None, live: bool = False) -> Rate:
        return rate(self.n(key, live), self.live_s if live else self.exposure_s)

    @property
    def dead(self) -> bool:
        return self.dead_s >= 0.5 * self.exposure_s if self.exposure_s > 0 else False


def choose_bin_s(duration_s: float) -> float:
    """Largeur des tranches : 1 h (≥ 3 h), 10 min (≥ 30 min), 1 min (≥ 5 min), sinon ~ 6 tranches."""
    d = _fnum(duration_s)
    if not (math.isfinite(d) and d > 0):
        return 60.0
    if d >= 3 * 3600:
        return 3600.0
    if d >= 1800:
        return 600.0
    if d >= 300:
        return 60.0
    for b in (30.0, 20.0, 15.0, 10.0):
        if d / b >= 6:
            return b
    return max(d / 6.0, 1.0)


def time_bins(rows: Sequence[dict], duration_s: float, bin_s: float | None = None,
              intervals: Sequence[dict] | None = None, periods: Sequence[DeadPeriod] = (),
              use_family: bool | None = None, cmap: ClockMap | None = None,
              times: np.ndarray | None = None) -> list[TimeBin]:
    """Comptages par tranche de temps (horloge d'analyse) et par classe.

    Les comptages viennent des traces (classe finale, corrections manuelles comprises) ;
    sans instants exploitables, ils viennent du journal du détecteur. Le journal fournit
    aussi le bruit et la fraction d'images perturbées de chaque tranche.
    """
    rows = list(rows)
    dur = _fnum(duration_s)
    if not (math.isfinite(dur) and dur > 0):
        return []
    b = _fnum(bin_s) if bin_s is not None else NAN
    if not (b > 0):
        b = choose_bin_s(dur)
    if use_family is None:
        use_family = has_family(rows)
    nb = max(1, int(math.ceil(dur / b - 1e-9)))
    edges = np.minimum(np.arange(nb + 1, dtype=float) * b, dur)
    edges[-1] = dur
    if times is None:
        times, _clock = times_and_clock(rows, dur)
    times = np.clip(np.asarray(times, float), 0.0, dur)
    cls = np.array([str(r.get("classe") or "incertain") for r in rows], dtype=object)
    fine = np.array([is_fine(r, use_family) for r in rows], bool)
    dead_mask = in_periods(times, periods) if len(rows) else np.zeros(0, bool)
    idx = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, nb - 1) if len(rows) else np.zeros(0, int)
    from_timeline = (not rows) and bool(intervals)
    iv = list(intervals or [])
    iv_mid = np.array([(i["t0"] + i["t1"]) / 2 for i in iv]) if iv else np.zeros(0)
    iv_idx = np.clip(np.searchsorted(edges, iv_mid, side="right") - 1, 0, nb - 1) if iv else np.zeros(0, int)
    out = []
    for k in range(nb):
        t0, t1 = float(edges[k]), float(edges[k + 1])
        expo = t1 - t0
        sel = idx == k
        live_sel = sel & ~dead_mask
        if from_timeline:
            cnt: Counter = Counter()
            for i, j in zip(iv, iv_idx):
                if j == k:
                    cnt.update(i["counts"])
            counts = {c: int(cnt.get(c, 0)) for c in CLASSES}
            for c, v in cnt.items():
                if c not in counts:
                    counts[c] = int(v)
            fines = int(sum(cnt.get(c, 0) for c in THIN_CLASSES))
            live_counts, live_fines = dict(counts), fines
        else:
            counts = {c: int(np.sum(sel & (cls == c))) for c in CLASSES}
            extra = set(cls[sel]) - set(CLASSES)
            for c in extra:
                counts[str(c)] = int(np.sum(sel & (cls == c)))
            fines = int(np.sum(sel & fine))
            live_counts = {c: int(np.sum(live_sel & (cls == c))) for c in counts}
            live_fines = int(np.sum(live_sel & fine))
        dead_s = sum(max(0.0, min(t1, p.t1) - max(t0, p.t0)) for p in periods or [])
        noise, dist = NAN, NAN
        if iv:
            mine = [i for i, j in zip(iv, iv_idx) if j == k]
            nz = [i["noise"] for i in mine if math.isfinite(i["noise"])]
            noise = float(np.median(nz)) if nz else NAN
            fr = sum(i["frames"] for i in mine)
            dd = sum(i["disturbed"] for i in mine)
            dist = dd / (fr + dd) if (fr + dd) > 0 else NAN
        out.append(TimeBin(
            index=k, t0=t0, t1=t1, exposure_s=expo, counts=counts, fines=fines,
            live_s=max(expo - dead_s, 0.0), live_counts=live_counts, live_fines=live_fines,
            wall0=float(cmap(t0)) if cmap is not None else NAN,
            wall1=float(cmap(t1)) if cmap is not None else NAN,
            noise=noise, disturbed_frac=dist, dead_s=float(dead_s),
            partial=expo < 0.999 * b))
    return out


# =========================================================================== #
# Stabilité du taux
# =========================================================================== #
@dataclass
class StabilityTest:
    label: str                   # « total », « fines » ou une classe
    bin_s: float
    n_bins: int
    n_total: int
    chi2: float
    dof: int
    p_value: float
    method: str                  # « asymptotique » ou « Monte-Carlo »
    min_expected: float
    slope_rel_per_h: float       # dérive relative du taux (fraction par heure)
    slope_err: float
    p_trend: float
    excluded_s: float = 0.0      # temps retiré (périodes mortes)

    @property
    def index(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else NAN

    @property
    def constant_ok(self) -> bool:
        return self.p_value > 0.01

    @property
    def trend_significant(self) -> bool:
        return math.isfinite(self.p_trend) and self.p_trend < 0.01

    def verdict(self) -> str:
        if self.constant_ok:
            return "compatible avec un taux constant"
        return "le taux a varié de façon significative"


_STAB_MC = 4000
_STAB_SEED = 20261020


def stability_from_counts(n, e, tmid, label: str = "total", bin_s: float = NAN,
                          excluded_s: float = 0.0, seed: int = _STAB_SEED) -> StabilityTest | None:
    """Test du χ² d'un taux constant : n_i ~ Poisson(r·e_i).

    Statistique de Pearson avec les valeurs attendues E_i = N·e_i/Σe (conditionnelle au total) ;
    p-valeur asymptotique (χ² à k − 1 ddl) si tous les E_i ≥ 5, sinon Monte-Carlo exacte
    (tirages multinomiaux, graine fixe). La dérive est estimée par moindres carrés pondérés
    du taux en fonction du temps (poids e_i / r).
    """
    n = np.asarray(n, float)
    e = np.asarray(e, float)
    tm = np.asarray(tmid, float)
    ok = np.isfinite(n) & np.isfinite(e) & (e > 0)
    if math.isfinite(_fnum(bin_s)) and bin_s > 0:
        ok &= e >= 0.05 * bin_s
    n, e, tm = n[ok], e[ok], tm[ok]
    k = int(n.size)
    N = int(round(n.sum()))
    if k < 3 or N < 10:
        return None
    prob = e / e.sum()
    E = N * prob
    chi2 = float(np.sum((n - E) ** 2 / E))
    dof = k - 1
    min_e = float(E.min())
    if min_e >= 5:
        p = float(stats.chi2.sf(chi2, dof))
        method = "asymptotique"
    else:
        rng = np.random.default_rng(seed)
        sim = rng.multinomial(N, prob, size=_STAB_MC)
        chi_sim = np.sum((sim - E) ** 2 / E, axis=1)
        p = float((1 + np.sum(chi_sim >= chi2 - 1e-9)) / (1 + _STAB_MC))
        method = "Monte-Carlo"
    r_hat = N / e.sum()
    w = e / r_hat
    y = n / e
    tw = np.sum(w * tm) / np.sum(w)
    sxx = float(np.sum(w * (tm - tw) ** 2))
    if sxx > 0 and r_hat > 0:
        yw = np.sum(w * y) / np.sum(w)
        slope = float(np.sum(w * (tm - tw) * (y - yw)) / sxx)
        se = 1.0 / math.sqrt(sxx)
        rel, rel_err = slope * 3600.0 / r_hat, se * 3600.0 / r_hat
        p_tr = float(2 * stats.norm.sf(abs(slope / se)))
    else:
        rel, rel_err, p_tr = NAN, NAN, NAN
    return StabilityTest(label, float(bin_s) if math.isfinite(_fnum(bin_s)) else NAN, k, N, chi2, dof,
                         min(max(p, 0.0), 1.0), method, min_e, rel, rel_err, p_tr, float(excluded_s))


def stability_test(bins: Sequence[TimeBin], key: str | None = None, live: bool = True) -> StabilityTest | None:
    """Stabilité du taux d'une classe (ou du total / des traces fines) d'une tranche à l'autre."""
    bins = list(bins)
    if not bins:
        return None
    n = [b.n(key, live) for b in bins]
    e = [b.live_s if live else b.exposure_s for b in bins]
    tm = [(b.t0 + b.t1) / 2 for b in bins]
    bin_s = max(b.exposure_s for b in bins)
    excl = sum(b.dead_s for b in bins) if live else 0.0
    return stability_from_counts(n, e, tm, key or "total", bin_s, excl)


# =========================================================================== #
# Comptages corrigés des erreurs de classification (déconvolution)
# =========================================================================== #
@dataclass
class UnfoldResult:
    classes: list                # vraies classes estimées
    observed: dict               # comptages bruts par classe attribuée (dont « incertain »)
    estimate: dict               # estimation corrigée par vraie classe
    err: dict                    # écart-type (bootstrap)
    lo: dict                     # IC 68 % (16e centile)
    hi: dict                     # IC 68 % (84e centile)
    efficiency: dict             # P(trace acceptée | vraie classe), sur simulation
    recall: dict                 # P(bonne classe | vraie classe), sur simulation
    manual: dict                 # traces corrigées à la main (comptées telles quelles)
    n_iter: int
    n_boot: int
    n_matrix: int                # traces simulées de la matrice (vraies classes estimées)
    artefact_leak: float         # fraction des artefacts simulés acceptés comme traces
    unattributed: int = 0        # traces observées qu'aucune vraie classe ne produit (matrice)

    @property
    def total_estimate(self) -> float:
        return float(sum(self.estimate.values()))


def validate_confusion(conf) -> tuple[list, list, np.ndarray] | None:
    """Vérifie une matrice de confusion {labels_true, labels_pred, matrix} ; None si invalide."""
    if not isinstance(conf, dict):
        return None
    lt, lp, m = conf.get("labels_true"), conf.get("labels_pred"), conf.get("matrix")
    if not (isinstance(lt, (list, tuple)) and isinstance(lp, (list, tuple)) and isinstance(m, (list, tuple))):
        return None
    lt, lp = [str(x) for x in lt], [str(x) for x in lp]
    if not lt or not lp or len(m) != len(lt):
        return None
    try:
        M = np.array([[float(v) for v in row] for row in m], float)
    except (TypeError, ValueError):
        return None
    if M.shape != (len(lt), len(lp)) or not np.all(np.isfinite(M)) or np.any(M < 0):
        return None
    return lt, lp, M


def _ibu(n: np.ndarray, R: np.ndarray, eff: np.ndarray, n_iter: int, prior: np.ndarray | None = None) -> np.ndarray:
    """Déconvolution bayésienne itérative (D'Agostini, 1995).

    n : comptages observés (J) ; R[i, j] = P(observé j | vrai i) ; eff[i] = Σ_j R[i, j].
    """
    I = R.shape[0]
    p = np.full(I, 1.0 / I) if prior is None else np.asarray(prior, float) / np.sum(prior)
    mu = np.zeros(I)
    for _ in range(max(1, int(n_iter))):
        denom = p @ R
        safe = np.where(denom > 0, denom, 1.0)
        theta = (R * p[:, None]) / safe[None, :]
        theta[:, denom <= 0] = 0.0
        mu = (theta @ n) / np.maximum(eff, 1e-9)
        s = float(mu.sum())
        if s <= 0:
            break
        p = mu / s
    return mu


UNFOLD_ITER = 4
UNFOLD_BOOT = 500
UNFOLD_SEED = 1995
UNFOLD_MIN_EFF = 0.2


def unfold_counts(observed: dict, confusion: dict, n_iter: int = UNFOLD_ITER, n_boot: int = UNFOLD_BOOT,
                  seed: int = UNFOLD_SEED, manual: dict | None = None,
                  true_classes: Sequence[str] = REAL_CLASSES) -> UnfoldResult | None:
    """Estime les vrais nombres de traces de chaque type à partir des nombres observés.

    ``confusion`` : {"labels_true": [...], "labels_pred": [... + "incertain"], "matrix": [[...]]}
    (lignes = vraie classe, colonnes = classe attribuée sur le jeu de TEST simulé, seuil
    d'abstention appliqué). Seules les vraies classes physiques sont estimées : on suppose qu'il
    ne reste pas d'artefacts parmi les traces valides (la fraction d'artefacts simulés acceptés
    est indiquée dans ``artefact_leak``). Les traces attribuées « artefact » ne sont pas
    observées (rejetées) : l'efficacité eff_i = P(trace acceptée | vrai i) corrige cette perte.

    Incertitudes : bootstrap (comptages observés tirés selon Poisson, lignes de la matrice
    selon une loi multinomiale), graine fixe → résultat reproductible. Renvoie None si la
    matrice est absente, invalide ou ne couvre pas les classes observées.
    """
    v = validate_confusion(confusion)
    if v is None:
        return None
    lt, lp, M = v
    obs = {str(k): int(round(_fnum(c))) for k, c in (observed or {}).items() if _fnum(c) > 0}
    manual = {str(k): int(round(_fnum(c))) for k, c in (manual or {}).items() if _fnum(c) > 0}
    # « fine » (β ou μ, famille sûre) est une abstention du modèle : colonne « incertain » de la matrice
    if "fine" in obs and "fine" not in lp:
        obs["incertain"] = obs.get("incertain", 0) + obs.pop("fine")
    if "fine" in manual:
        manual.pop("fine")
    obs_cols = [j for j, c in enumerate(lp) if c in CLASSES]
    obs_labels = [lp[j] for j in obs_cols]
    if any(c not in obs_labels for c in obs):
        return None
    rows_idx = [i for i, c in enumerate(lt) if c in true_classes and M[i].sum() > 0]
    if not rows_idx or not obs_cols:
        return None
    classes = [lt[i] for i in rows_idx]
    T = M[rows_idx].sum(axis=1)
    P = M[rows_idx] / T[:, None]
    integer = bool(np.all(np.abs(M - np.round(M)) < 1e-6))
    n = np.array([obs.get(c, 0) for c in obs_labels], float)

    def solve(Pm: np.ndarray, nv: np.ndarray) -> np.ndarray:
        R = Pm[:, obs_cols]
        eff = np.maximum(R.sum(axis=1), UNFOLD_MIN_EFF)
        # A priori : proportions observées (+ 1 trace fictive par classe) → convergence rapide
        prior = np.array([nv[obs_labels.index(c)] if c in obs_labels else 0.0 for c in classes]) + 1.0
        return _ibu(nv, R, eff, n_iter, prior)

    man_vec = np.array([manual.get(c, 0) for c in classes], float)
    est = solve(P, n) + man_vec
    rng = np.random.default_rng(seed)
    boots = np.empty((max(int(n_boot), 0), len(classes)))
    for b in range(boots.shape[0]):
        nb = rng.poisson(n).astype(float)
        if integer:
            Mb = np.array([rng.multinomial(int(round(t)), P[i]) for i, t in enumerate(T)], float)
            Tb = Mb.sum(axis=1)
            Pb = Mb / np.where(Tb > 0, Tb, 1.0)[:, None]
        else:
            Pb = P
        boots[b] = solve(Pb, nb) + rng.poisson(man_vec)
    if boots.shape[0] >= 2:
        err = boots.std(axis=0, ddof=1)
        lo, hi = np.percentile(boots, 16, axis=0), np.percentile(boots, 84, axis=0)
    else:
        err = np.sqrt(np.maximum(est, 0.0))
        lo, hi = est - err, est + err
    R_obs = P[:, obs_cols]
    eff = R_obs.sum(axis=1)
    recall = {c: float(P[k, lp.index(c)]) if c in lp else NAN for k, c in enumerate(classes)}
    leak = NAN
    if "artefact" in lt:
        ia = lt.index("artefact")
        if M[ia].sum() > 0:
            leak = float(M[ia, obs_cols].sum() / M[ia].sum())
    denom = P[:, obs_cols].sum(axis=0)
    unattributed = int(sum(n[j] for j in range(len(obs_cols)) if denom[j] <= 0))
    return UnfoldResult(
        classes=classes, observed={c: obs.get(c, 0) for c in obs_labels},
        estimate={c: float(x) for c, x in zip(classes, est)},
        err={c: float(x) for c, x in zip(classes, err)},
        lo={c: float(x) for c, x in zip(classes, lo)},
        hi={c: float(x) for c, x in zip(classes, hi)},
        efficiency={c: float(x) for c, x in zip(classes, eff)}, recall=recall,
        manual={c: int(manual.get(c, 0)) for c in classes},
        n_iter=int(n_iter), n_boot=int(boots.shape[0]), n_matrix=int(round(float(T.sum()))),
        artefact_leak=leak, unattributed=unattributed)


def get_reference_confusion() -> dict | None:
    """Matrice de confusion de l'IA de référence (jeu de test simulé), ou None."""
    try:
        from ..detection import reference_model as rm
    except Exception:
        return None
    try:
        fn = getattr(rm, "reference_confusion", None)
        if callable(fn):
            c = fn()
            if validate_confusion(c) is not None:
                return c
        m = rm.ReferenceModel.load_default()
        if m is None:
            return None
        for src in (getattr(m, "info", None), getattr(m, "metrics", None)):
            c = (src or {}).get("confusion_test") if isinstance(src, dict) else None
            if validate_confusion(c) is not None:
                return c
    except Exception:
        return None
    return None


def get_reference_info() -> dict:
    """Résumé de l'IA de référence ({} si absente) : n_tracks_total, threshold, metrics…"""
    try:
        from ..detection.reference_model import ReferenceModel
        m = ReferenceModel.load_default()
        info = dict(m.info or {}) if m is not None else {}
        return info
    except Exception:
        return {}


def reference_threshold(default: float = 0.85) -> float:
    t = _fnum(get_reference_info().get("threshold"))
    return t if (math.isfinite(t) and 0 < t < 1) else default


# =========================================================================== #
# Confiance de la classification
# =========================================================================== #
def confidence_summary(rows: Sequence[dict], threshold: float = 0.85, events: Sequence | None = None) -> dict:
    """Distribution des confiances par classe, part des incertaines et pourquoi, accords règles/IA.

    Clés : threshold, n, n_classified, frac_confident (confiance ≥ seuil parmi les traces
    classées), frac_ia_confident, classes {cls: {n, median, q1, q3, frac_ge, n_ia, median_ia,
    frac_ia_ge}}, incertain {n, frac, by_family, competitors, capped, disagree, low_conf},
    agreement {kind, n, n_agree, frac} | None, manual {n, n_same} | None, has_ia.
    """
    rows = list(rows)
    thr = float(threshold)
    out: dict = {"threshold": thr, "n": len(rows), "classes": {}, "has_ia": False}
    conf_all, ia_all = [], []
    for c in CLASSES:
        rc = [r for r in rows if r.get("classe") == c]
        cv = np.array([_fnum(r.get("confiance")) for r in rc], float)
        cv = cv[np.isfinite(cv)]
        iv = np.array([_fnum(r.get("proba_ia")) for r in rc], float)
        iv = iv[np.isfinite(iv)]
        d = {"n": len(rc), "median": NAN, "q1": NAN, "q3": NAN, "frac_ge": NAN,
             "n_ia": int(iv.size), "median_ia": NAN, "frac_ia_ge": NAN}
        if cv.size:
            d.update(median=float(np.median(cv)), q1=float(np.percentile(cv, 25)),
                     q3=float(np.percentile(cv, 75)), frac_ge=float(np.mean(cv >= thr)))
        if iv.size:
            d.update(median_ia=float(np.median(iv)), frac_ia_ge=float(np.mean(iv >= thr)))
            out["has_ia"] = True
        out["classes"][c] = d
        if c != "incertain":
            conf_all += list(cv)
            ia_all += list(iv)
    n_cls = sum(out["classes"][c]["n"] for c in REAL_CLASSES)
    out["n_classified"] = n_cls
    out["frac_confident"] = float(np.mean(np.asarray(conf_all) >= thr)) if conf_all else NAN
    out["frac_ia_confident"] = float(np.mean(np.asarray(ia_all) >= thr)) if ia_all else NAN
    inc = [r for r in rows if r.get("classe") == "incertain"]
    by_fam = Counter((row_family(r) or "inconnue") for r in inc)
    comp = Counter(str(r.get("classe_ia")) for r in inc if r.get("classe_ia") not in (None, "", "incertain"))
    out["incertain"] = {"n": len(inc), "frac": len(inc) / len(rows) if rows else NAN,
                        "by_family": dict(by_fam), "competitors": dict(comp),
                        "capped": None, "disagree": None, "low_conf": None}
    out["agreement"] = None
    out["manual"] = None
    evs = [e for e in (events or []) if getattr(e, "valid", True)]
    if evs:
        try:
            inc_e = [e for e in evs if getattr(e, "final_cls", "") == "incertain"]
            ex = [getattr(e, "extra", {}) or {} for e in inc_e]
            if any("capped" in x or "agree" in x for x in ex):
                out["incertain"]["capped"] = sum(1 for x in ex if x.get("capped"))
                out["incertain"]["disagree"] = sum(1 for x in ex if x.get("agree") is False)
                out["incertain"]["low_conf"] = sum(1 for x in ex if not x.get("capped") and x.get("agree") is not False)
            ag = [bool((getattr(e, "extra", {}) or {}).get("agree")) for e in evs
                  if "agree" in (getattr(e, "extra", {}) or {}) and not getattr(e, "manual_cls", "")]
            if ag:
                out["agreement"] = {"kind": "regles_ia", "n": len(ag), "n_agree": int(sum(ag)),
                                    "frac": float(np.mean(ag))}
            man = [e for e in evs if getattr(e, "manual_cls", "")]
            if man:
                out["manual"] = {"n": len(man), "n_same": sum(1 for e in man if e.cls == e.manual_cls)}
        except Exception:
            pass
    if out["agreement"] is None:
        pairs = [(r.get("classe"), r.get("classe_ia")) for r in rows
                 if r.get("classe_ia") not in (None, "") and r.get("classe") in REAL_CLASSES
                 and not r.get("classe_manuelle")]
        if pairs:
            ag = [a == b for a, b in pairs]
            out["agreement"] = {"kind": "final_ia", "n": len(ag), "n_agree": int(sum(ag)),
                                "frac": float(np.mean(ag))}
    return out


# =========================================================================== #
# Analyse approfondie d'un enregistrement
# =========================================================================== #
def classification_mode(run, rows: Sequence[dict]) -> str:
    """« hybride », « ia », « regles » (d'après les traces, sinon la configuration)."""
    meth = [str(r.get("methode") or "") for r in rows if not r.get("classe_manuelle")]
    meth = [m for m in meth if m]
    if meth:
        hyb = sum(1 for m in meth if m.lower().startswith("hybride"))
        if hyb >= 0.5 * len(meth):
            return "hybride"
        if sum(1 for m in meth if "règle" in m.lower() or "regle" in m.lower()) >= 0.5 * len(meth):
            return "regles"
        return "ia" if any("ia" in m.lower() for m in meth) else "autre"
    cfg = getattr(run, "config", None)
    mode = ""
    if isinstance(cfg, dict):
        mode = str((cfg.get("classification") or {}).get("mode") or "")
    return mode or "regles"


def unfold_run(run, rows: Sequence[dict] | None = None, confusion="auto") -> tuple[UnfoldResult | None, str]:
    """Comptages corrigés d'un enregistrement, ou (None, raison en français)."""
    rows = list(run.table() if rows is None else rows)
    if not rows:
        return None, "aucune trace"
    mode = classification_mode(run, rows)
    if mode != "hybride":
        return None, ("la matrice de confusion de l'IA de référence ne décrit que la classification hybride "
                      "(règles + IA de référence) ; cet enregistrement a été classé autrement")
    conf = get_reference_confusion() if (isinstance(confusion, str) and confusion == "auto") else confusion
    if conf is None:
        return None, "matrice de confusion de l'IA de référence indisponible"
    auto = Counter(str(r.get("classe") or "incertain") for r in rows if not r.get("classe_manuelle"))
    manual = Counter(str(r.get("classe")) for r in rows if r.get("classe_manuelle"))
    res = unfold_counts(dict(auto), conf, manual=dict(manual))
    if res is None:
        return None, "matrice de confusion invalide ou ne couvrant pas toutes les classes observées"
    return res, ""


def deep_analysis(run, confusion="auto", bin_s: float | None = None, threshold: float | None = None,
                  rows: Sequence[dict] | None = None) -> dict:
    """Analyse approfondie d'un enregistrement (journal temporel facultatif).

    ``confusion`` : "auto" (matrice de l'IA de référence), un dict {labels_true, labels_pred, matrix}
    ou None (pas de comptages corrigés). ``rows`` : traces déjà extraites (évite un 2e ``run.table()``).

    Renvoie un dictionnaire :
      has_timeline, clock ("analyse" | "video"), intervals, clock_map (ClockMap | None),
      bin_s, bins (list[TimeBin]), stability {"total", "alpha", "fines"} (StabilityTest | None),
      dead (voir detect_dead_periods), live {"duration_s", "n", "rates" {classe|"fines"|"total": Rate}},
      noise (noise_summary | None), families (family_counts), confidence (confidence_summary),
      unfold (UnfoldResult | None), unfold_reason (str), mode (classification), use_family,
      duration_s, n.
    """
    rows = list(run.table() if rows is None else rows)
    dur = _fnum(getattr(run, "duration_s", 0.0))
    dur = dur if math.isfinite(dur) else 0.0
    timeline = getattr(run, "timeline", None) or []
    intervals = timeline_intervals(timeline if isinstance(timeline, (list, tuple)) else [])
    times, clock = times_and_clock(rows, dur)
    cmap = clock_map(intervals, rows if clock == "analyse" else None)
    use_family = has_family(rows)
    dead = detect_dead_periods(times, dur, intervals, cmap)
    periods = dead["periods"]
    b = bin_s if bin_s else choose_bin_s(dur)
    bins = time_bins(rows, dur, b, intervals, periods, use_family, cmap, times)
    stability = {k: stability_test(bins, None if k == "total" else k) for k in ("total", "alpha", "fines")}
    live_rows, live_s = live_selection(rows, dur, periods, times)
    live_rates = {c: rate(sum(1 for r in live_rows if r.get("classe") == c), live_s) for c in CLASSES}
    live_rates["fines"] = rate(sum(1 for r in live_rows if is_fine(r, use_family)), live_s)
    live_rates["total"] = rate(len(live_rows), live_s)
    thr = threshold if threshold is not None else reference_threshold()
    unfold, why = unfold_run(run, rows, confusion)
    return {
        "has_timeline": bool(intervals), "clock": clock, "intervals": intervals, "clock_map": cmap,
        "bin_s": float(b), "bins": bins, "stability": stability, "dead": dead,
        "live": {"duration_s": live_s, "n": len(live_rows), "rates": live_rates},
        "noise": noise_summary(intervals),
        "families": family_counts(rows, use_family), "use_family": use_family,
        "confidence": confidence_summary(rows, thr, getattr(run, "events", None)),
        "unfold": unfold, "unfold_reason": why, "mode": classification_mode(run, rows),
        "duration_s": dur, "n": len(rows),
    }
