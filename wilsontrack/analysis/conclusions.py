"""Conclusions scientifiques automatiques (en français) à partir des comptages.

Trois niveaux :
1. ``conclude_run``        : un enregistrement seul (taux, caractère aléatoire,
   énergie des alphas, isotropie des muons, qualité de la mesure) et, par défaut,
   l'analyse approfondie (stabilité du taux, périodes inactives, bruit, familles de
   traces, comptages corrigés, fiabilité de la classification, note de qualité) ;
2. ``conclude_comparison`` : échantillon (KCl, banane…) contre bruit de fond
   (rayons cosmiques + radon) → excès net, signification statistique, activité
   attendue du K-40 et efficacité apparente de détection ;
3. ``quality_score``       : note de qualité 0–100 d'un enregistrement, expliquée
   critère par critère, avec des conseils concrets pour le prochain enregistrement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import physics
from .stats import (CLASSES, REAL_CLASSES, choose_bin_s, common_family_mode, compare, deep_analysis,
                    detect_dead_periods, fines_label, fr_clock, fr_duration, fr_g, fr_int, fr_num, fr_p_eq, is_fine,
                    live_selection, stability_test, summarize, time_bins, timeline_intervals, times_and_clock)

NOMS = {c: physics.CLASS_INFO[c]["nom"] for c in CLASSES}
SYMB = {"alpha": "α", "beta": "β", "muon": "μ", "proton": "p", "fine": "β/μ", "incertain": "?", "artefact": "×"}
# (singulier, pluriel) pour les phrases du type « 3 alphas (α) »
NOMS_N = {"alpha": ("alpha (α)", "alphas (α)"), "beta": ("bêta / électron (β⁻)", "bêtas / électrons (β⁻)"),
          "muon": ("muon cosmique (μ)", "muons cosmiques (μ)"),
          "proton": ("proton / particule lourde", "protons / particules lourdes"),
          "incertain": ("trace incertaine", "traces incertaines")}


def _nom_n(c: str, n: int) -> str:
    sg, pl = NOMS_N.get(c, (NOMS.get(c, c).lower(), NOMS.get(c, c).lower()))
    return f"{n} {pl if n > 1 else sg}"


def activity_text(exp: physics.SampleActivity) -> str:
    """Activité attendue d'un échantillon (texte français, virgule décimale)."""
    return (f"{fr_g(exp.mass_g)} g d'échantillon → {fr_g(exp.k_mass_g)} g de K → activité ≈ "
            f"{fr_g(exp.activity_bq)} Bq ({fr_g(exp.beta_per_s)} β⁻/s, {fr_g(exp.gamma_per_s)} γ/s)")


@dataclass
class Conclusion:
    title: str
    text: str
    level: str = "info"          # "info" | "succes" | "attention"

    def as_markdown(self) -> str:
        icon = {"succes": "✅", "attention": "⚠️"}.get(self.level, "ℹ️")
        return f"**{icon} {self.title}** — {self.text}"


def _fmt_p(p: float) -> str:
    return fr_p_eq(p)


def _sigma_txt(sig: float) -> str:
    return f"{fr_num(sig, 1)} σ" if sig < 30 else "> 30 σ"


def _pct(x: float, d: int = 0) -> str:
    return f"{fr_num(100.0 * x, d)} %" if math.isfinite(x) else "—"


def conclude_run(run, deep: bool = True) -> tuple[dict, list[Conclusion]]:
    """Résumé statistique et conclusions d'un enregistrement.

    Avec ``deep=True`` (défaut), le résumé contient aussi ``s["deep"]`` (voir
    ``stats.deep_analysis``) et ``s["quality"]`` (``QualityScore``), et les conclusions de
    l'analyse approfondie sont ajoutées (la note de qualité juste après le bilan).
    """
    rows = run.table()
    s = summarize(rows, run.duration_s, run.roi_area_mm2)
    out: list[Conclusion] = []
    if not (run.duration_s > 0):
        out.append(Conclusion(
            "Durée analysée nulle",
            "Aucune image n'a pu être analysée (image fixe, trop peu d'images pour apprendre le fond, ou "
            "perturbation permanente) : aucune statistique n'est possible. Utilisez une vidéo ou une série "
            "d'au moins quelques dizaines d'images.", "attention"))
        return s, out
    dmin = run.duration_s / 60.0
    tot = s["total"]
    area = s["area_cm2"]

    parts = []
    for c in CLASSES:
        n = s["classes"][c]["rate"].n
        if n:
            parts.append(f"{_nom_n(c, n)} ({s['classes'][c]['rate'].fmt()})")
    dtxt = f"{fr_num(dmin, 1)} min" if dmin < 120 else fr_duration(run.duration_s)
    out.append(Conclusion(
        "Bilan des observations",
        f"En {dtxt} d'observation sur ≈ {fr_num(area, 0)} cm², {tot.n} traces ont été détectées "
        f"(taux global {tot.fmt()}). Détail : " + (" ; ".join(parts) if parts else "aucune trace") + ".",
        "info"))

    if not run.calibrated:
        out.append(Conclusion(
            "Échelle non calibrée",
            "Les longueurs en mm (et donc l'énergie des alphas) reposent sur un champ de vision supposé. "
            "Calibrez avec une règle filmée dans la chambre pour des mesures quantitatives.",
            "attention"))

    if tot.n < 20:
        out.append(Conclusion(
            "Statistique limitée",
            f"Seulement {tot.n} traces : l'incertitude relative est d'environ {100 / math.sqrt(max(tot.n, 1)):.0f} %. "
            "Allongez l'enregistrement (≥ 10 min recommandé) pour des conclusions solides.",
            "attention"))

    d = s["dispersion"]
    if d is not None:
        if d.poisson_ok:
            out.append(Conclusion(
                "Caractère aléatoire (loi de Poisson)",
                f"Comptages par tranches de {fr_num(d.bin_s, 0)} s : moyenne {fr_num(d.mean)}, variance "
                f"{fr_num(d.var)} (rapport {fr_num(d.index)} ≈ 1, {_fmt_p(d.p_value)}). Les traces arrivent de "
                "façon indépendante et aléatoire, comme attendu pour des désintégrations radioactives et des "
                "rayons cosmiques.",
                "succes"))
        else:
            out.append(Conclusion(
                "Écart à la loi de Poisson",
                f"Rapport variance/moyenne = {fr_num(d.index)} ({_fmt_p(d.p_value)}). Les comptages ne sont pas "
                "stationnaires : conditions de la chambre qui ont changé (sursaturation, glace sèche, éclairage) "
                "ou perturbations. Vérifiez la courbe du taux en fonction du temps.",
                "attention"))

    ia = s["interarrival"]
    if ia is not None:
        mu, p = ia
        verdict = "compatible" if p > 0.01 else "non compatible"
        out.append(Conclusion(
            "Temps entre deux traces",
            f"Intervalle moyen {fr_num(mu, 1)} s ; distribution {verdict} avec une loi exponentielle "
            f"(test de Kolmogorov-Smirnov/Lilliefors, {_fmt_p(p)}).",
            "succes" if p > 0.01 else "attention"))

    na = s["classes"]["alpha"]["rate"].n
    if na:
        em, ee, ne = s["alpha_energy"]
        ph = run.config.get("physics", {}) if isinstance(run.config, dict) else {}
        temp = ph.get("chamber_temp_c", -25.0)
        depth = ph.get("sensitive_depth_mm", 8.0)
        L, _Le, _ = s["classes"]["alpha"]["length"]
        if ne >= 1 and math.isfinite(em) and run.calibrated:
            unc = f" ± {fr_num(ee)}" if math.isfinite(ee) else ""
            try:                                      # portées Po-210 (5,30 MeV) → Po-212 (8,78 MeV)
                r_lo = 10 * physics.alpha_range_cm(5.304, temp, ph.get("pressure_kpa", 101.325))
                r_hi = 10 * physics.alpha_range_cm(8.785, temp, ph.get("pressure_kpa", 101.325))
                rng_txt = f"{fr_num(r_lo, 0)} à {fr_num(r_hi, 0)} mm"
            except Exception:
                rng_txt = "35 à 75 mm"
            txt = (f"{na} alphas ({s['classes']['alpha']['rate'].fmt()}), longueur projetée moyenne {fr_num(L, 1)} mm. "
                   f"Vue de dessus, on ne mesure que la projection horizontale de la partie de la trace située "
                   f"dans la couche sensible (≈ {fr_num(depth, 0)} mm d'épaisseur, alors qu'une alpha parcourt "
                   f"{rng_txt} dans l'air à {fr_num(temp, 0)} °C selon l'émetteur) : beaucoup de traces sont raccourcies. "
                   f"L'énergie déduite de la longueur (relation portée-énergie de Geiger) est donc une énergie "
                   f"MINIMALE (moyenne {fr_num(em)}{unc} MeV).")
            e_edge, n_e = s.get("alpha_energy_edge", (float("nan"), ne))
            if n_e >= 10 and math.isfinite(e_edge):
                name, e_ref = physics.closest_alpha_emitter(e_edge)
                ok = abs(e_edge - e_ref) < 0.5
                txt += (f" Les traces les plus longues (entièrement contenues) estiment le mieux l'énergie : "
                        f"{fr_num(e_edge)} MeV (90e centile)"
                        + (f", compatible avec {name} ({fr_num(e_ref)} MeV)." if ok
                           else ", sans émetteur identifiable avec certitude."))
            else:
                txt += " Trop peu d'alphas (< 10) pour identifier l'émetteur."
            txt += (" Les alphas observées sans source proviennent vraisemblablement du radon-222 de l'air "
                    "et de ses descendants (Po-218, Po-214).")
            out.append(Conclusion("Particules alpha", txt, "info"))
        else:
            why = ("Calibrez l'échelle pour estimer leur énergie." if not run.calibrated else
                   "Leur énergie n'a pas pu être estimée (longueurs non exploitables).")
            out.append(Conclusion(
                "Particules alpha",
                f"{na} alphas détectées ({s['classes']['alpha']['rate'].fmt()}), longueur projetée moyenne "
                f"{fr_num(L, 1)} mm. {why} Sans source, elles proviennent "
                "vraisemblablement du radon-222 de l'air et de ses descendants.", "info"))

    nm = s["classes"]["muon"]["rate"].n
    if nm:
        iso = s["muon_isotropy"]
        txt = (f"{nm} traces fines et rectilignes ({s['classes']['muon']['rate'].fmt()}, "
               f"{fr_num(s['classes']['muon']['per_cm2_min'], 3)} /cm²/min). Le flux de muons au sol est "
               f"≈ {fr_g(physics.MUON_FLUX_PER_CM2_MIN)} /cm²/min, mais la plupart arrivent presque à la verticale : "
               "vue de dessus, leur trace ne mesure que ≈ épaisseur de la couche × tan θ (quelques mm), et les "
               "traces très fines sont souvent trop pâles pour être détectées : on en compte donc bien moins que "
               "ce flux. Seuls les muons très inclinés laissent de longues traces. Des électrons énergétiques "
               "(> 1 MeV) peuvent aussi paraître rectilignes.")
        if iso is not None:
            R, p = iso
            txt += (" Orientations " + ("isotropes" if p > 0.05 else "NON isotropes") +
                    f" dans le plan (test de Rayleigh, {_fmt_p(p)}).")
        out.append(Conclusion("Muons cosmiques", txt, "info"))

    nb = s["classes"]["beta"]["rate"].n
    if nb:
        out.append(Conclusion(
            "Électrons (bêta)",
            f"{nb} traces fines et sinueuses ({s['classes']['beta']['rate'].fmt()}). Sans source, elles viennent "
            "surtout d'électrons mis en mouvement par les rayons gamma ambiants (effet Compton) et d'électrons "
            "secondaires (rayons delta) produits par les rayons cosmiques.",
            "info"))

    if deep:
        try:
            dd = deep_analysis(run, rows=rows)
            q = quality_score(run, s, dd)
            s["deep"], s["quality"] = dd, q
            first, rest = conclude_deep(run, s, dd, q)
            out[1:1] = first
            out += rest
        except Exception as ex:                      # l'analyse de base reste disponible
            out.append(Conclusion(
                "Analyse approfondie indisponible",
                f"L'analyse approfondie de l'enregistrement n'a pas pu être calculée ({type(ex).__name__} : {ex}).",
                "attention"))
    return s, out


# =========================================================================== #
# Note de qualité d'un enregistrement
# =========================================================================== #
@dataclass
class QualityComponent:
    key: str
    label: str
    weight: float
    value: float | None          # 0..1 ; None = non évalué (donnée absente)
    measured: str
    comment: str
    advice: str = ""

    @property
    def points(self) -> float | None:
        return None if self.value is None else self.weight * self.value


@dataclass
class QualityScore:
    score: int                   # 0..100 (sur les critères évalués)
    grade: str                   # « excellente », « bonne », « passable », « insuffisante »
    components: list = field(default_factory=list)
    advice: list = field(default_factory=list)
    note: str = ""               # ex. plafonnement pour un enregistrement trop court

    @property
    def n_evaluated(self) -> int:
        return sum(1 for c in self.components if c.value is not None)

    @property
    def max_points(self) -> float:
        return float(sum(c.weight for c in self.components if c.value is not None))

    def summary(self) -> str:
        parts = [f"{c.label.lower()} {fr_num(c.points, 0)}/{fr_num(c.weight, 0)}"
                 for c in self.components if c.value is not None]
        return ", ".join(parts)


QUALITY_WEIGHTS = {"duree": 25, "statistique": 15, "stabilite": 15, "bruit": 10, "perturbations": 10,
                   "calibration": 10, "incertaines": 10, "alphas": 5}
QUALITY_FULL_DURATION_MIN = 120.0     # durée donnant tous les points (échelle logarithmique depuis 1 min)
QUALITY_SHORT_MIN = 10.0              # en dessous : note plafonnée à 59 (« passable » au mieux)


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0)) if math.isfinite(x) else 0.0


def _grade(score: float) -> str:
    if score >= 85:
        return "excellente"
    if score >= 70:
        return "bonne"
    if score >= 50:
        return "passable"
    return "insuffisante"


def quality_score(run, s: dict | None = None, deep: dict | None = None) -> QualityScore:
    """Note 0–100 de la qualité d'un enregistrement, critère par critère.

    Critères (poids, voir QUALITY_WEIGHTS) : durée (25 ; tous les points à 2 h, échelle
    logarithmique), nombre de traces (15), stabilité du taux (15), bruit de l'image et sa dérive
    (10), temps perturbé ou inactif (10), calibration (10), part de traces incertaines (10), alphas
    disponibles pour l'analyse d'énergie (5). Un critère dont la donnée manque (ex. journal du
    détecteur absent dans un ancien enregistrement) n'est pas évalué : la note est rapportée aux
    critères évalués. Moins de 10 min analysées : note plafonnée à 59 (``QualityScore.note``).
    """
    rows = run.table() if (s is None or deep is None) else None
    if s is None:
        s = summarize(rows, run.duration_s, run.roi_area_mm2)
    if deep is None:
        deep = deep_analysis(run, rows=rows)
    W = QUALITY_WEIGHTS
    comps: list[QualityComponent] = []
    dur = float(run.duration_s or 0.0)
    d_min = dur / 60.0
    v = _clip01(math.log10(d_min) / math.log10(QUALITY_FULL_DURATION_MIN)) if d_min >= 1 else 0.0
    comps.append(QualityComponent(
        "duree", "Durée", W["duree"], v, fr_duration(dur),
        "au moins 2 h : excellent pour la statistique" if v >= 1 else
        ("durée correcte" if v >= 0.75 else "enregistrement court"),
        "Enregistrez plus longtemps : au moins 30 min par condition, idéalement une nuit entière (option « filmer "
        "toute la nuit »). L'incertitude relative d'un taux diminue comme 1/√N."))
    N = int(s["total"].n)
    v = _clip01(math.log10(N / 10.0) / math.log10(40.0)) if N >= 10 else 0.0
    comps.append(QualityComponent(
        "statistique", "Nombre de traces", W["statistique"], v,
        f"{fr_int(N)} traces (incertitude relative ≈ {fr_num(100 / math.sqrt(max(N, 1)), 0)} %)",
        "statistique solide" if v >= 0.75 else "statistique limitée",
        "Augmentez le nombre de traces : enregistrement plus long ou zone analysée plus grande (sans inclure les "
        "bords flous de la chambre)."))
    st = (deep.get("stability") or {}).get("total")
    if st is None:
        comps.append(QualityComponent("stabilite", "Stabilité du taux", W["stabilite"], None,
                                      "non évaluée", "trop peu de tranches de temps ou de traces"))
    else:
        p = st.p_value
        v = 1.0 if p > 0.05 else (0.75 if p > 0.01 else (0.4 if p > 1e-3 else 0.1))
        comps.append(QualityComponent(
            "stabilite", "Stabilité du taux", W["stabilite"], v,
            f"χ² = {fr_num(st.chi2, 1)} pour {st.dof} ddl, {_fmt_p(p)}", st.verdict(),
            "Le taux a varié : laissez la chambre se mettre en régime 10 à 15 min avant de lancer l'enregistrement, "
            "vérifiez qu'il reste assez d'alcool (feutre bien imbibé) et de glace sèche pour toute la durée, et "
            "évitez les courants d'air et les vibrations."))
    nz = deep.get("noise")
    if not nz or not math.isfinite(nz.get("drift_ratio", float("nan"))):
        comps.append(QualityComponent("bruit", "Bruit de l'image", W["bruit"], None, "non évalué",
                                      "journal du détecteur absent (enregistrement d'une version antérieure)"))
    else:
        r = nz["drift_ratio"]
        rs = max(r, 1.0 / r) if r > 0 else float("inf")
        f_drift = _clip01(1.0 - (rs - 1.2) / 1.3)
        f_spike = 1.0 - 0.5 * _clip01(nz["spike_frac"] / 0.2)
        v = f_drift * f_spike
        comps.append(QualityComponent(
            "bruit", "Bruit de l'image", W["bruit"], v,
            f"dérive ×{fr_num(r)} ; {nz['n_spikes']} pointe(s) de bruit",
            "bruit stable" if v >= 0.75 else "bruit variable",
            "Le bruit de l'image a changé pendant l'enregistrement : désactivez le gain et l'exposition automatiques "
            "de la caméra, essuyez la buée du couvercle et gardez un éclairage latéral constant."))
    dead = deep.get("dead") or {}
    unan = float("nan")
    if nz:
        unan = nz.get("unanalysed_frac", float("nan"))
        if not math.isfinite(unan):
            unan = nz.get("disturbed_frac", float("nan"))
    dead_frac = (dead.get("dead_s", 0.0) / dur) if (dur > 0 and dead.get("possible")) else float("nan")
    if not math.isfinite(unan) and not math.isfinite(dead_frac):
        comps.append(QualityComponent("perturbations", "Temps perturbé ou inactif", W["perturbations"], None,
                                      "non évalué", "journal absent et enregistrement trop court pour rechercher "
                                      "des périodes inactives"))
    else:
        u = unan if math.isfinite(unan) else 0.0
        df = dead_frac if math.isfinite(dead_frac) else 0.0
        lost = u + (1.0 - u) * df
        v = 1.0 - _clip01(lost / 0.3)
        meas = f"{_pct(lost, 1)} du temps perdu"
        if math.isfinite(unan) and math.isfinite(dead_frac):
            meas += f" (perturbé ou non analysé : {_pct(unan, 1)} ; chambre inactive : {_pct(dead_frac, 1)})"
        elif math.isfinite(dead_frac):
            meas += " (périodes inactives seulement ; temps perturbé inconnu)"
        comps.append(QualityComponent(
            "perturbations", "Temps perturbé ou inactif", W["perturbations"], v, meas,
            "peu de temps perdu" if v >= 0.75 else "beaucoup de temps perdu",
            "Une partie du temps a été perdue (lumière allumée, caméra bougée, chambre inactive) : fixez solidement "
            "la caméra, gardez la pièce dans le noir et surveillez l'alcool et la glace sèche."))
    cal = bool(getattr(run, "calibrated", False))
    comps.append(QualityComponent(
        "calibration", "Calibration", W["calibration"], 1.0 if cal else 0.0,
        "échelle calibrée" if cal else "échelle supposée (non calibrée)",
        "longueurs en mm fiables" if cal else "longueurs et énergies approximatives",
        "Filmez une règle posée au fond de la chambre et calibrez l'échelle (px/mm) : les longueurs et "
        "l'énergie des alphas deviennent quantitatives."))
    if N == 0:
        comps.append(QualityComponent("incertaines", "Traces incertaines", W["incertaines"], None,
                                      "non évaluée", "aucune trace"))
    else:
        n_inc = s["classes"]["incertain"]["rate"].n
        frac = n_inc / N
        v = 1.0 - _clip01((frac - 0.05) / 0.30)
        fam = deep.get("families") or {}
        com = "peu de traces incertaines" if v >= 0.75 else "beaucoup de traces incertaines"
        if fam.get("mode") == "famille" and n_inc and fam.get("fines_incertaines", 0) >= 0.5 * n_inc:
            com += " (surtout des traces fines : ambiguïté physique β/μ)"
        comps.append(QualityComponent(
            "incertaines", "Traces incertaines", W["incertaines"], v, f"{n_inc} sur {fr_int(N)} ({_pct(frac)})", com,
            "Beaucoup de traces incertaines : améliorez la mise au point et le contraste (éclairage latéral rasant, "
            "fond noir mat, caméra bien perpendiculaire), puis vérifiez quelques traces à la main."))
    na = s["classes"]["alpha"]["rate"].n
    v = _clip01(na / 30.0)
    comps.append(QualityComponent(
        "alphas", "Alphas pour l'énergie", W["alphas"], v, f"{na} alphas",
        "assez d'alphas pour identifier l'émetteur" if na >= 30 else
        ("analyse d'énergie possible mais imprécise" if na >= 10 else "trop peu d'alphas pour l'énergie"),
        "Au moins 30 alphas sont nécessaires pour identifier l'émetteur par l'énergie : allongez l'enregistrement "
        "(sans source, les alphas viennent du radon de l'air, quelques-unes par minute)."))
    ev = [c for c in comps if c.value is not None]
    tot_w = sum(c.weight for c in ev)
    score = int(round(100.0 * sum(c.points for c in ev) / tot_w)) if tot_w > 0 else 0
    capped = d_min < QUALITY_SHORT_MIN and score > 59
    if capped:                                   # trop court pour une conclusion solide
        score = 59
    lost = sorted((c for c in ev if c.value < 0.75 and c.advice), key=lambda c: -(c.weight - c.points))
    advice = [c.advice for c in lost]
    man = (deep.get("confidence") or {}).get("manual")
    if not man or man.get("n", 0) < 50:
        advice.append("Pour mesurer la fiabilité réelle de la classification sur VOTRE chambre, vérifiez à la main "
                      "au moins 50 traces tirées au hasard (correction manuelle dans l'application) : c'est la seule "
                      "mesure de l'exactitude sur de vraies vidéos.")
    note = (f"Note plafonnée à 59 : moins de {fr_num(QUALITY_SHORT_MIN, 0)} min analysées, c'est trop court pour "
            "une conclusion solide." if capped else "")
    return QualityScore(score, _grade(score), comps, advice, note)


# =========================================================================== #
# Conclusions de l'analyse approfondie
# =========================================================================== #
def _period_txt(p, cmap_source: str = "") -> str:
    txt = f"{fr_clock(p.t0)} – {fr_clock(p.t1)}"
    if math.isfinite(p.wall0) and math.isfinite(p.wall1):
        txt += f" (vidéo : {fr_clock(p.wall0)} – {fr_clock(p.wall1)})"
    txt += f" : {p.n_obs} traces au lieu de ≈ {fr_num(p.n_expected, 0)}"
    if p.kind == "dégradée":
        txt += (f", bruit ×{fr_num(p.noise_ratio, 1)}" if math.isfinite(p.noise_ratio) else "") + " (dégradée)"
    return txt


def _trend_txt(st, duration_s: float = float("nan")) -> str:
    """Dérive du taux : en % par heure (≥ 2 h), sinon en % sur toute la durée ; rien si inexploitable."""
    if not (math.isfinite(st.slope_rel_per_h) and math.isfinite(st.slope_err)):
        return ""
    if not (math.isfinite(duration_s) and duration_s >= 1800):       # trop court : dérive sans intérêt
        return ""
    sig = "significative" if st.trend_significant else "non significative"
    if math.isfinite(duration_s) and duration_s >= 2 * 3600:
        k, unit = 1.0, "par heure"
    else:
        k, unit = (duration_s / 3600.0 if math.isfinite(duration_s) else float("nan")), "du début à la fin"
    if not math.isfinite(k) or 100 * k * st.slope_err > 50:       # incertitude trop grande : sans intérêt
        return ""
    return (f" Dérive estimée : {fr_num(100 * k * st.slope_rel_per_h, 1, sign=True)} ± "
            f"{fr_num(100 * k * st.slope_err, 1)} % {unit} ({sig}, {_fmt_p(st.p_trend)}).")


def conclude_deep(run, s: dict, deep: dict, q: QualityScore | None = None) -> tuple[list[Conclusion], list[Conclusion]]:
    """Conclusions de l'analyse approfondie : (à placer après le bilan, à placer à la fin)."""
    first: list[Conclusion] = []
    rest: list[Conclusion] = []
    if q is not None:
        lvl = "succes" if q.score >= 70 else ("info" if q.score >= 50 else "attention")
        n_ev, n_all = q.n_evaluated, len(q.components)
        txt = (f"Note calculée sur {n_ev} critère{'s' if n_ev > 1 else ''} évalué{'s' if n_ev > 1 else ''} "
               f"(sur {n_all}) : {q.summary()}.")
        if q.note:
            txt += " " + q.note
        if q.advice:
            a0 = q.advice[0]
            txt += " Pour le prochain enregistrement : " + (a0[:1].lower() + a0[1:])
        first.append(Conclusion(f"Qualité de l'enregistrement : {q.score}/100 ({q.grade})", txt, lvl))

    st = (deep.get("stability") or {}).get("total")
    if st is not None:
        bin_txt = fr_duration(st.bin_s)
        excl = " (périodes inactives exclues)" if st.excluded_s > 0 else ""
        base = (f"Taux total par tranches de {bin_txt}{excl} : χ² = {fr_num(st.chi2, 1)} pour {st.dof} degrés de "
                f"liberté ({_fmt_p(st.p_value)}, p-valeur {st.method}).")
        dur = float(deep.get("duration_s", float("nan")))
        if st.constant_ok:
            rest.append(Conclusion(
                "Taux stable au cours du temps",
                base + " Les variations d'une tranche à l'autre sont compatibles avec les fluctuations aléatoires "
                "d'un taux constant : la chambre est restée dans les mêmes conditions." + _trend_txt(st, dur),
                "succes"))
        else:
            hint = ""
            sa = deep["stability"].get("alpha")
            sf = deep["stability"].get("fines")
            if sa is not None and sf is not None:
                if not sa.constant_ok and sf.constant_ok:
                    hint = (" Seules les alphas varient : cela peut venir du radon de la pièce, dont la concentration "
                            "change naturellement au cours de la journée (ventilation, chauffage, portes ouvertes).")
                elif not sf.constant_ok and sa.constant_ok:
                    hint = (" Ce sont surtout les traces fines qui varient : la détection des traces pâles a changé "
                            "(sursaturation, alcool, glace sèche, buée, éclairage, bruit de la caméra).")
                elif not sf.constant_ok:
                    hint = (" Les alphas et les traces fines varient ensemble : la sensibilité de toute la chambre a "
                            "probablement changé (alcool, glace sèche, température, buée).")
            rest.append(Conclusion(
                "Taux variable au cours du temps",
                base + " Le taux a varié plus que ne le permettent les fluctuations aléatoires." + _trend_txt(st, dur)
                + hint + " Consultez le graphique des taux par tranche de temps ; pour une comparaison "
                "échantillon / fond, n'utilisez que des périodes où la chambre était stable.",
                "attention"))

    dead = deep.get("dead") or {}
    periods = dead.get("periods") or []
    if periods:
        lr = deep["live"]["rates"]["total"]
        lst = " ; ".join(_period_txt(p) for p in periods[:5]) + (" ; …" if len(periods) > 5 else "")
        rest.append(Conclusion(
            "Périodes d'inactivité de la chambre",
            f"{len(periods)} période{'s' if len(periods) > 1 else ''} ({fr_duration(dead['dead_s'])} au total) où le "
            f"taux est tombé très au-dessous de la normale ({fr_num(dead['ref_rate_per_min'])} /min), ce qui est "
            f"très improbable par hasard (p < 0,001, corrigé pour les tests multiples) : {lst}. Elles sont exclues "
            f"des taux « sur le temps actif » "
            f"(taux total {lr.fmt()} sur {fr_duration(deep['live']['duration_s'])}), jamais des comptages bruts. "
            "Vérifiez la vidéo à ces moments (alcool épuisé, glace sèche finie, buée, éclairage). Retirer une période "
            "parce qu'elle compte peu de traces augmente légèrement le taux moyen : ce choix n'est justifié que si la "
            "chambre était réellement inactive.",
            "attention"))
    elif dead.get("possible"):
        rest.append(Conclusion(
            "Aucune période d'inactivité",
            f"La chambre a détecté des traces tout au long de l'enregistrement : le taux n'est jamais tombé sous "
            f"25 % du taux habituel (tranches de {fr_duration(dead['bin_s'])}).",
            "succes"))

    nz = deep.get("noise")
    if nz:
        r = nz.get("drift_ratio", float("nan"))
        notable = ((math.isfinite(r) and (r >= 1.5 or r <= 1 / 1.5)) or nz.get("n_spikes", 0) > 0
                   or (math.isfinite(nz.get("disturbed_frac", float("nan"))) and nz["disturbed_frac"] > 0.05)
                   or (math.isfinite(nz.get("unanalysed_frac", float("nan"))) and nz["unanalysed_frac"] > 0.10))
        if notable:
            bits = []
            if math.isfinite(r):
                bits.append(f"bruit de l'image ×{fr_num(r)} entre le début et la fin")
            if nz.get("n_spikes"):
                bits.append(f"{nz['n_spikes']} intervalle(s) avec un bruit plus de 2 fois supérieur à la normale")
            if math.isfinite(nz.get("disturbed_frac", float("nan"))):
                bits.append(f"{_pct(nz['disturbed_frac'], 1)} des images perturbées (lumière, mouvement)")
            if math.isfinite(nz.get("unanalysed_frac", float("nan"))):
                bits.append(f"{_pct(nz['unanalysed_frac'], 1)} du temps non analysé")
            rest.append(Conclusion(
                "Bruit de l'image et perturbations",
                "Pendant l'enregistrement : " + " ; ".join(bits) + ". Un bruit qui augmente rend les traces fines "
                "plus difficiles à détecter et peut faire baisser leur taux sans changement physique.",
                "attention"))

    fam = deep.get("families") or {}
    if fam.get("mode") == "famille" and fam.get("fines_incertaines", 0) > 0:
        nfi, nf = fam["fines_incertaines"], fam["fines"]
        rf = (s.get("families") or {}).get("rate")
        rtxt = f" (taux des traces fines : {rf.fmt()})" if rf is not None and math.isfinite(rf.per_min) else ""
        rest.append(Conclusion(
            "Traces fines : ambiguïté β/μ",
            f"{nfi} des {fr_int(nf)} traces fines{rtxt} ({_pct(nfi / nf if nf else float('nan'))}) sont classées "
            "incertaines. "
            "Vue de dessus, une trace fine, courte et droite peut être un muon presque vertical ou un électron "
            "énergétique (> 1 MeV) : c'est une ambiguïté physique réelle, pas une erreur du programme. Pour la "
            "recherche de radioactivité, toutes les traces fines (incertaines comprises) sont regroupées, ce qui "
            "conserve toute la statistique.",
            "info"))

    uf = deep.get("unfold")
    if uf is not None:
        parts = [f"{SYMB[c]} ≈ {fr_num(uf.estimate[c], 0)} ± {fr_num(uf.err[c], 0)}" for c in uf.classes]
        raw = [f"{SYMB.get(c, c)} {n}" for c, n in uf.observed.items() if n]
        rest.append(Conclusion(
            "Estimation corrigée des comptages (d'après la validation sur simulation)",
            f"En tenant compte des erreurs de classification mesurées sur {fr_int(uf.n_matrix)} traces simulées dont "
            f"la nature est connue, les nombres de traces de chaque type sont estimés à : {', '.join(parts)} "
            f"(comptages bruts : {', '.join(raw) or 'aucun'}). Cette correction dépend du modèle (une simulation ne "
            "reproduit jamais parfaitement une vraie chambre) : elle ne remplace pas les comptages bruts, elle "
            "indique l'ordre de grandeur des erreurs de classification.",
            "info"))

    cf = deep.get("confidence") or {}
    if cf.get("n_classified"):
        thr = cf["threshold"]
        per = [f"{SYMB[c]} {_pct(cf['classes'][c]['frac_ge'])}" for c in REAL_CLASSES
               if cf["classes"][c]["n"] and math.isfinite(cf["classes"][c]["frac_ge"])]
        txt = (f"{_pct(cf['frac_confident'])} des traces classées ont une confiance d'au moins "
               f"{fr_num(100 * thr, 0)} % ({', '.join(per)}).")
        inc = cf.get("incertain") or {}
        if inc.get("n"):
            txt += f" {inc['n']} traces restent incertaines ({_pct(inc['frac'])})"
            bf = inc.get("by_family") or {}
            if bf.get("fine") or bf.get("lourde"):
                txt += f" : {bf.get('fine', 0)} fines, {bf.get('lourde', 0)} lourdes"
            comp = inc.get("competitors") or {}
            if comp:
                top = sorted(comp.items(), key=lambda kv: -kv[1])[:2]
                txt += " ; l'IA de référence y voyait surtout " + " et ".join(
                    f"{NOMS.get(k, k).lower()} ({v})" for k, v in top)
            txt += "."
        ag = cf.get("agreement")
        if ag and ag.get("n"):
            who = "règles physiques et IA de référence" if ag["kind"] == "regles_ia" else \
                "classification finale et IA de référence"
            txt += f" Accord {who} : {_pct(ag['frac'])} des {ag['n']} traces."
        man = cf.get("manual")
        if man and man.get("n"):
            txt += (f" Sur les {man['n']} traces corrigées à la main, la classification automatique était juste "
                    f"pour {man['n_same']} (échantillon non aléatoire : les traces douteuses sont plus souvent "
                    "vérifiées).")
        if cf.get("has_ia") and math.isfinite(cf.get("frac_ia_confident", float("nan"))):
            txt += (f" Probabilité de l'IA de référence d'au moins {fr_num(100 * thr, 0)} % pour "
                    f"{_pct(cf['frac_ia_confident'])} des traces classées. Cette probabilité est calibrée sur des "
                    "traces simulées ; la confiance finale combine les règles physiques et l'IA.")
        elif deep.get("mode") == "hybride":
            txt += (" La confiance finale combine les règles physiques et l'IA de référence, dont les probabilités "
                    "sont calibrées sur des traces simulées.")
        else:
            txt += (" Avec les règles physiques seules, la « confiance » est un score flou entre 0 et 1, pas une "
                    "probabilité calibrée.")
        txt += (" Sur votre chambre, la fiabilité réelle ne se mesure qu'en vérifiant à la main un échantillon de "
                "traces tirées au hasard.")
        rest.append(Conclusion("Fiabilité de la classification", txt, "info"))
    return first, rest


# =========================================================================== #
# Comparaison échantillon / fond
# =========================================================================== #
def expected_sample(run) -> physics.SampleActivity | None:
    frac = physics.SAMPLES.get(run.sample_type)
    if not frac or run.sample_mass_g <= 0:
        return None
    return physics.sample_activity(run.sample_mass_g, frac)


def _live_rows(run, exclude_dead: bool):
    """(traces, durée, détection des périodes inactives) d'un enregistrement pour la comparaison."""
    rows = run.table()
    dur = float(run.duration_s or 0.0)
    if not exclude_dead or not (dur > 0):
        return rows, dur, None
    times, _clock = times_and_clock(rows, dur)
    tl = getattr(run, "timeline", None) or []
    dead = detect_dead_periods(times, dur, timeline_intervals(tl if isinstance(tl, (list, tuple)) else []))
    if dead["periods"]:
        live, live_s = live_selection(rows, dur, dead["periods"], times)
        if live_s > 0:
            return live, live_s, dead
    return rows, dur, dead


def _stability_total(run, dead: dict | None):
    """Stabilité du taux total d'un enregistrement (périodes inactives exclues si détectées)."""
    rows = run.table()
    dur = float(run.duration_s or 0.0)
    if not (dur > 0) or len(rows) < 10:
        return None
    times, _clock = times_and_clock(rows, dur)
    periods = (dead or {}).get("periods") or []
    bins = time_bins(rows, dur, choose_bin_s(dur), periods=periods, times=times)
    return stability_test(bins, None, live=True)


def conclude_comparison(sample, background, exclude_dead: bool = True) -> tuple[dict, list[Conclusion]]:
    """Compare un enregistrement « échantillon » au « bruit de fond ».

    Traces fines : famille « fine » (incertaines fines comprises) si les deux enregistrements
    ont la colonne ``famille``, sinon β + μ (anciens résultats). Avec ``exclude_dead`` (défaut),
    les périodes où la chambre était inactive (voir ``stats.detect_dead_periods``) sont retirées
    des deux enregistrements avant la comparaison, et c'est signalé.
    """
    ts0, tb0 = sample.duration_s, background.duration_s
    if not (ts0 > 0 and tb0 > 0):
        comps = {c: compare(0, 0.0, 0, 0.0, c) for c in CLASSES + ["fines", "total"]}
        return comps, [Conclusion(
            "Comparaison impossible",
            "L'un des deux enregistrements a une durée analysée nulle : aucune comparaison statistique n'est "
            "possible.", "attention")]
    rs, ts, dead_s = _live_rows(sample, exclude_dead)
    rb, tb, dead_b = _live_rows(background, exclude_dead)
    use_family = common_family_mode(rs, rb)
    flabel = fines_label(use_family)
    # Surfaces analysées : le fond (uniforme) est ramené à la surface de l'échantillon
    As = float(getattr(sample, "roi_area_mm2", 0) or 0)
    Ab = float(getattr(background, "roi_area_mm2", 0) or 0)
    k = As / Ab if (As > 0 and Ab > 0 and math.isfinite(As / Ab)) else 1.0
    preds = {c: (lambda r, c=c: r.get("classe") == c) for c in CLASSES}
    preds["fines"] = lambda r: is_fine(r, use_family)      # particules au minimum d'ionisation
    preds["total"] = lambda r: True
    comps = {}
    counts_s, counts_b = {}, {}
    for c, pred in preds.items():
        ns = sum(1 for r in rs if pred(r))
        nb = sum(1 for r in rb if pred(r))
        counts_s[c], counts_b[c] = ns, nb
        comps[c] = compare(ns, ts, nb, tb, c, area_ratio=k)
    comps["fines"].label = flabel
    out: list[Conclusion] = []
    excl = [(lab, d) for lab, d in (("échantillon", dead_s), ("fond", dead_b)) if d and d.get("periods")]
    if excl:
        txt = "; ".join(f"{lab} : {len(d['periods'])} période(s), {fr_duration(d['dead_s'])} retirées"
                        for lab, d in excl)
        out.append(Conclusion(
            "Périodes inactives exclues de la comparaison",
            f"La chambre a cessé de détecter des traces pendant une partie du temps ({txt}). Ces périodes sont "
            f"retirées des comptages ET des durées avant la comparaison (durées utilisées : {fr_duration(ts)} pour "
            f"l'échantillon, {fr_duration(tb)} pour le fond). Vérifiez la vidéo : la comparaison n'a de sens que "
            "si la chambre était réellement inactive pendant ces périodes.",
            "attention"))
    if abs(k - 1.0) > 0.05:
        out.append(Conclusion(
            "Surfaces analysées différentes",
            f"Zone analysée : {fr_num(As / 100, 0)} cm² (échantillon) contre {fr_num(Ab / 100, 0)} cm² (fond). Le taux "
            f"du fond a été ramené à la surface de l'échantillon (×{fr_num(k)}) avant la comparaison. Pour une "
            "comparaison directe, utilisez la même zone analysée et la même calibration pour les deux enregistrements.",
            "attention"))
    ps, pb = float(getattr(sample, "px_per_mm", 0) or 0), float(getattr(background, "px_per_mm", 0) or 0)
    if bool(getattr(sample, "calibrated", False)) != bool(getattr(background, "calibrated", False)) or (
            ps > 0 and pb > 0 and abs(ps / pb - 1.0) > 0.05):
        out.append(Conclusion(
            "Calibrations différentes",
            "Les deux enregistrements n'ont pas la même échelle (px/mm) : les surfaces, les longueurs et la "
            "normalisation du fond sont moins fiables. Calibrez les deux avec la même règle.",
            "attention"))
    unstable = []
    for lab, run_, dead_ in (("échantillon", sample, dead_s), ("fond", background, dead_b)):
        try:
            stt = _stability_total(run_, dead_)
        except Exception:
            stt = None
        if stt is not None and not stt.constant_ok:
            unstable.append(f"{lab} ({_fmt_p(stt.p_value)})")
    if unstable:
        out.append(Conclusion(
            "Taux instable pendant un enregistrement",
            f"Le taux de traces n'est pas resté constant pendant : {', '.join(unstable)}. La comparaison suppose "
            "que la chambre était dans les mêmes conditions du début à la fin de chaque enregistrement : "
            "interprétez l'excès avec prudence et consultez les courbes de taux.",
            "attention"))
    label = sample.condition or "échantillon"
    beta = comps["beta"]
    fines = comps["fines"]
    tot = comps["total"]
    if not beta.significant and fines.significant:
        extra = (" Les traces fines classées incertaines (ambiguïté β/μ) sont incluses, ce qui conserve toute la "
                 "statistique." if use_family else "")
        out.append(Conclusion(
            f"Radioactivité détectée sur les traces fines ({label})",
            f"{flabel} : {fines.a.fmt()} avec l'échantillon contre {fines.b.fmt()} pour le fond, excès "
            f"{fr_num(fines.diff, sign=True)} ± {fr_num(fines.diff_err)} /min, {_sigma_txt(fines.sigma)} "
            f"({_fmt_p(fines.p_value)}). Les électrons énergétiques du K-40 (jusqu'à 1,31 MeV) sont presque "
            "rectilignes et peuvent être classés comme muons : l'ensemble des traces fines est l'indicateur le plus "
            "fiable." + extra,
            "succes"))
    if beta.significant:
        out.append(Conclusion(
            f"Radioactivité β détectée ({label})",
            f"Taux de traces β : {beta.a.fmt()} avec l'échantillon contre {beta.b.fmt()} pour le fond. "
            f"Excès net {fr_num(beta.diff, sign=True)} ± {fr_num(beta.diff_err)} /min (×{fr_num(beta.ratio)}), "
            f"signification {_sigma_txt(beta.sigma)} ({_fmt_p(beta.p_value)}). "
            "L'échantillon émet des électrons : cohérent avec la désintégration β⁻ du potassium-40.",
            "succes"))
    elif not fines.significant:
        out.append(Conclusion(
            f"Pas d'excès β significatif ({label})",
            f"Taux de traces β : {beta.a.fmt()} contre {beta.b.fmt()} pour le fond "
            f"(différence {fr_num(beta.diff, sign=True)} ± {fr_num(beta.diff_err)} /min, {_fmt_p(beta.p_value)}). "
            f"{flabel} : différence {fr_num(fines.diff, sign=True)} ± {fr_num(fines.diff_err)} /min "
            f"({_fmt_p(fines.p_value)}). La radioactivité de l'échantillon, si elle existe, est trop faible pour être "
            "distinguée du fond avec cette durée d'observation.",
            "attention"))
    if tot.significant and not beta.significant and not fines.significant:
        out.append(Conclusion(
            "Excès global",
            f"Le nombre total de traces augmente significativement ({tot.a.fmt()} contre {tot.b.fmt()}, "
            f"{_sigma_txt(tot.sigma)}), même si l'excès n'est pas attribuable clairement aux β.",
            "info"))
    # Contrôle : test BILATÉRAL (une baisse des alphas est aussi un signal d'alerte)
    al2 = compare(counts_s["alpha"], ts, counts_b["alpha"], tb, "alpha", area_ratio=k, alternative="two-sided")
    comps["alpha_bilateral"] = al2
    if al2.p_value > 0.01:
        out.append(Conclusion(
            "Contrôle : alphas inchangées",
            f"Taux d'alphas {al2.a.fmt()} contre {al2.b.fmt()} (test bilatéral, {_fmt_p(al2.p_value)}) : pas de "
            "variation significative, comme attendu (le K-40 n'émet pas de particules alpha). Ce contrôle soutient "
            "l'hypothèse que la chambre était dans les mêmes conditions pendant les deux enregistrements.",
            "succes"))
    elif al2.diff > 0:
        out.append(Conclusion(
            "Alphas en hausse",
            f"Taux d'alphas {al2.a.fmt()} contre {al2.b.fmt()} (test bilatéral, {_fmt_p(al2.p_value)}). Le K-40 "
            "n'émet pas d'alphas : cet écart vient probablement d'un changement du radon ambiant ou d'une "
            "contamination. Interprétez la comparaison β avec prudence.",
            "attention"))
    else:
        out.append(Conclusion(
            "Alphas en baisse",
            f"Taux d'alphas {al2.a.fmt()} contre {al2.b.fmt()} (test bilatéral, {_fmt_p(al2.p_value)}). Les alphas "
            "ont significativement diminué pendant l'enregistrement de l'échantillon : radon ambiant plus faible ou "
            "chambre moins sensible (sursaturation, glace sèche, éclairage). Les conditions n'étaient pas "
            "identiques : la comparaison β doit être interprétée avec prudence.",
            "attention"))
    exp = expected_sample(sample)
    if exp is not None:
        txt = activity_text(exp) + "."
        g = fines                                   # indicateur choisi a priori (traces fines)
        if exp.beta_per_s > 0 and math.isfinite(g.diff_err):
            kk = 100.0 / (60.0 * exp.beta_per_s)     # (/min) → % de l'activité β⁻ attendue
            if g.significant:
                txt += (f" Excès mesuré sur les {flabel[0].lower() + flabel[1:]} ≈ {fr_num(g.diff / 60.0, 3)} ± "
                        f"{fr_num(g.diff_err / 60.0, 3)} traces/s, soit une efficacité apparente de "
                        f"{fr_g(g.diff * kk, 2)} ± {fr_g(g.diff_err * kk, 2)} %. Elle est faible car la plupart des "
                        "électrons sont absorbés dans l'échantillon lui-même, partent vers le bas, ou traversent la "
                        "couche sensible trop verticalement.")
            else:
                ul = max(g.diff, 0.0) + 1.645 * g.diff_err   # limite supérieure unilatérale ≈ 95 %
                txt += (f" Aucun excès significatif sur les traces fines ({_fmt_p(g.p_value)}) : l'efficacité "
                        f"apparente de détection n'est pas mesurable ; elle est inférieure à ≈ {fr_g(ul * kk, 2)} % "
                        "(limite supérieure à 95 %).")
        out.append(Conclusion("Activité attendue (K-40)", txt, "info"))
    dur_min = min(ts, tb) / 60.0
    if dur_min < 10:
        out.append(Conclusion(
            "Durée d'observation",
            f"L'enregistrement le plus court dure {fr_num(dur_min, 1)} min. Pour détecter un faible excès, "
            "enregistrez au moins 10 à 20 min pour chaque condition, dans les mêmes conditions (glace sèche, "
            "alcool, éclairage).",
            "attention"))
    return comps, out


def conclusions_text(items: list[Conclusion]) -> str:
    return "\n\n".join(c.as_markdown() for c in items)
