"""Analyse approfondie : longs enregistrements (nuits entières), familles de traces, comptages
corrigés (déconvolution), note de qualité, rapport et graphiques.

Tous les tests utilisent de petits objets factices (aucune vidéo, aucun modèle réel) : la matrice
de confusion de l'IA de référence est simulée (monkeypatch de ``stats.get_reference_confusion``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from wilsontrack.analysis import conclusions as C
from wilsontrack.analysis import plots, report
from wilsontrack.analysis import stats as S

HOUR = 3600.0
METHOD = "Hybride (règles + IA de référence)"
RATES = {"alpha": 2.0, "beta": 6.0, "muon": 4.0, "proton": 0.05, "incertain": 0.8}   # traces / min


# --------------------------------------------------------------------------- #
# Fabrication d'enregistrements factices
# --------------------------------------------------------------------------- #
def make_rows(duration_s: float, seed: int = 0, factor=None, rates=None, family: bool = True,
              ia: bool = True, method: str = METHOD) -> list[dict]:
    """Traces poissonniennes ; ``factor(t)`` (≤ 1) module le taux (marche, période morte…)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c, r in (rates or RATES).items():
        n = rng.poisson(r / 60.0 * duration_s)
        t = rng.uniform(0.0, duration_s, n)
        if factor is not None:
            t = t[rng.uniform(size=t.size) < np.asarray(factor(t), float)]
        for ti in t:
            if c in ("alpha", "proton"):
                fam = "lourde"
            elif c in ("beta", "muon"):
                fam = "fine"
            else:
                fam = "fine" if rng.uniform() < 0.8 else "lourde"
            conf = float(rng.uniform(0.86, 1.0) if c != "incertain" else rng.uniform(0.15, 0.3))
            row = {"classe": c, "confiance": round(conf, 3), "t_analyse_s": float(ti), "t_debut_s": float(ti),
                   "longueur_mm": float(rng.uniform(5, 60)), "epaisseur_mm": float(rng.uniform(0.5, 2.0)),
                   "duree_visible_s": float(rng.uniform(0.3, 2.0)), "orientation_deg": float(rng.uniform(0, 180)),
                   "methode": method, "classe_manuelle": ""}
            if family:
                row["famille"] = fam
            if ia:
                row["classe_ia"] = c if c != "incertain" else ("beta" if fam == "fine" else "alpha")
                row["proba_ia"] = round(float(rng.uniform(0.8, 1.0) if c != "incertain" else rng.uniform(0.4, 0.7)), 3)
            rows.append(row)
    rows.sort(key=lambda r: r["t_analyse_s"])
    for i, r in enumerate(rows, 1):
        r["id"] = i
    return rows


def make_timeline(rows: list[dict], duration_s: float, period: float = 60.0, fps: float = 10.0,
                  noise_fn=None, disturbed_fn=None) -> list[dict]:
    """Journal cumulatif conforme au contrat du détecteur (une entrée par minute + entrée finale)."""
    times = np.array([r["t_analyse_s"] for r in rows], float)
    cls = np.array([r["classe"] for r in rows], dtype=object)
    ticks = list(np.arange(period, duration_s + 1e-9, period))
    ticks.append(duration_s)                       # entrée finale (finalize_run)
    out, frames, dist, wall, t_prev = [], 0, 0, 0.0, 0.0
    for t in ticks:
        dt = t - t_prev
        dd = int(disturbed_fn(t_prev, t)) if disturbed_fn else 0
        frames += int(round(dt * fps))
        dist += dd
        wall += dt + dd / fps
        out.append({"t_s": float(t), "wall_s": float(wall), "frames": frames, "disturbed_frames": dist,
                    "noise": float(noise_fn(t) if noise_fn else 2.0), "lit": 0.01,
                    "counts": {c: int(np.sum((times < t) & (cls == c))) for c in S.CLASSES}, "rejected": 0})
        t_prev = t
    return out


@dataclass
class FakeRun:
    rows: list
    duration_s: float
    timeline: list = field(default_factory=list)
    session: dict = field(default_factory=dict)
    name: str = "2026-10-20 Nuit de test essai-01"
    source: str = "nuit.mp4"
    condition: str = "Fond (rayons cosmiques)"
    sample_type: str = "Aucun"
    sample_mass_g: float = 0.0
    trial: int = 1
    date: str = "2026-10-20T07:00:00"
    fps: float = 10.0
    n_frames: int = 0
    calibrated: bool = True
    px_per_mm: float = 4.0
    roi_area_mm2: float = 35000.0
    config: dict = field(default_factory=lambda: {"classification": {"mode": "hybride"},
                                                  "physics": {"chamber_temp_c": -25.0}})
    notes: str = ""
    events: list = field(default_factory=list)
    folder: str = ""

    def table(self) -> list[dict]:
        return self.rows

    @property
    def total(self) -> int:
        return len(self.rows)

    def counts(self) -> dict:
        out: dict = {}
        for r in self.rows:
            out[r["classe"]] = out.get(r["classe"], 0) + 1
        return out


def night(hours: float = 8.0, seed: int = 1, factor=None, timeline: bool = True, noise_fn=None,
          disturbed_fn=None, **kw) -> FakeRun:
    dur = hours * HOUR
    rows = make_rows(dur, seed=seed, factor=factor, **kw)
    tl = make_timeline(rows, dur, noise_fn=noise_fn, disturbed_fn=disturbed_fn) if timeline else []
    return FakeRun(rows=rows, duration_s=dur, timeline=tl, n_frames=int(dur * 10))


LT = ["alpha", "beta", "muon", "proton", "artefact"]
LP = ["alpha", "beta", "muon", "proton", "artefact", "incertain"]
P_TRUE = np.array([
    [0.88, 0.01, 0.00, 0.04, 0.02, 0.05],
    [0.01, 0.70, 0.12, 0.00, 0.05, 0.12],
    [0.00, 0.15, 0.72, 0.00, 0.03, 0.10],
    [0.10, 0.00, 0.02, 0.75, 0.03, 0.10],
    [0.02, 0.05, 0.02, 0.00, 0.85, 0.06],
])


def fake_confusion(seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    M = np.array([rng.multinomial(n, p) for n, p in zip([3000, 5000, 4000, 800, 2000], P_TRUE)])
    return {"labels_true": LT, "labels_pred": LP, "matrix": M.tolist()}


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """Jamais le vrai modèle : seuil fixe, matrice de confusion simulée (remplaçable par test)."""
    monkeypatch.setattr(S, "get_reference_info", lambda: {"threshold": 0.85})
    monkeypatch.setattr(S, "get_reference_confusion", lambda: fake_confusion())


# --------------------------------------------------------------------------- #
# Formats français
# --------------------------------------------------------------------------- #
def test_french_number_formats():
    assert S.fr_num(3.14159) == "3,14"
    assert S.fr_num(-0.001) == "0,00"
    assert S.fr_num(2.5, 1, sign=True) == "+2,5"
    assert S.fr_num(float("nan")) == "—"
    assert S.fr_int(12345) == "12 345"
    assert S.fr_p(0.0342) == "0,034"
    assert S.fr_p(0.0012) == "0,0012"
    assert S.fr_p(2.1e-5) == "2,1 × 10⁻⁵"
    assert S.fr_g(1660.0) == "1 660" and S.fr_g(52.4) == "52,4" and S.fr_g(1.0) == "1"
    assert S.fr_duration(27000) == "7 h 30 min"
    assert S.fr_duration(8 * HOUR) == "8 h"
    assert S.fr_duration(90) == "90 s"
    assert S.fr_p(0.0) == "< 10⁻¹⁶"
    assert S.fr_clock(3 * HOUR + 600) == "3 h 10"
    assert S.stats.chi2 is not None          # scipy toujours disponible (module de base)
    assert S.rate(100, 120.0).fmt() == "50,00 ± 5,00 /min"


# --------------------------------------------------------------------------- #
# Journal temporel
# --------------------------------------------------------------------------- #
def test_timeline_intervals_deltas_and_duplicates():
    tl = [
        {"t_s": 60.0, "wall_s": 60.0, "frames": 600, "disturbed_frames": 0, "noise": 2.0, "lit": 0.01,
         "counts": {"alpha": 2, "beta": 5}, "rejected": 1},
        {"t_s": 120.0, "wall_s": 150.0, "frames": 1200, "disturbed_frames": 300, "noise": 4.0, "lit": 0.02,
         "counts": {"alpha": 3, "beta": 9}, "rejected": 1},
        {"t_s": 120.0, "wall_s": 150.0, "frames": 1200, "disturbed_frames": 300, "noise": 4.0, "lit": 0.02,
         "counts": {"alpha": 4, "beta": 9}, "rejected": 2},           # entrée finale au même instant
        "invalide", {"wall_s": 3.0},
    ]
    iv = S.timeline_intervals(tl)
    assert len(iv) == 2
    a, b = iv
    assert a["dt"] == pytest.approx(60.0) and a["counts"] == {"alpha": 2, "beta": 5}
    assert a["unanalysed_frac"] == pytest.approx(0.0)
    assert b["counts"] == {"alpha": 2, "beta": 4}                    # 1 + 1 (fusion de l'entrée finale)
    assert b["disturbed_frac"] == pytest.approx(300 / 900)
    assert b["unanalysed_frac"] == pytest.approx(1 - 60 / 90)
    assert b["rejected"] == 1
    ns = S.noise_summary(iv)
    assert ns["drift_ratio"] == pytest.approx(2.0)
    assert ns["unanalysed_frac"] == pytest.approx(1 - 120 / 150)
    assert S.timeline_intervals(None) == [] and S.noise_summary([]) is None


def test_clock_map_from_timeline():
    run = night(hours=1.0, seed=2, disturbed_fn=lambda t0, t1: 300 if t0 == 600.0 else 0)
    iv = S.timeline_intervals(run.timeline)
    cm = S.clock_map(iv)
    assert cm.source == "journal"
    assert cm(600.0) == pytest.approx(600.0)
    assert cm(660.0) == pytest.approx(690.0)                         # 30 s perturbées dans la 11e minute
    assert cm(3600.0) == pytest.approx(3630.0)


# --------------------------------------------------------------------------- #
# Taux par tranche et stabilité
# --------------------------------------------------------------------------- #
def test_constant_rate_night_is_stable():
    run = night(hours=8.0, seed=11)
    d = S.deep_analysis(run)
    assert d["has_timeline"] and d["clock"] == "analyse"
    assert d["bin_s"] == 3600.0 and len(d["bins"]) == 8
    assert sum(b.total for b in d["bins"]) == len(run.rows)
    st = d["stability"]["total"]
    assert st is not None and st.constant_ok and st.p_value > 0.01
    assert st.dof == 7 and st.method == "asymptotique"
    assert abs(st.slope_rel_per_h) < 3 * st.slope_err
    assert d["dead"]["possible"] and d["dead"]["periods"] == []
    # Taux par classe dans chaque tranche ≈ taux simulé
    for b in d["bins"]:
        assert b.rate("alpha").per_min == pytest.approx(RATES["alpha"], abs=5 * math.sqrt(RATES["alpha"] / 60))
    assert d["live"]["duration_s"] == pytest.approx(run.duration_s)


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_no_false_dead_periods_on_constant_rate(seed):
    d = S.deep_analysis(night(hours=3.0, seed=seed))
    assert d["dead"]["possible"] and d["dead"]["periods"] == []


def test_rate_step_fails_stability():
    run = night(hours=8.0, seed=12, factor=lambda t: np.where(t < 4 * HOUR, 1.0, 0.6))
    d = S.deep_analysis(run)
    st = d["stability"]["total"]
    assert st is not None and not st.constant_ok and st.p_value < 1e-3
    assert st.trend_significant and st.slope_rel_per_h < 0
    # La baisse de 40 % n'est pas une « période morte » (critère : < 25 % du taux habituel)
    assert d["dead"]["periods"] == []
    s, items = C.conclude_run(run)
    titles = [c.title for c in items]
    assert "Taux variable au cours du temps" in titles
    assert s["quality"].components[2].key == "stabilite" and s["quality"].components[2].value <= 0.1


def test_dead_hour_detected_and_excluded():
    dead = lambda t: np.where((t >= 3 * HOUR) & (t < 4 * HOUR), 0.0, 1.0)      # noqa: E731
    run = night(hours=8.0, seed=13, factor=dead)
    d = S.deep_analysis(run)
    per = d["dead"]["periods"]
    assert len(per) == 1
    p = per[0]
    assert p.kind == "morte" and p.n_obs <= 2 and p.p_value < 1e-3
    assert p.t0 >= 3 * HOUR - 300 and p.t1 <= 4 * HOUR + 300
    assert p.duration_s >= 0.8 * HOUR
    # Taux sur le temps actif ≈ taux simulé ; taux brut dilué par l'heure morte
    true_total = sum(RATES.values())
    live = d["live"]["rates"]["total"]
    raw = S.rate(len(run.rows), run.duration_s)
    assert live.per_min == pytest.approx(true_total, abs=3 * live.err)
    assert raw.per_min < live.per_min
    # Les tranches qui chevauchent l'heure morte sont marquées ; la stabilité porte sur le temps actif
    assert any(b.dead for b in d["bins"])
    st = d["stability"]["total"]
    assert st.excluded_s > 0 and st.constant_ok
    s, items = C.conclude_run(run)
    titles = [c.title for c in items]
    assert "Périodes d'inactivité de la chambre" in titles
    assert s["quality"].score < C.quality_score(night(hours=8.0, seed=13)).score


def test_degraded_period_needs_noise():
    low = lambda t: np.where((t >= 2 * HOUR) & (t < 3 * HOUR), 0.45, 1.0)      # noqa: E731
    noisy = lambda t: 6.0 if 2 * HOUR < t <= 3 * HOUR else 2.0                  # noqa: E731
    d_noisy = S.deep_analysis(night(hours=6.0, seed=14, factor=low, noise_fn=noisy))
    kinds = [p.kind for p in d_noisy["dead"]["periods"]]
    assert kinds == ["dégradée"]
    assert d_noisy["dead"]["periods"][0].noise_ratio == pytest.approx(3.0, rel=0.05)
    assert d_noisy["noise"]["n_spikes"] >= 50
    # Même baisse sans bruit élevé : ce n'est pas une période inactive
    d_quiet = S.deep_analysis(night(hours=6.0, seed=14, factor=low))
    assert d_quiet["dead"]["periods"] == []


def test_short_run_uses_minute_bins_and_old_rows():
    run = night(hours=0.25, seed=15, timeline=False)
    for r in run.rows:                                   # ancien format : pas d'horloge d'analyse
        del r["t_analyse_s"]
    d = S.deep_analysis(run)
    assert d["clock"] == "video" and not d["has_timeline"] and d["noise"] is None
    assert d["bin_s"] == 60.0 and len(d["bins"]) == 15
    assert not d["dead"]["possible"]
    assert S.choose_bin_s(90) == 15.0 and S.choose_bin_s(45 * 60) == 600.0 and S.choose_bin_s(5 * HOUR) == 3600.0


def test_stability_small_counts_uses_monte_carlo():
    st = S.stability_from_counts([1, 0, 2, 1, 3, 1, 2, 0, 1, 1], [60.0] * 10, np.arange(10) * 60 + 30, bin_s=60)
    assert st.method == "Monte-Carlo" and st.constant_ok
    assert S.stability_from_counts([1, 2], [60, 60], [30, 90]) is None


# --------------------------------------------------------------------------- #
# Familles de traces et comparaison
# --------------------------------------------------------------------------- #
def test_family_counts_include_thin_uncertain():
    rows = [{"classe": "beta", "famille": "fine"}, {"classe": "muon", "famille": "fine"},
            {"classe": "incertain", "famille": "fine"}, {"classe": "incertain", "famille": "lourde"},
            {"classe": "alpha", "famille": "lourde"}]
    f = S.family_counts(rows)
    assert f["mode"] == "famille" and f["fines"] == 3 and f["fines_incertaines"] == 1 and f["lourdes"] == 2
    old = [{k: v for k, v in r.items() if k != "famille"} for r in rows]
    g = S.family_counts(old)
    assert g["mode"] == "classes" and g["fines"] == 2 and g["lourdes"] == 1
    assert S.common_family_mode(rows, rows) and not S.common_family_mode(rows, old)
    assert S.common_family_mode(rows, [])
    # La classe finale prime sur la famille mesurée : fines = β + μ + incertaines fines (≥ β + μ)
    odd = rows + [{"classe": "muon", "famille": "lourde"}, {"classe": "alpha", "famille": "fine"}]
    h = S.family_counts(odd)
    assert h["fines"] == 4 and h["lourdes"] == 3 and h["famille_desaccord"] == 2


def test_comparison_uses_family_fines():
    rng = np.random.default_rng(5)
    bg = FakeRun(rows=make_rows(1200, seed=31), duration_s=1200)
    # Échantillon : excès de traces fines dont une partie classée incertaine (ambiguïté β/μ)
    extra = [{"classe": "incertain", "famille": "fine", "confiance": 0.25, "t_analyse_s": float(t),
              "t_debut_s": float(t), "methode": METHOD} for t in rng.uniform(0, 1200, 150)]
    rows = make_rows(1200, seed=32) + extra
    sample = FakeRun(rows=rows, duration_s=1200, condition="KCl", sample_type="KCl (chlorure de potassium)",
                     sample_mass_g=100.0)
    comps, items = C.conclude_comparison(sample, bg)
    n_fine = sum(1 for r in rows if r["famille"] == "fine")
    assert comps["fines"].a.n == n_fine
    assert "incertaines" in comps["fines"].label
    assert comps["fines"].significant
    assert any(c.title.startswith("Radioactivité détectée sur les traces fines") for c in items)
    act = [c for c in items if c.title == "Activité attendue (K-40)"][0]
    assert "1 660 Bq" in act.text and "," in act.text
    # Ancien fond sans colonne « famille » : retour à β + μ des deux côtés
    old_bg = FakeRun(rows=[{k: v for k, v in r.items() if k != "famille"} for r in bg.rows], duration_s=1200)
    comps2, _ = C.conclude_comparison(sample, old_bg)
    assert comps2["fines"].a.n == sum(1 for r in rows if r["classe"] in ("beta", "muon"))
    assert comps2["fines"].label == "Traces fines (β + μ)"


def test_comparison_excludes_dead_periods():
    dead = lambda t: np.where((t >= 2 * HOUR) & (t < 3 * HOUR), 0.0, 1.0)      # noqa: E731
    sample = night(hours=6.0, seed=41, factor=dead)
    bg = night(hours=6.0, seed=42)
    comps, items = C.conclude_comparison(sample, bg)
    titles = [c.title for c in items]
    assert "Périodes inactives exclues de la comparaison" in titles
    assert comps["total"].a.t_s < sample.duration_s - 0.8 * HOUR
    assert not comps["total"].significant
    assert comps["total"].p_value > 0.01
    comps_raw, _ = C.conclude_comparison(sample, bg, exclude_dead=False)
    assert comps_raw["total"].a.t_s == pytest.approx(sample.duration_s)


# --------------------------------------------------------------------------- #
# Comptages corrigés (déconvolution bayésienne itérative)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_unfolding_recovers_known_truth(seed):
    rng = np.random.default_rng(100 + seed)
    truth = {"alpha": 150, "beta": 600, "muon": 300, "proton": 15}
    obs = np.zeros(len(LP))
    for i, c in enumerate(["alpha", "beta", "muon", "proton"]):
        obs += rng.multinomial(rng.poisson(truth[c]), P_TRUE[i])
    observed = {lp: int(obs[j]) for j, lp in enumerate(LP) if lp != "artefact"}
    res = S.unfold_counts(observed, fake_confusion(), seed=seed)
    assert res is not None and res.classes == ["alpha", "beta", "muon", "proton"]
    for c, n in truth.items():
        assert abs(res.estimate[c] - n) < 3 * res.err[c] + 2, (c, res.estimate[c], n, res.err[c])
        assert res.lo[c] <= res.estimate[c] + 1e-6 <= res.hi[c] + 2 * res.err[c]
    # Les comptages bruts sont biaisés (β ↔ μ, incertaines) ; l'estimation corrige vers la vérité
    assert abs(res.estimate["beta"] - truth["beta"]) < abs(observed["beta"] - truth["beta"])
    assert res.efficiency["alpha"] == pytest.approx(0.98, abs=0.02)
    assert res.recall["beta"] == pytest.approx(0.70, abs=0.03)
    assert res.artefact_leak == pytest.approx(0.15, abs=0.03)
    assert res.n_matrix == 12800


def test_unfolding_is_reproducible_and_adds_manual_counts():
    obs = {"alpha": 140, "beta": 450, "muon": 260, "proton": 20, "incertain": 100}
    a = S.unfold_counts(obs, fake_confusion())
    b = S.unfold_counts(obs, fake_confusion())
    assert a.estimate == b.estimate and a.err == b.err
    m = S.unfold_counts(obs, fake_confusion(), manual={"alpha": 10})
    assert m.estimate["alpha"] == pytest.approx(a.estimate["alpha"] + 10)
    assert m.manual["alpha"] == 10


def test_unfolding_invalid_inputs():
    assert S.unfold_counts({"alpha": 3}, None) is None
    assert S.unfold_counts({"alpha": 3}, {"labels_true": ["alpha"], "labels_pred": ["alpha"], "matrix": [[1, 2]]}) is None
    no_inc = {"labels_true": ["alpha", "beta"], "labels_pred": ["alpha", "beta"], "matrix": [[9, 1], [1, 9]]}
    assert S.unfold_counts({"alpha": 3, "incertain": 2}, no_inc) is None    # classe observée non couverte
    assert S.unfold_counts({"alpha": 3, "beta": 4}, no_inc) is not None


def test_unfold_run_modes(monkeypatch):
    run = night(hours=1.0, seed=51)
    res, why = S.unfold_run(run)
    assert res is not None and why == ""
    rules = night(hours=1.0, seed=51, method="Règles physiques")
    res, why = S.unfold_run(rules)
    assert res is None and "hybride" in why
    monkeypatch.setattr(S, "get_reference_confusion", lambda: None)
    res, why = S.unfold_run(run)
    assert res is None and "indisponible" in why


# --------------------------------------------------------------------------- #
# Confiance de la classification
# --------------------------------------------------------------------------- #
def test_confidence_summary():
    run = night(hours=1.0, seed=61)
    cf = S.confidence_summary(run.rows, 0.85)
    assert cf["n"] == len(run.rows) and cf["has_ia"]
    assert cf["frac_confident"] == pytest.approx(1.0)                   # confiances simulées ≥ 0,86
    assert cf["classes"]["incertain"]["median"] < 0.31
    inc = cf["incertain"]
    assert inc["n"] == cf["classes"]["incertain"]["n"]
    assert set(inc["by_family"]) <= {"fine", "lourde"}
    assert set(inc["competitors"]) <= {"alpha", "beta"}
    assert cf["agreement"]["kind"] == "final_ia" and cf["agreement"]["frac"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Note de qualité
# --------------------------------------------------------------------------- #
def _score(run) -> int:
    return C.quality_score(run).score


def test_quality_score_monotonic():
    short, long_ = night(hours=0.2, seed=71), night(hours=6.0, seed=71)
    assert _score(long_) > _score(short)
    calib = night(hours=1.0, seed=72)
    uncal = night(hours=1.0, seed=72)
    uncal.calibrated = False
    assert _score(calib) > _score(uncal)
    clean = night(hours=2.0, seed=73)
    unsure = night(hours=2.0, seed=73, rates={**RATES, "incertain": 6.0})
    assert _score(clean) > _score(unsure)
    noisy = night(hours=2.0, seed=73, noise_fn=lambda t: 2.0 + 4.0 * t / (2 * HOUR))
    assert _score(clean) > _score(noisy)
    disturbed = night(hours=2.0, seed=73, disturbed_fn=lambda t0, t1: 300)
    assert _score(clean) > _score(disturbed)
    q = C.quality_score(clean)
    assert 0 <= q.score <= 100 and q.grade in ("excellente", "bonne", "passable", "insuffisante")
    assert [c.key for c in q.components] == list(C.QUALITY_WEIGHTS)
    assert sum(c.weight for c in q.components) == 100
    assert q.n_evaluated == 8
    assert any("50 traces" in a for a in q.advice)


def test_quality_without_timeline_skips_components():
    run = night(hours=0.1, seed=74, timeline=False)
    q = C.quality_score(run)
    keys = {c.key: c for c in q.components}
    assert keys["bruit"].value is None and keys["perturbations"].value is None
    assert q.n_evaluated == 6 and q.max_points == pytest.approx(80)


# --------------------------------------------------------------------------- #
# Conclusions
# --------------------------------------------------------------------------- #
FRENCH_MARKERS = ("traces", "taux", "fond", "échantillon", "alphas", "enregistrement", "loi", "min")


def test_conclude_run_deep_items_are_french():
    run = night(hours=8.0, seed=81, factor=lambda t: np.where((t >= 5 * HOUR) & (t < 6 * HOUR), 0.0, 1.0))
    s, items = C.conclude_run(run)
    titles = [c.title for c in items]
    assert titles[0] == "Bilan des observations"
    assert titles[1].startswith("Qualité de l'enregistrement : ")
    assert "Analyse approfondie indisponible" not in titles
    for t in ("Périodes d'inactivité de la chambre", "Traces fines : ambiguïté β/μ",
              "Estimation corrigée des comptages (d'après la validation sur simulation)",
              "Fiabilité de la classification"):
        assert t in titles, t
    assert any(t in titles for t in ("Taux stable au cours du temps", "Taux variable au cours du temps"))
    for c in items:
        low = (c.title + " " + c.text).lower()
        assert any(w in low for w in FRENCH_MARKERS), c
        for en in (" the ", " rate ", " background ", " significant "):
            assert en not in f" {low} ", (en, c)
        assert c.level in ("info", "succes", "attention")
    corr = [c for c in items if c.title.startswith("Estimation corrigée")][0]
    assert "ne remplace pas les comptages bruts" in corr.text
    assert s["deep"]["unfold"] is not None and s["quality"].score > 0
    # Rien en notation décimale anglaise (« 2.5 ») dans les textes de l'analyse approfondie
    import re
    deep_txt = " ".join(c.text for c in items[1:2] + items[-5:])
    assert not re.search(r"\d\.\d", deep_txt), deep_txt


def test_conclude_run_without_deep():
    run = night(hours=1.0, seed=82)
    s, items = C.conclude_run(run, deep=False)
    assert "deep" not in s and not any(c.title.startswith("Qualité") for c in items)


# --------------------------------------------------------------------------- #
# Graphiques
# --------------------------------------------------------------------------- #
PNG = b"\x89PNG"


def test_deep_figures_render():
    run = night(hours=8.0, seed=91, factor=lambda t: np.where((t >= 3 * HOUR) & (t < 4 * HOUR), 0.0, 1.0),
                noise_fn=lambda t: 2.0 + 0.2 * math.sin(t / 3000.0))
    deep = S.deep_analysis(run)
    figs = plots.run_figures(run, deep=deep)
    keys = [k for k, _t, _f in figs]
    for k in ("taux_tranches", "cumul", "bruit_temps", "confiance", "comptages_corriges"):
        assert k in keys, k
        assert k in report.FIG_CAPTIONS
    for _k, _t, f in figs[6:]:
        assert plots.fig_to_png_bytes(f, dpi=50).startswith(PNG)


def test_deep_figures_empty_inputs():
    for f in (plots.fig_time_rates({}), plots.fig_noise_timeline({}), plots.fig_cumulative([]),
              plots.fig_confidence([]), plots.fig_unfolded(None)):
        assert plots.fig_to_png_bytes(f, dpi=40).startswith(PNG)
    short = night(hours=0.1, seed=92, timeline=False)
    keys = [k for k, _t, _f in plots.run_figures(short)]
    assert "taux_tranches" not in keys and "bruit_temps" not in keys and "cumul" in keys


def test_decimal_comma_on_axes():
    fig, ax = plots._new_fig(3, 2)
    ax.plot([0, 1], [0.0, 0.5])
    ax.set_ylim(0, 0.5)
    plots.fig_to_png_bytes(fig, dpi=40)
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert labels and any("," in s for s in labels) and not any("." in s for s in labels)


# --------------------------------------------------------------------------- #
# Rapport HTML
# --------------------------------------------------------------------------- #
def test_report_with_timeline_session_and_model(tmp_path):
    run = night(hours=8.0, seed=101, factor=lambda t: np.where((t >= 3 * HOUR) & (t < 4 * HOUR), 0.0, 1.0))
    run.session = {"start": "2026-10-19T22:30:00", "end": "2026-10-20T06:40:00", "planned_s": 9 * HOUR,
                   "actual_s": 8 * HOUR + 600, "checkpoints": [{"t_s": 1800.0}, {"t_s": 3600.0}],
                   "warnings": ["Caméra déconnectée 12 s à 02:14 (reconnexion automatique)"],
                   "camera": "Logitech C920"}
    out = report.build_run_report(run, tmp_path / "rapport.html")
    html = out.read_text(encoding="utf-8")
    assert 'id="approfondie"' in html and "Analyse approfondie" in html
    assert 'id="session"' in html and "Session de nuit" in html
    assert "Caméra déconnectée 12 s" in html and "Session écourtée" in html and "Logitech C920" in html
    assert "Estimation corrigée (d&#x27;après la validation sur simulation)" in html or \
        "Estimation corrigée (d'après la validation sur simulation)" in html
    assert "/100" in html and "Note de qualité" in html
    assert "Taux par tranche de 1 h<" in html
    assert "22:30 – 23:30" in html                                       # heure réelle (début de session)
    assert "inactive" in html
    assert html.count("data:image/png;base64,") >= 11                    # 6 graphiques de base + 5 approfondis
    assert "<script" not in html                                          # autonome, sans JavaScript
    md = report.report_text(run)
    assert "## Analyse approfondie" in md and "Session de nuit" in md and "| Tranche |" in md


def test_report_session_from_long_run_backend(tmp_path):
    """Clés écrites par la session de nuit (longrun.session_info) : avertissements en dicts, durée prévue
    illimitée (None), résultats provisoires."""
    run = night(hours=0.5, seed=103)
    run.session = {"started_at": "2026-10-19T22:00:00", "ended_at": "", "planned_end": "", "planned_s": None,
                   "elapsed_s": 1830.5, "stop_reason": "", "partial": True, "checkpoints": 6,
                   "video_segments": ["video/segment_001.mp4"], "video_codec": "mp4v", "photos": 3,
                   "reconnections": 1, "simulated_clock": False, "record_video": "reduite", "keep_awake": True,
                   "record_scale": 0.5,
                   "warnings": [{"heure": "2026-10-19T22:14:05", "temps_s": 845.0,
                                 "message": "Caméra perdue 8 s : reconnexion réussie."}]}
    html = report.build_run_report(run, tmp_path / "p.html").read_text(encoding="utf-8")
    assert "Session de nuit" in html and "Résultats provisoires" in html
    assert "22:14 — Caméra perdue 8 s : reconnexion réussie." in html
    assert "jusqu&#x27;à l&#x27;arrêt manuel" in html or "jusqu'à l'arrêt manuel" in html
    assert "Reconnexions de la caméra" in html and "Échelle de la vidéo" in html and "0,5" in html
    assert "Session écourtée" not in html
    md = report.report_text(run)
    assert "Caméra perdue 8 s" in md


def test_report_without_timeline_session_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "get_reference_confusion", lambda: None)
    run = night(hours=0.3, seed=102, timeline=False, family=False, ia=False)
    out = report.build_run_report(run, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "Analyse approfondie" in html and 'id="session"' not in html
    assert "Estimation corrigée non disponible" in html
    assert "Journal du détecteur absent" in html
    assert "non évalué" in html


def test_comparison_report_mentions_family_fines(tmp_path):
    bg = FakeRun(rows=make_rows(1800, seed=111), duration_s=1800)
    sample = FakeRun(rows=make_rows(1800, seed=112, rates={**RATES, "beta": 9.0}), duration_s=1800, condition="KCl",
                     sample_type="KCl (chlorure de potassium)", sample_mass_g=100.0)
    out = report.build_comparison_report(sample, bg, tmp_path / "c.html")
    html = out.read_text(encoding="utf-8")
    assert "fines classées incertaines" in html
    assert "Traces fines (β, μ et fines incertaines)" in html
    md = report.comparison_text(sample, bg)
    assert "Traces fines (β, μ et fines incertaines)" in md
