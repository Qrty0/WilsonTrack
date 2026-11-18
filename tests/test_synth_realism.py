"""Réalisme physique du simulateur v2 (wilsontrack/synth.py) et évaluation avec artefacts.

Chaque test compare le simulateur à une source citée (voir synth.JUSTIFICATIONS) :
- portées NIST (ASTAR à −25 °C, ESTAR, PSTAR) ;
- diffusion multiple de Highland (PDG) ;
- longueurs projetées des muons : P(L > x) = (1 + (x/d)²)⁻² (cos²θ, PDG) ;
- durées de visibilité : MÊME loi pour les électrons et les muons (Sussman) ;
- épaisseur et densité de gouttelettes : même loi pour toutes les traces fines ;
- artefacts présents dans la vérité terrain, détectés et comptés comme faux positifs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import ks_2samp

from wilsontrack import physics
from wilsontrack import synth as S
from wilsontrack.config import AppConfig
from wilsontrack.detection import TrackDetector
from wilsontrack.detection.classifier import alpha_max_length_for
from wilsontrack.evaluate import evaluate, format_report, label_detections

RHO = S.air_density_g_cm3(-25.0)
X0_MM = S.X0_AIR_G_CM2 / RHO * 10.0


# --------------------------------------------------------------------------- #
# Tables NIST et transport
# --------------------------------------------------------------------------- #
def test_air_density_and_radiation_length_cold():
    assert RHO == pytest.approx(1.4225e-3, rel=2e-3)             # gaz parfait à −25 °C
    assert X0_MM / 1000.0 == pytest.approx(257.0, rel=0.01)      # ≈ 257 m (PDG, ramené à −25 °C)


@pytest.mark.parametrize("e, r_astar_cm", [(5.304, 3.36), (5.49, 3.54), (6.003, 4.06), (7.687, 6.00), (8.785, 7.44)])
def test_alpha_range_matches_astar_minus25(e, r_astar_cm):
    """Portée ASTAR à −25 °C (fait « alpha_cold_air ») à 10 % près, table ET transport pas à pas."""
    assert S.alpha_range_mm(e, -25.0) / 10.0 == pytest.approx(r_astar_cm, rel=0.10)
    rng = np.random.default_rng(1)
    paths = [S.transport(rng, "a", e, (0, 0, 500), (1, 0, 0), RHO, X0_MM, 0, 1000)[3] for _ in range(5)]
    assert np.mean(paths) / 10.0 == pytest.approx(r_astar_cm, rel=0.10)


def test_electron_and_proton_ranges_match_nist():
    for e, r in S.ESTAR_CSDA:
        assert S.csda_range_g_cm2("e", e) == pytest.approx(r, rel=0.03), e
    for e, r in S.PSTAR_CSDA:
        assert S.csda_range_g_cm2("p", e) == pytest.approx(r, rel=0.03), e
    assert S.stopping_power("p", 3.0) == pytest.approx(102.0, rel=0.03)


def test_ionisation_ratios():
    assert S.ionisation_ratio("mu", 4000.0) == 1.0
    assert S.ionisation_ratio("e", 1.0) == pytest.approx(1.0, abs=0.01)   # électron rapide ≈ minimum d'ionisation
    assert S.ionisation_ratio("e", 20.0) == 1.0
    assert S.ionisation_ratio("e", 0.02) == pytest.approx(11.57 / 1.655, rel=0.05)
    assert S.ionisation_ratio("a", 5.49) == pytest.approx(713 / 1.815, rel=0.05)   # ≈ 390× (fait alpha_vs_mip)
    assert 30 < S.ionisation_ratio("p", 3.0) < 80                     # ≈ 56× (PSTAR)


@pytest.mark.parametrize("e, x_mm", [(0.5, 10.0), (0.5, 50.0), (1.31, 50.0)])
def test_highland_multiple_scattering(e, x_mm):
    """Écart latéral rms simulé ≈ Highland avec terme logarithmique (PDG éq. 34.16/34.20), ± 30 %."""
    rng = np.random.default_rng(2)
    ys = []
    for _ in range(300):
        pts, _q, _e, _s = S.transport(rng, "e", e, (0, 0, 500), (1, 0, 0), RHO, X0_MM, 0, 1000, max_path_mm=1.6 * x_mm)
        k = np.nonzero(pts[:, 0] >= x_mm)[0]
        if len(k):
            ys.append(pts[k[0], 1])
    w = e + S.ME
    p = math.sqrt(w * w - S.ME ** 2)
    b = p / w
    th = 13.6 / (b * p) * math.sqrt(x_mm / X0_MM) * (1 + 0.038 * math.log(x_mm / X0_MM / b ** 2))
    assert np.sqrt(np.mean(np.square(ys))) == pytest.approx(x_mm * th / math.sqrt(3), rel=0.30)


def test_no_visible_delta_rays_on_alphas():
    """Tmax = 2mₑc²β²γ² ≈ 3 keV pour une alpha de 5,5 MeV : aucun rayon delta visible (≥ 10 keV)."""
    sim = S.SyntheticChamber(S.SynthConfig(width=320, height=180, duration_s=0.1))
    pts = np.array([[0.0, 0.0, 4.0], [40.0, 0.0, 4.0]])
    assert sim._delta_rays("a", 5.49, pts, (1, 0, 0), 0.0, 8.0) == []


# --------------------------------------------------------------------------- #
# Muons : géométrie cos²θ
# --------------------------------------------------------------------------- #
def test_muon_length_distribution_cos2():
    rng = np.random.default_rng(3)
    d = 10.0
    L = np.array([S.muon_projected_length(rng, d) for _ in range(20000)])
    for x in (2.0, 5.0, 10.0, 20.0, 30.0):
        expected = (1 + (x / d) ** 2) ** -2
        assert np.mean(L > x) == pytest.approx(expected, abs=0.012), x
    assert np.median(L) == pytest.approx(0.64 * d, rel=0.05)                 # médiane ≈ 0,64·d
    # Tirage conditionné à L ≥ l_min : loi conditionnelle exacte
    Lc = np.array([S.muon_projected_length(rng, d, 5.0) for _ in range(20000)])
    assert Lc.min() >= 5.0 - 1e-9
    p5 = (1 + 0.25) ** -2
    assert np.mean(Lc > 10.0) == pytest.approx((1 + 1.0) ** -2 / p5, abs=0.015)


def _gt_only(cfg: S.SynthConfig) -> list[dict]:
    sim = S.SyntheticChamber(cfg)
    for _ in range(sim.n_frames):
        sim.render_next()
    return sim.gt


@pytest.fixture(scope="module")
def thin_population():
    """Grande population de traces fines (petite image : rendu rapide), sans artefacts."""
    cfg = S.SynthConfig(width=320, height=180, fov_mm=400.0, fps=25.0, duration_s=90.0, seed=7,
                        rates_per_min={"alpha": 40.0, "beta": 160.0, "muon": 160.0, "proton": 0.0},
                        artefact_rates_per_min={}, shower_per_min=0.0, cosmic_e_frac=0.0, crossing_min_mm=1.0)
    return cfg, _gt_only(cfg)


def test_muon_lengths_in_chamber_follow_cos2(thin_population):
    cfg, gt = thin_population
    L = np.array([g["length_mm"] for g in gt if g["kind"] == "muon"])
    assert len(L) > 150
    d, lmin = cfg.deff_mm, cfg.crossing_min_mm
    p_min = (1 + (lmin / d) ** 2) ** -2
    for x in (5.0, 10.0, 20.0):
        assert np.mean(L > x) == pytest.approx((1 + (x / d) ** 2) ** -2 / p_min, abs=0.06), x


def test_thin_track_lifetimes_identical_for_electrons_and_muons(thin_population):
    cfg, gt = thin_population
    le = np.array([g["life_frames"] for g in gt if g["cls"] == "beta"]) / cfg.fps
    lm = np.array([g["life_frames"] for g in gt if g["cls"] == "muon"]) / cfg.fps
    la = np.array([g["life_frames"] for g in gt if g["cls"] == "alpha"]) / cfg.fps
    assert len(le) > 100 and len(lm) > 100 and len(la) > 20
    assert ks_2samp(le, lm).pvalue > 0.01                         # même loi (Sussman : traces fines confondues)
    for x in (le, lm):
        assert np.median(x) == pytest.approx(cfg.thin_life_median_s, rel=0.2)
    assert np.mean(la) == pytest.approx(cfg.alpha_life_mean_s, rel=0.1)   # alphas 1,5 s (Sussman)


def test_thin_track_widths_identical_for_electrons_and_muons(thin_population):
    cfg, gt = thin_population
    we = [g["width_mm"] for g in gt if g["cls"] == "beta"]
    wm = [g["width_mm"] for g in gt if g["cls"] == "muon"]
    assert ks_2samp(we, wm).pvalue > 0.01                         # même diffusion latérale pour toutes
    qm = [g["q_mean"] for g in gt if g["cls"] == "muon"]
    assert max(qm) == pytest.approx(1.0)                          # muons au minimum d'ionisation


def test_k40_beta_spectrum_mean():
    rng = np.random.default_rng(4)
    e = np.array([S._k40_beta_energy(rng) for _ in range(20000)])
    assert e.mean() == pytest.approx(physics.K40_BETA_MEAN_MEV, abs=0.01)
    assert e.max() <= physics.K40_BETA_EMAX_MEV


def test_compton_electrons_mostly_low_energy():
    """Électrons sortant des parois : front Compton 1,243 MeV, spectre dominé par les basses énergies."""
    rng = np.random.default_rng(5)
    e0 = np.array([S._klein_nishina_electron(rng, S.K40_GAMMA_MEV) for _ in range(4000)])
    assert e0.max() <= 1.2434 and e0.max() > 1.1                 # front Compton 2k/(1+2k)·Eγ = 1,2434 MeV
    e = np.array([S._escaping_energy(rng, x) for x in e0])
    assert np.median(e) < 0.5 * np.median(e0)


def test_alpha_max_length_rule_matches_po212_range():
    assert alpha_max_length_for(-25.0) == pytest.approx(78.0, rel=0.03)
    assert alpha_max_length_for(0.0) == pytest.approx(86.0, rel=0.03)


def test_thoron_pairs_share_origin():
    cfg = S.SynthConfig(width=640, height=360, fov_mm=160.0, duration_s=20.0, seed=9,
                        rates_per_min={"alpha": 120.0, "beta": 0.0, "muon": 0.0, "proton": 0.0},
                        alpha_mix={"thoron": 1.0}, artefact_rates_per_min={}, shower_per_min=0.0)
    gt = _gt_only(cfg)
    groups: dict[int, list] = {}
    for g in gt:
        groups.setdefault(g["group"], []).append(g)
    pairs = [v for k, v in groups.items() if k > 0 and len(v) == 2]
    assert len(pairs) >= 10
    dts = []
    for a, b in pairs:
        assert {a["kind"], b["kind"]} == {"Rn-220", "Po-216"}
        dts.append(abs(b["t_birth"] - a["t_birth"]))
    assert np.median(dts) == pytest.approx(0.145, abs=0.12)      # T½(Po-216) = 0,145 s


# --------------------------------------------------------------------------- #
# Artefacts : vérité terrain, détections, faux positifs
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def artefact_run():
    cfg = S.SynthConfig(width=640, height=360, fov_mm=160.0, fps=25.0, duration_s=30.0, seed=21,
                        rates_per_min={"alpha": 10.0, "beta": 10.0, "muon": 20.0, "proton": 0.0},
                        artefact_rates_per_min={"brume": 12.0, "amas": 12.0, "scintillement": 12.0, "reflet": 8.0,
                                                "reflet_bord": 6.0})
    sim = S.SyntheticChamber(cfg)
    app = AppConfig()
    app.classification.mode = "regles"
    app.physics.px_per_mm = cfg.px_per_mm
    det = TrackDetector(app, cfg.fps, (cfg.height, cfg.width))
    for _ in range(sim.n_frames):
        det.process(sim.render_next(), None)
    det.finalize_run()
    gt = {"px_per_mm": cfg.px_per_mm, "config": {"fps": cfg.fps, "width": cfg.width, "height": cfg.height},
          "tracks": sim.gt}
    return cfg, sim, det, gt


def test_artefacts_produce_ground_truth(artefact_run):
    _cfg, sim, _det, _gt = artefact_run
    kinds = {g["kind"] for g in sim.gt if g["cls"] == "artefact"}
    assert kinds == set(S.ARTEFACT_KINDS)
    for g in sim.gt:
        assert g["cls"] in ("alpha", "beta", "muon", "proton", "artefact")
        assert len(g["polyline"]) >= 2


def test_artefacts_fool_the_detector_and_count_as_false_positives(artefact_run):
    _cfg, _sim, det, gt = artefact_run
    res = evaluate(det.events, gt)
    art = res["artefact"]
    assert art["n_gt"] > 20
    assert art["n_detected"] >= 3, format_report(res)              # ils créent bien des détections
    # Une détection posée sur un artefact est un faux positif (pas une vraie trace)
    n_valid = sum(1 for e in det.events if e.valid)
    assert res["false_positives"] >= art["n_detected"]
    assert res["precision"] <= (n_valid - art["n_detected"]) / max(1, n_valid) + 1e-9
    assert "artefacts simulés" in format_report(res)
    # Le rappel reste calculé sur les 4 classes de particules
    assert set(res["recall"]) == {"alpha", "beta", "muon", "proton"}


def test_label_detections_covers_every_valid_detection(artefact_run):
    _cfg, _sim, det, gt = artefact_run
    labs = label_detections(det.events, gt)
    valid = [e for e in det.events if e.valid]
    assert len(labs) == len(valid)
    assert {id(d["event"]) for d in labs} == {id(e) for e in valid}
    hows = {d["how"] for d in labs}
    assert hows <= {"match", "artefact", "dup", "far"}
    for d in labs:
        if d["how"] in ("artefact", "far"):
            assert d["label"] == "artefact"
        else:
            assert d["label"] in ("alpha", "beta", "muon", "proton")
    assert any(d["how"] == "artefact" for d in labs)


def test_evaluate_reads_old_ground_truth_without_kind():
    """Vérité terrain d'une ancienne vidéo d'exemple (sans « kind » ni artefacts) : toujours lisible."""
    gt = {"px_per_mm": 4.0, "config": {"fps": 30.0, "width": 320, "height": 180},
          "tracks": [{"id": 1, "cls": "muon", "t_birth": 1.0, "frame_birth": 30, "life_frames": 20,
                      "length_mm": 20.0, "polyline": [[10.0, 10.0], [90.0, 10.0]]}]}
    res = evaluate([], gt)
    assert res["n_gt"] == 1 and res["recall_total"] == 0.0 and res["artefact"]["n_gt"] == 0
    assert label_detections([], gt) == []
