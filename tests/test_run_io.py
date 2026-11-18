"""Sauvegarde / rechargement d'un enregistrement (run.json, traces.csv, traces.xlsx, vignettes)."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime

import numpy as np
import pytest

from wilsontrack.analysis.run import RunResult, _jsonable, default_run_name, safe_name
from wilsontrack.config import AppConfig
from wilsontrack.detection.events import EXPORT_COLUMNS, TrackEvent


def _event(i: int, cls: str, t: float, length: float, width: float, valid: bool = True,
           energy: float | None = None, radius: float = float("inf")) -> TrackEvent:
    feats = {"length_mm": length, "chord_mm": 0.98 * length, "width_mm": width, "straightness": 0.98,
             "tortuosity": 0.01, "density_rel": 3.0 if cls == "alpha" else 1.0, "width_rel": 1.1,
             "density": 120.5, "peak_snr": 40.0, "asymmetry": 1.2, "branches": 0,
             "orientation_deg": 12.5 * i % 180, "radius_mm": radius}
    if energy is not None:
        feats["energy_mev"] = energy
    thumb = np.full((20, 30, 3), 40 + 10 * i, np.uint8)
    return TrackEvent(
        id=i if valid else -1, first_frame=int(t * 30), last_frame=int(t * 30) + 20, best_frame=int(t * 30) + 5,
        t_first=t, t_best=t + 0.15, t_last=t + 0.66, n_seen=20, bbox_proc=(10, 10, 60, 30),
        centroid_proc=(100.0 + i, 50.0 + i), features=feats, path_full=np.zeros((0, 2)), thumb=thumb,
        cls=cls, confidence=0.87, reason=f"raison {cls} é", valid=valid, px_per_mm_full=5.12,
        extra={"scale": 0.75, "calibrated": True})


@pytest.fixture
def run_with_events():
    cfg = AppConfig()
    cfg.physics.px_per_mm = 5.12
    cfg.run.condition = "KCl"
    evs = [
        _event(1, "alpha", 1.0, 36.1, 1.4, energy=5.58),
        _event(2, "beta", 2.5, 22.0, 0.6, radius=41.3),
        _event(3, "muon", 4.0, 120.0, 0.5),
        _event(4, "muon", 7.5, 88.0, 0.55),
        _event(5, "beta", 9.0, 3.0, 0.4, valid=False),      # rejetée : ne doit pas être exportée
    ]
    evs[3].manual_cls = "beta"                               # correction manuelle de l'utilisateur
    return RunResult(
        name=default_run_name("KCl", 3, datetime(2026, 9, 29)), source="video.mp4", condition="KCl",
        sample_type="KCl (chlorure de potassium)", sample_mass_g=100.0, trial=3, date="2026-09-29T10:00:00",
        fps=30.0, n_frames=300, duration_s=9.6, frame_w=1280, frame_h=720, px_per_mm=5.12, calibrated=True,
        roi_area_mm2=35156.25, config=cfg.to_dict(), events=evs, expansion_times=[], rejected=7,
        notes="Essai « test » à −25 °C")


# --------------------------------------------------------------------------- #
def test_default_run_name_format():
    assert default_run_name("KCl", 3, datetime(2026, 9, 29)) == "2026-09-29 KCl essai-03"
    assert default_run_name("Banane", 12, datetime(2026, 1, 5)) == "2026-01-05 Banane essai-12"
    name = default_run_name("Fond (rayons cosmiques)", 1)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} Fond \(rayons cosmiques\) essai-01", name)
    assert name.startswith(datetime.now().strftime("%Y-%m-%d"))


def test_safe_name():
    assert safe_name('KCl/banane: essai?') == "KCl-banane- essai-"
    assert safe_name("  ") == "enregistrement"
    assert safe_name("Fond (rayons cosmiques)") == "Fond (rayons cosmiques)"


def test_table_counts_from_events(run_with_events):
    r = run_with_events
    rows = r.table()
    assert len(rows) == 4 == r.total                       # l'événement invalide est exclu
    assert r.counts() == {"alpha": 1, "beta": 2, "muon": 1}  # manual_cls prime sur cls
    assert all(set(row) == set(EXPORT_COLUMNS) for row in rows)


def test_save_load_round_trip(run_with_events, tmp_path):
    r = run_with_events
    orig_rows = r.table()
    orig_meta = r.meta()
    folder = r.save(tmp_path)
    assert folder == tmp_path / "2026-09-29 KCl essai-03"
    for name in ("run.json", "traces.csv", "traces.xlsx"):
        assert (folder / name).is_file(), name
    thumbs = sorted(p.name for p in (folder / "vignettes").glob("*.png"))
    assert thumbs == ["trace_00001_alpha.png", "trace_00002_beta.png", "trace_00003_muon.png",
                      "trace_00004_beta.png"]

    meta_json = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert meta_json["summary"] == {"alpha": 1, "beta": 2, "muon": 1}

    loaded = RunResult.load(folder)
    assert loaded.counts() == r.counts()
    assert loaded.total == 4
    assert loaded.folder == str(folder)
    # Métadonnées identiques (après normalisation JSON : tuples → listes, inf → null)
    expected_meta = json.loads(json.dumps(_jsonable(orig_meta)))
    expected_meta["folder"] = str(folder)
    assert loaded.meta() == expected_meta
    assert loaded.notes == "Essai « test » à −25 °C"
    assert AppConfig.from_dict(loaded.config).to_dict() == AppConfig.from_dict(orig_meta["config"]).to_dict()

    # Lignes identiques (les champs vides « » redeviennent None)
    assert len(loaded.rows) == len(orig_rows)
    for a, b in zip(orig_rows, loaded.rows):
        for k in EXPORT_COLUMNS:
            va, vb = a[k], b[k]
            if va == "" and k not in ("classe_manuelle", "raison"):
                assert vb is None, k
            elif isinstance(va, float):
                assert vb == pytest.approx(va), k
            else:
                assert vb == va, k
    assert isinstance(loaded.rows[0]["id"], int)
    assert loaded.rows[0]["energie_MeV"] == pytest.approx(5.58)
    assert loaded.rows[1]["rayon_courbure_mm"] == pytest.approx(41.3)
    assert loaded.rows[0]["rayon_courbure_mm"] is None      # rayon infini → vide
    assert loaded.rows[3]["classe_manuelle"] == "beta"


def test_save_load_twice_is_stable(run_with_events, tmp_path):
    folder = run_with_events.save(tmp_path / "a")
    first = RunResult.load(folder)
    folder2 = first.save(tmp_path / "b")
    second = RunResult.load(folder2)
    assert second.rows == first.rows
    assert second.counts() == first.counts()
    assert not (folder2 / "vignettes").exists()              # pas d'événements en mémoire → pas de vignettes


def test_xlsx_content(run_with_events, tmp_path):
    from openpyxl import load_workbook
    folder = run_with_events.save(tmp_path)
    wb = load_workbook(folder / "traces.xlsx")
    assert wb.sheetnames == ["Résumé", "Traces"]
    ws = wb["Traces"]
    header = [c.value for c in ws[1]]
    assert header == EXPORT_COLUMNS
    assert ws.max_row == 1 + 4
    summary = {row[0]: row[1] for row in wb["Résumé"].iter_rows(values_only=True) if row and row[0]}
    assert summary["Nombre total de traces"] == 4
    assert summary["Condition"] == "KCl"
    assert summary["Calibré"] == "oui"
    # Taux par classe : bêta = 2 traces en 9.6 s = 12.5 /min
    assert summary["beta"] == 2


def test_load_sample_run(kcl_run, sample_run_dirs):
    """Les enregistrements d'exemple se rechargent et restent cohérents avec run.json."""
    meta = json.loads((sample_run_dirs[1] / "run.json").read_text(encoding="utf-8"))
    assert kcl_run.counts() == meta["summary"]
    assert kcl_run.calibrated and kcl_run.px_per_mm == pytest.approx(5.12)
    assert kcl_run.sample_type == "KCl (chlorure de potassium)" and kcl_run.sample_mass_g == 100.0
    assert all(isinstance(r["id"], int) for r in kcl_run.rows)
    assert all(math.isfinite(r["longueur_mm"]) for r in kcl_run.rows)
