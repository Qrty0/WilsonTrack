"""Tests hors écran des écrans de longues analyses (session de nuit, vidéos reçues / lot, bandeau d'état,
file d'attente) et du Centre d'apprentissage de l'IA."""
from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import datetime as dt  # noqa: E402
import re  # noqa: E402
import string  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from PySide6.QtCore import QMimeData, QPointF, Qt, QTime, QUrl  # noqa: E402
from PySide6.QtGui import QColor, QDropEvent, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QProgressBar, QPushButton  # noqa: E402

from wilsontrack.gui import learning as lc_mod  # noqa: E402
from wilsontrack.gui import session_dialogs as sd  # noqa: E402

NIGHT_KEYS = {"camera_index", "duration_h", "end_time", "segment_min", "record_video", "record_scale", "record_fps",
              "snapshot_every_min", "autosave_every_min", "keep_awake", "dead_after_min", "condition", "sample",
              "mass_g", "px_per_mm", "output_dir"}
BATCH_KEYS = {"files", "watch_folder", "condition", "sample", "mass_g", "px_per_mm", "output_dir", "make_reports"}
CAMS = [{"index": 0, "label": "Caméra 0 (1920×1080)", "width": 1920, "height": 1080},
        {"index": 2, "label": "Caméra 2 (1280×720)", "width": 1280, "height": 720}]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    from wilsontrack.gui.app import QSS, dark_palette
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(QSS)
    return app


def collect(signal) -> list:
    got = []
    signal.connect(lambda *a: got.append(a[0] if len(a) == 1 else a))
    return got


def fake_longrun(est: int, free: int):
    class Cfg:
        seen = []

        @staticmethod
        def from_dict(d):
            Cfg.seen.append(dict(d))
            return d
    return Cfg, (lambda cfg, wh, fps: est), (lambda path: free)


def missing_drive() -> str | None:
    for letter in reversed(string.ascii_uppercase):
        if not os.path.exists(f"{letter}:\\"):
            return f"{letter}:\\WilsonTrack"
    return None


# ================================================================ session de nuit ====
def test_normalize_night_defaults_and_unknown_keys():
    d = sd.normalize_night({"inconnue": 1, "segment_min": "45", "record_video": "4K", "keep_awake": "non",
                            "end_time": "25:99", "duration_h": None, "mass_g": -3})
    assert set(d) == NIGHT_KEYS
    assert d["segment_min"] == 45 and isinstance(d["segment_min"], int)
    assert d["record_video"] == "reduite"                # valeur inconnue → défaut
    assert d["keep_awake"] is False
    assert d["end_time"] == "07:30"                      # heure invalide et durée 0 → heure de fin par défaut
    assert d["mass_g"] == 0.0
    assert sd.normalize_night(None) == sd.normalize_night({})


def test_night_dialog_schema_and_defaults(qapp):
    dlg = sd.NightSessionDialog()
    d = dlg.result_dict()
    assert set(d) == NIGHT_KEYS
    types = {"camera_index": int, "duration_h": float, "end_time": str, "segment_min": int, "record_video": str,
             "record_scale": float, "record_fps": float, "snapshot_every_min": int, "autosave_every_min": int,
             "keep_awake": bool, "dead_after_min": int, "condition": str, "sample": str, "mass_g": float,
             "px_per_mm": float, "output_dir": str}
    for k, t in types.items():
        assert isinstance(d[k], t), (k, d[k])
    assert d["record_video"] == "reduite" and d["record_scale"] == 0.5 and d["record_fps"] == 10.0
    assert d["segment_min"] == 30 and d["snapshot_every_min"] == 10 and d["autosave_every_min"] == 5
    assert d["keep_awake"] is True and d["dead_after_min"] == 20 and d["output_dir"] == ""
    assert d["duration_h"] == 0.0 and d["end_time"] == "07:30"   # « jusqu'à » 7 h 30 par défaut
    assert "Assez d'alcool et de glace sèche" in dlg.checklist_items[0].text()
    dlg.deleteLater()


def test_night_dialog_duration_or_end_time(qapp):
    dlg = sd.NightSessionDialog({"duration_h": 8.5, "end_time": "06:00", "foo": "bar"})
    assert dlg.mode_duration.isChecked() and not dlg.end_time.isEnabled()
    d = dlg.result_dict()
    assert d["duration_h"] == 8.5 and d["end_time"] == ""
    dlg.mode_until.setChecked(True)
    dlg.end_time.setTime(QTime(6, 15))
    d = dlg.result_dict()
    assert d["duration_h"] == 0.0 and d["end_time"] == "06:15"
    dlg._now = lambda: dt.datetime(2026, 10, 20, 22, 0)
    assert dlg.planned_hours() == pytest.approx(8.25)            # 22 h 00 → demain 6 h 15
    dlg._refresh()
    assert "demain" in dlg.end_preview.text() and "6 h 15" in dlg.end_preview.text()
    dlg._now = lambda: dt.datetime(2026, 10, 20, 5, 0)
    assert dlg.planned_hours() == pytest.approx(1.25)            # l'heure n'est pas passée : aujourd'hui
    dlg._refresh()
    assert "aujourd'hui" in dlg.end_preview.text()
    dlg.deleteLater()


def test_night_dialog_cameras_and_record_modes(qapp):
    dlg = sd.NightSessionDialog({"camera_index": 2, "record_video": "aucune"}, CAMS)
    assert dlg.camera.count() == 2 and dlg.result_dict()["camera_index"] == 2
    assert dlg.selected_frame_wh() == (1280, 720)
    assert not dlg.segment.isEnabled() and not dlg.reduced_row.isVisibleTo(dlg)
    dlg.record_mode.setCurrentIndex(sd.RECORD_MODES.index("reduite"))
    assert dlg.segment.isEnabled() and dlg.reduced_row.isVisibleTo(dlg)
    assert "½ résolution, 10 img/s" in dlg.record_mode.itemText(1)
    dlg.record_scale.setCurrentIndex(2)
    dlg.record_fps.setValue(5)
    assert "¼ résolution, 5 img/s" in dlg.record_mode.itemText(1)
    d = dlg.result_dict()
    assert d["record_video"] == "reduite" and d["record_scale"] == pytest.approx(0.25) and d["record_fps"] == 5.0
    dlg.deleteLater()
    dlg = sd.NightSessionDialog({"camera_index": 3}, [])           # aucune caméra connue
    assert dlg.result_dict()["camera_index"] == 3 and "Caméra 3" in dlg.camera.currentText()
    dlg.deleteLater()


def test_night_dialog_disk_estimate(qapp, tmp_path):
    dlg = sd.NightSessionDialog({"output_dir": str(tmp_path), "duration_h": 10})
    dlg._longrun = fake_longrun(est=5 * 1024 ** 3, free=200 * 1024 ** 3)
    dlg._refresh()
    assert dlg.estimate_bytes == 5 * 1024 ** 3 and dlg.space_ok
    assert "5,0 Go" in dlg.est_main.text() and "200 Go" in dlg.est_sub.text()
    assert dlg.est_box.property("state") == "ok"
    cfg_cls = dlg._longrun[0]
    assert cfg_cls.seen[-1]["duration_h"] == 10.0 and set(cfg_cls.seen[-1]) == NIGHT_KEYS
    dlg._longrun = fake_longrun(est=300 * 1024 ** 3, free=100 * 1024 ** 3)
    dlg._refresh()
    assert not dlg.space_ok and dlg.est_box.property("state") == "bad"
    assert "Espace insuffisant" in dlg.est_warn.text() and "Lancer quand même" in dlg.ok_btn.text()
    dlg._longrun = None                                          # moteur absent : pas d'estimation
    dlg._refresh()
    assert dlg.estimate_bytes is None and "Estimation indisponible" in dlg.est_main.text()
    assert dlg.est_box.property("state") == "na"
    dlg.deleteLater()


def test_night_dialog_accept_and_validation(qapp, tmp_path, monkeypatch):
    dlg = sd.NightSessionDialog({"output_dir": str(tmp_path)})
    dlg.ok_btn.click()
    assert dlg.result() == QDialog.DialogCode.Accepted
    dlg.deleteLater()
    bad = missing_drive()
    if bad:
        dlg = sd.NightSessionDialog({"output_dir": bad})
        dlg.ok_btn.click()
        assert dlg.result() != QDialog.DialogCode.Accepted
        assert "introuvable" in dlg.error_label.text()
        dlg.deleteLater()
    # dossier choisi avec le sélecteur
    dlg = sd.NightSessionDialog()
    monkeypatch.setattr(sd.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    dlg.output.button.click()
    assert dlg.result_dict()["output_dir"] == os.path.normpath(str(tmp_path))
    dlg.deleteLater()


# ================================================================ vidéos reçues ====
def make_files(folder: Path, names) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = folder / n
        p.write_bytes(b"x" * 2048)
        out.append(str(p))
    return out


def test_batch_dialog_validation_and_schema(qapp, tmp_path):
    dlg = sd.BatchDialog()
    assert not dlg.ok_btn.isEnabled()
    assert "au moins une vidéo" in dlg.validate()
    files = make_files(tmp_path / "recues", ["a.mp4", "b.AVI", "notes.txt"])
    assert dlg.add_files(files) == 2                          # le .txt est ignoré
    assert dlg.add_files(files[:1]) == 0                      # doublon
    assert dlg.ok_btn.isEnabled() and dlg.validate() == ""
    assert "Analyser 2 vidéos" in dlg.ok_btn.text()
    assert "ignoré" in dlg.summary.text()
    d = dlg.result_dict()
    assert set(d) == BATCH_KEYS
    assert d["files"] == [os.path.normpath(f) for f in files[:2]]
    assert d["watch_folder"] == "" and d["make_reports"] is True and isinstance(d["mass_g"], float)
    dlg.file_list.item(0).setSelected(True)
    dlg.remove_selected()
    assert len(dlg.files()) == 1
    dlg.clear_files()
    assert dlg.files() == [] and not dlg.ok_btn.isEnabled()
    dlg.deleteLater()


def test_batch_dialog_watch_folder_only(qapp, tmp_path):
    dlg = sd.BatchDialog({"watch_folder": str(tmp_path), "make_reports": False, "condition": "KCl", "mass_g": 50})
    assert dlg.watch_check.isChecked() and dlg.ok_btn.isEnabled()
    assert "Surveiller le dossier" in dlg.ok_btn.text()
    d = dlg.result_dict()
    assert d["files"] == [] and d["watch_folder"] == str(tmp_path) and d["make_reports"] is False
    assert d["condition"] == "KCl" and d["mass_g"] == 50.0 and d["sample"].lower().startswith("kcl")
    dlg.watch_field.set_path(str(tmp_path / "absent"))
    assert not dlg.ok_btn.isEnabled() and "introuvable" in dlg.validate()
    dlg.watch_check.setChecked(False)                         # surveillance désactivée : plus rien à faire
    assert dlg.result_dict()["watch_folder"] == "" and not dlg.ok_btn.isEnabled()
    dlg.deleteLater()


def test_batch_dialog_folder_drop_and_file_picker(qapp, tmp_path, monkeypatch):
    folder = tmp_path / "OneDrive"
    make_files(folder, ["v1.mp4", "v2.mov", "lisez-moi.txt"])
    dlg = sd.BatchDialog()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(folder))])
    ev = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier)
    dlg.file_list.dropEvent(ev)
    assert [Path(f).name for f in dlg.files()] == ["v1.mp4", "v2.mov"]
    more = make_files(tmp_path / "autres", ["v3.mkv"])
    monkeypatch.setattr(sd.QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: (more, "")))
    dlg.add_btn.click()
    assert len(dlg.files()) == 3
    dlg.ok_btn.click()
    assert dlg.result() == QDialog.DialogCode.Accepted
    dlg.deleteLater()


# ================================================================ bandeau de nuit ====
STATUS = {"elapsed_s": 3 * 3600 + 720, "remaining_s": 5 * 3600 + 2880, "next_autosave_s": 130,
          "disk_free_bytes": 231 * 1024 ** 3, "segment_index": 6, "recording": True, "n_tracks": 12873,
          "rate_per_min": 61.4, "last_checkpoint": "2026-10-21T02:41:10",
          "warnings": ["Aucune trace <valide> depuis 22 min"], "session_folder": "C:/Sessions/nuit"}


def test_status_bar_update(qapp):
    bar = sd.SessionStatusBar()
    bar.update_status({})                                    # clés manquantes tolérées
    bar.update_status(STATUS)
    assert bar.elapsed.text() == "3 h 12 min"
    assert "Reste 5 h 48 min" in bar.remaining.text()
    assert bar.progress.maximum() == 1000 and bar.progress.value() == pytest.approx(355, abs=2)
    assert bar.tile_tracks.value.text() == "12\u00a0873"
    assert "61,4" in bar.tile_rate.value.text()
    assert bar.tile_disk.value.text() == "231 Go"
    assert "Segment" in bar.tile_rec.value.text() and "7" in bar.tile_rec.value.text()
    assert "2 h 41" in bar.footer.text() and "2 min 10 s" in bar.footer.text()
    assert bar.warnings.isVisibleTo(bar) and "&lt;valide&gt;" in bar.warnings.text()
    assert bar.property("alert") is True
    assert bar.folder_btn.isVisibleTo(bar)
    bar.update_status(dict(STATUS, remaining_s=-1, warnings=[], disk_free_bytes=1024 ** 3, recording=False))
    assert bar.progress.maximum() == 0                       # durée inconnue : barre « occupée »
    assert "arrêt manuel" in bar.remaining.text()
    assert not bar.warnings.isVisibleTo(bar) and bar.property("alert") is False
    assert "color" in bar.tile_disk.value.styleSheet()       # disque presque plein : en rouge
    assert bar.tile_rec.value.text() == "Désactivée"
    bar.deleteLater()


def test_status_bar_stop_needs_confirmation(qapp):
    bar = sd.SessionStatusBar()
    got = collect(bar.stop_requested)
    bar.stop_btn.click()
    assert got == [] and "encore" in bar.stop_btn.text()
    bar.stop_btn.click()
    assert len(got) == 1 and not bar.stop_btn.isEnabled()
    bar.deleteLater()
    bar = sd.SessionStatusBar(confirm_stop=False)
    got = collect(bar.stop_requested)
    bar.stop_btn.click()
    assert len(got) == 1
    bar.set_finished()
    assert not bar.stop_btn.isVisibleTo(bar) and bar.property("finished") is True
    assert bar.progress.value() == 1000 and "Durée totale" in bar.remaining.text()
    bar.reset()
    assert bar.stop_btn.isVisibleTo(bar) and bar.stop_btn.isEnabled() and bar.property("finished") is False
    bar.deleteLater()


def test_status_bar_open_folder_signal(qapp):
    bar = sd.SessionStatusBar()
    got = collect(bar.open_folder_requested)
    bar.update_status(STATUS)
    bar.folder_btn.click()
    assert got == ["C:/Sessions/nuit"]
    bar.deleteLater()


# ================================================================ file du lot ====
JOBS = [
    {"path": "C:/r/a.mp4", "status": "terminé", "progress": 1, "eta_s": -1, "result_folder": "C:/res/a",
     "counts": {"alpha": 3, "beta": 20, "muon": 9, "proton": 1}, "error": ""},
    {"path": "C:/r/b.mp4", "name": "b.mp4", "status": "en cours", "progress": 0.4, "eta_s": 300, "result_folder": "",
     "counts": {"alpha": 1}, "error": ""},
    {"path": "C:/r/c.mp4", "status": "en attente", "progress": 0, "eta_s": -1, "from_watch": True,
     "warnings": ["arrivée pendant la copie"]},
    {"path": "C:/r/d.mp4", "status": "erreur", "progress": 0.1, "error": "Codec inconnu"},
]


def test_batch_queue_widget(qapp):
    w = sd.BatchQueueWidget()
    assert w.empty.isVisibleTo(w) and not w.cancel_btn.isEnabled()
    w.update_jobs(JOBS)
    t = w.table
    assert t.rowCount() == 4 and not w.empty.isVisibleTo(w)
    assert t.item(0, 0).text() == "a.mp4" and "Terminé" in t.item(0, 1).text()
    assert t.item(2, 0).text().startswith("👁") and "dossier surveillé" in t.item(2, 0).toolTip()
    assert "En cours" in t.item(1, 1).text() and "Erreur" in t.item(3, 1).text()
    assert "Codec inconnu" in t.item(3, 1).toolTip()
    bar = t.cellWidget(1, 2).findChild(QProgressBar)
    assert bar.value() == 400
    assert t.item(0, 5).text() == "20" and t.item(2, 5).text() == "—"
    assert t.item(1, 3).text() == "5 min 00 s"
    summary = re.sub("<[^>]+>", "", w.summary.text()).replace("&nbsp;", " ")
    assert "1 terminée" in summary and "1 en cours" in summary and "1 erreur" in summary
    assert w.cancel_btn.isEnabled()
    first_bar = bar
    w.update_jobs([dict(JOBS[0]), dict(JOBS[1], progress=0.9), JOBS[2], JOBS[3]])   # lignes réutilisées
    assert t.cellWidget(1, 2).findChild(QProgressBar) is first_bar and first_bar.value() == 900
    cancel = collect(w.cancel_requested)
    w.cancel_btn.click()
    assert len(cancel) == 1
    opened = collect(w.open_folder_requested)
    btn = t.cellWidget(0, 8)
    assert isinstance(btn, QPushButton) and btn.isEnabled()
    btn.click()
    assert opened == ["C:/res/a"]
    t.cellWidget(3, 8).click()                               # erreur : message, pas de dossier
    assert opened == ["C:/res/a"]
    w.update_jobs([dict(j, status="terminé") for j in JOBS[:2]])
    assert t.rowCount() == 2 and not w.cancel_btn.isEnabled()
    w.update_jobs([])
    assert w.empty.isVisibleTo(w) and t.rowCount() == 0
    w.deleteLater()


def test_normalize_job_is_tolerant():
    j = sd.normalize_job({"path": "x/y.mp4", "status": "TERMINÉ", "progress": "abc", "eta_s": None})
    assert j["status"] == "terminé" and j["progress"] == 1.0 and j["eta_s"] == -1.0 and j["name"] == "y.mp4"
    j = sd.normalize_job({"status": "running", "progress": 7})
    assert j["status"] == "en cours" and j["progress"] == 1.0
    assert sd.normalize_job(None)["status"] == "en attente"


# ================================================================ centre d'apprentissage ====
def test_learning_center_without_model(qapp, monkeypatch):
    from wilsontrack.detection import reference_model as rm
    monkeypatch.setattr(rm.ReferenceModel, "load_default", classmethod(lambda cls: None))
    w = lc_mod.LearningCenter(examples_dir=Path("absent_dir_xyz"))
    s = w.summary
    assert s["available"] is False and s["splits"] == {"test": None, "stress": None}
    assert w.missing_box.isVisibleTo(w)
    assert all(t.value.text() == "—" for t in w.tiles)
    assert not w.confusion.isVisibleTo(w)
    assert "Aucune matrice" in w.confusion_note.text()
    assert len(w.result_cards) == 2
    assert "SIMULÉES" in w.honest_text.text() and "incertaine" in w.honest_text.text()
    assert not w._examples_host.isVisibleTo(w)
    got = collect(w.validate_requested)
    w.validate_btn.click()
    assert len(got) == 1
    w.deleteLater()


def test_learning_center_with_installed_model(qapp):
    from wilsontrack.detection.reference_model import ReferenceModel
    model = ReferenceModel.load_default()
    if model is None:
        pytest.skip("modèle de référence absent ou incompatible")
    w = lc_mod.LearningCenter()
    s = w.summary
    assert s["available"] and s["n_tracks_total"] == model.info["n_tracks_total"]
    assert not w.missing_box.isVisibleTo(w)
    test = s["splits"]["test"]
    assert test is not None and test["n"] > 0
    assert set(test["per_class"]) >= {"alpha", "beta", "muon"}
    for v in test["per_class"].values():
        assert v["recall"] is None or 0.0 <= v["recall"] <= 1.0
    conf = s["confusion"]
    assert conf is not None and len(conf["matrix"]) == len(conf["labels_true"])
    assert w.confusion.isVisibleTo(w)
    assert w.tiles[0].value.text().replace("\u00a0", " ") == f"{model.info['n_tracks_total']:,}".replace(",", " ")
    w.deleteLater()


class _ContractModel:
    """Faux modèle au format du contrat (clés de info définies par l'agent « IA »)."""
    def __init__(self):
        per = {"alpha": {"precision": 0.97, "recall": 0.95, "n": 400}, "beta": {"precision": 0.9, "recall": 0.88, "n": 900},
               "muon": {"precision": 0.86, "recall": 0.8, "n": 700}, "proton": {"precision": 0.9, "recall": 0.7, "n": 60},
               "artefact": {"precision": 0.8, "recall": 0.75, "n": 120}}
        m = {"n": 2180, "coverage": 0.87, "accuracy_confident": 0.94, "balanced_accuracy": 0.82, "per_class": per}
        self.info = {"n_tracks_total": 52000, "n_train": 40000, "n_jobs": 900,
                     "classes": ["alpha", "beta", "muon", "proton", "artefact"], "threshold": 0.85,
                     "created": "2026-10-20T03:00:00", "model_type": "HistGradientBoostingClassifier",
                     "generator": {"fps": [15, 60], "noise_sigma": [1.0, 8.0]},
                     "kinds": {"alpha_rn222": 9000, "thoron_v": 800, "compton": 5000, "delta": 700},
                     "metrics": {"test": m, "stress": dict(m, coverage=0.7)},
                     "confusion_test": {"labels_true": ["alpha", "beta", "muon", "proton", "artefact"],
                                        "labels_pred": ["alpha", "beta", "muon", "proton", "artefact", "incertain"],
                                        "matrix": [[380, 0, 2, 3, 0, 15], [0, 790, 60, 0, 5, 45],
                                                   [0, 70, 560, 0, 2, 68], [8, 0, 4, 42, 0, 6],
                                                   [1, 6, 2, 0, 90, 21]]}}


def test_learning_center_contract_format(qapp, monkeypatch, tmp_path):
    from wilsontrack.detection import reference_model as rm
    monkeypatch.setattr(rm.ReferenceModel, "load_default", classmethod(lambda cls: _ContractModel()))
    for k, cls in enumerate(("alpha", "beta", "muon", "artefact")):
        pm = QPixmap(40, 40)
        pm.fill(QColor(20, 20, 20))
        pm.save(str(tmp_path / f"{cls}_{k + 1}.png"))
    (tmp_path / "pas_une_vignette.png").write_bytes(b"")
    w = lc_mod.LearningCenter(examples_dir=tmp_path)
    s = w.summary
    assert s["threshold"] == 0.85 and s["n_jobs"] == 900
    assert s["splits"]["test"]["legacy"] is False and s["splits"]["stress"]["coverage"] == 0.7
    assert s["confusion"]["labels_pred"][-1] == "incertain"
    assert ("paires en « V » du thoron", 800) in s["kinds"]
    assert any("85" in t.value.text() for t in w.tiles)
    assert w._examples_host.isVisibleTo(w)
    assert len(w._examples_host.findChildren(lc_mod._Thumb)) == 4
    assert "± " in w.honest_text.text() or "points" in w.honest_text.text()
    w.deleteLater()


def test_learning_center_show_validation(qapp):
    w = lc_mod.LearningCenter(examples_dir=Path("absent_dir_xyz"))
    w.show_validation({"n": 64, "accuracy": 0.81,
                       "per_class": {"alpha": {"precision": 0.95, "recall": 0.9, "n": 20}},
                       "confusion": {"labels_true": ["alpha", "beta"], "labels_pred": ["alpha", "beta", "incertain"],
                                     "matrix": [[18, 1, 1], [0, 20, 4]]},
                       "note": "64 traces corrigées"})
    texts = " ".join(lab.text() for lab in w.valid_box.findChildren(lc_mod.QLabel))
    assert "64" in texts and "81" in texts and "64 traces corrigées" in texts
    assert w.valid_box.findChildren(lc_mod.ConfusionMatrix)
    w.show_validation({"n": 0, "accuracy": None, "per_class": {}, "confusion": None, "note": ""})
    texts = " ".join(lab.text() for lab in w.valid_box.findChildren(lc_mod.QLabel))
    assert "Aucune trace corrigée" in texts
    w.show_validation(None)
    assert w.validation is None
    w.deleteLater()


def test_pilot_model_read_by_new_info_format():
    """info au format du contrat mais vide (modèle pilote relu) → détail repris de model.metrics, sans seuil."""
    legacy = {"alpha": {"alpha": 9, "beta": 1, "incertain": 0}, "beta": {"alpha": 0, "beta": 10, "incertain": 0}}

    class Pilot:
        info = {"n_tracks_total": 1036, "threshold": 0.85, "format": 1, "confusion_test": None,
                "metrics": {"test": {"n": 20, "coverage": None, "accuracy_confident": 0.95, "per_class": {},
                                     "rules": 0.6, "ml": 0.9, "hybrid": 0.92}}}
        metrics = {"test": {"hybrid": {"confusion": legacy, "balanced_accuracy": 0.92},
                            "rules": {"balanced_accuracy": 0.6}}}
    s = lc_mod.summarize_model(Pilot())
    t = s["splits"]["test"]
    assert t["legacy"] and set(t["per_class"]) == {"alpha", "beta"} and t["coverage"] == 1.0
    assert t["methods"] == {"rules": 0.6, "ml": 0.9, "hybrid": 0.92}
    assert s["confusion"]["matrix"] == [[9, 1, 0], [0, 10, 0]] and s["legacy"] is True
    card = lc_mod._ResultCard("📝", "Examen", "", t, 0.85)
    texts = " ".join(lab.text() for lab in card.findChildren(lc_mod.QLabel))
    assert "(seuil" not in texts and "modèle pilote" in texts      # le seuil 85 % n'a pas servi à ces chiffres
    card.deleteLater()


def test_show_validation_accepts_evaluate_on_labeled_output(qapp):
    w = lc_mod.LearningCenter(examples_dir=Path("absent_dir_xyz"))
    w.show_validation({"n": 50, "accuracy": 0.9, "coverage": 0.8, "accuracy_all": 0.72,
                       "accuracy_ci95": [0.77, 0.96], "n_confident": 40, "threshold": 0.85,
                       "per_class": {"alpha": {"precision": 1.0, "recall": 0.9, "n": 20},
                                     "artefact": {"precision": None, "recall": 0.5, "n": 4}},
                       "confusion": {"labels_true": ["alpha", "artefact"],
                                     "labels_pred": ["alpha", "artefact", "incertain"],
                                     "matrix": [[18, 0, 2], [0, 2, 2]]},
                       "note": "Mesuré sur 50 traces étiquetées."})
    texts = " ".join(lab.text() for lab in w.valid_box.findChildren(lc_mod.QLabel)).replace("\xa0", " ")
    assert "dont 40 classées" in texts and "± 9<" in texts      # IC fourni : [77 ; 96] % → ± 9 points
    w.deleteLater()


def test_learning_helpers():
    legacy = {"alpha": {"alpha": 9, "beta": 0, "incertain": 1}, "beta": {"alpha": 1, "beta": 8, "incertain": 1}}
    c = lc_mod.normalize_confusion(legacy)
    assert c["labels_true"] == ["alpha", "beta"] and c["labels_pred"] == ["alpha", "beta", "incertain"]
    assert c["matrix"] == [[9, 0, 1], [1, 8, 1]]
    assert lc_mod.normalize_confusion({"labels_true": ["a"], "labels_pred": ["a"], "matrix": [[1, 2]]}) is None
    assert lc_mod.normalize_confusion(None) is None
    m = lc_mod.normalize_metrics(None, {"hybrid": {"confusion": legacy, "balanced_accuracy": 0.85}})
    assert m["legacy"] and m["n"] == 20 and m["coverage"] == pytest.approx(0.9)
    assert m["accuracy_confident"] == pytest.approx(17 / 18) and m["balanced_accuracy"] == 0.85
    assert lc_mod.normalize_metrics({"rules": 0.6}, None) is None
    lo, hi = lc_mod.wilson_interval(90, 100)
    assert lo < 0.9 < hi and 0.82 < lo and hi < 0.95
    assert lc_mod.wilson_interval(0, 0) is None
