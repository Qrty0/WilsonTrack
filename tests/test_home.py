"""Tests de la page d'accueil (wilsontrack.gui.home) et de la boîte « Analyser une vidéo » (hors écran)."""
from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from wilsontrack import physics  # noqa: E402
from wilsontrack.gui import home as home_mod  # noqa: E402
from wilsontrack.gui.home import (MAX_RECENT_VIDEOS, HomePage, VideoSetupDialog, find_example_videos,  # noqa: E402
                                  fmt_num, guess_condition)
from wilsontrack.settings import Settings  # noqa: E402

VIDEO_KEYS = {"path", "condition", "sample", "mass_g", "px_per_mm", "realtime"}


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    from wilsontrack.gui.app import QSS, dark_palette
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(QSS)
    return app


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "settings.json")


@pytest.fixture
def page(qapp, settings):
    p = HomePage(settings)
    p.resize(1200, 800)
    p.show()
    qapp.processEvents()
    yield p
    p.close()
    p.deleteLater()
    qapp.processEvents()


def make_video(path: Path, n: int = 12, size=(64, 48), fps: float = 10.0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened()
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), 20 + 3 * i, np.uint8))
    w.release()
    return path


def collect(signal) -> list:
    got = []
    signal.connect(lambda *a: got.append(a[0] if len(a) == 1 else a))
    return got


def accept_dialog(monkeypatch, page, captured: list | None = None):
    def run(dlg):
        if captured is not None:
            captured.append(dlg.result_dict())
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(page, "_run_dialog", run)


# ------------------------------------------------------------------------------------------------
def test_find_example_videos():
    vids = find_example_videos()
    assert all(isinstance(p, Path) and p.suffix.lower() in (".mp4", ".avi", ".mov") for p in vids)
    assert [p.name.lower() for p in vids] == sorted(p.name.lower() for p in vids)
    root_ex = Path(home_mod.__file__).resolve().parents[2] / "exemples"
    if root_ex.is_dir():
        expected = sorted(p.name.lower() for p in root_ex.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".mov"))
        assert [p.name.lower() for p in vids] == expected


def test_find_example_videos_frozen(tmp_path, monkeypatch):
    ex = tmp_path / "exemples"
    ex.mkdir()
    for n in ("b.MP4", "a.avi", "c.mov", "notes.txt", "d.mp4.gt.json"):
        (ex / n).write_bytes(b"x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "WilsonTrack.exe"))
    assert [p.name for p in find_example_videos()] == ["a.avi", "b.MP4", "c.mov"]


def test_fmt_num_uses_decimal_comma():
    assert fmt_num(3.84, 2) == "3,84"
    assert fmt_num(1800, 0) == "1\u00a0800"


def test_buttons_emit_signals(page):
    sig = {name: collect(getattr(page, name)) for name in
           ("photos_requested", "demo_requested", "compare_requested", "results_requested", "assistant_requested",
            "guide_requested", "camera_refresh_requested", "learning_requested")}
    page.learning_btn.click()
    assert len(sig["learning_requested"]) == 1
    page.photos_btn.click()
    page.demo_btn.click()
    page.demo_kcl_btn.click()
    page.compare_btn.click()
    page.results_btn.click()
    page.results_btn2.click()
    page.assistant_btn.click()
    page.guide_btn.click()
    page.cam_refresh_btn.click()
    assert len(sig["photos_requested"]) == 1
    assert sig["demo_requested"] == ["demo", "demo_kcl"]
    assert len(sig["compare_requested"]) == 1
    assert len(sig["results_requested"]) == 2
    assert len(sig["assistant_requested"]) == 1
    assert len(sig["guide_requested"]) == 1
    assert len(sig["camera_refresh_requested"]) == 1
    assert not page.cam_refresh_btn.isEnabled()          # recherche en cours…
    page.set_cameras([])
    assert page.cam_refresh_btn.isEnabled()              # … terminée


def test_cards_are_clickable_and_keyboard_accessible(page):
    demos = collect(page.demo_requested)
    assistant = collect(page.assistant_requested)
    photos = collect(page.photos_requested)
    QTest.mouseClick(page.cards["demo"], Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(12, 12))
    QTest.keyClick(page.cards["assistant"], Qt.Key.Key_Return)
    QTest.keyClick(page.cards["photos"], Qt.Key.Key_Space)
    learning = collect(page.learning_requested)
    QTest.keyClick(page.cards["learning"], Qt.Key.Key_Return)
    assert demos == ["demo"]
    assert len(assistant) == 1 and len(photos) == 1 and len(learning) == 1
    assert set(page.cards) == {"camera", "video", "night", "photos", "batch", "demo", "results", "learning",
                               "assistant"}
    for card in page.cards.values():
        assert card.focusPolicy() & Qt.FocusPolicy.TabFocus
        assert card.cursor().shape() == Qt.CursorShape.PointingHandCursor


def test_keyboard_tab_order_follows_reading_order(page, tmp_path):
    f = tmp_path / "recente.mp4"
    f.write_bytes(b"x")
    page.add_recent_video(str(f))
    chain, w = [], page.cards["camera"]
    for _ in range(80):
        chain.append(w)
        w = w.nextInFocusChain()
        if w is page.cards["camera"]:
            break
    idx = {k: chain.index(c) for k, c in page.cards.items()}
    # cartes essentielles (2 × 2) puis la rangée « Plus d'outils »
    order = ["camera", "video", "night", "results", "photos", "demo", "batch", "learning", "assistant"]
    assert [idx[k] for k in order] == sorted(idx[k] for k in order)
    rows = page.recent_rows + page.example_rows
    assert rows and all(chain.index(page.video_btn) < chain.index(r) < idx["night"] for r in rows)


def test_cameras(page, settings):
    got = collect(page.camera_requested)
    page.set_cameras([{"index": 0, "label": "Caméra 0 (1920×1080)", "width": 1920, "height": 1080},
                      {"index": 2, "label": "Caméra 2 (1280×720)", "width": 1280, "height": 720}])
    assert page.cam_combo.count() == 2 and page.cam_combo.isEnabled()
    page.cam_combo.setCurrentIndex(1)
    page.cam_start_btn.click()
    assert got == [2]
    assert settings.get("cam_index") == 2
    # la caméra mémorisée est re-sélectionnée
    page.set_cameras([{"index": 0, "label": "A"}, {"index": 2, "label": "B"}])
    assert page.cam_combo.currentData() == 2
    page.set_cameras([])
    assert page.cam_combo.count() == 0
    assert page.cam_combo.placeholderText() == "Aucune caméra détectée"
    page.cam_start_btn.click()
    assert got == [2, 0]


def test_add_recent_video_dedupe_and_limit(page, settings, tmp_path):
    files = []
    for i in range(10):
        f = tmp_path / f"video {i}.mp4"
        f.write_bytes(b"x")
        files.append(f)
    for f in files:
        page.add_recent_video(str(f))
    rec = page.recent_videos()
    assert len(rec) == MAX_RECENT_VIDEOS == 8
    assert [Path(p).name for p in rec] == [f"video {i}.mp4" for i in range(9, 1, -1)]
    page.add_recent_video(str(files[5]))                 # déjà présente → remonte en tête, sans doublon
    rec = page.recent_videos()
    assert Path(rec[0]).name == "video 5.mp4" and len(rec) == 8
    assert len({os.path.normcase(p) for p in rec}) == 8
    if sys.platform == "win32":                         # même fichier, autre casse
        page.add_recent_video(str(files[7]).upper())
        assert len(page.recent_videos()) == 8
    page.add_recent_video(str(tmp_path / "absente.mp4"))  # fichier inexistant : ignoré
    assert all(Path(p).exists() for p in page.recent_videos())
    files[9].unlink()                                    # fichier supprimé : retiré de la liste
    assert "video 9.mp4" not in [Path(p).name for p in page.recent_videos()]
    stored = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["recent_videos"]
    assert len(stored) <= 8 and len(stored) == len({os.path.normcase(p) for p in stored})
    # les lignes « Vidéos récentes » sont affichées (au plus 4) et cliquables
    assert 1 <= len(page.recent_rows) <= 4


def test_dialog_autofills_scale_from_gt_json(qapp, tmp_path):
    vid = make_video(tmp_path / "essai KCl.avi")
    Path(str(vid) + ".gt.json").write_text(json.dumps({"px_per_mm": 3.84, "tracks": []}), encoding="utf-8")
    dlg = VideoSetupDialog(str(vid), {"px_per_mm": 1.0})
    assert dlg.scale.value() == pytest.approx(3.84)
    assert dlg.scale_note.isVisibleTo(dlg)
    assert "échelle connue" in dlg.scale_note.text()
    assert dlg.ok_btn.isEnabled()
    assert dlg.info["width"] == 64 and dlg.info["height"] == 48 and dlg.info["n_frames"] == 12
    assert dlg.info_values["Résolution"].text().replace("\u00a0", " ") == "64 × 48"
    assert dlg.info_values["Nombre d'images"].text() == "12"
    d = dlg.result_dict()
    assert set(d) == VIDEO_KEYS
    assert d["px_per_mm"] == pytest.approx(3.84) and d["condition"] == "KCl"
    assert d["sample"].lower().startswith("kcl")
    dlg.deleteLater()


def test_dialog_gt_json_with_suffix_variant_and_without(qapp, tmp_path):
    vid = make_video(tmp_path / "autre.avi")
    dlg = VideoSetupDialog(str(vid), {"px_per_mm": 2.5})
    assert dlg.scale.value() == pytest.approx(2.5)        # pas de vérité terrain : valeur par défaut
    assert not dlg.scale_note.isVisibleTo(dlg)
    dlg.deleteLater()
    (tmp_path / "autre.gt.json").write_text(json.dumps({"px_per_mm": 5.0}), encoding="utf-8")
    dlg = VideoSetupDialog(str(vid))
    assert dlg.scale.value() == pytest.approx(5.0)
    dlg.deleteLater()


def test_dialog_unreadable_file(qapp, tmp_path):
    bad = tmp_path / "corrompue.mp4"
    bad.write_bytes(b"pas une video" * 40)
    dlg = VideoSetupDialog(str(bad))
    assert not dlg.ok_btn.isEnabled()
    assert dlg.error_label.isVisibleTo(dlg)
    assert "Impossible de lire" in dlg.error_label.text()
    dlg.deleteLater()
    dlg = VideoSetupDialog(str(tmp_path / "introuvable.mp4"))
    assert not dlg.ok_btn.isEnabled()
    dlg.deleteLater()


def test_dialog_condition_rules(qapp, tmp_path):
    vid = make_video(tmp_path / "video.avi")
    dlg = VideoSetupDialog(str(vid), {"condition": "Autre", "sample": "Aucun", "mass_g": 0})
    kcl = next(k for k in physics.SAMPLES if k.lower().startswith("kcl"))
    dlg.condition.setCurrentText("KCl")
    assert dlg.sample.currentText() == kcl
    dlg.mass.setValue(100)
    assert "Bq" in dlg.activity.text()
    dlg.condition.setCurrentText("Fond (rayons cosmiques)")
    assert dlg.sample.currentText() == "Aucun"
    assert dlg.mass.value() == 0
    dlg.realtime.setChecked(True)
    d = dlg.result_dict()
    assert d["condition"] == "Fond (rayons cosmiques)" and d["mass_g"] == 0.0 and d["realtime"] is True
    assert isinstance(d["px_per_mm"], float) and isinstance(d["mass_g"], float)
    dlg.deleteLater()
    # condition devinée d'après le nom du fichier, et défauts respectés
    fond = make_video(tmp_path / "mesure fond 2.avi")
    assert guess_condition(fond) == "Fond (rayons cosmiques)"
    dlg = VideoSetupDialog(str(fond))
    assert dlg.result_dict()["condition"] == "Fond (rayons cosmiques)"
    dlg.deleteLater()
    dlg = VideoSetupDialog(str(vid), {"condition": "Banane", "mass_g": 150, "realtime": True})
    d = dlg.result_dict()
    assert d["condition"] == "Banane" and d["sample"].lower().startswith("banane")
    assert d["mass_g"] == 150.0 and d["realtime"] is True
    dlg.deleteLater()


def test_open_video_setup_emits_and_remembers(page, settings, tmp_path, monkeypatch):
    vid = make_video(tmp_path / "essai KCl 3.avi")
    got = collect(page.video_requested)
    monkeypatch.setattr(page, "_run_dialog", lambda dlg: QDialog.DialogCode.Rejected)
    assert page.open_video_setup(str(vid)) is None
    assert got == []
    accept_dialog(monkeypatch, page)
    d = page.open_video_setup(str(vid))
    assert got == [d] and set(d) == VIDEO_KEYS
    assert d["path"] == str(vid) and d["condition"] == "KCl"
    assert os.path.normcase(page.recent_videos()[0]) == os.path.normcase(str(vid))
    assert settings.get("video_defaults")["condition"] == "KCl"


def test_scale_default_from_calibration_then_remembered_per_video(page, settings, tmp_path, monkeypatch):
    vid = make_video(tmp_path / "enregistrement.avi")
    settings.set("config", {"physics": {"px_per_mm": 4.2}})
    d = page._video_defaults(str(vid))
    assert d["px_per_mm"] == pytest.approx(4.2) and d["scale_source"] == "calibration"
    dlg = VideoSetupDialog(str(vid), d)
    assert dlg.scale.value() == pytest.approx(4.2)
    assert dlg.scale_note.isVisibleTo(dlg) and "calibration" in dlg.scale_note.text()
    dlg.scale.setValue(3.0)                              # valeur modifiée : la note ne s'applique plus
    assert not dlg.scale_note.isVisibleTo(dlg)
    dlg.deleteLater()

    def run(dlg):
        dlg.scale.setValue(2.75)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(page, "_run_dialog", run)
    assert page.open_video_setup(str(vid))["px_per_mm"] == pytest.approx(2.75)
    d2 = page._video_defaults(str(vid))
    assert d2["px_per_mm"] == pytest.approx(2.75) and d2["scale_source"] == "video"
    settings.set("config", {})
    d3 = page._video_defaults(str(tmp_path / "jamais vue.avi"))
    assert d3["px_per_mm"] == 0 and d3["scale_source"] == ""


def test_choose_video_uses_file_dialog_and_last_dir(page, settings, tmp_path, monkeypatch):
    vid = make_video(tmp_path / "choisie.avi")
    seen = {}

    def fake_open(parent, title, start, flt):
        seen["start"], seen["filter"] = start, flt
        return str(vid), flt
    monkeypatch.setattr(home_mod.QFileDialog, "getOpenFileName", staticmethod(fake_open))
    accept_dialog(monkeypatch, page)
    got = collect(page.video_requested)
    page.video_btn.click()
    assert len(got) == 1 and got[0]["path"] == str(vid)
    assert settings.get("last_dir") == str(tmp_path)
    for ext in ("mp4", "avi", "mov", "mkv", "webm", "wmv", "m4v"):
        assert f"*.{ext}" in seen["filter"]
    # annulation du sélecteur : rien n'est émis
    monkeypatch.setattr(home_mod.QFileDialog, "getOpenFileName", staticmethod(lambda *a: ("", "")))
    page.video_btn.click()
    assert len(got) == 1


def test_example_rows_open_dialog(page, monkeypatch):
    if not page.example_rows:
        pytest.skip("pas de vidéos d'exemple dans ce dépôt")
    captured = []
    accept_dialog(monkeypatch, page, captured)
    got = collect(page.video_requested)
    page.example_rows[-1].clicked.emit()
    assert len(got) == 1
    if home_mod.ground_truth_scale(got[0]["path"]):
        assert got[0]["px_per_mm"] > 0


def test_drag_and_drop_video(qapp, page, tmp_path, monkeypatch):
    vid = make_video(tmp_path / "glissee.avi")
    accept_dialog(monkeypatch, page)
    got = collect(page.video_requested)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(vid))])
    enter = QDragEnterEvent(QPoint(50, 50), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    page.dragEnterEvent(enter)
    assert enter.isAccepted()
    drop = QDropEvent(QPointF(50, 50), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier)
    page.dropEvent(drop)
    for _ in range(5):
        qapp.processEvents()
    assert len(got) == 1 and got[0]["path"] == str(vid)
    # un fichier qui n'est pas une vidéo est refusé
    txt = tmp_path / "notes.txt"
    txt.write_text("x", encoding="utf-8")
    mime2 = QMimeData()
    mime2.setUrls([QUrl.fromLocalFile(str(txt))])
    enter2 = QDragEnterEvent(QPoint(50, 50), Qt.DropAction.CopyAction, mime2, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier)
    enter2.setAccepted(False)
    page.dragEnterEvent(enter2)
    assert not enter2.isAccepted()


def test_recent_runs(page, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(home_mod.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url) or True))
    runs = [{"name": f"run {i}", "condition": "KCl" if i % 2 else "Fond (rayons cosmiques)", "total": 10 + i,
             "counts": {"alpha": 1, "beta": 5 + i, "muon": 4}, "duration_min": 1.5, "folder": str(tmp_path)}
            for i in range(7)]
    page.set_recent_runs(runs)
    assert len(page.run_rows) == 5
    assert page.run_rows[0].folder == str(tmp_path)
    QTest.mouseClick(page.run_rows[0], Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10))
    assert len(opened) == 1 and opened[0].toLocalFile().rstrip("/\\") == str(tmp_path).replace("\\", "/")
    asked = collect(page.run_open_requested)            # si la fenêtre principale écoute : signal au lieu de l'Explorateur
    page.run_rows[1].clicked.emit()
    assert asked == [str(tmp_path)] and len(opened) == 1
    page.set_recent_runs([])
    assert page.run_rows == []


def test_responsive_grid(qapp, page):
    page.resize(1400, 900)
    qapp.processEvents()
    assert page.columns == 2                  # 4 cartes essentielles : grille 2 × 2
    page.resize(950, 700)
    qapp.processEvents()
    assert page.columns == 2
    page.resize(560, 700)
    qapp.processEvents()
    assert page.columns == 1
    assert page.scroll.widgetResizable()


def test_grid_positions_with_new_cards(qapp, page):
    page.resize(1400, 900)
    qapp.processEvents()
    g = page._cards_grid

    def pos(k):
        return g.getItemPosition(g.indexOf(page.cards[k]))
    assert pos("camera")[:2] == (0, 0) and pos("video")[:2] == (0, 1)
    assert pos("night")[:2] == (1, 0) and pos("results")[:2] == (1, 1)
    for k in ("photos", "demo", "batch", "learning", "assistant"):      # rangée « Plus d'outils »
        assert g.indexOf(page.cards[k]) < 0 and page.cards[k].parent() is page._tools_host
    page.resize(700, 700)
    qapp.processEvents()
    assert [pos(k)[:2] for k in ("camera", "video", "night", "results")] == [(0, 0), (1, 0), (2, 0), (3, 0)]
    for k in ("camera", "video", "night", "results"):          # rien ne déborde horizontalement
        assert page.cards[k].geometry().right() <= page._cards_host.width()
    for k in ("photos", "demo", "batch", "learning", "assistant"):
        assert page.cards[k].geometry().right() <= page._tools_host.width()


def test_hero_compact_on_short_window(qapp, page):
    page.resize(1366, 640)
    qapp.processEvents()
    assert page.hero.compact and not page.hero.eyebrow_label.isVisibleTo(page.hero)
    page.resize(1366, 900)
    qapp.processEvents()
    assert not page.hero.compact


NIGHT_KEYS = {"camera_index", "duration_h", "end_time", "segment_min", "record_video", "record_scale", "record_fps",
              "snapshot_every_min", "autosave_every_min", "keep_awake", "dead_after_min", "condition", "sample",
              "mass_g", "px_per_mm", "output_dir"}
BATCH_KEYS = {"files", "watch_folder", "condition", "sample", "mass_g", "px_per_mm", "output_dir", "make_reports"}


def test_night_card_opens_dialog_and_emits(page, settings, monkeypatch):
    page.set_cameras([{"index": 0, "label": "A", "width": 640, "height": 480},
                      {"index": 2, "label": "B", "width": 1920, "height": 1080}])
    page.cam_combo.setCurrentIndex(1)
    settings.set("config", {"physics": {"px_per_mm": 3.5}})
    seen = []

    def run(dlg):
        seen.append(dlg)
        assert type(dlg).__name__ == "NightSessionDialog"
        assert dlg.selected_frame_wh() == (1920, 1080)
        dlg.dead_after.setValue(45)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(page, "_run_dialog", run)
    got = collect(page.night_requested)
    page.night_btn.click()
    assert len(got) == 1 and set(got[0]) == NIGHT_KEYS
    assert got[0]["camera_index"] == 2 and got[0]["px_per_mm"] == pytest.approx(3.5)
    assert got[0]["dead_after_min"] == 45
    assert settings.get("night_defaults")["dead_after_min"] == 45      # mémorisé pour la prochaine nuit
    monkeypatch.setattr(page, "_run_dialog", lambda dlg: QDialog.DialogCode.Rejected)
    QTest.keyClick(page.cards["night"], Qt.Key.Key_Return)              # carte accessible au clavier
    assert len(got) == 1


def test_batch_card_and_multi_video_drop(qapp, page, settings, tmp_path, monkeypatch):
    v1 = make_video(tmp_path / "recue 1.avi")
    v2 = make_video(tmp_path / "recue 2.avi")
    captured = []

    def run(dlg):
        assert type(dlg).__name__ == "BatchDialog"
        captured.append(dlg.files())
        return QDialog.DialogCode.Accepted if dlg.files() else QDialog.DialogCode.Rejected
    monkeypatch.setattr(page, "_run_dialog", run)
    got = collect(page.batch_requested)
    page.batch_btn.click()                                  # liste vide → l'utilisateur annule
    assert got == [] and captured == [[]]
    d = page.open_batch_setup([str(v1)])
    assert got == [d] and set(d) == BATCH_KEYS and d["files"] == [str(v1)]
    assert "files" not in settings.get("batch_defaults")
    mime = QMimeData()                                      # plusieurs vidéos déposées → analyse par lot
    mime.setUrls([QUrl.fromLocalFile(str(v1)), QUrl.fromLocalFile(str(v2))])
    enter = QDragEnterEvent(QPoint(50, 50), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    page.dragEnterEvent(enter)
    assert enter.isAccepted() and "lot" in page._overlay.title
    page.dropEvent(QDropEvent(QPointF(50, 50), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                              Qt.KeyboardModifier.NoModifier))
    for _ in range(5):
        qapp.processEvents()
    assert len(got) == 2 and [Path(f).name for f in got[1]["files"]] == ["recue 1.avi", "recue 2.avi"]
    mime2 = QMimeData()                                     # un dossier déposé → ses vidéos
    mime2.setUrls([QUrl.fromLocalFile(str(tmp_path))])
    page.dropEvent(QDropEvent(QPointF(50, 50), Qt.DropAction.CopyAction, mime2, Qt.MouseButton.LeftButton,
                              Qt.KeyboardModifier.NoModifier))
    for _ in range(5):
        qapp.processEvents()
    assert len(got) == 3 and len(got[2]["files"]) == 2


def test_reference_model_blurb(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"n_tracks_total": 52000, "n_jobs": 900}), encoding="utf-8")
    assert home_mod.reference_model_blurb(p) == "Entraînée sur 52 000 traces simulées (900 enregistrements " \
                                                "simulés)."
    assert "règles physiques" in home_mod.reference_model_blurb(tmp_path / "absent.json")
    p.write_text("{pas du json", encoding="utf-8")
    assert "règles physiques" in home_mod.reference_model_blurb(p)


def test_app_qss_has_modern_rules():
    from wilsontrack.gui.app import QSS, dark_palette, run_gui  # noqa: F401
    assert isinstance(QSS, str)
    for rule in ('QPushButton[primary="true"]', 'QPushButton[success="true"]', 'QPushButton[danger="true"]',
                 "QToolTip", "QScrollBar:vertical", "QRadioButton::indicator", "QProgressBar[slim=\"true\"]",
                 "QHeaderView::section", "QTabBar::tab:selected"):
        assert rule in QSS
    for url in [part.split('")')[0] for part in QSS.split('url("')[1:]]:
        assert Path(url).is_file(), url
