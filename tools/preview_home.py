"""Captures d'écran hors écran de la page d'accueil, des boîtes et des écrans de longues analyses.

Usage : python tools/preview_home.py [DOSSIER_SORTIE]      (défaut : _previews à la racine du projet)

Produit (PNG) :
  home_1400x900.png, home_1400_complet.png             page d'accueil, fenêtre 1400×900 (+ page entière)
  home_1366x768.png, home_1366_bas.png                 portable 1366×768 (haut et bas de la page)
  home_1920x1080.png                                   écran 1920×1080
  home_1000x700.png, home_1000_complet.png             fenêtre étroite (2 colonnes)
  home_640_complet.png                                 très étroite (1 colonne)
  home_vide_1000x700.png                               sans caméra ni résultat
  home_survol.png, home_depot.png, home_depot_lot.png  survol d'une carte, glisser-déposer (1 / plusieurs vidéos)
  dialogue_video.png, dialogue_video_illisible.png     boîte « Analyser une vidéo »
  dialogue_nuit.png, dialogue_nuit_1366.png            boîte « Session de nuit » (écran 1920 / 1366×768)
  dialogue_nuit_disque_plein.png                       même boîte, estimation disque insuffisante (simulée)
  dialogue_lot.png, dialogue_lot_vide.png              boîte « Vidéos reçues / analyse par lot »
  bandeau_nuit.png, bandeau_nuit_1000.png              bandeau d'état d'une session de nuit (faux état)
  bandeau_nuit_fin.png                                 bandeau en fin de session
  file_lot.png, file_lot_vide.png                      file d'analyse par lot (fausses tâches)
  centre_ia.png, centre_ia_complet.png                 centre d'apprentissage avec le modèle pilote installé
  centre_ia_validation.png                             … avec une mesure (fictive) sur des traces corrigées
  centre_ia_sans_modele.png                            … quand le modèle est absent
  centre_ia_900_complet.png                            … fenêtre étroite
  qss_reglages.png, qss_fenetre_*.png                  QSS appliquée aux écrans existants (si importables)
Aucun réglage de l'utilisateur n'est modifié (APPDATA et préférences temporaires, aucune caméra ouverte).
Les vignettes d'exemple du centre d'apprentissage sont dessinées pour l'aperçu seulement (dossier temporaire).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QMimeData, QPointF, QRect, QRectF, QSize, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QColor, QDragEnterEvent, QDragLeaveEvent, QFont, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from wilsontrack.gui.app import QSS, dark_palette  # noqa: E402
from wilsontrack.gui.home import HomePage, VideoSetupDialog, find_example_videos, paint_signature  # noqa: E402
from wilsontrack.settings import Settings  # noqa: E402

EXAMPLE = ROOT / "exemples" / "exemple KCl (source simulee).mp4"
SCREEN_1366 = QRect(0, 0, 1366, 728)      # zone utile d'un écran 1366×768 (barre des tâches déduite)
SCREEN_1920 = QRect(0, 0, 1920, 1040)

FAKE_CAMS = [{"index": 0, "label": "Caméra 0 (1920×1080)", "width": 1920, "height": 1080},
             {"index": 1, "label": "Caméra 1 (1280×720)", "width": 1280, "height": 720}]
FAKE_RUNS = [
    {"name": "2026-10-09 KCl essai-03", "condition": "KCl", "total": 412,
     "counts": {"alpha": 31, "beta": 268, "muon": 109, "proton": 1, "incertain": 3}, "duration_min": 20.0,
     "folder": str(ROOT)},
    {"name": "2026-10-09 Fond (rayons cosmiques) essai-03", "condition": "Fond (rayons cosmiques)", "total": 236,
     "counts": {"alpha": 28, "beta": 96, "muon": 110, "proton": 2}, "duration_min": 20.0, "folder": str(ROOT)},
    {"name": "2026-10-02 Banane essai-01", "condition": "Banane", "total": 151,
     "counts": {"alpha": 14, "beta": 70, "muon": 66, "proton": 0, "incertain": 1}, "duration_min": 12.5,
     "folder": str(ROOT)},
    {"name": "2026-09-29 KCl essai-01", "condition": "KCl", "total": 98,
     "counts": {"alpha": 6, "beta": 61, "muon": 31}, "duration_min": 1.5, "folder": str(ROOT)},
    {"name": "2026-09-29 Fond (rayons cosmiques) essai-01", "condition": "Fond (rayons cosmiques)", "total": 52,
     "counts": {"alpha": 5, "beta": 20, "muon": 27}, "duration_min": 1.5, "folder": str(ROOT)},
    {"name": "plus ancien (ne doit pas apparaître)", "condition": "Autre", "total": 1, "counts": {},
     "duration_min": 1.0, "folder": ""},
]
FAKE_STATUS = {"elapsed_s": 3 * 3600 + 12 * 60, "remaining_s": 5 * 3600 + 48 * 60, "next_autosave_s": 130,
               "disk_free_bytes": 231 * 1024 ** 3, "segment_index": 6, "recording": True, "n_tracks": 12873,
               "rate_per_min": 61.4, "last_checkpoint": "2026-10-21T02:41:10",
               "warnings": ["Aucune trace valide depuis 22 min : la chambre manque peut-être d'alcool ou de "
                            "glace sèche."],
               "session_folder": str(ROOT)}
FAKE_JOBS = [
    {"path": r"C:\Reçues\2026-10-09 KCl essai-03.mp4", "name": "2026-10-09 KCl essai-03.mp4", "status": "terminé",
     "progress": 1.0, "eta_s": -1, "result_folder": str(ROOT),
     "counts": {"alpha": 31, "beta": 268, "muon": 109, "proton": 1}, "error": ""},
    {"path": r"C:\Reçues\vidéo de Yanis — fond, salle B-204.mp4", "name": "vidéo de Yanis — fond, salle B-204.mp4",
     "status": "en cours", "progress": 0.42, "eta_s": 312, "result_folder": "",
     "counts": {"alpha": 4, "beta": 30, "muon": 17, "proton": 0}, "error": "", "from_watch": True},
    {"path": r"C:\Reçues\banane 150 g.mov", "name": "banane 150 g.mov", "status": "en attente", "progress": 0.0,
     "eta_s": 540, "result_folder": "", "counts": {}, "error": ""},
    {"path": r"C:\Reçues\corrompue.mp4", "name": "corrompue.mp4", "status": "erreur", "progress": 0.05,
     "eta_s": -1, "result_folder": "", "counts": {}, "error": "Format ou codec non pris en charge"},
    {"path": r"C:\Reçues\essai annulé.mp4", "name": "essai annulé.mp4", "status": "annulé", "progress": 0.3,
     "eta_s": -1, "result_folder": "", "counts": {"alpha": 1, "beta": 3}, "error": ""},
]
FAKE_VALIDATION = {
    "n": 64, "accuracy": 0.81,
    "per_class": {"alpha": {"precision": 0.95, "recall": 0.9, "n": 20},
                  "beta": {"precision": 0.74, "recall": 0.8, "n": 25},
                  "muon": {"precision": 0.76, "recall": 0.68, "n": 19}},
    "confusion": {"labels_true": ["alpha", "beta", "muon"],
                  "labels_pred": ["alpha", "beta", "muon", "proton", "incertain"],
                  "matrix": [[18, 0, 0, 1, 1], [0, 20, 3, 0, 2], [0, 4, 13, 0, 2]]},
    "note": "EXEMPLE FICTIF pour l'aperçu : 64 traces corrigées dans 3 vidéos.",
}


def settle(app: QApplication, n: int = 8):
    for _ in range(n):
        app.processEvents()


def fake_thumbnails(folder: Path) -> Path:
    """Vignettes dessinées (aperçu de mise en page seulement)."""
    folder.mkdir(parents=True, exist_ok=True)
    for cls in ("alpha", "beta", "muon", "proton"):
        for k in range(1, 6):
            pm = QPixmap(120, 120)
            pm.fill(QColor(12, 13, 17))
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            paint_signature(p, QRectF(0, 0, 120, 120), cls, seed=k * 7 + len(cls))
            p.end()
            pm.save(str(folder / f"{cls}_{k}.png"))
    return folder


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_previews"
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))

    tmp = Path(tempfile.mkdtemp(prefix="wilsontrack_preview_"))
    appdata = tmp / "appdata"
    (appdata / "WilsonTrack").mkdir(parents=True)
    (appdata / "WilsonTrack" / "settings.json").write_text(
        json.dumps({"cameras_cache": FAKE_CAMS}, ensure_ascii=False), encoding="utf-8")
    os.environ["APPDATA"] = str(appdata)      # aucune écriture dans les vraies préférences
    fake_recent = []
    for name in ("2026-10-09 KCl essai-03.mp4", "2026-10-09 Fond (rayons cosmiques) essai-03.mp4",
                 "Enregistrement très long du laboratoire de physique du cégep — séance du 2 octobre 2026.mp4"):
        f = tmp / name
        f.write_bytes(b"")
        fake_recent.append(str(f))
    settings = Settings(tmp / "settings.json")
    settings.set("recent_videos", list(reversed(fake_recent)))
    page = HomePage(settings)
    page.set_cameras(FAKE_CAMS)
    page.set_recent_runs(FAKE_RUNS)
    for f in fake_recent:
        page.add_recent_video(f)
    shots = []

    def shot(widget: QWidget, name: str):
        settle(app)
        p = out / name
        widget.grab().save(str(p))
        shots.append(p)

    # ---------------------------------------------------------------- page d'accueil --
    page.resize(1400, 900)
    page.show()
    settle(app, 12)
    shot(page, "home_1400x900.png")
    print("colonnes à 1400 px :", page.columns)
    shot(page.content, "home_1400_complet.png")
    card = page.cards["night"]                 # survol simulé d'une carte
    card.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)
    card.style().unpolish(card)
    card.style().polish(card)
    card.update()
    shot(page, "home_survol.png")
    card.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, False)
    card.style().unpolish(card)
    card.style().polish(card)

    for (w, h), name in (((1366, 768), "home_1366x768.png"), ((1920, 1080), "home_1920x1080.png")):
        page.resize(w, h)
        settle(app, 12)
        page.scroll.verticalScrollBar().setValue(0)
        shot(page, name)
        print(f"colonnes à {w} px :", page.columns)
        if w == 1366:
            sb = page.scroll.verticalScrollBar()
            sb.setValue(sb.maximum() // 2)
            shot(page, "home_1366_milieu.png")
            sb.setValue(sb.maximum())
            shot(page, "home_1366_bas.png")
            sb.setValue(0)

    page.resize(1000, 700)
    settle(app, 12)
    shot(page, "home_1000x700.png")
    print("colonnes à 1000 px :", page.columns)
    shot(page.content, "home_1000_complet.png")
    page.resize(640, 700)
    settle(app, 12)
    print("colonnes à 640 px :", page.columns)
    shot(page.content, "home_640_complet.png")
    page.resize(1000, 700)
    settle(app, 12)

    # glisser-déposer : une vidéo, puis plusieurs (analyse par lot)
    for urls, name in (([EXAMPLE], "home_depot.png"), (find_example_videos(), "home_depot_lot.png")):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(u)) for u in urls])
        ev = QDragEnterEvent(QPointF(400, 300).toPoint(), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier)
        page.dragEnterEvent(ev)
        shot(page, name)
        page.dragLeaveEvent(QDragLeaveEvent())

    s2 = Settings(tmp / "vide.json")
    empty = HomePage(s2)
    empty.set_cameras([])
    empty.set_recent_runs([])
    empty.resize(1000, 700)
    empty.show()
    settle(app, 12)
    shot(empty, "home_vide_1000x700.png")
    empty.hide()

    # ---------------------------------------------------------------- boîte vidéo --
    if EXAMPLE.exists():
        dlg = VideoSetupDialog(str(EXAMPLE), {"condition": "KCl", "mass_g": 100.0, "realtime": False})
        dlg.show()
        settle(app, 10)
        shot(dlg, "dialogue_video.png")
        print("dialogue vidéo :", dlg.result_dict(), dlg.size().toTuple())
        dlg.close()
    bad = tmp / "video illisible.mp4"
    bad.write_bytes(b"ceci n'est pas une video" * 50)
    dlg = VideoSetupDialog(str(bad), None)
    dlg.show()
    settle(app, 10)
    shot(dlg, "dialogue_video_illisible.png")
    dlg.close()

    # ---------------------------------------------------------------- longues analyses --
    from wilsontrack.gui import session_dialogs as sd
    night_defaults = {"condition": "KCl", "mass_g": 100.0, "camera_index": 1}
    for avail, name in ((SCREEN_1920, "dialogue_nuit.png"), (SCREEN_1366, "dialogue_nuit_1366.png")):
        nd = sd.NightSessionDialog(night_defaults, FAKE_CAMS, (1280, 720), 30.0)
        sd._fit_to_screen(nd, nd.sizeHint().expandedTo(QSize(940, 0)), avail)
        nd.show()
        settle(app, 10)
        shot(nd, name)
        print(f"session de nuit ({avail.width()}) :", nd.size().toTuple(), "indice", nd.sizeHint().toTuple())
        if name == "dialogue_nuit.png":
            print("dictionnaire nuit :", nd.result_dict())
            # estimation simulée : disque insuffisant (montre l'avertissement rouge)
            nd._longrun = (type("Cfg", (), {"from_dict": staticmethod(lambda d: d)}),
                           lambda cfg, wh, fps: 180 * 1024 ** 3, lambda path: 120 * 1024 ** 3)
            nd.record_mode.setCurrentIndex(2)
            settle(app, 6)
            shot(nd, "dialogue_nuit_disque_plein.png")
        nd.close()

    files = [str(p) for p in find_example_videos()]
    bd = sd.BatchDialog({"files": files, "condition": "KCl", "mass_g": 100.0})
    sd._fit_to_screen(bd, bd.sizeHint().expandedTo(QSize(980, 600)), SCREEN_1366)
    bd.show()
    settle(app, 10)
    shot(bd, "dialogue_lot.png")
    print("dictionnaire lot :", bd.result_dict(), bd.size().toTuple())
    bd.close()
    bd = sd.BatchDialog(None)
    sd._fit_to_screen(bd, bd.sizeHint().expandedTo(QSize(980, 600)), SCREEN_1366)
    bd.show()
    settle(app, 10)
    shot(bd, "dialogue_lot_vide.png")
    bd.close()

    for width, name in ((1340, "bandeau_nuit.png"), (1000, "bandeau_nuit_1000.png")):
        bar = sd.SessionStatusBar()
        bar.update_status(FAKE_STATUS)
        bar.resize(width, 200)
        bar.show()
        settle(app, 8)
        h = bar.layout().heightForWidth(width) if bar.layout().hasHeightForWidth() else bar.sizeHint().height()
        bar.resize(width, max(h, bar.minimumSizeHint().height()))
        settle(app, 8)
        shot(bar, name)
        if width == 1340:
            bar.set_finished("Session terminée — résultats enregistrés")
            bar.update_status(dict(FAKE_STATUS, remaining_s=0, warnings=[], recording=False, segment_index=-1))
            settle(app, 6)
            shot(bar, "bandeau_nuit_fin.png")
        bar.close()

    q = sd.BatchQueueWidget()
    q.update_jobs(FAKE_JOBS)
    q.resize(1100, 360)
    q.show()
    settle(app, 8)
    shot(q, "file_lot.png")
    q.update_jobs([])
    q.resize(1100, 220)
    settle(app, 6)
    shot(q, "file_lot_vide.png")
    q.close()

    # ---------------------------------------------------------------- centre d'apprentissage --
    from wilsontrack.gui.learning import LearningCenter
    thumbs = fake_thumbnails(tmp / "ia_exemples")
    lc = LearningCenter(examples_dir=thumbs)
    lc.resize(1366, 900)
    lc.show()
    settle(app, 12)
    shot(lc, "centre_ia.png")
    shot(lc.scroll.widget(), "centre_ia_complet.png")
    lc.show_validation(FAKE_VALIDATION)
    settle(app, 10)
    sb = lc.scroll.verticalScrollBar()
    sb.setValue(sb.maximum())
    shot(lc, "centre_ia_validation.png")
    lc.resize(900, 900)
    settle(app, 12)
    shot(lc.scroll.widget(), "centre_ia_900_complet.png")
    lc.close()
    try:
        from wilsontrack.detection import reference_model as rm
        orig = rm.ReferenceModel.__dict__["load_default"]
        rm.ReferenceModel.load_default = classmethod(lambda cls: None)
        try:
            lc2 = LearningCenter(examples_dir=tmp / "absent")
            lc2.resize(1366, 900)
            lc2.show()
            settle(app, 12)
            shot(lc2.scroll.widget(), "centre_ia_sans_modele.png")
            lc2.close()
        finally:
            rm.ReferenceModel.load_default = orig
    except Exception as e:  # pragma: no cover
        print("centre sans modèle non rendu :", e)

    # ---------------------------------------------------------------- QSS sur l'existant --
    try:
        from wilsontrack.gui.settings_panel import SettingsPanel
        panel = SettingsPanel()
        panel.resize(380, 900)
        panel.show()
        settle(app, 10)
        shot(panel, "qss_reglages.png")
        panel.hide()
    except Exception as e:  # pragma: no cover - panneau en cours de modification par ailleurs
        print("panneau de réglages non rendu :", e)
    try:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
        from wilsontrack.gui.main_window import MainWindow
        win = MainWindow()
        win.resize(1366, 768)
        win.show()
        settle(app, 15)
        for i, name in ((0, "qss_fenetre_accueil.png"), (1, "qss_fenetre_direct.png"), (2, "qss_fenetre_traces.png")):
            win.tabs.setCurrentIndex(i)
            shot(win, name)
        win.close()
    except Exception as e:  # pragma: no cover - fenêtre en cours de modification par ailleurs
        print("fenêtre principale non rendue :", e)

    print("exemples :", [p.name for p in find_example_videos()])
    for p in shots:
        print("capture :", p)
    page.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
