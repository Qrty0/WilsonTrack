"""Aperçus hors écran de l'identité visuelle du Cégep Marie-Victorin (contrôle visuel de l'interface).

Usage : python tools/preview_cmv.py [DOSSIER_SORTIE]      (défaut : _previews à la racine du projet)

Produit (PNG, préfixe cmv_) :
  cmv_splash_1/2/3.png           écran de chargement à trois instants de l'animation
  cmv_home_anim.png              accueil pendant l'animation d'accueil (pilules en train de monter)
  cmv_home_1400.png              accueil, fenêtre 1400×900 (animation terminée)
  cmv_home_1366.png              accueil, portable 1366×768
  cmv_settings.png               panneau « Réglages » ouvert (réglages essentiels)
  cmv_settings_avance.png        … avec « Réglages avancés » déplié
  cmv_direct.png                 onglet Direct, démonstration en cours (≈ 5 s)
  cmv_traces.png                 onglet Traces, une trace sélectionnée (explication « pourquoi »)
  cmv_conclusions.png            onglet Conclusions (synthèse) ; cmv_graphiques.png (sous-onglet Graphiques)
  cmv_night_dialog.png           boîte « Session de nuit »
  cmv_learning_tab.png           centre d'apprentissage de l'IA (onglet IA)
  cmv_aide.png                   onglet Aide
Aucun réglage de l'utilisateur n'est modifié (APPDATA temporaire, aucune caméra ouverte).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_previews"
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="wilsontrack_preview_"))
os.environ["APPDATA"] = str(TMP / "appdata")
(TMP / "appdata").mkdir(exist_ok=True)

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from wilsontrack.gui.app import apply_theme  # noqa: E402

app = QApplication(sys.argv[:1])
apply_theme(app)
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
QDesktopServices.openUrl = staticmethod(lambda url: True)


def wait(sec, cond=None):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.01)
        if cond is not None and cond():
            return True
    return True


def save(widget, name):
    widget.grab().save(str(OUT / f"{name}.png"))
    print("écrit :", OUT / f"{name}.png")


# ---------------------------------------------------------------- écran de chargement --
from wilsontrack.gui.splash import SplashScreen  # noqa: E402

sp = SplashScreen()
sp.show()
for ms, k, status, prog in ((420, 1, "Démarrage…", 0.08), (1150, 2, "Chargement de l'IA de référence…", 0.35),
                            (2300, 3, "Préparation de la caméra…", 0.8)):
    sp.set_status(status, prog)
    sp.set_time(ms)
    img = QImage(sp.width() + 40, sp.height() + 40, QImage.Format.Format_ARGB32)
    img.fill(QColor("#e4ebe5"))
    p = QPainter(img)
    sp.render(p, QPoint(20, 20))
    p.end()
    img.save(str(OUT / f"cmv_splash_{k}.png"))
sp.close()

# ---------------------------------------------------------------- fenêtre principale --
from wilsontrack.gui.main_window import MainWindow  # noqa: E402

w = MainWindow()
w.panel.out_dir.setText(str(TMP / "resultats"))
w._apply_settings()
w.resize(1400, 900)
w.show()
wait(0.55)
save(w, "cmv_home_anim")
wait(1.6, lambda: not w.home.hero.intro_running)
wait(0.3)
save(w, "cmv_home_1400")
w.resize(1366, 768)
wait(0.6)
save(w, "cmv_home_1366")
w.resize(1400, 900)
w.settings_dock.show()
wait(0.5)
save(w, "cmv_settings")
w.panel.sec_advanced.btn.setChecked(True)
wait(0.3)
save(w, "cmv_settings_avance")
w.panel.sec_advanced.btn.setChecked(False)
w.settings_dock.hide()

w.home.demo_requested.emit("demo")
wait(5.5)
save(w, "cmv_direct")
w.stop_source(wait=True)
wait(20, lambda: w.worker is None and w.run is not None)
wait(1.0)
w.tabs.setCurrentWidget(w.tab_tracks)
if w.table.rowCount():
    w.table.selectRow(0)
wait(0.8)
save(w, "cmv_traces")
w.tabs.setCurrentWidget(w.tab_concl)
wait(0.8)
save(w, "cmv_conclusions")
w.tabs.setCurrentWidget(w.tab_charts)
wait(1.2)
save(w, "cmv_graphiques")
w.tabs.setCurrentWidget(w.tab_learning)
wait(0.8)
save(w, "cmv_learning_tab")
w.tabs.setCurrentWidget(w.tab_help)
wait(0.5)
save(w, "cmv_aide")

from wilsontrack.gui.session_dialogs import NightSessionDialog  # noqa: E402

dlg = NightSessionDialog({"camera_index": 0}, [{"index": 0, "label": "Caméra 0 (1920×1080)", "width": 1920,
                                                "height": 1080}], (1920, 1080), 30.0, w)
dlg.show()
wait(0.5)
save(dlg, "cmv_night_dialog")
dlg.close()
w.close()
print("terminé")
