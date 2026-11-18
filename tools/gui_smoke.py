"""Test automatisé de l'interface : démo en direct, analyse de deux vidéos (fond puis KCl),
comparaison, rapports, IA locale, assistant sans clé ; captures d'écran de chaque onglet.

Usage : python tools/gui_smoke.py VIDEO_FOND VIDEO_KCL PX_PAR_MM SORTIE_DIR [--offscreen]
"""
import os
import sys
import time
from pathlib import Path

vid_bg, vid_k, ppm, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(out / "appdata")
Path(os.environ["APPDATA"]).mkdir(exist_ok=True)
if "--offscreen" in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QDesktopServices  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from wilsontrack.gui.app import QSS, dark_palette  # noqa: E402

app = QApplication(sys.argv)
app.setStyle("Fusion")
app.setPalette(dark_palette())
app.setStyleSheet(QSS)
errors = []
QMessageBox.information = staticmethod(lambda *a, **k: print("INFO:", a[1:3]))
QMessageBox.warning = staticmethod(lambda *a, **k: print("WARN:", a[1:3]))
QMessageBox.critical = staticmethod(lambda *a, **k: errors.append(f"CRITIQUE: {a[1:3]}"))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
opened = []
QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url.toLocalFile()) or True)

from wilsontrack.gui.main_window import MainWindow  # noqa: E402

w = MainWindow()
w.resize(1600, 950)
w.panel.out_dir.setText(str(out / "resultats"))
w._apply_settings()
w.show()


def wait(sec, cond=None):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.01)
        if cond is not None and cond():
            return True
    return cond() if cond else True


def grab(name):
    w.grab().save(str(out / f"{name}.png"))


# 0) Menu d'accueil
wait(1.5)
if w.tabs.currentWidget() is not w.home:
    errors.append("l'application ne s'ouvre pas sur l'accueil")
grab("00_accueil")

# 1) Démo en direct (lancée depuis l'accueil)
w.home.demo_requested.emit("demo")
wait(10)
grab("01_demo_direct")
st = w.last_stats
print("démo :", {k: st.get(k) for k in ("t", "counts", "proc_fps", "noise")})
if not st or st.get("frame", 0) < 30:
    errors.append("démo : trop peu d'images traitées")
w.view_mode.setCurrentIndex(3)
wait(2)
grab("02_demo_carte")
w.view_mode.setCurrentIndex(0)
w.stop_source(wait=True)
wait(3, lambda: w.worker is None)


def analyse(video, cond, sample, mass, trial):
    w.panel.trial.setValue(trial)
    done = {"run": None}
    # Même chemin que le bouton « Choisir une vidéo… » de l'accueil (fenêtre de préparation validée)
    w.home.video_requested.emit({"path": video, "condition": cond, "sample": sample, "mass_g": mass,
                                 "px_per_mm": ppm, "realtime": False})
    if w.worker is None:
        errors.append(f"l'analyse ne démarre pas depuis l'accueil : {video}")
        return None
    if abs(w.cfg.physics.px_per_mm - ppm) > 1e-6 or w.cfg.run.condition != cond:
        errors.append(f"réglages de la fenêtre vidéo non appliqués ({w.cfg.physics.px_per_mm}, {w.cfg.run.condition})")
    w.worker.run_finished.connect(lambda r: done.__setitem__("run", r))
    ok = wait(900, lambda: done["run"] is not None and w.worker is None)
    wait(2)
    if not ok:
        errors.append(f"analyse non terminée : {video}")
    return w.run


run_bg = analyse(vid_bg, "Fond (rayons cosmiques)", "Aucun", 0, 1)
print("fond :", run_bg.counts() if run_bg else None, run_bg.folder if run_bg else None)
run_k = analyse(vid_k, "KCl", "KCl (chlorure de potassium)", 100, 1)
print("KCl  :", run_k.counts() if run_k else None, run_k.folder if run_k else None)
grab("03_fichier_fin")
for tab, name in [(w.tab_tracks, "04_traces"), (w.tab_charts, "05_graphiques"), (w.tab_concl, "06_conclusions"),
                  (w.tab_ai, "08_assistant"), (w.tab_help, "09_aide")]:
    w.tabs.setCurrentWidget(tab)
    wait(1.5)
    grab(name)
for c in w.charts.values():
    if c.canvas is None:
        errors.append(f"graphique non affiché : {c.msg.text()}")
if w.table.rowCount():
    w.tabs.setCurrentWidget(w.tab_tracks)
    w.table.selectRow(0)
    wait(0.5)
    txt = w.detail_txt.toPlainText()
    if "Pourquoi" not in txt and "pourquoi" not in txt:
        errors.append("le panneau de détail n'affiche pas l'explication « pourquoi »")
    grab("10_trace_pourquoi")
    for r in range(min(w.table.rowCount(), 40)):
        why = w.table.item(r, 10).text() if w.table.item(r, 10) else ""
        if not why or "nan" in why.lower().split():
            errors.append(f"colonne Pourquoi vide ou invalide (ligne {r})")
            break
    w.manual.setCurrentIndex(2)
    w._on_manual_class(2)
    wait(0.5)
    grab("11_trace_corrigee")
w.go_home()
wait(1.0)
grab("12_accueil_resultats")
w.train_local_ai()
wait(1)
print("modèle IA locale :", w.cfg.classification.model_path)
w.make_report()
# Comparaison KCl / fond
w.refresh_runs()
names = [w.cmp_sample.itemText(i) for i in range(w.cmp_sample.count())]
print("enregistrements :", names)
si = next((i for i, n in enumerate(names) if "KCl" in n), -1)
bi = next((i for i, n in enumerate(names) if "Fond" in n), -1)
if si < 0 or bi < 0:
    errors.append("enregistrements introuvables pour la comparaison")
else:
    w.cmp_sample.setCurrentIndex(si)
    w.cmp_bg.setCurrentIndex(bi)
    w.compare_runs()
    wait(2)
    w.tabs.setCurrentWidget(w.tab_compare)
    wait(1)
    grab("07_comparer")
    w.compare_report()
    txt = w.cmp_text.toPlainText()
    print("comparaison :", txt[:600].replace("\n", " | "))
    if "Radioactivit" not in txt:
        errors.append("la comparaison ne conclut pas à une radioactivité détectée")
# Assistant sans clé : message, pas de plantage
w.api_key.setText("")
os.environ.pop("ANTHROPIC_API_KEY", None)
w.ai_interpret_run()
wait(0.5)
print("rapports ouverts :", opened)
for f in opened:
    if not f or not os.path.exists(f) or os.path.getsize(f) < 20000:
        errors.append(f"rapport invalide : {f}")
w.close()
print("ERREURS :", errors)
sys.exit(1 if errors else 0)
