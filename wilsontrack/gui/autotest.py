"""Auto-test de bout en bout (utilisé pour vérifier l'exécutable compilé).

    WilsonTrack-console.exe --autotest     (ou python main.py --autotest)

Lance l'interface, démarre la démonstration en direct, vérifie que les images
sont traitées et que des traces sont détectées, génère les graphiques, termine
l'enregistrement, sauvegarde les résultats et le rapport HTML, entraîne l'IA locale,
puis quitte. Le résultat est écrit dans la console et dans %TEMP%/wilsontrack_autotest.txt.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def run_autotest(seconds: float = 20.0) -> int:
    log_path = Path(tempfile.gettempdir()) / "wilsontrack_autotest.txt"
    lines: list[str] = []

    def log(msg: str):
        lines.append(msg)
        try:
            print(msg, flush=True)
        except Exception:
            pass

    out = Path(tempfile.mkdtemp(prefix="wilsontrack_autotest_"))
    os.environ["APPDATA"] = str(out / "appdata")
    (out / "appdata").mkdir(exist_ok=True)
    ok = True
    try:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QApplication, QMessageBox

        from .app import QSS, dark_palette
        app = QApplication.instance() or QApplication(sys.argv[:1])
        app.setStyle("Fusion")
        app.setPalette(dark_palette())
        app.setStyleSheet(QSS)
        QMessageBox.information = staticmethod(lambda *a, **k: log(f"info: {a[1:3]}"))
        QMessageBox.warning = staticmethod(lambda *a, **k: log(f"avertissement: {a[1:3]}"))
        QMessageBox.critical = staticmethod(lambda *a, **k: log(f"ERREUR: {a[1:3]}"))
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
        QDesktopServices.openUrl = staticmethod(lambda url: True)
        from .main_window import MainWindow

        w = MainWindow()
        w.panel.out_dir.setText(str(out / "resultats"))
        w._apply_settings()
        w.show()
        if w.tabs.currentWidget() is not w.home:
            ok = False
            log("   ÉCHEC : l'application ne s'ouvre pas sur le menu d'accueil")
        else:
            log("0. Menu d'accueil affiché")
        try:
            from ..detection.reference_model import ReferenceModel
            ref = ReferenceModel.load_default()
            if ref is None:
                log("   IA de référence : non disponible (règles physiques seules)")
            else:
                inf = ref.info or {}
                log(f"   IA de référence chargée : {inf.get('n_tracks_total', '?')} traces d'entraînement, "
                    f"méthode de classification : {w.cfg.classification.mode}")
        except Exception as ex:
            ok = False
            log(f"   ÉCHEC : chargement de l'IA de référence : {ex}")

        def wait(sec, cond=None):
            t0 = time.time()
            while time.time() - t0 < sec:
                app.processEvents()
                time.sleep(0.01)
                if cond is not None and cond():
                    return True
            return bool(cond()) if cond else True

        log("1. Démonstration en direct…")
        w.start_source("demo_kcl")
        wait(seconds)
        st = w.last_stats
        n_img = st.get("frame", -1) + 1
        log(f"   images traitées : {n_img}, cadence ≈ {st.get('proc_fps', 0):.0f} img/s, traces : {st.get('counts')}")
        if n_img < 50:
            ok = False
            log("   ÉCHEC : trop peu d'images traitées")
        w.refresh_charts()
        log("2. Arrêt et calcul des résultats…")
        w.stop_source()
        if not wait(60, lambda: w.run is not None and w.worker is None):
            ok = False
            log("   ÉCHEC : l'enregistrement ne s'est pas terminé")
        run = w.run
        if run is not None:
            log(f"   {run.total} traces : {run.counts()} — durée {run.duration_s:.1f} s")
            if run.total < 1:
                ok = False
                log("   ÉCHEC : aucune trace détectée dans la démonstration")
            folder = w.save_results(quiet=True)
            rep = Path(folder) / "rapport.html" if folder else None
            if rep is None or not rep.exists() or rep.stat().st_size < 20000:
                ok = False
                log(f"   ÉCHEC : rapport HTML absent ({rep})")
            else:
                log(f"   rapport : {rep} ({rep.stat().st_size // 1024} Ko)")
            for k, c in w.charts.items():
                if c.canvas is None:
                    ok = False
                    log(f"   ÉCHEC : graphique « {k} » non affiché ({c.msg.text()})")
            from ..detection.explain import explain_event
            shown = set()
            n_expl = 0
            for e in run.events:
                if not e.valid:
                    continue
                ex = explain_event(e, w.cfg.classification, w.cfg.physics)
                html_txt = ex.to_html()
                n_expl += 1
                if len(html_txt) < 200 or "nan" in ex.to_text().lower().split():
                    ok = False
                    log(f"   ÉCHEC : explication invalide pour la trace #{e.id}")
                if e.final_cls not in shown:
                    shown.add(e.final_cls)
                    log(f"   pourquoi #{e.id} : {ex.short()}")
            log(f"   {n_expl} explications « pourquoi » générées")
            w.tabs.setCurrentWidget(w.tab_tracks)
            if w.table.rowCount():
                w.table.selectRow(0)
                wait(0.3)
                if "Pourquoi" not in w.detail_txt.toPlainText() and "pourquoi" not in w.detail_txt.toPlainText():
                    ok = False
                    log("   ÉCHEC : le panneau de détail n'affiche pas l'explication")
            from ..detection.classifier import MLClassifier
            feats = [e.features for e in run.events if e.valid]
            labels = [e.cls for e in run.events if e.valid]
            if len(set(labels)) >= 2 and len(labels) >= 6:
                msg = MLClassifier().train(feats, labels)
                log("3. IA locale : " + msg.splitlines()[0] if msg else "3. IA locale entraînée")
            import anthropic  # noqa: F401  (disponibilité du module pour l'assistant Claude)
            log(f"4. Module Anthropic disponible (version {anthropic.__version__})")
        w.close()
    except Exception:
        ok = False
        log("EXCEPTION :\n" + traceback.format_exc())
    log("RÉSULTAT : " + ("SUCCÈS" if ok else "ÉCHEC"))
    try:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass
    return 0 if ok else 1
