"""Vidéos reçues : file d'attente, analyse par lot, résumé resume_lot.csv et dossier surveillé."""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from wilsontrack.batch import (STATE_FILE, STATUS_CANCELLED, STATUS_DONE, STATUS_ERROR, STATUS_WAITING,  # noqa: E402
                               SUMMARY_COLUMNS, BatchConfig, BatchRunner, FolderWatcher, VideoQueue,
                               is_partial_name, is_video_file, video_probe)
from wilsontrack.config import AppConfig  # noqa: E402

JOB_KEYS = {"path", "name", "status", "progress", "eta_s", "result_folder", "counts", "error"}


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """Vidéo synthétique de 7 s en 640×360 (+ vérité terrain .gt.json)."""
    from wilsontrack.synth import SynthConfig, generate_video
    folder = tmp_path_factory.mktemp("recues")
    sc = SynthConfig(width=640, height=360, duration_s=7.0, seed=8,
                     rates_per_min={"alpha": 20, "beta": 50, "muon": 50, "proton": 0})
    vid = folder / "essai reçu été.mp4"
    generate_video(vid, sc)
    return vid, sc


# --------------------------------------------------------------------------- #
def test_names_and_config():
    assert is_video_file("a.MP4") and is_video_file("b.webm") and not is_video_file("c.txt")
    for n in ("v.mp4.part", "v.crdownload", "~v.mp4", ".v.mp4", "v.tmp", "~$v.mp4", "v.mp4.tmp.mp4"):
        assert is_partial_name(n), n
    assert not is_partial_name("nuit 2026-10-20.mp4")
    b = BatchConfig.from_dict({"files": "x.mp4", "mass_g": "50,5", "px_per_mm": "abc", "inconnu": 1,
                               "make_reports": False})
    assert b.files == ["x.mp4"] and b.mass_g == pytest.approx(50.5) and b.px_per_mm == 0.0 and not b.make_reports
    assert BatchConfig.from_dict(None) == BatchConfig()
    assert set(BatchConfig().to_dict()) == {"files", "watch_folder", "condition", "sample", "mass_g", "px_per_mm",
                                             "output_dir", "make_reports"}


def test_queue_deduplicates_pending_jobs(tmp_path):
    q = VideoQueue()
    a = tmp_path / "a.mp4"
    assert len(q.add([a, str(a), tmp_path / "b.mp4"])) == 2
    assert len(q.add(a)) == 0                                   # déjà en attente
    snap = q.snapshot()
    assert [j["name"] for j in snap] == ["a.mp4", "b.mp4"] and all(JOB_KEYS <= set(j) for j in snap)
    assert all(j["status"] == STATUS_WAITING and j["eta_s"] == -1 for j in snap)
    job = q.next_waiting()
    job.status = STATUS_DONE
    assert len(q.add(a)) == 1                                   # terminée : peut être reprise
    assert q.pending() == 2 and q.cancel_waiting() == 2 and q.pending() == 0
    q.clear_finished()
    assert q.jobs() == []


def test_batch_runner_files_and_summary(tiny_video, tmp_path):
    vid, sc = tiny_video
    out = tmp_path / "résultats lot"
    changes, finished, done = [], [], []
    runner = BatchRunner({"files": [str(vid), str(tmp_path / "absente.mp4")], "condition": "KCl",
                          "sample": "KCl (chlorure de potassium)", "mass_g": 50, "output_dir": str(out),
                          "make_reports": True}, AppConfig(),
                         on_jobs_changed=changes.append, on_job_finished=finished.append,
                         on_all_finished=lambda: done.append(True))
    runner.run()                                                # même fil (synchrone)
    assert done == [True]
    jobs = {j["name"]: j for j in finished}
    ok, bad = jobs[vid.name], jobs["absente.mp4"]
    assert ok["status"] == STATUS_DONE and ok["progress"] == 1.0 and ok["total"] == sum(ok["counts"].values())
    assert ok["total"] > 0 and ok["duration_s"] > 5.0
    assert any("vérité terrain" in w for w in ok["warnings"])  # échelle exacte lue dans le .gt.json
    assert bad["status"] == STATUS_ERROR and "introuvable" in bad["error"]
    folder = Path(ok["result_folder"])
    assert folder.parent == out and "KCl" in folder.name and "essai reçu été" in folder.name
    for name in ("run.json", "traces.csv", "traces.xlsx", "rapport.html"):
        assert (folder / name).is_file(), name
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert meta["sample_mass_g"] == 50 and meta["calibrated"] and meta["px_per_mm"] == pytest.approx(sc.px_per_mm)
    # Progression transmise pendant l'analyse
    running = [j for snap in changes for j in snap if j["status"] == "en cours" and j["name"] == vid.name]
    assert running and any(0 < j["progress"] < 1 for j in running)
    # Résumé : une ligne par vidéo
    with open(out / "resume_lot.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == SUMMARY_COLUMNS
    by = {r["video"]: r for r in rows}
    r = by[vid.name]
    assert r["statut"] == STATUS_DONE and r["condition"] == "KCl" and r["calibre"] == "oui"
    assert int(r["total"]) == ok["total"] and int(r["beta"]) == ok["counts"].get("beta", 0)
    assert float(r["taux_total_par_min"]) == pytest.approx(ok["total"] / (ok["duration_s"] / 60), rel=1e-3)
    assert float(r["incertitude_total_par_min"]) > 0 and r["dossier"] == str(folder)
    assert by["absente.mp4"]["statut"] == STATUS_ERROR and by["absente.mp4"]["total"] == ""
    assert (out / "resume_lot.xlsx").is_file()
    # Un second lot dans le même dossier COMPLÈTE le résumé (et ne crée pas de doublon de dossier)
    runner2 = BatchRunner({"files": [str(vid)], "condition": "KCl", "output_dir": str(out), "make_reports": False},
                          AppConfig())
    runner2.run()
    with open(out / "resume_lot.csv", encoding="utf-8-sig", newline="") as fh:
        rows2 = list(csv.DictReader(fh))
    assert len(rows2) == 2                                      # même fichier : ligne mise à jour
    folders = [p for p in out.iterdir() if p.is_dir()]
    assert len(folders) == 2 and any(p.name.endswith("(2)") for p in folders)


def test_batch_runner_stop_cancels_current_job(tiny_video, tmp_path):
    vid, _ = tiny_video
    finished = []
    runner = BatchRunner({"files": [str(vid), str(vid) + ".copie.mp4"], "output_dir": str(tmp_path)}, AppConfig(),
                         on_job_finished=finished.append,
                         on_jobs_changed=lambda jobs: runner.stop() if any(
                             j["status"] == "en cours" and j["progress"] > 0 for j in jobs) else None)
    runner.EMIT_EVERY_S = 0.0
    runner.run()
    assert finished[0]["status"] == STATUS_CANCELLED
    assert [j.status for j in runner.queue.jobs()] == [STATUS_CANCELLED, STATUS_CANCELLED]
    assert not (tmp_path / "resume_lot.csv").exists() and not any(p.is_dir() for p in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# Dossier surveillé
# --------------------------------------------------------------------------- #
def test_folder_watcher_waits_for_stable_size(tmp_path):
    w = FolderWatcher(tmp_path, poll_s=5, stable_s=10, probe=None)
    v = tmp_path / "nuit.mp4"
    v.write_bytes(b"x" * 1000)
    for name in ("nuit2.mp4.part", "nuit3.crdownload", "~nuit4.mp4", ".nuit5.mp4", "notes.txt", "vide.avi"):
        (tmp_path / name).write_bytes(b"" if name == "vide.avi" else b"y" * 50)
    assert w.scan(now=0.0) == []                                # première observation
    assert w.scan(now=5.0) == []                                # stable depuis 5 s seulement
    with open(v, "ab") as fh:                                   # le téléchargement continue
        fh.write(b"x" * 500)
    assert w.scan(now=11.0) == []                               # taille changée : on repart de zéro
    assert w.scan(now=16.0) == []
    assert w.scan(now=21.5) == [v]                              # stable 10 s, ouvrable
    assert w.scan(now=30.0) == []                               # déjà confiée à l'analyse
    assert w.pending_count == 0
    assert w.due(now=36.0) and not w.due(now=31.0)


def test_folder_watcher_probe_and_state_file(tmp_path):
    v = tmp_path / "reçue.avi"
    v.write_bytes(b"z" * 2000)
    readable = {"ok": False}
    w = FolderWatcher(tmp_path, stable_s=10, probe=lambda p: readable["ok"], give_up_unreadable_s=60)
    w.scan(now=0.0)
    assert w.scan(now=10.0) == []                               # pas encore lisible
    readable["ok"] = True
    assert w.scan(now=12.0) == [v]
    w.mark_processed(v, STATUS_DONE, "dossier résultats")
    state = json.loads((tmp_path / STATE_FILE).read_text(encoding="utf-8"))
    assert state["fichiers"]["reçue.avi"]["statut"] == STATUS_DONE and state["fichiers"]["reçue.avi"]["taille"] == 2000
    # Nouveau surveillant (programme relancé) : la vidéo n'est pas reprise
    w2 = FolderWatcher(tmp_path, stable_s=10, probe=None)
    assert w2.is_processed("reçue.avi", 2000)
    assert w2.scan(now=0.0) == [] and w2.scan(now=20.0) == [] and w2.pending_count == 0
    # … sauf si elle est remplacée par une autre version (taille différente)
    v.write_bytes(b"z" * 3000)
    assert w2.scan(now=30.0) == [] and w2.scan(now=41.0) == [v]
    # Vidéo jamais lisible : confiée quand même après le délai (l'analyse la signalera en erreur)
    u = tmp_path / "abimee.mkv"
    u.write_bytes(b"q" * 100)
    w3 = FolderWatcher(tmp_path, stable_s=1, probe=lambda p: False, give_up_unreadable_s=5)
    w3.mark_processed(v)
    w3.scan(now=0.0)
    assert w3.scan(now=2.0) == [] and w3.scan(now=8.0) == [u]
    # Fichier d'état corrompu : on repart d'un état vide sans planter
    (tmp_path / STATE_FILE).write_text("{pas du json", encoding="utf-8")
    w4 = FolderWatcher(tmp_path, probe=None)
    assert w4.state["fichiers"] == {} and "illisible" in w4.last_error


def test_watch_folder_runner_end_to_end(tiny_video, tmp_path):
    vid, _ = tiny_video
    watch = tmp_path / "Vidéos reçues"
    watch.mkdir()
    shutil.copy2(vid, watch / "nuit du 20.mp4")
    (watch / "en cours.mp4.crdownload").write_bytes(b"0" * 1000)
    out = tmp_path / "sortie"
    infos = []
    runner = BatchRunner({"watch_folder": str(watch), "condition": "Fond", "output_dir": str(out),
                          "make_reports": False}, AppConfig(), poll_s=0.2, stable_s=0.5, watch_once=True,
                         on_info=infos.append)
    t0 = time.monotonic()
    runner.run()
    assert time.monotonic() - t0 < 120
    jobs = runner.queue.jobs()
    assert [j.name for j in jobs] == ["nuit du 20.mp4"] and jobs[0].status == STATUS_DONE and jobs[0].from_watch
    assert any("nouvelle(s) vidéo(s)" in m for m in infos)
    state = json.loads((watch / STATE_FILE).read_text(encoding="utf-8"))
    assert list(state["fichiers"]) == ["nuit du 20.mp4"]
    assert state["fichiers"]["nuit du 20.mp4"]["dossier"] == jobs[0].result_folder
    # Relance : rien n'est analysé deux fois
    runner2 = BatchRunner({"watch_folder": str(watch), "output_dir": str(out)}, AppConfig(), poll_s=0.2,
                          stable_s=0.5, watch_once=True)
    runner2.run()
    assert runner2.queue.jobs() == []


def test_qt_batch_controller(tiny_video, tmp_path):
    from PySide6.QtWidgets import QApplication

    from wilsontrack.batch import QtBatchController
    app = QApplication.instance() or QApplication([])
    vid, _ = tiny_video
    ctl = QtBatchController()
    got = {"jobs": [], "finished": [], "all": 0, "errors": []}
    ctl.jobs_changed.connect(lambda jobs: got["jobs"].append(jobs))
    ctl.job_finished.connect(lambda d: got["finished"].append(d))
    ctl.all_finished.connect(lambda: got.__setitem__("all", got["all"] + 1))
    ctl.error.connect(lambda m: got["errors"].append(m))
    assert ctl.start({"files": [str(vid)], "output_dir": str(tmp_path), "make_reports": False})
    t0 = time.monotonic()
    while got["all"] == 0 and time.monotonic() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    assert got["all"] == 1 and not got["errors"]
    assert got["finished"][0]["status"] == STATUS_DONE and JOB_KEYS <= set(got["finished"][0])
    assert got["jobs"] and isinstance(got["jobs"][-1], list)
    assert ctl.wait(5) and not ctl.is_running() and ctl.jobs()[0]["status"] == STATUS_DONE


def test_cli_lot_and_surveiller(tiny_video, tmp_path, capsys):
    from wilsontrack.cli import main
    vid, _ = tiny_video
    out = tmp_path / "cli"
    code = main(["lot", str(vid), str(tmp_path / "manquante.mp4"), "--condition", "KCl", "--echantillon", "kcl",
                 "--masse", "50,0", "--sortie", str(out), "--pas-de-rapport"])
    txt = capsys.readouterr()
    assert code == 1                                            # une vidéo en erreur
    assert "Lot terminé : 1 vidéo(s) analysée(s), 1 en erreur" in txt.out and "resume_lot.csv" in txt.out
    assert "manquante.mp4" in txt.err
    with open(out / "resume_lot.csv", encoding="utf-8-sig", newline="") as fh:
        rows = {r["video"]: r for r in csv.DictReader(fh)}
    assert rows[vid.name]["echantillon"] == "KCl (chlorure de potassium)" and float(rows[vid.name]["masse_g"]) == 50
    # Jokers développés (l'invite de commandes Windows ne le fait pas)
    watch = tmp_path / "boîte de réception"
    watch.mkdir()
    shutil.copy2(vid, watch / "reçue 1.mp4")
    code = main(["surveiller", str(watch), "--une-fois", "--intervalle", "0,2", "--stabilite", "0,5",
                 "--sortie", str(out), "--pas-de-rapport"])
    assert code == 0 and (watch / STATE_FILE).is_file()
    assert "1 vidéo(s) analysée(s)" in capsys.readouterr().out
    assert main(["surveiller", str(tmp_path / "absent")]) == 1


def test_video_probe(tiny_video, tmp_path):
    vid, _ = tiny_video
    assert video_probe(vid)
    bad = tmp_path / "faux.mp4"
    bad.write_bytes(b"pas une video" * 10)
    assert not video_probe(bad)
