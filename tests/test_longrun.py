"""Sessions de longue durée (« toute la nuit ») : réglages, disque, veille, enregistrement segmenté,
points de sauvegarde, délestage mémoire et session accélérée complète."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wilsontrack.analysis import run as run_mod  # noqa: E402
from wilsontrack.analysis.run import RunResult, analyze_video, default_run_name  # noqa: E402
from wilsontrack.capture import VideoRecorder  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.detection.events import TrackEvent  # noqa: E402
from wilsontrack.longrun import (ChamberDeadDetector, KeepAwake, NightSessionConfig, RateWindow,  # noqa: E402
                                 ReconnectPolicy, SessionClock, SessionController, SessionJournal, SourceTime,
                                 estimate_disk_breakdown, estimate_disk_usage, fmt_bytes, fmt_duration, fmt_num,
                                 free_disk_bytes, parse_end_time, run_session)

CONTRACT_KEYS = {"camera_index", "duration_h", "end_time", "segment_min", "record_video", "record_scale",
                 "record_fps", "snapshot_every_min", "autosave_every_min", "keep_awake", "dead_after_min",
                 "condition", "sample", "mass_g", "px_per_mm", "output_dir"}
STATUS_KEYS = {"elapsed_s", "remaining_s", "next_autosave_s", "disk_free_bytes", "segment_index", "recording",
               "n_tracks", "rate_per_min", "last_checkpoint", "warnings", "session_folder"}


# --------------------------------------------------------------------------- #
# Réglages
# --------------------------------------------------------------------------- #
def test_config_from_dict_ignores_unknown_and_defaults_missing():
    cfg = NightSessionConfig.from_dict({"inconnue": 3, "duration_h": "7,5", "record_video": "Réduite",
                                        "keep_awake": "oui", "segment_min": 15, "mass_g": None})
    assert cfg.duration_h == pytest.approx(7.5)
    assert cfg.record_video == "reduite"
    assert cfg.keep_awake is True
    assert cfg.segment_min == 15
    assert cfg.mass_g == 0.0                                  # None → défaut
    assert cfg.autosave_every_min == 5 and cfg.snapshot_every_min == 10 and cfg.dead_after_min == 20
    d = cfg.to_dict()
    assert CONTRACT_KEYS <= set(d)
    assert isinstance(d["segment_min"], int) and isinstance(d["camera_index"], int)
    assert NightSessionConfig.from_dict(d) == cfg              # aller-retour
    assert NightSessionConfig.from_dict({"record_video": "complète"}).record_video == "complete"
    assert NightSessionConfig.from_dict({"record_video": "aucune"}).video_params() is None
    assert NightSessionConfig.from_dict({"record_video": "complete"}).video_params() == (1.0, 0.0)
    assert NightSessionConfig.from_dict({"autosave_every_min": 0}).autosave_every_min == 5.0   # jamais désactivé
    assert NightSessionConfig.from_dict(None) == NightSessionConfig()


def test_end_time_across_midnight():
    cfg = NightSessionConfig.from_dict({"duration_h": 0, "end_time": "07:00"})
    start = datetime(2026, 10, 20, 22, 15)
    assert cfg.end_datetime(start) == datetime(2026, 10, 21, 7, 0)
    assert cfg.planned_seconds(start) == pytest.approx(8 * 3600 + 45 * 60)
    # Heure de fin plus tard le même jour
    assert cfg.end_datetime(datetime(2026, 10, 21, 5, 0)) == datetime(2026, 10, 21, 7, 0)
    # Exactement l'heure de fin : le lendemain
    assert cfg.end_datetime(datetime(2026, 10, 21, 7, 0)) == datetime(2026, 10, 22, 7, 0)
    # La durée a priorité sur l'heure de fin
    cfg2 = NightSessionConfig.from_dict({"duration_h": 2, "end_time": "07:00"})
    assert cfg2.planned_seconds(start) == pytest.approx(7200)
    # Ni durée ni heure : jusqu'à l'arrêt manuel
    assert NightSessionConfig.from_dict({"duration_h": 0}).planned_seconds(start) is None
    # Formats acceptés / refusés
    assert parse_end_time("7h30") == (7, 30)
    assert parse_end_time("23:59:00") == (23, 59)
    assert parse_end_time("6h") == (6, 0)
    assert parse_end_time("") is None
    for bad in ("25:00", "7", "12:75", "midi"):
        with pytest.raises(ValueError):
            parse_end_time(bad)
    bad = NightSessionConfig.from_dict({"duration_h": 0, "end_time": "25:00"})
    assert bad.validate() and bad.end_datetime(start) is None
    assert NightSessionConfig().validate() == []
    txt = cfg.describe(start)
    assert "demain à 07:00" in txt and "8 h 45 min" in txt


def test_french_formatting():
    assert fmt_num(1234.56, 1) == "1 234,6"
    assert fmt_duration(8 * 3600 + 5 * 60) == "8 h 05 min"
    assert fmt_duration(75) == "1 min 15 s"
    assert fmt_bytes(1.5 * 1024 ** 3) == "1,5 Go"


# --------------------------------------------------------------------------- #
# Espace disque
# --------------------------------------------------------------------------- #
def test_disk_estimate_sanity():
    base = {"duration_h": 8, "snapshot_every_min": 10}
    wh, fps = (1280, 720), 30.0
    none = estimate_disk_breakdown(NightSessionConfig.from_dict({**base, "record_video": "aucune"}), wh, fps)
    red = estimate_disk_breakdown(NightSessionConfig.from_dict({**base, "record_video": "reduite"}), wh, fps)
    full = estimate_disk_breakdown(NightSessionConfig.from_dict({**base, "record_video": "complete"}), wh, fps)
    assert none["video"] == 0
    assert 0 < red["video"] < full["video"]
    # Vidéo réduite (640×360, 10 img/s) : 0,35 bit/px/image → ≈ 3,6 Go en 8 h (prudent)
    assert 1e9 < red["video"] < 6e9
    # Vidéo complète 1280×720 à 30 img/s : ≈ 35 Go en 8 h (mesuré ≈ 21 Go en mp4v)
    assert 20e9 < full["video"] < 60e9
    assert red["photos"] == 49 * 2 * 1.5 * 1280 * 720         # 48 intervalles de 10 min + la première
    assert red["total"] == sum(red[k] for k in ("video", "vignettes", "photos", "donnees"))
    assert estimate_disk_usage(NightSessionConfig.from_dict({**base, "record_video": "reduite"}), wh, fps) == red["total"]
    # Plus long → plus gros ; sans fin prévue → 12 h supposées
    longer = estimate_disk_usage(NightSessionConfig.from_dict({**base, "duration_h": 10}), wh, fps)
    assert longer > red["total"]
    unl = estimate_disk_breakdown(NightSessionConfig.from_dict({"duration_h": 0}), wh, fps)
    assert unl["unlimited"] and unl["hours"] == pytest.approx(12.0)


def test_free_disk_bytes(tmp_path):
    assert free_disk_bytes(tmp_path) > 0
    assert free_disk_bytes(tmp_path / "pas" / "encore" / "créé") > 0      # premier parent existant


# --------------------------------------------------------------------------- #
# Veille, horloge, journal, chambre « éteinte », reconnexion
# --------------------------------------------------------------------------- #
def test_keep_awake_no_crash():
    seen = {}

    def body():
        with KeepAwake(True) as ka:
            seen["inside"] = ka.active
        seen["after"] = ka.active
        with KeepAwake(False) as kb:
            seen["disabled"] = kb.active

    th = threading.Thread(target=body)           # comme dans le fil de traitement
    th.start()
    th.join()
    assert seen["after"] is False and seen["disabled"] is False
    if os.name == "nt":
        assert seen["inside"] is True


def test_session_clock_simulated_and_wall():
    c = SessionClock(planned_s=10.0, simulated=True, started_at=datetime(2026, 10, 20, 22, 0))
    for t in (0.0, 4.0, 2.0, 9.0):              # un recul (caméra rouverte) n'est pas compté
        c.tick(t)
    assert c.elapsed_s == pytest.approx(11.0) and c.should_stop() and c.remaining_s == 0.0
    assert c.now() == datetime(2026, 10, 20, 22, 0, 11)
    fake = [100.0]
    w = SessionClock(planned_s=None, simulated=False, monotonic=lambda: fake[0])
    fake[0] = 160.0
    assert w.elapsed_s == pytest.approx(60.0) and w.remaining_s == -1.0 and not w.should_stop()


def test_journal_appends_utf8(tmp_path):
    j = SessionJournal(tmp_path, SessionClock(simulated=True, started_at=datetime(2026, 10, 20, 23, 0)))
    j.write("Démarrage — alcool isopropylique 99 %")
    j.write("Chambre inactive", "attention")
    lines = j.lines()
    assert len(lines) == 2
    assert lines[0].startswith("2026-10-20 23:00:00  [+0:00:00]  Démarrage")
    assert "ATTENTION : Chambre inactive" in lines[1]


def test_chamber_dead_detector_warns_once_and_clears():
    d = ChamberDeadDetector(20 * 60)
    assert d.update(0.0, 0) is None
    assert d.update(600.0, 3) is None                     # traces → dernier instant = 600 s
    assert d.update(600.0 + 1199, 3) is None
    assert d.update(600.0 + 1200, 3) == "eteinte" and d.dead
    assert d.update(600.0 + 1500, 3) is None              # une seule alerte
    assert d.update(600.0 + 1510, 4) == "reprise" and not d.dead
    assert ChamberDeadDetector(0).update(1e6, 0) is None  # désactivé


def test_rate_window():
    r = RateWindow(600)
    for t in range(0, 1200, 20):                          # 3 traces par minute
        r.add(float(t))
    assert r.rate_per_min(1200.0) == pytest.approx(3.0, abs=0.11)


def test_reconnect_policy_and_source_time():
    now = [0.0]
    p = ReconnectPolicy(first_retry_s=3, retry_every_s=10, slow_after_s=60, slow_retry_s=30, give_up_after_s=300,
                        monotonic=lambda: now[0])
    assert p.ok() is None
    assert p.fail() == "attendre"                         # hoquet : pas encore signalé
    now[0] = 1.0
    assert p.fail() == "attendre"
    now[0] = 2.5
    assert p.fail() == "perdue"                           # signalé une seule fois
    now[0] = 3.5
    assert p.fail() == "reconnecter" and p.attempts == 1
    now[0] = 10.0
    assert p.fail() == "attendre"
    now[0] = 14.0
    assert p.fail() == "reconnecter"
    now[0] = 20.0
    assert p.ok() == pytest.approx(20.0)                  # durée de la coupure
    now[0] = 30.0
    assert p.fail() == "attendre"
    now[0] = 30.5
    assert p.ok() is None                                 # hoquet de 0,5 s : rien à signaler
    now[0] = 40.0
    p.fail()
    now[0] = 42.5
    assert p.fail() == "perdue"
    now[0] = 400.0
    assert p.fail() == "abandon"
    st = SourceTime()
    assert [st(t) for t in (0.0, 0.1, 0.2)] == pytest.approx([0.0, 0.1, 0.2])
    st.gap(12.0)                                          # caméra rouverte : repart de 0
    assert st(0.0) == pytest.approx(12.2) and st(0.1) == pytest.approx(12.3)
    assert st(0.05) > 12.3                                # jamais de recul


# --------------------------------------------------------------------------- #
# Enregistrement vidéo segmenté
# --------------------------------------------------------------------------- #
def _frames(n, w=160, h=120):
    out = []
    for i in range(n):
        fr = np.full((h, w, 3), (10 + 4 * i) % 256, np.uint8)
        cv2.putText(fr, str(i), (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.append(fr)
    return out


def _count(path) -> tuple[int, tuple]:
    cap = cv2.VideoCapture(str(path))
    n, shape = 0, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        shape = fr.shape
    cap.release()
    return n, shape


@pytest.mark.parametrize("suffix", [".mp4", ".avi"])
def test_segmented_recorder_creates_playable_segments(tmp_path, suffix):
    rec = VideoRecorder()
    rec.start(tmp_path / "vidéos nuit" / f"nuit{suffix}", 20.0, (160, 120), segment_s=1.0)
    assert rec.active and rec.segment_index == 1
    for fr in _frames(50):
        assert rec.write(fr)
    assert rec.segment_index == 3
    segs = rec.stop()
    assert isinstance(segs, list) and len(segs) == 3
    assert [Path(s).name for s in segs] == [f"nuit-00{k}{suffix}" for k in (1, 2, 3)]
    counts = [_count(s)[0] for s in segs]
    assert counts == [20, 20, 10]
    assert rec.frames == 50 and not rec.active and rec.segment_index == -1
    assert rec.bytes_written == sum(os.path.getsize(s) for s in segs)


def test_recorder_scale_and_max_fps(tmp_path):
    rec = VideoRecorder()
    rec.start(tmp_path / "r.avi", 30.0, (320, 240), scale=0.5, max_fps=10.0)
    assert rec.size_out == (160, 120) and rec.out_fps == 10.0
    written = sum(rec.write(fr, t=i / 30.0) for i, fr in enumerate(_frames(60, 320, 240)))
    assert written == 20 and rec.frames == 20 and rec.frames_in == 60
    path = rec.stop()
    assert path == str(tmp_path / "r.avi")                 # non segmenté : chemin (comportement historique)
    n, shape = _count(path)
    assert n == 20 and shape == (120, 160, 3)
    cap = cv2.VideoCapture(path)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0)
    cap.release()
    # Sans horodatage : une image sur trois (rapport des cadences)
    rec2 = VideoRecorder()
    rec2.start(tmp_path / "r2.mp4", 30.0, (320, 240), max_fps=10.0)
    assert sum(rec2.write(fr) for fr in _frames(30, 320, 240)) == 10
    rec2.stop()
    # Image d'une autre taille (caméra rouverte dans un autre mode) : redimensionnée
    rec3 = VideoRecorder()
    rec3.start(tmp_path / "r3.avi", 10.0, (160, 120))
    assert rec3.write(np.zeros((240, 320, 3), np.uint8)) and rec3.write(np.zeros((120, 160), np.uint8))
    rec3.stop()
    assert _count(tmp_path / "r3.avi")[0] == 2


def test_recorder_write_error_is_handled(tmp_path):
    class Broken:
        def write(self, fr):
            raise OSError("disque plein (simulé)")

        def release(self):
            pass

    rec = VideoRecorder()
    rec.start(tmp_path / "e.avi", 10.0, (160, 120), segment_s=60)
    rec.writer = Broken()
    assert rec.write(_frames(1)[0]) is False                # aucune exception
    assert not rec.active and "disque plein" in rec.last_error
    assert rec.write(_frames(1)[0]) is False
    assert rec.stop() == [str(tmp_path / "e-001.avi")]
    # Fichier impossible à créer : start() lève IOError (comme avant)
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    with pytest.raises(OSError):
        VideoRecorder().start(blocker / "sous" / "v.mp4", 10.0, (160, 120))


def test_recorder_stops_when_disk_is_full(tmp_path, monkeypatch):
    import wilsontrack.capture as cap_mod
    rec = VideoRecorder()
    rec.DISK_CHECK_EVERY = 5
    rec.start(tmp_path / "d.avi", 10.0, (160, 120))
    monkeypatch.setattr(cap_mod, "_free_bytes", lambda folder: 10 * 1024 ** 2)
    for fr in _frames(5):
        rec.write(fr)
    assert not rec.active and "Espace disque insuffisant" in rec.last_error
    assert _count(tmp_path / "d.avi")[0] == 5               # le fichier reste lisible


# --------------------------------------------------------------------------- #
# Points de sauvegarde et délestage des vignettes
# --------------------------------------------------------------------------- #
def _event(i: int, cls: str = "beta", valid: bool = True) -> TrackEvent:
    feats = {"length_mm": 20.0 + i, "chord_mm": 19.0, "width_mm": 0.6, "straightness": 0.9, "tortuosity": 0.05,
             "density_rel": 1.0, "width_rel": 1.0, "density": 100.0, "peak_snr": 30.0, "asymmetry": 1.1,
             "branches": 0, "orientation_deg": 10.0, "radius_mm": float("inf")}
    return TrackEvent(id=i if valid else -1, first_frame=10 * i, last_frame=10 * i + 9, best_frame=10 * i + 3,
                      t_first=i, t_best=i + 0.1, t_last=i + 0.3, n_seen=9, bbox_proc=(0, 0, 10, 10),
                      centroid_proc=(50.0, 50.0), features=feats, path_full=np.zeros((0, 2)),
                      thumb=np.full((20, 30, 3), 30 + i, np.uint8), cls=cls, confidence=0.8, reason="test",
                      valid=valid, px_per_mm_full=5.0, extra={"scale": 1.0, "calibrated": True})


def _run(events, name="2026-10-20 Fond nuit 22h00") -> RunResult:
    cfg = AppConfig()
    return RunResult(name=name, source="camera:0", condition="Fond", sample_type="Aucun", sample_mass_g=0.0, trial=1,
                     date="2026-10-20T22:00:00", fps=30.0, n_frames=1000, duration_s=600.0, frame_w=1280,
                     frame_h=720, px_per_mm=5.0, calibrated=True, roi_area_mm2=30000.0, config=cfg.to_dict(),
                     events=events, rejected=3,
                     timeline=[{"t_s": 60.0, "wall_s": 61.0, "frames": 1800, "disturbed_frames": 0, "noise": 2.1,
                                "lit": 0.01, "counts": {"beta": 2}, "rejected": 1}],
                     session={"duration_h": 8, "stop_reason": "", "partial": True})


def test_checkpoint_save_load_roundtrip_partial(tmp_path):
    folder = tmp_path / "session"
    run = _run([_event(1), _event(2, "muon"), _event(3, valid=False)])
    out = run.save(folder=folder, partial=True, xlsx=False)
    assert out == folder and run.partial
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert meta["partiel"] is True and meta["summary"] == {"beta": 1, "muon": 1}
    assert meta["timeline"][0]["counts"] == {"beta": 2} and meta["session"]["duration_h"] == 8
    assert (folder / "traces.csv").is_file() and not (folder / "traces.xlsx").exists()
    assert sorted(p.name for p in (folder / "vignettes").iterdir()) == ["trace_00001_beta.png",
                                                                       "trace_00002_muon.png"]
    loaded = RunResult.load(folder)
    assert loaded.partial is True and loaded.total == 2
    assert loaded.timeline == meta["timeline"] and loaded.session["duration_h"] == 8
    # Sauvegarde finale au même endroit : « partiel » faux, Excel écrit
    run.save(folder=folder, partial=False)
    assert json.loads((folder / "run.json").read_text(encoding="utf-8"))["partiel"] is False
    assert (folder / "traces.xlsx").is_file()
    assert RunResult.load(folder).partial is False
    # Ancien enregistrement (sans timeline / session / partiel) : rechargé avec les valeurs par défaut
    meta.pop("timeline")
    meta.pop("session")
    meta.pop("partiel")
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    old = RunResult.load(folder)
    assert old.timeline == [] and old.session == {} and old.partial is False
    with pytest.raises(ValueError):
        run.save()


def test_spilled_thumbnails_are_moved_and_renamed_on_save(tmp_path):
    events = [_event(i) for i in range(1, 6)] + [_event(99, valid=False)]
    spill = run_mod.make_spill_dir()
    n = run_mod.spill_event_thumbnails(events, spill, keep_recent=2)
    assert n == 3
    assert [e.thumb is None for e in events[:5]] == [True, True, True, False, False]
    assert all(Path(e.extra["thumb_path"]).parent == spill for e in events[:3])
    events[0].manual_cls = "muon"                           # correction après délestage
    run = _run(events)
    folder = run.save(tmp_path)
    names = sorted(p.name for p in (folder / "vignettes").iterdir())
    assert names == ["trace_00001_muon.png", "trace_00002_beta.png", "trace_00003_beta.png",
                     "trace_00004_beta.png", "trace_00005_beta.png"]
    assert not spill.exists()                               # dossier temporaire vidé puis supprimé
    assert events[0].extra["thumb_path"] == str(folder / "vignettes" / "trace_00001_muon.png")
    # Reclassification finale dans le même dossier : renommage, l'ancien nom disparaît
    events[1].cls = "alpha"
    run.save(tmp_path)
    names = sorted(p.name for p in (folder / "vignettes").iterdir())
    assert "trace_00002_alpha.png" in names and "trace_00002_beta.png" not in names and len(names) == 5


def test_prune_and_limit_memory():
    class Det:
        def __init__(self):
            self.events = [_event(i, valid=(i % 3 == 0)) for i in range(1, 31)]

    d = Det()
    n_inv = sum(not e.valid for e in d.events)
    assert run_mod.prune_invalid_events(d, keep=5) == n_inv - 5
    assert sum(not e.valid for e in d.events) == 5 and sum(e.valid for e in d.events) == 10
    # Le moteur fournit ses propres crochets : ils sont utilisés
    calls = []

    class EngineDet:
        keep_invalid_events = True
        events = []

        def spill_thumbnails(self, folder, keep_recent=300):
            calls.append((str(folder), keep_recent))
            return 7

    e = EngineDet()
    assert run_mod.limit_memory(e, "dossier", keep_recent=12) == 7
    assert e.keep_invalid_events is False and calls == [("dossier", 12)]


def test_analyze_video_long_mode_spills_and_reports_eta(tmp_path, monkeypatch):
    from wilsontrack.synth import SynthConfig, generate_video
    vid = tmp_path / "longue.mp4"
    sc = SynthConfig(width=640, height=360, duration_s=6.0, seed=3,
                     rates_per_min={"alpha": 20, "beta": 60, "muon": 60, "proton": 0})
    generate_video(vid, sc)
    cfg = AppConfig()
    cfg.physics.px_per_mm = sc.px_per_mm
    monkeypatch.setattr(run_mod, "KEEP_RECENT_THUMBS", 1)
    monkeypatch.setattr(run_mod, "SPILL_EVERY_S", 1.0)
    seen = []
    spill = tmp_path / "délestage"
    run, det = analyze_video(str(vid), cfg, progress=lambda f, eta: seen.append((f, eta)), spill_dir=spill,
                             long_video_s=1.0)
    assert seen and all(0 < f <= 1 for f, _ in seen)
    assert all(eta >= 0 for _, eta in seen)                 # temps restant estimé (s)
    valid = [e for e in run.events if e.valid]
    assert len(valid) >= 3
    spilled = [e for e in valid if e.thumb is None and e.extra.get("thumb_path")]
    assert spilled and all(Path(e.extra["thumb_path"]).is_file() for e in spilled)
    assert sum(not e.valid for e in det.events) <= run_mod.KEEP_INVALID
    folder = run.save(tmp_path / "out")
    assert len(list((folder / "vignettes").glob("trace_*.png"))) == len([e for e in valid if e.id >= 0])
    assert run.analysis_warnings == []


# --------------------------------------------------------------------------- #
# Session accélérée complète (sans interface)
# --------------------------------------------------------------------------- #
def _synth_factory(minutes: float, seed: int = 4):
    from wilsontrack.gui.worker import SynthSource
    from wilsontrack.synth import SynthConfig
    sc = SynthConfig(width=640, height=360, fps=10.0, duration_s=minutes * 60 + 30, seed=seed,
                     rates_per_min={"alpha": 10, "beta": 25, "muon": 25, "proton": 0.5})
    return (lambda: SynthSource(sc, realtime=False)), sc


def test_accelerated_session_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(SessionController, "KEEP_THUMBS", 5)       # délestage dès 5 vignettes
    factory, sc = _synth_factory(2.0)
    cfg = {"duration_h": 2.0 / 60.0, "autosave_every_min": 0.5, "snapshot_every_min": 0.5, "segment_min": 0.5,
           "record_video": "reduite", "record_fps": 5, "record_scale": 0.5, "condition": "Fond (rayons cosmiques)",
           "output_dir": str(tmp_path), "dead_after_min": 20, "inconnue": "ignorée"}
    statuses, partial_seen = [], []

    def on_status(st):
        statuses.append(st)
        p = Path(st["session_folder"]) / "run.json"
        if p.is_file():
            try:
                partial_seen.append(json.loads(p.read_text(encoding="utf-8")).get("partiel"))
            except (OSError, ValueError):
                pass

    run = run_session(cfg, AppConfig(), factory, source_name="simulation", on_status=on_status)
    folder = Path(run.folder)
    s = run.session
    assert s["stop_reason"] == "durée prévue atteinte" and s["simulated_clock"] is True
    assert s["planned_s"] == pytest.approx(120.0)
    assert 100.0 < run.duration_s <= 121.0                  # temps d'analyse (hors apprentissage du fond)
    assert run.total >= 10 and not run.partial
    assert s["checkpoints"] >= 3
    # Points de sauvegarde : run.json « partiel » pendant la session, définitif à la fin
    assert True in partial_seen
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert meta["partiel"] is False and meta["session"]["stop_reason"] == "durée prévue atteinte"
    for name in ("traces.csv", "traces.xlsx", "rapport.html", "journal.txt", "session.json"):
        assert (folder / name).is_file(), name
    # Vignettes : une par trace valide (délestées puis renommées si la classe finale a changé)
    thumbs = sorted(p.name for p in (folder / "vignettes").glob("trace_*.png"))
    assert len(thumbs) == run.total
    assert {f"trace_{e.id:05d}_{e.final_cls}.png" for e in run.events if e.valid} == set(thumbs)
    assert any(e.thumb is None for e in run.events if e.valid)       # délestage effectif
    # Photos (brute + annotée) toutes les 30 s
    photos = list((folder / "photos").glob("*.png"))
    assert len(photos) >= 6 and len([p for p in photos if p.name.endswith("annotee.png")]) >= 3
    # Vidéo segmentée (30 s par fichier) et lisible
    segs = [folder / p for p in s["video_segments"]]
    assert len(segs) >= 3 and all(p.is_file() for p in segs)
    n, shape = _count(segs[0])
    assert shape == (180, 320, 3) and 140 <= n <= 151           # 30 s à 5 img/s
    # Journal et chronologie
    journal = (folder / "journal.txt").read_text(encoding="utf-8")
    assert "Point de sauvegarde n° 1" in journal and "Santé :" in journal and "Fin de la session" in journal
    assert run.timeline and run.timeline[-1]["counts"]
    assert {"t_s", "wall_s", "frames", "disturbed_frames", "noise", "lit", "counts", "rejected"} <= set(run.timeline[0])
    # État envoyé régulièrement, avec les clés du contrat
    assert statuses and STATUS_KEYS <= set(statuses[-1])
    assert statuses[-1]["recording"] is True and statuses[-1]["segment_index"] >= 1
    # Rechargement du dossier de session comme un enregistrement ordinaire
    loaded = RunResult.load(folder)
    assert loaded.total == run.total and loaded.session["checkpoints"] == s["checkpoints"]


class _DarkSource:
    """Chambre « éteinte » : images sombres et bruitées, aucune trace (source accélérée)."""
    live = True
    realtime = False
    fps = 10.0
    shape = (270, 480)
    n_frames = 3000

    def __init__(self):
        self._i = 0
        self._rng = np.random.default_rng(0)

    def read(self):
        if self._i >= self.n_frames:
            return False, None, self._i / self.fps
        fr = np.clip(20 + self._rng.normal(0, 2, (270, 480, 3)), 0, 255).astype(np.uint8)
        t = self._i / self.fps
        self._i += 1
        return True, fr, t

    @property
    def position(self):
        return self._i

    def release(self):
        pass


class _FlakyCamera(_DarkSource):
    """« Caméra » qui ne renvoie plus d'images pendant ~4 s (réelles) après 200 images, puis repart
    de t = 0 une fois rouverte (comme une vraie CameraSource)."""
    n_frames = 0                                   # caméra : durée inconnue

    def __init__(self):
        super().__init__()
        self.opened = 1
        self.fail_until = None

    def read(self):
        import time as _t
        if self._i == 200 and self.fail_until is None:
            self.fail_until = _t.monotonic() + 4.0
        if (self.fail_until is not None and _t.monotonic() < self.fail_until) or self._i >= 3000:
            return False, None, 0.0
        fr = np.clip(20 + self._rng.normal(0, 2, (270, 480, 3)), 0, 255).astype(np.uint8)
        t = self._i / self.fps
        self._i += 1
        return True, fr, t


def test_session_survives_camera_dropout(tmp_path):
    cam = _FlakyCamera()

    def factory():
        cam.opened += 1
        return cam

    warnings = []
    run = run_session({"duration_h": 40 / 3600, "record_video": "aucune", "snapshot_every_min": 0,
                       "output_dir": str(tmp_path)}, AppConfig(), factory, simulated_clock=True,
                      on_warning=warnings.append, make_report=False, max_outage_s=60)
    journal = (Path(run.folder) / "journal.txt").read_text(encoding="utf-8")
    assert any("ne renvoie plus d'images" in w for w in warnings)
    assert "Reconnexion à la caméra (tentative 1) : réussie" in journal
    assert "Caméra de nouveau disponible après" in journal
    assert run.session["reconnections"] >= 1 and run.session["stop_reason"] == "durée prévue atteinte"
    assert cam.opened >= 3                          # ouverture initiale + au moins une réouverture
    # La coupure n'est pas comptée comme du temps d'observation
    assert run.duration_s < run.session["elapsed_s"] - 3.0


def test_session_until_manual_stop_and_dead_chamber(tmp_path):
    """Sans fin prévue : arrêt par stop_event ; chambre sans traces → alerte « chambre »."""
    stop = threading.Event()
    warnings = []

    def on_status(st):
        if st["elapsed_s"] > 65:                         # après la 1re ligne de santé (60 s)
            stop.set()

    run = run_session({"duration_h": 0, "record_video": "aucune", "snapshot_every_min": 0,
                       "autosave_every_min": 10, "dead_after_min": 0.5, "output_dir": str(tmp_path)},
                      AppConfig(), _DarkSource, on_status=on_status,
                      on_warning=warnings.append, stop_event=stop, make_report=False)
    assert run.total == 0
    assert run.session["stop_reason"] == "arrêt demandé"
    assert run.session["planned_s"] is None and run.session["video_segments"] == []
    assert any("chambre ne fonctionne peut-être plus" in w for w in warnings)
    assert not (Path(run.folder) / "video").exists()


def test_processing_worker_session(tmp_path):
    """Le fil de traitement de l'interface pilote une session accélérée jusqu'à l'arrêt automatique."""
    from PySide6.QtWidgets import QApplication

    from wilsontrack.gui.worker import ProcessingWorker
    app = QApplication.instance() or QApplication([])
    factory, sc = _synth_factory(1.0, seed=6)
    cfg = AppConfig()
    w = ProcessingWorker(factory, cfg, "simulation", session={"duration_h": 1.0 / 60.0, "autosave_every_min": 0.5,
                                                              "snapshot_every_min": 0, "record_video": "aucune",
                                                              "condition": "KCl", "output_dir": str(tmp_path)})
    got = {"status": [], "run": None, "errors": []}
    w.session_status.connect(lambda d: got["status"].append(d))
    w.run_finished.connect(lambda r: got.__setitem__("run", r))
    w.error.connect(lambda m: got["errors"].append(m))
    w.start()
    import time as _t
    t0 = _t.monotonic()
    while (w.isRunning() or got["run"] is None) and _t.monotonic() - t0 < 300:
        app.processEvents()
        _t.sleep(0.02)
    app.processEvents()
    assert not got["errors"], got["errors"]
    run = got["run"]
    assert run is not None and run.session["stop_reason"] == "durée prévue atteinte"
    assert run.condition == "KCl" and Path(run.folder, "run.json").is_file()
    assert got["status"] and STATUS_KEYS <= set(got["status"][0])
    assert run.session["checkpoints"] >= 1


def test_cli_nuit_simulation(tmp_path, capsys):
    from wilsontrack.cli import main
    code = main(["nuit", "--simulation", "--duree-h", "0,008", "--video", "reduite", "--segment-min", "0,25",
                 "--photo-min", "0", "--sauvegarde-min", "0,2", "--condition", "Fond (rayons cosmiques)",
                 "--sortie", str(tmp_path), "--pas-de-rapport", "--affichage-s", "0"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Session terminée (durée prévue atteinte)" in out and "Espace disque estimé" in out
    folders = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(folders) == 1 and "nuit" in folders[0].name
    f = folders[0]
    assert (f / "run.json").is_file() and (f / "journal.txt").is_file() and not (f / "rapport.html").exists()
    assert len(list((f / "video").glob("*.avi"))) >= 2
    # Arguments invalides : code 2
    assert main(["nuit", "--duree-h", "8", "--jusqua", "07:00"]) == 2     # options exclusives
    assert main(["nuit", "--jusqua", "25:00"]) == 2
    capsys.readouterr()


def test_session_folder_name_and_journal_header(tmp_path):
    cfg = NightSessionConfig.from_dict({"condition": "Banane", "output_dir": str(tmp_path), "duration_h": 1})
    app = AppConfig()
    ctl = SessionController(cfg, app, started_at=datetime(2026, 10, 20, 22, 15))
    assert ctl.folder.name == "2026-10-20 Banane nuit 22h15" and app.run.condition == "Banane"
    ctl2 = SessionController(cfg, AppConfig(), started_at=datetime(2026, 10, 20, 22, 15))
    assert ctl2.folder.name == "2026-10-20 Banane nuit 22h15 (2)"
    assert default_run_name("Banane", 1).endswith("essai-01")
    assert ctl.clock.end_at == datetime(2026, 10, 20, 23, 15) - timedelta(0)
