"""Acquisition : fichiers vidéo, dossiers de photos, enregistreur MP4, écriture d'images."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from wilsontrack.capture import (FileSource, ImageFolderSource, SerialTrigger, VideoRecorder, _imread,
                                 imwrite)

W, H = 64, 48


def _frame(i: int) -> np.ndarray:
    """Image unie dont le niveau encode l'indice (robuste à la compression mp4v)."""
    fr = np.full((H, W, 3), (15 + 20 * i) % 256, np.uint8)
    return fr


def _write_mp4(path, n=10, fps=10.0):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert wr.isOpened()
    for i in range(n):
        wr.write(_frame(i))
    wr.release()
    return path


# --------------------------------------------------------------------------- #
# FileSource
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("folder", ["videos", "vidéos été"])
def test_file_source_reads_all_frames(tmp_path, folder):
    d = tmp_path / folder
    d.mkdir()
    vid = _write_mp4(d / "petit.mp4", n=10, fps=10.0)
    src = FileSource(vid)
    assert not src.live
    assert src.fps == pytest.approx(10.0)
    assert src.n_frames == 10
    assert src.shape == (H, W)
    levels, times = [], []
    while True:
        ok, fr, t = src.read()
        if not ok:
            break
        assert fr.shape == (H, W, 3)
        levels.append(float(fr.mean()))
        times.append(t)
    assert len(levels) == 10
    assert src.position == 10
    assert times == pytest.approx([i / 10.0 for i in range(10)])
    # Le premier read() renvoie bien la PREMIÈRE image (la lecture de contrôle a été rembobinée)
    assert levels[0] == pytest.approx(15, abs=6)
    assert all(b > a for a, b in zip(levels, levels[1:]))
    src.release()


def test_file_source_seek_and_grab(tmp_path):
    src = FileSource(_write_mp4(tmp_path / "v.mp4", n=10))
    src.read()
    src.read()
    fr = src.grab_frame(6)
    assert fr is not None and float(fr.mean()) == pytest.approx(15 + 20 * 6, abs=6)
    assert src.position == 2                  # grab_frame ne déplace pas la lecture
    ok, fr, t = src.read()
    assert ok and float(fr.mean()) == pytest.approx(15 + 20 * 2, abs=6)
    src.seek(8)
    ok, fr, t = src.read()
    assert ok and t == pytest.approx(0.8) and float(fr.mean()) == pytest.approx(15 + 20 * 8, abs=6)
    src.release()


def test_file_source_missing(tmp_path):
    with pytest.raises(IOError):
        FileSource(tmp_path / "absent.mp4")


# --------------------------------------------------------------------------- #
# ImageFolderSource
# --------------------------------------------------------------------------- #
def test_image_folder_source_accented_folder(tmp_path):
    d = tmp_path / "images été"
    d.mkdir()
    for i in (2, 0, 1):                                  # ordre de création ≠ ordre alphabétique
        assert imwrite(d / f"photo_{i:03d}.png", _frame(i))
    imwrite(d / "photo_003.jpg", cv2.resize(_frame(3), (2 * W, 2 * H)))   # taille différente
    (d / "notes.txt").write_text("pas une image", encoding="utf-8")
    src = ImageFolderSource(d, fps=4.0)
    assert src.n_frames == 4
    assert [p.name for p in src.files] == ["photo_000.png", "photo_001.png", "photo_002.png", "photo_003.jpg"]
    assert src.shape == (H, W)
    out = []
    while True:
        ok, fr, t = src.read()
        if not ok:
            break
        assert fr.shape == (H, W, 3)                     # redimensionnée à la taille de la 1re image
        out.append((t, float(fr.mean())))
    assert [t for t, _ in out] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert [m for _, m in out[:3]] == pytest.approx([15, 35, 55], abs=0.5)   # PNG sans perte
    assert out[3][1] == pytest.approx(75, abs=3)
    assert src.position == 4


def test_image_folder_source_empty(tmp_path):
    d = tmp_path / "vide é"
    d.mkdir()
    (d / "lisez-moi.txt").write_text("x", encoding="utf-8")
    with pytest.raises(IOError):
        ImageFolderSource(d)


# --------------------------------------------------------------------------- #
# VideoRecorder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("folder", ["rec", "WilsonTrack Résultats"])
def test_video_recorder_round_trip(tmp_path, folder):
    rec = VideoRecorder()
    assert not rec.active
    target = tmp_path / folder / "sous-dossier" / "brut.mp4"     # dossiers créés automatiquement
    rec.start(target, 15.0, (W, H))
    assert rec.active
    for i in range(12):
        rec.write(_frame(i))
    assert rec.frames == 12
    path = rec.stop()
    assert path == str(target)
    assert not rec.active
    rec.write(_frame(0))                                  # sans effet une fois arrêté
    assert rec.frames == 12
    assert target.is_file() and target.stat().st_size > 0
    src = FileSource(target)
    assert src.fps == pytest.approx(15.0)
    n = 0
    while src.read()[0]:
        n += 1
    assert n == 12
    src.release()


def test_video_recorder_restart_closes_previous(tmp_path):
    rec = VideoRecorder()
    rec.start(tmp_path / "a.mp4", 10, (W, H))
    for i in range(5):
        rec.write(_frame(i))
    rec.start(tmp_path / "b.mp4", 10, (W, H))           # start() arrête d'abord l'enregistrement en cours
    rec.write(_frame(0))
    assert rec.frames == 1
    rec.stop()
    cap = cv2.VideoCapture(str(tmp_path / "a.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()


# --------------------------------------------------------------------------- #
# imwrite / _imread (chemins accentués sous Windows)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["trace é à.png", "vignette ç.jpg"])
def test_imwrite_accented_path(tmp_path, name):
    d = tmp_path / "données chambre"
    d.mkdir()
    img = np.zeros((30, 40, 3), np.uint8)
    img[10:20, 5:35] = (40, 160, 255)
    p = d / name
    assert imwrite(p, img)
    assert p.is_file()
    back = _imread(p)
    assert back is not None and back.shape == img.shape
    if p.suffix == ".png":
        assert np.array_equal(back, img)
    else:
        assert np.abs(back.astype(int) - img.astype(int)).mean() < 6


def test_imwrite_unsupported_extension(tmp_path):
    """Extension inconnue : échec signalé (False ou cv2.error), aucun fichier écrit."""
    p = tmp_path / "x.inconnu"
    try:
        ok = imwrite(p, np.zeros((4, 4, 3), np.uint8))
    except cv2.error:
        ok = False
    assert ok is False
    assert not p.exists()


def test_serial_ports_listing():
    ports = SerialTrigger.ports()
    assert isinstance(ports, list)
    assert all(isinstance(p, str) for p in ports)
