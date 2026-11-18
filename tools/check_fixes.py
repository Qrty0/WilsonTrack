"""Vérifie les corrections issues des revues : croisements, pointillés, bords, spectre K-40, horloge."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from wilsontrack import synth  # noqa: E402
from wilsontrack.config import AppConfig  # noqa: E402
from wilsontrack.detection import TrackDetector  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK   " if cond else "ÉCHEC") + f" {name} {detail}")


# 1) Spectre K-40
rng = np.random.default_rng(0)
E = np.array([synth._k40_beta_energy(rng) for _ in range(20000)])
check("spectre K-40 : énergie moyenne ≈ 0.56 MeV", abs(E.mean() - 0.56) < 0.03, f"({E.mean():.3f} MeV)")


def frames_with(draw_fn, n=90, shape=(540, 960)):
    rng = np.random.default_rng(1)
    for i in range(n):
        img = np.full(shape, 20, np.float32) + rng.normal(0, 2, shape).astype(np.float32)
        draw_fn(img, i)
        yield cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def run(draw_fn, n=90):
    cfg = AppConfig()
    cfg.physics.px_per_mm = 4.0
    det = TrackDetector(cfg, 30, (540, 960))
    for fr in frames_with(draw_fn, n):
        det.process(fr)
    return det.finalize_run(), det


# 2) Deux traces qui se croisent (B apparaît pendant que A est visible)
def cross(img, i):
    if 20 <= i < 80:
        cv2.line(img, (150, 270), (800, 270), 90, 2, cv2.LINE_AA)
    if 40 <= i < 75:
        cv2.line(img, (475, 60), (475, 480), 90, 2, cv2.LINE_AA)


ev, det = run(cross)
check("traces qui se croisent : 2 traces comptées", len(ev) == 2, f"({len(ev)} traces, rejetées {det.rejected})")


# 3) Trace en pointillés (muon) : 6 tirets de 45 px séparés de 22 px
def dashed(img, i):
    if 20 <= i < 70:
        x = 180
        for _ in range(6):
            cv2.line(img, (x, 300), (x + 45, 300), 80, 2, cv2.LINE_AA)
            x += 67


ev, det = run(dashed)
L = [e.features["length_mm"] for e in ev]
check("trace en pointillés : une seule trace ≈ 99 mm", len(ev) == 1 and abs(L[0] - 99) < 12, f"(longueurs {np.round(L, 1)})")


# 4) Trace qui touche le bord de l'image
def border(img, i):
    if 20 <= i < 60:
        cv2.line(img, (0, 270), (120, 270), 160, 6, cv2.LINE_AA)


ev, det = run(border)
check("trace coupée par le bord : touche_bord = vrai", len(ev) >= 1 and all(e.touches_border for e in ev),
      f"({[e.touches_border for e in ev]})")

# 5) Horloge réelle : caméra annoncée à 30 img/s mais livrant 10 img/s
cfg = AppConfig()
det = TrackDetector(cfg, 30, (540, 960))
for i, fr in enumerate(frames_with(lambda img, i: None, 300)):
    det.process(fr, t=i / 10.0)
check("horloge réelle : durée ≈ 30 s pour 300 images à 10 img/s", abs(det.analyzed_s - 28.8) < 2.0,
      f"({det.analyzed_s:.1f} s, cadence estimée {det.fps:.1f} img/s)")

print("TOUT EST BON" if ok else "DES VÉRIFICATIONS ONT ÉCHOUÉ")
sys.exit(0 if ok else 1)
