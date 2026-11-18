"""WilsonTrack — détection, classification et analyse des traces de particules
dans une chambre de Wilson (chambre à brouillard à diffusion ou à expansion).

Projet d'intégration en sciences — Cégep Marie-Victorin.
"""
import os as _os

__version__ = "2.0.0"
APP_NAME = "WilsonTrack"

# Le parallélisme interne d'OpenCV est contre-productif pour nos images (surcoût de
# synchronisation > gain) : en mono-fil, la conversion couleur passe de ~15 ms à ~2 ms
# et l'étiquetage des composantes de ~40 ms à ~6 ms sur un processeur 12 fils.
try:
    import cv2 as _cv2
    _cv2.setNumThreads(int(_os.environ.get("WILSONTRACK_CV_THREADS", "1")))
except Exception:  # pragma: no cover
    pass
