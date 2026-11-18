"""Acquisition d'images : caméra (webcam USB, caméra de téléphone via logiciel webcam…),
fichier vidéo, enregistrement MP4 et déclencheur série (Arduino, chambre à expansion).
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

BACKENDS = {
    "auto": cv2.CAP_ANY,
    "DirectShow": cv2.CAP_DSHOW,
    "Media Foundation": cv2.CAP_MSMF,
}


def mac_video_devices() -> list[dict] | None:
    """Caméras vues par macOS (AVFoundation), dans l'ordre des index d'OpenCV.

    OpenCV numérote les caméras dans l'ordre de AVCaptureDevice.devicesWithMediaType (vidéo, puis
    « muxed »). On lit cette même liste pour donner leur vrai nom aux caméras (« Caméra FaceTime HD »,
    « iPhone de … ») et reconnaître l'iPhone (Caméra de continuité). None si indisponible (hors Mac,
    PyObjC absent…) : on retombe alors sur la numérotation simple.
    """
    if sys.platform != "darwin":
        return None
    try:
        import AVFoundation as AVF  # pyobjc-framework-AVFoundation
        devs = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo) or [])
        devs += list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeMuxed) or [])
    except Exception:
        return None
    out = []
    for d in devs:
        try:
            name = str(d.localizedName())
            dtype = str(d.deviceType() or "")
            model = str(d.modelID() or "")
        except Exception:
            name, dtype, model = "Caméra", "", ""
        low = name.lower()
        phone = ("Continuity" in dtype or "iPhone" in model or "iphone" in low or "ipad" in low
                 or any(k in low for k in ("iriun", "droidcam", "epoccam", "camo")))
        out.append({"name": name, "phone": phone})
    return out


def list_cameras(max_index: int = 6, backend: str = "DirectShow") -> list[dict]:
    """Détecte les caméras disponibles (index 0..max_index-1)."""
    found = []
    mac = mac_video_devices()
    if mac is not None:
        # Sur Mac on connaît la liste exacte : inutile de sonder des index vides (lent avec AVFoundation)
        max_index = len(mac)
    api = BACKENDS.get(backend, cv2.CAP_ANY) if sys.platform == "win32" else cv2.CAP_ANY
    # Pendant la recherche, OpenCV affiche des avertissements bruts pour chaque index vide
    try:
        old = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        old = None
    try:
        for i in range(max_index):
            cap = cv2.VideoCapture(i, api)
            if cap is not None and cap.isOpened():
                ok, fr = cap.read()
                if ok and fr is not None:
                    name, phone = f"Caméra {i}", False
                    if mac is not None and i < len(mac):
                        name, phone = mac[i]["name"], mac[i]["phone"]
                    found.append({"index": i, "width": fr.shape[1], "height": fr.shape[0], "phone": phone,
                                  "name": name, "label": f"{'📱 ' if phone else ''}{name} ({fr.shape[1]}×{fr.shape[0]})"})
            if cap is not None:
                cap.release()
    finally:
        if old is not None:
            try:
                cv2.utils.logging.setLogLevel(old)
            except Exception:
                pass
    return found


class FrameSource:
    """Interface commune : read() -> (ok, frame BGR, t en secondes)."""
    fps: float = 30.0
    live: bool = False
    n_frames: int = 0
    shape: tuple = (0, 0)

    def read(self):  # pragma: no cover - interface
        raise NotImplementedError

    def release(self):
        pass

    @property
    def position(self) -> int:
        return 0


class CameraSource(FrameSource):
    live = True

    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080, fps: float = 30.0,
                 backend: str = "DirectShow", mjpg: bool = True):
        # Sous Windows on essaie le pilote demandé, puis les autres, puis le choix automatique :
        # certaines webcams n'ouvrent qu'avec Media Foundation, d'autres qu'avec DirectShow.
        apis = []
        if sys.platform == "win32":
            first = BACKENDS.get(backend, cv2.CAP_ANY)
            apis = [first] + [a for a in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if a != first]
        else:
            apis = [cv2.CAP_ANY]
        self.cap = None
        for api in apis:
            cap = cv2.VideoCapture(index, api)
            if cap is not None and cap.isOpened():
                self.cap = cap
                break
            if cap is not None:
                cap.release()
        if self.cap is None or not self.cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la caméra {index}. Est-elle branchée / utilisée par un autre logiciel "
                          f"(Zoom, Teams, l'application Caméra de Windows…) ?")
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Beaucoup de webcams renvoient plusieurs images vides le temps de démarrer,
        # surtout si on a demandé une résolution qu'elles ne gèrent pas.
        fr = self._warmup()
        if fr is None:
            # Repli : on réouvre sans forcer de résolution, la caméra choisit son mode natif.
            self.cap.release()
            self.cap = cv2.VideoCapture(index, apis[0])
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fr = self._warmup()
        if fr is None:
            if self.cap is not None:
                self.cap.release()
            raise IOError(f"La caméra {index} ne renvoie pas d'image. Essayez un autre pilote (DirectShow / Media "
                          f"Foundation) ou une résolution plus basse dans les réglages.")
        self.shape = fr.shape[:2]
        f = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = f if f and 1 <= f <= 240 else fps
        self._t0 = time.perf_counter()
        self._first = fr
        self._count = 0
        # Lecture dans un fil à part : cap.read() peut rester bloqué indéfiniment (iPhone qui se met en
        # veille, câble USB débranché…). Le traitement ne fait qu'attendre la dernière image avec un délai
        # maximum : il ne gèle jamais, l'arrêt reste immédiat, et les images en retard sont sautées.
        self._cond = threading.Condition()
        self._latest = None
        self._seq = 0
        self._seen = 0
        self._alive = True
        self._reader = threading.Thread(target=self._reader_loop, name="CameraReader", daemon=True)
        self._reader.start()

    read_timeout = 0.25   # s sans nouvelle image → read() renvoie False (le traitement gère la reconnexion)

    def _reader_loop(self):
        while self._alive:
            try:
                ok, fr = self.cap.read()
            except Exception:
                ok, fr = False, None
            if not self._alive:
                break
            if not ok or fr is None or fr.size == 0:
                time.sleep(0.01)
                continue
            with self._cond:
                self._latest = fr
                self._seq += 1
                self._cond.notify_all()

    def _warmup(self, attempts: int = 30):
        """Lit quelques images pour laisser la caméra démarrer ; renvoie la première image valide."""
        for _ in range(attempts):
            ok, fr = self.cap.read()
            if ok and fr is not None and fr.size > 0:
                return fr
            time.sleep(0.05)
        return None

    def read(self):
        if self._first is not None:
            fr, self._first = self._first, None
            self._count += 1
            return True, fr, time.perf_counter() - self._t0
        with self._cond:
            if self._seq == self._seen:
                self._cond.wait(self.read_timeout)
            if self._seq == self._seen:
                return False, None, time.perf_counter() - self._t0
            self._seen = self._seq
            fr = self._latest
        self._count += 1
        return True, fr, time.perf_counter() - self._t0

    @property
    def position(self) -> int:
        return self._count

    def set(self, prop: str, value: float) -> bool:
        ids = {"exposition": cv2.CAP_PROP_EXPOSURE, "gain": cv2.CAP_PROP_GAIN,
               "luminosite": cv2.CAP_PROP_BRIGHTNESS, "contraste": cv2.CAP_PROP_CONTRAST,
               "mise_au_point": cv2.CAP_PROP_FOCUS, "autofocus": cv2.CAP_PROP_AUTOFOCUS,
               "auto_exposition": cv2.CAP_PROP_AUTO_EXPOSURE}
        if prop not in ids:
            return False
        return bool(self.cap.set(ids[prop], value))

    def get(self, prop: str) -> float:
        ids = {"exposition": cv2.CAP_PROP_EXPOSURE, "gain": cv2.CAP_PROP_GAIN,
               "luminosite": cv2.CAP_PROP_BRIGHTNESS, "contraste": cv2.CAP_PROP_CONTRAST,
               "mise_au_point": cv2.CAP_PROP_FOCUS}
        return float(self.cap.get(ids[prop])) if prop in ids else float("nan")

    def open_driver_settings(self):
        """Ouvre la fenêtre de réglages du pilote (DirectShow sous Windows)."""
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def release(self):
        self._alive = False
        r = getattr(self, "_reader", None)
        if r is not None and r.is_alive():
            r.join(1.5)
            if r.is_alive():
                # cap.read() bloqué dans le pilote : libérer la caméra pendant la lecture ferait planter
                # l'application. On l'abandonne ; macOS la libère à la fermeture de l'application.
                return
        try:
            self.cap.release()
        except Exception:
            pass


class FileSource(FrameSource):
    live = False

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {path}")
        f = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = f if f and 1 <= f <= 1000 else 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, fr = self.cap.read()
        if not ok:
            raise IOError("La vidéo ne contient aucune image lisible.")
        self.shape = fr.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._i = 0

    def read(self):
        ok, fr = self.cap.read()
        t = self._i / self.fps
        if ok:
            self._i += 1
        return ok, fr, t

    def seek(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        self._i = max(0, frame_idx)

    def grab_frame(self, frame_idx: int) -> np.ndarray | None:
        pos = self._i
        self.seek(frame_idx)
        ok, fr = self.cap.read()
        self.seek(pos)
        return fr if ok else None

    @property
    def position(self) -> int:
        return self._i

    def release(self):
        self.cap.release()


class ImageFolderSource(FrameSource):
    """Suite de photos (appareil photo en rafale / time-lapse) traitée comme une vidéo."""
    live = False
    EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, folder: str | Path, fps: float = 5.0):
        self.files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in self.EXT)
        if not self.files:
            raise IOError("Aucune image trouvée dans ce dossier.")
        self.fps = fps
        self.n_frames = len(self.files)
        fr = _imread(self.files[0])
        if fr is None:
            raise IOError(f"Image illisible : {self.files[0]}")
        self.shape = fr.shape[:2]
        self._i = 0

    skipped: list[str] = []

    def read(self):
        # Une photo illisible (fichier tronqué, en cours d'écriture…) est sautée, pas fatale
        while self._i < len(self.files):
            path = self.files[self._i]
            t = self._i / self.fps
            self._i += 1
            try:
                fr = _imread(path)
            except Exception:
                fr = None
            if fr is None:
                self.skipped = self.skipped + [path.name]
                continue
            if fr.shape[:2] != self.shape:
                fr = cv2.resize(fr, (self.shape[1], self.shape[0]))
            return True, fr, t
        return False, None, self._i / self.fps

    @property
    def position(self) -> int:
        return self._i


def _imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    """Écriture d'image compatible avec les chemins accentués (Windows)."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


_OPENH264_CACHE: list[bool] = []


def _openh264_available() -> bool:
    """La bibliothèque OpenH264 (encodeur H.264 d'OpenCV/FFmpeg) est-elle présente ?

    Sans elle, tenter « avc1 » échoue en affichant des erreurs dans la console : on ne
    l'essaie donc que si la DLL est trouvée (dossier d'OpenCV, de l'exécutable, PATH)."""
    if _OPENH264_CACHE:
        return _OPENH264_CACHE[0]
    found = False
    if sys.platform == "win32":
        dirs = [Path(cv2.__file__).parent, Path(sys.executable).parent, Path.cwd()]
        dirs += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        for d in dirs:
            try:
                if any(d.glob("openh264*.dll")):
                    found = True
                    break
            except OSError:
                continue
    _OPENH264_CACHE.append(found)
    return found


def _codec_candidates(suffix: str) -> list[tuple[str, int]]:
    """Codecs essayés, du plus compact au plus sûr, selon le conteneur.

    Mesures sur des images de chambre simulées (bruit du capteur inclus) : mp4v / XVID
    ≈ 0,17 à 0,21 bit par pixel et par image ; le H.264 de Media Foundation, à son débit par
    défaut, ≈ 1,0 bit/px (5 fois plus gros) : il n'est donc PAS utilisé. Le H.264 d'OpenH264
    n'est essayé que si sa DLL est présente."""
    suffix = suffix.lower()
    ff = getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY)
    if suffix == ".avi":
        cands = [("XVID", ff), ("MJPG", ff), ("XVID", cv2.CAP_ANY), ("MJPG", cv2.CAP_ANY)]
    else:
        cands = []
        if suffix in (".mp4", ".m4v", ".mov") and _openh264_available():
            cands.append(("avc1", ff))
        cands += [("mp4v", ff), ("mp4v", cv2.CAP_ANY)]
    return cands


class VideoRecorder:
    """Enregistre les images brutes (non annotées) dans un fichier vidéo, éventuellement découpé
    en segments (« nom-001.mp4 », « nom-002.mp4 »…), réduit (échelle) et allégé (cadence max).

    - ``start(path, fps, size_wh)`` : comportement historique (un seul fichier MP4, mp4v).
    - ``start(..., segment_s=1800, scale=0.5, max_fps=10)`` : segments de 30 min, images
      réduites de moitié, au plus 10 images/s (les images en trop sont sautées).
    - ``write()`` ne lève jamais d'exception : en cas d'erreur (disque plein, fichier
      impossible à créer…) l'enregistrement s'arrête et la cause est dans ``last_error``.
    - Un segment terminé est un fichier complet et lisible : en cas de coupure de courant,
      seul le segment en cours est perdu (MP4) ou tronqué mais lisible (AVI).
    """

    MIN_FREE_BYTES = 200 * 1024 ** 2     # en dessous : arrêt propre (disque presque plein)
    DISK_CHECK_EVERY = 150               # vérification de l'espace libre toutes les N images écrites

    def __init__(self):
        self.writer: cv2.VideoWriter | None = None
        self.path: str = ""              # fichier en cours (ou dernier fichier écrit)
        self.frames = 0                  # images écrites depuis start() (tous segments)
        self.frames_in = 0               # images reçues par write() (y compris sautées)
        self.last_error: str = ""
        self.codec: str = ""
        self._segments: list[str] = []
        self._segmented = False
        self._segment_s = 0.0
        self._seg_index = 0
        self._seg_frames = 0
        self._base: Path | None = None
        self._in_fps = 30.0
        self._out_fps = 30.0
        self._max_fps = 0.0
        self._size_out: tuple[int, int] = (0, 0)
        self._next_t: float | None = None
        self._acc = 0.0
        self._pending_fps: float | None = None

    # ------------------------------------------------------------------ état --
    @property
    def active(self) -> bool:
        return self.writer is not None

    @property
    def segments(self) -> list[str]:
        """Fichiers créés depuis le dernier start() (le dernier peut être en cours d'écriture)."""
        return list(self._segments)

    @property
    def segment_index(self) -> int:
        """Numéro (1, 2, 3…) du segment en cours, −1 si aucun enregistrement actif."""
        return self._seg_index if self.writer is not None else -1

    @property
    def out_fps(self) -> float:
        return self._out_fps

    @property
    def size_out(self) -> tuple[int, int]:
        return self._size_out

    @property
    def bytes_written(self) -> int:
        n = 0
        for p in self._segments:
            try:
                n += os.path.getsize(p)
            except OSError:
                pass
        return n

    # --------------------------------------------------------------- commandes --
    def start(self, path: str | Path, fps: float, size_wh: tuple[int, int], segment_s: float = 0,
              scale: float = 1.0, max_fps: float = 0) -> None:
        """Démarre l'enregistrement (arrête d'abord l'enregistrement en cours).

        Lève IOError si le premier fichier ne peut pas être créé."""
        self.stop()
        self.last_error = ""
        self.frames = 0
        self.frames_in = 0
        self._segments = []
        self._seg_index = 0
        self._seg_frames = 0
        self._next_t = None
        self._acc = 0.0
        self._pending_fps = None
        path = Path(path)
        self._base = path
        self._segment_s = float(segment_s) if segment_s and segment_s > 0 else 0.0
        self._segmented = self._segment_s > 0
        self._in_fps = float(fps) if fps and fps > 0 else 30.0
        self._max_fps = float(max_fps) if max_fps and max_fps > 0 else 0.0
        self._out_fps = min(self._in_fps, self._max_fps) if self._max_fps else self._in_fps
        w, h = int(size_wh[0]), int(size_wh[1])
        scale = float(scale) if scale and scale > 0 else 1.0
        if abs(scale - 1.0) > 1e-6:
            # Dimensions paires : exigées par certains codecs
            w = max(2, int(round(w * scale)) // 2 * 2)
            h = max(2, int(round(h * scale)) // 2 * 2)
        self._size_out = (w, h)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._open_next():
            err = self.last_error or f"Impossible de créer {path}"
            raise IOError(err)

    def set_fps(self, fps: float) -> None:
        """Cadence d'entrée mesurée (caméra plus lente qu'annoncé) : appliquée au PROCHAIN segment."""
        if fps and fps > 0:
            self._pending_fps = float(fps)

    def write(self, frame: np.ndarray, t: float | None = None) -> bool:
        """Ajoute une image (renvoie True si elle a été écrite). Ne lève jamais d'exception.

        ``t`` (s, horloge croissante) sert à respecter la cadence maximale ; sans ``t``, une image
        sur N est gardée selon le rapport des cadences."""
        if self.writer is None or frame is None:
            return False
        self.frames_in += 1
        try:
            if self._out_fps < self._in_fps - 1e-6:
                if t is not None:
                    step = 1.0 / self._out_fps
                    if self._next_t is None or t - self._next_t > 2 * step:
                        self._next_t = t          # premier appel ou trou (pause, caméra figée)
                    if t + 1e-9 < self._next_t:
                        return False
                    self._next_t += step
                else:
                    self._acc += self._out_fps / self._in_fps
                    if self._acc < 1.0 - 1e-9:
                        return False
                    self._acc -= 1.0
            img = frame
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if (img.shape[1], img.shape[0]) != self._size_out:
                img = cv2.resize(img, self._size_out, interpolation=cv2.INTER_AREA)
            self.writer.write(img)
        except Exception as e:  # noqa: BLE001 - une erreur d'écriture ne doit jamais remonter
            self._fail(f"Erreur d'écriture de la vidéo « {Path(self.path).name} » : {e}")
            return False
        self.frames += 1
        self._seg_frames += 1
        if self.frames % self.DISK_CHECK_EVERY == 0:
            free = _free_bytes(Path(self.path).parent)
            if 0 <= free < self.MIN_FREE_BYTES:
                self._fail(f"Espace disque insuffisant ({free / 1024 ** 2:.0f} Mo libres) : "
                           "enregistrement vidéo arrêté.")
                return True
        if self._segmented and self._seg_frames >= self._segment_s * self._out_fps:
            self._roll()
        return True

    def stop(self):
        """Arrête l'enregistrement.

        Renvoie le chemin du fichier (str, comportement historique) ou, en mode segmenté,
        la liste des segments écrits."""
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:  # noqa: BLE001
                pass
        self.writer = None
        if self._segmented:
            return list(self._segments)
        return self.path

    # ---------------------------------------------------------------- interne --
    def _segment_path(self, k: int) -> Path:
        base = self._base
        if not self._segmented:
            return base
        return base.with_name(f"{base.stem}-{k:03d}{base.suffix or '.mp4'}")

    def _open_writer(self, path: Path, fps: float, size_wh: tuple[int, int]):
        """Ouvre un cv2.VideoWriter avec le premier codec qui fonctionne → (writer, codec) ou (None, "")."""
        for fourcc, api in _codec_candidates(path.suffix or ".mp4"):
            try:
                w = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*fourcc), float(fps), size_wh)
            except Exception:  # noqa: BLE001
                continue
            if w is not None and w.isOpened():
                return w, fourcc
            try:
                w.release()
            except Exception:  # noqa: BLE001
                pass
        return None, ""

    def _open_next(self) -> bool:
        if self._pending_fps:
            self._in_fps = self._pending_fps
            self._out_fps = min(self._in_fps, self._max_fps) if self._max_fps else self._in_fps
            self._pending_fps = None
        k = self._seg_index + 1
        p = self._segment_path(k)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            w, codec = self._open_writer(p, self._out_fps, self._size_out)
        except Exception as e:  # noqa: BLE001
            w, codec = None, ""
            self.last_error = f"Impossible de créer {p} : {e}"
        if w is None:
            self.last_error = self.last_error or f"Impossible de créer {p}"
            self.writer = None
            return False
        self.writer, self.codec, self.path = w, codec, str(p)
        self._seg_index = k
        self._seg_frames = 0
        self._segments.append(str(p))
        return True

    def _roll(self):
        try:
            self.writer.release()
        except Exception:  # noqa: BLE001
            pass
        self.writer = None
        free = _free_bytes(Path(self.path).parent)
        if 0 <= free < self.MIN_FREE_BYTES:
            self.last_error = (f"Espace disque insuffisant ({free / 1024 ** 2:.0f} Mo libres) : "
                               "enregistrement vidéo arrêté.")
            return
        self._open_next()

    def _fail(self, msg: str):
        self.last_error = msg
        w, self.writer = self.writer, None
        if w is not None:
            try:
                w.release()
            except Exception:  # noqa: BLE001
                pass


def _free_bytes(folder: Path) -> int:
    try:
        return int(shutil.disk_usage(str(folder)).free)
    except OSError:
        return -1


class SerialTrigger:
    """Liaison série avec un Arduino (chambre à expansion).

    Protocole (voir arduino/declencheur_expansion/declencheur_expansion.ino) :
      PC → Arduino : « E\\n » déclenche une expansion (ouverture de l'électrovanne)
      Arduino → PC : « EXP <millis>\\n » à chaque expansion (bouton ou commande)
    """

    def __init__(self, port: str, baud: int = 115200, on_expansion=None):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.on_expansion = on_expansion
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    @staticmethod
    def ports() -> list[str]:
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()]
        except Exception:
            return []

    def _run(self):
        buf = b""
        while not self._stop.is_set():
            try:
                data = self.ser.read(64)
            except Exception:
                break
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip().upper().startswith(b"EXP") and self.on_expansion:
                    self.on_expansion()

    def fire(self):
        self.ser.write(b"E\n")

    def close(self):
        self._stop.set()
        try:
            self.ser.close()
        except Exception:
            pass
