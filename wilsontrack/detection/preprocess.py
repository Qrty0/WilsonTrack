"""Pré-traitement des images : mise à l'échelle, zone d'intérêt, fond adaptatif,
estimation robuste du bruit et image d'« évidence » (rapport signal/bruit + crêtes).

Performance (version 2.0) : les mêmes opérations OpenCV / NumPy, dans le même ordre et sur
les mêmes types, mais écrites dans des tampons RÉUTILISÉS d'une image à l'autre (``Workspace``).
Sous Windows, chaque nouvelle image de 2 Mo coûte autant que le calcul lui-même (pages
mémoire à obtenir du système) : le résultat est identique au bit près, environ 2× plus vite.
Les objets créés par ``TrackDetector`` activent ``reuse_buffers`` : les tableaux renvoyés
sont alors des tampons internes, valables jusqu'à l'image suivante. Les autres appelants
reçoivent, comme avant, des copies indépendantes.
"""
from __future__ import annotations

import math
from collections import deque
from functools import lru_cache

import cv2
import numpy as np


def make_roi_mask(shape_hw: tuple[int, int], roi_norm: list, margin_frac: float = 0.0) -> np.ndarray:
    """Masque uint8 (255 = analysé) à partir d'un polygone normalisé [[x,y],...] (0..1)."""
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    if roi_norm and len(roi_norm) >= 3:
        pts = np.array([[p[0] * (w - 1), p[1] * (h - 1)] for p in roi_norm], np.float32)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    else:
        m = int(round(margin_frac * min(h, w)))
        mask[m:h - m if m else h, m:w - m if m else w] = 255
    return mask


def robust_sigma(values: np.ndarray) -> float:
    """Écart-type robuste (1.4826 × MAD)."""
    if values.size == 0:
        return 1.0
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(max(1.4826 * mad, 1e-3))


# --------------------------------------------------------------------------- #
# Outils de performance (résultats identiques à np.median / cv2.*)
# --------------------------------------------------------------------------- #
class Workspace:
    """Tampons de travail réutilisés d'une image à l'autre (clé + forme + type)."""

    __slots__ = ("_bufs", "_meta")

    def __init__(self):
        self._bufs: dict = {}
        self._meta: dict = {}

    def get_meta(self, key):
        return self._meta.get(key)

    def set_meta(self, key, value) -> None:
        self._meta[key] = value

    def get(self, key: str, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Tampon NON initialisé (son contenu précédent est quelconque). ``shape`` : tuple."""
        k = (key, shape, dtype)
        a = self._bufs.get(k)
        if a is None:
            a = np.empty(shape, dtype)
            self._bufs[k] = a
        return a

    def zeros(self, key: str, shape, dtype=np.uint8) -> np.ndarray:
        """Tampon mis à zéro à sa création seulement : l'appelant le remet à zéro après usage."""
        shape = tuple(int(s) for s in shape)
        k = (key, shape, np.dtype(dtype).str, "z")
        a = self._bufs.get(k)
        if a is None:
            a = np.zeros(shape, dtype)
            self._bufs[k] = a
        return a

    def full(self, key: str, shape, value, dtype) -> np.ndarray:
        """Tampon rempli de ``value`` à sa création seulement (l'appelant le restaure)."""
        shape = tuple(int(s) for s in shape)
        k = (key, shape, np.dtype(dtype).str, "f", value)
        a = self._bufs.get(k)
        if a is None:
            a = np.full(shape, value, dtype)
            self._bufs[k] = a
        return a

    def clear(self) -> None:
        self._bufs.clear()
        self._meta.clear()

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self._bufs.values()))


@lru_cache(maxsize=64)
def ellipse_kernel(size: int) -> np.ndarray:
    """Élément structurant elliptique (mis en cache ; ne pas modifier le tableau renvoyé)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))


@lru_cache(maxsize=16)
def rect_kernel(size: int) -> np.ndarray:
    return np.ones((int(size), int(size)), np.uint8)


def median_sorted(v: np.ndarray):
    """(médiane, valeurs triées) ; la médiane est EXACTEMENT ``np.median(v)`` (même type,
    np.float32 pour un tableau float32) : np.median prend la moyenne (np.mean) des deux
    statistiques d'ordre centrales, que l'on lit ici dans le tableau trié (tri SIMD, plus
    rapide que la sélection partielle de np.median)."""
    n = v.size
    if n < 2:
        return np.median(v), None
    s = np.sort(v, axis=None)
    if np.isnan(s[-1]):
        return np.median(v), None
    k = n // 2
    return np.mean(s[k - 1:k + 1] if n % 2 == 0 else s[k:k + 1]), s


def _percentile_replica(v: np.ndarray, q: float, srt: np.ndarray | None = None):
    """Même calcul que np.percentile(v, q) (méthode « linear », q nombre Python, v 1-D réel)."""
    n = v.size
    vi = (n - 1) * (q / 100)
    if vi >= n - 1:
        i0 = i1 = n - 1
        prev = -1
    else:
        i0 = int(math.floor(vi))
        i1 = i0 + 1
        prev = i0
    if srt is None:
        srt = np.partition(v, (i0, i1, n - 1)) if n > 64 else np.sort(v)
        if np.isnan(srt[n - 1]):
            return np.percentile(v, q)
    t = float(vi - prev)
    a, b = srt[i0], srt[i1]
    d = b - a
    r = a + d * t
    if t >= 0.5:
        r = b - d * (1 - t)
    return r


def _percentile_selfcheck() -> bool:
    rng = np.random.default_rng(12345)
    try:
        for n in (1, 2, 3, 7, 10, 33, 100, 1001):
            for dt in (np.float32, np.float64):
                v = rng.normal(0, 3, n).astype(dt)
                v[:n // 3] = np.round(v[:n // 3])
                for q in (0.0, 30.0, 60.0, 70.0, 84.0, 90.0, 99.5, 100.0):
                    a, b = _percentile_replica(v, q), np.percentile(v, q)
                    if type(a) is not type(b) or not (a == b):
                        return False
        return True
    except Exception:  # pragma: no cover
        return False


_PERCENTILE_OK = _percentile_selfcheck()


def percentile(v: np.ndarray, q: float, srt: np.ndarray | None = None):
    """EXACTEMENT ``np.percentile(v, q)`` pour un tableau 1-D réel sans NaN et q nombre Python
    (méthode linéaire de NumPy reproduite à l'identique ; vérifiée à l'import, sinon NumPy).
    ``srt`` : les mêmes valeurs déjà triées (évite la sélection)."""
    if (not _PERCENTILE_OK or v.ndim != 1 or v.size == 0 or v.dtype.kind != "f"
            or type(q) not in (int, float) or not (0 <= q <= 100)):
        return np.percentile(v, q)
    return _percentile_replica(v, q, srt)


def _kth_two_sorted(a: np.ndarray, b: np.ndarray, k: int):
    """k-ième plus petite valeur (k à partir de 0) de la réunion de deux suites croissantes."""
    na, nb = a.size, b.size
    lo, hi = max(0, k + 1 - nb), min(k + 1, na)
    while lo < hi:                      # plus petit i tel que b[k-i] <= a[i]
        i = (lo + hi) // 2
        j = k + 1 - i
        if i < na and j > 0 and b[j - 1] > a[i]:
            lo = i + 1
        else:
            hi = i
    i = lo
    j = k + 1 - i
    if i > 0 and j > 0:
        return max(a[i - 1], b[j - 1])
    return a[i - 1] if i > 0 else b[j - 1]


def mad_median(v: np.ndarray, srt: np.ndarray | None, m: float, tmp: np.ndarray | None = None):
    """EXACTEMENT ``np.median(np.abs(v - m))`` (type np.float32 pour v float32) sans second tri.

    v − m est calculé comme NumPy le fait (m converti au type de v), sur les valeurs TRIÉES :
    la suite des écarts est croissante, |écart| est la réunion de deux suites croissantes
    (côté négatif retourné, côté positif) dont on lit les statistiques d'ordre centrales."""
    if srt is None or v.dtype != np.float32 or not np.isfinite(srt[0]) or not np.isfinite(srt[-1]):
        return np.median(np.abs(v - m))
    n = srt.size
    d = np.subtract(srt, m, out=tmp if tmp is not None and tmp.shape == srt.shape else None)
    if not (np.isfinite(d[0]) and np.isfinite(d[-1])):
        return np.median(np.abs(v - m))
    j = int(np.searchsorted(d, 0.0, side="left"))       # d[:j] < 0 <= d[j:]
    left = np.negative(d[:j][::-1])                     # |d| du côté négatif, croissant
    right = d[j:]
    k = n // 2
    if n % 2 == 0:
        pair = np.array([_kth_two_sorted(left, right, k - 1), _kth_two_sorted(left, right, k)], np.float32)
    else:
        pair = np.array([_kth_two_sorted(left, right, k)], np.float32)
    return np.mean(pair)


# --------------------------------------------------------------------------- #
class Preprocessor:
    """Convertit une image couleur native en image de travail (float32, niveaux de gris)."""

    def __init__(self, frame_shape_hw: tuple[int, int], process_width: int, blur_sigma: float,
                 temporal_avg: int, roi: list, roi_margin_frac: float = 0.0, stabilize: bool = False):
        h, w = frame_shape_hw
        if process_width and process_width < w:
            self.scale = process_width / float(w)
        else:
            self.scale = 1.0
        self.size_wh = (max(8, int(round(w * self.scale))), max(8, int(round(h * self.scale))))
        self.blur_sigma = blur_sigma
        self.buf: deque[np.ndarray] = deque(maxlen=max(1, int(temporal_avg)))
        self.acc: np.ndarray | None = None
        self.roi_mask = make_roi_mask((self.size_wh[1], self.size_wh[0]), roi, roi_margin_frac)
        self.roi_bool = self.roi_mask > 0
        self.roi_area_px = int(self.roi_bool.sum())
        self.roi_full = bool(self.roi_bool.all())
        self.stabilize = stabilize
        self._stab_ref: np.ndarray | None = None
        self._hann: np.ndarray | None = None
        self.last_shift = (0.0, 0.0)
        # Tampons réutilisés (voir l'en-tête du module) ; False : renvoie des copies
        self.reuse_buffers = False
        self.ws = Workspace()
        self._free: list[np.ndarray] = []        # images float libérées par la moyenne glissante
        # Sous-échantillonnage régulier (grille) pour les statistiques robustes : rapide
        ys, xs = np.nonzero(self.roi_bool)
        if ys.size:
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        else:
            y0, y1, x0, x1 = 0, self.size_wh[1], 0, self.size_wh[0]
        step = max(1, int(np.sqrt(max(1, ys.size) / 25000.0)))
        self._stat_slice = (slice(y0, y1, step), slice(x0, x1, step))
        sub = self.roi_bool[self._stat_slice]
        self._stat_mask = None if sub.all() else sub

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            g = frame
        if self.scale != 1.0:
            g = cv2.resize(g, self.size_wh, interpolation=cv2.INTER_AREA)
        return g.astype(np.float32)

    def _gray_native_dtype(self, frame: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Mêmes opérations que ``to_gray`` (sans la conversion float), dans des tampons.
        ``out`` : tampon de destination de la dernière étape (copie si aucune étape)."""
        ws = self.ws
        if frame.ndim == 3:
            dst = out if (out is not None and self.scale == 1.0) else ws.get("gray_native", frame.shape[:2], frame.dtype)
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=dst)
        else:
            g = frame
        if self.scale != 1.0:
            dst = out if out is not None else ws.get("gray_small", (self.size_wh[1], self.size_wh[0]), g.dtype)
            g = cv2.resize(g, self.size_wh, dst=dst, interpolation=cv2.INTER_AREA)
        elif out is not None and g is not out:
            np.copyto(out, g)
            g = out
        return g

    def _int_slot(self, frame: np.ndarray) -> np.ndarray | None:
        """Tampon entier (niveaux de gris à l'échelle de traitement) pour la moyenne glissante."""
        dt = frame.dtype
        if dt not in (np.uint8, np.uint16):
            return None
        shape = (self.size_wh[1], self.size_wh[0]) if self.scale != 1.0 else frame.shape[:2]
        while self._free:
            cand = self._free.pop()
            if cand.shape == shape and cand.dtype == dt:
                return cand
        return np.empty(shape, dt)

    def _stabilize(self, g: np.ndarray) -> np.ndarray:
        small = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if self._stab_ref is None:
            self._stab_ref = small
            self._hann = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
            return g
        (dx, dy), _resp = cv2.phaseCorrelate(self._stab_ref, small, self._hann)
        dx, dy = 2 * dx, 2 * dy
        self.last_shift = (dx, dy)
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return g
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(g, m, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def motion(self, g: np.ndarray) -> float:
        """Déplacement global (px, échelle de traitement) depuis l'image précédente."""
        small = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA)
        prev = self._motion_prev
        self._motion_prev = small
        if prev is None or prev.shape != small.shape:
            return 0.0
        if self._motion_hann is None or self._motion_hann.shape != small.shape:
            self._motion_hann = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(prev, small, self._motion_hann)
        if resp < 0.05:
            return 0.0
        return 4.0 * float(np.hypot(dx, dy))

    _motion_prev: np.ndarray | None = None
    _motion_hann: np.ndarray | None = None
    last_motion: float = 0.0
    track_motion: bool = False

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        n = self.buf.maxlen
        slot = self._int_slot(frame) if (n > 1 and not self.stabilize and not self.track_motion) else None
        if slot is not None:
            # Moyenne glissante sur les niveaux de gris ENTIERS : la conversion en float32 (exacte)
            # est faite à la volée par la soustraction / l'addition — mêmes valeurs float32 que
            # la version d'origine (acc −= ancienne image ; acc += nouvelle image), sans copie.
            g_n = self._gray_native_dtype(frame, out=slot)
            if self.acc is None or self.acc.shape != g_n.shape:
                self.acc = np.zeros(g_n.shape, np.float32)
                self.buf.clear()
            old = None
            if len(self.buf) == n:
                old = self.buf[0]
                if old.dtype == g_n.dtype:
                    cv2.subtract(self.acc, old, dst=self.acc, dtype=cv2.CV_32F)   # float32 exact
                else:
                    np.subtract(self.acc, old, out=self.acc)
            self.buf.append(g_n)
            cv2.accumulate(g_n, self.acc)                                       # acc += float32(g)
            if old is not None and old is not g_n:
                self._free.append(old)
            g = np.divide(self.acc, len(self.buf), out=self.ws.get("temporal", g_n.shape))
            if self.blur_sigma > 0:
                g = cv2.GaussianBlur(g, (0, 0), self.blur_sigma, dst=self.ws.get("blur", g.shape))
            return g if self.reuse_buffers else g.copy()
        g_n = self._gray_native_dtype(frame)
        if n > 1:
            # Image float stockée dans la moyenne glissante : un tableau distinct par image
            # (on recycle celui que la moyenne vient d'abandonner).
            gf = None
            while self._free:
                cand = self._free.pop()
                if cand.shape == g_n.shape and cand.dtype == np.float32:
                    gf = cand
                    break
            if gf is None:
                gf = np.empty(g_n.shape, np.float32)
        else:
            gf = self.ws.get("gray_f", g_n.shape)
        np.copyto(gf, g_n)                     # ≡ g.astype(np.float32) (conversion exacte)
        g = gf
        if self.stabilize:
            g = self._stabilize(g)
        if self.track_motion:
            self.last_motion = self.motion(g)
        # Moyenne temporelle glissante (les traces vivent ~1 s, le bruit du capteur change à chaque image)
        if n > 1:
            if self.acc is None or self.acc.shape != g.shape:
                self.acc = np.zeros_like(g)
                self.buf.clear()
            old = None
            if len(self.buf) == n:
                old = self.buf[0]
                self.acc -= old
            self.buf.append(g)
            self.acc += g
            if old is not None and old is not g and old.dtype == np.float32:
                self._free.append(old)
            g = np.divide(self.acc, len(self.buf), out=self.ws.get("temporal", g.shape))
        if self.blur_sigma > 0:
            g = cv2.GaussianBlur(g, (0, 0), self.blur_sigma, dst=self.ws.get("blur", g.shape))
        return g if self.reuse_buffers else g.copy()

    def reset_temporal(self):
        self.buf.clear()
        self.acc = None

    def sample_stats(self, img: np.ndarray) -> np.ndarray:
        v = img[self._stat_slice]
        return v[self._stat_mask] if self._stat_mask is not None else v.ravel()


class BackgroundModel:
    """Fond adaptatif à mise à jour sélective.

    Les pixels couverts par une trace détectée ne sont (presque) pas appris,
    ce qui évite que les traces « s'impriment » dans le fond. Le brouillard,
    l'éclairage et les reflets, qui évoluent lentement, sont absorbés.
    """

    def __init__(self, fps: float, time_s: float, warmup_frames: int):
        self.fps = max(1.0, fps)
        self.time_s = max(0.1, time_s)
        self.warmup = max(1, warmup_frames)
        self.bg: np.ndarray | None = None
        self.n = 0
        self.reuse_buffers = False
        self._diff: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.bg is not None and self.n >= self.warmup

    def reset(self):
        self.bg = None
        self.n = 0

    def diff(self, g: np.ndarray) -> np.ndarray:
        if self.bg is None:
            return np.zeros_like(g)
        if self.reuse_buffers:
            if self._diff is None or self._diff.shape != g.shape or self._diff.dtype != np.result_type(g, self.bg):
                self._diff = np.empty(g.shape, np.result_type(g, self.bg))
            return np.subtract(g, self.bg, out=self._diff)
        return g - self.bg

    def update(self, g: np.ndarray, fg_mask: np.ndarray | None, bg_mask: np.ndarray | None = None):
        """``fg_mask`` : pixels des traces (bool, ou uint8 0/1 avec ``bg_mask`` = 1 − fg_mask
        déjà calculé par l'appelant)."""
        if self.bg is None or self.bg.shape != g.shape:
            self.bg = g.copy()
            self.n = 1
            return
        self.n += 1
        a = 1.0 / (self.fps * self.time_s)
        if self.n < self.warmup:
            a = max(a, 1.0 / self.n)
            cv2.accumulateWeighted(g, self.bg, a)
            return
        if fg_mask is None or not fg_mask.any():
            cv2.accumulateWeighted(g, self.bg, a)
            return
        if bg_mask is not None and fg_mask.dtype == np.uint8:
            fg, bgm = fg_mask, bg_mask
        else:
            fg = fg_mask.astype(np.uint8)
            bgm = (1 - fg).astype(np.uint8)
        cv2.accumulateWeighted(g, self.bg, a, mask=bgm)
        # Apprentissage très lent sous les traces (évite qu'une zone reste « collée »)
        cv2.accumulateWeighted(g, self.bg, a * 0.05, mask=fg)


def _hessian_into(img: np.ndarray, sigma: float, ws: Workspace, out_key: str) -> np.ndarray:
    """Calcul de ``hessian_line_response`` dans des tampons (résultat identique au bit près)."""
    if sigma >= 2.5 and min(img.shape) >= 64:
        h, w = img.shape
        small = cv2.resize(img, (w // 2, h // 2), dst=ws.get("hess_down", (h // 2, w // 2), img.dtype),
                           interpolation=cv2.INTER_AREA)
        r = _hessian_into(small, sigma / 2.0, ws, "hess_half")
        return cv2.resize(r, (w, h), dst=ws.get(out_key, (h, w), r.dtype), interpolation=cv2.INTER_LINEAR)
    shp = img.shape
    g = cv2.GaussianBlur(img, (0, 0), sigma, dst=ws.get("hess_g", shp, img.dtype))
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, dst=ws.get("hess_dxx", shp), ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, dst=ws.get("hess_dyy", shp), ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, dst=ws.get("hess_dxy", shp), ksize=3)
    s = (sigma * sigma) / 4.0
    # λ1,2 = tr ∓ disc. Réponse = (λ2<0 ? λ2−λ1 : −λ1) = min(2·disc, disc − tr)  (identité exacte)
    tr = cv2.addWeighted(dxx, 0.5, dyy, 0.5, 0.0, dst=ws.get("hess_tr", shp))
    half = cv2.addWeighted(dxx, 0.5, dyy, -0.5, 0.0, dst=ws.get("hess_half_d", shp))
    disc = cv2.magnitude(half, dxy, magnitude=ws.get("hess_disc", shp))
    two = cv2.add(disc, disc, dst=dxx)            # = multiply(disc, 2.0) : doublement exact
    dmt = cv2.subtract(disc, tr, dst=dyy)
    resp = cv2.min(two, dmt, dst=two)
    resp = cv2.max(resp, 0.0, dst=resp)
    out = ws.get(out_key, shp, resp.dtype)
    s32 = np.float32(s)
    if float(s32) == s and resp.dtype == np.float32:
        # Facteur représentable en float32 : produit float32 correctement arrondi ≡ produit
        # double (exact) puis arrondi en float32, ce que fait cv2.multiply(resp, s).
        return np.multiply(resp, s32, out=out)
    return cv2.multiply(resp, s, dst=out)


def hessian_line_response(img: np.ndarray, sigma: float) -> np.ndarray:
    """Réponse de « ligne brillante » basée sur les valeurs propres de la hessienne.

    Pour une ligne claire sur fond sombre, la valeur propre λ₁ (la plus négative)
    est grande en valeur absolue tandis que λ₂ ≈ 0. Pour une goutte isolée
    (tache ronde), λ₁ ≈ λ₂ < 0 : la réponse λ₂ − λ₁ est alors faible.
    Normalisée par σ² (invariance d'échelle de Lindeberg). Aux grandes échelles
    (σ ≥ 2,5), calcul sur une image réduite de moitié (4× plus rapide).
    """
    return _hessian_into(img, sigma, Workspace(), "out").copy()


def _highpass_into(img: np.ndarray, sigma: float, ws: Workspace, out_key: str = "hp_out") -> np.ndarray:
    if sigma <= 0:
        return img
    f = 4
    h, w = img.shape
    sw, sh = max(2, w // f), max(2, h // f)
    small = cv2.resize(img, (sw, sh), dst=ws.get("hp_small", (sh, sw), img.dtype), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f, dst=ws.get("hp_small_blur", (sh, sw), img.dtype))
    low = cv2.resize(small, (w, h), dst=ws.get("hp_low", (h, w), img.dtype), interpolation=cv2.INTER_LINEAR)
    return np.subtract(img, low, out=ws.get(out_key, (h, w), np.result_type(img, low)))


def highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Retire les grandes structures (nappes de brouillard, gradient d'éclairage).

    Flou gaussien large calculé sur une image réduite (rapide), puis soustrait.
    """
    if sigma <= 0:
        return img
    return _highpass_into(img, sigma, Workspace()).copy()


class EvidenceBuilder:
    """Construit les images d'évidence (en unités de « sigma de bruit »).

    - ``seed`` : réponse de LIGNE (hessienne multi-échelle). Une goutte isolée de la
      « pluie » ou une nappe de brouillard n'y répond presque pas : seules des
      structures allongées peuvent amorcer une détection.
    - ``ext``  : max(contraste, ligne) — sert à étendre la détection à toute la trace
      (extrémités, gouttelettes voisines, cœur des traces alpha très épaisses).
    """

    def __init__(self, use_ridge: bool, scales: tuple, highpass_sigma: float = 12.0):
        self.use_ridge = use_ridge
        self.scales = tuple(scales) if scales else ()
        self.hp_sigma = highpass_sigma
        self._ridge_sigma: dict[float, float] = {}
        self.reuse_buffers = False
        self.ws = Workspace()
        # Médiane du contraste brut de la dernière image (= np.median(pre.sample_stats(diff))) :
        # réutilisée par le détecteur pour repérer un saut global de luminosité.
        self.last_diff_median: float = 0.0
        self._last: Evidence | None = None
        # True : l'évidence complète est calculée d'emblée (le détecteur l'active quand l'image
        # précédente comptait beaucoup de détections) ; même résultat, seul le coût change.
        self.eager_hint = False

    def __call__(self, diff: np.ndarray, pre: Preprocessor):
        """(seed, ext, sigma, hp) en images complètes (interface d'origine)."""
        ev = self.compute(diff, pre)
        ext = ev.ext_full()
        seed = ev.seed_full()
        if self.reuse_buffers:
            return seed, ext, ev.sig, ev.hp
        ext_c = ext.copy()
        seed_c = ext_c if seed is ext else seed.copy()
        return seed_c, ext_c, ev.sig, ev.hp.copy()

    def compute(self, diff: np.ndarray, pre: Preprocessor) -> "Evidence":
        """Calcule tout ce dont dépend l'évidence (contraste passe-haut centré, bruit, réponses
        de crêtes et leurs échelles de bruit) ; les images seed / ext elles-mêmes sont évaluées
        à la demande par l'objet ``Evidence`` renvoyé (valable jusqu'à l'appel suivant)."""
        ws = self.ws
        if self._last is not None:
            self._last._release()
        shp = diff.shape
        # Fond lisse estimé SANS les traces brillantes (écrêtage à médiane + 4σ) : sinon le
        # flou large étale chaque trace brillante et la soustraction crée un halo négatif,
        # dont les bords deviennent de fausses lignes parallèles pour le filtre hessien.
        s0 = pre.sample_stats(diff)
        med0, srt0 = median_sorted(s0)
        m0 = float(med0)
        self.last_diff_median = m0
        tmp = ws.get("stat_tmp", (s0.size,), np.float32) if s0.dtype == np.float32 else None
        sd0 = float(max(1.4826 * mad_median(s0, srt0, m0, tmp), 1e-3))
        clipped = cv2.min(diff, m0 + 4.0 * sd0, dst=ws.get("clipped", shp, diff.dtype))
        hp1 = _highpass_into(clipped, self.hp_sigma, ws)
        # hp = diff − (clipped − passe-haut(clipped)) : mêmes soustractions float32, même ordre
        t = np.subtract(clipped, hp1, out=ws.get("hp_tmp", shp, np.result_type(clipped, hp1)))
        hp = np.subtract(diff, t, out=ws.get("hp", shp, np.result_type(diff, t)))
        sample = pre.sample_stats(hp)
        med_s, srt = median_sorted(sample)
        med = float(med_s)
        tmp = ws.get("stat_tmp", (sample.size,), np.float32) if sample.dtype == np.float32 else None
        sig = float(max(1.4826 * mad_median(sample, srt, med, tmp), 1e-3))
        hp = cv2.subtract(hp, med, dst=hp)
        # snr = cv2.multiply(hp, 1/sig) : évalué à la demande (voir Evidence)
        ridges: list[tuple[np.ndarray, float]] = []
        if self.use_ridge and self.scales:
            for i, s in enumerate(self.scales):
                r = _hessian_into(hp, s, ws, f"ridge{i}")
                rs = pre.sample_stats(r)
                # Échelle de bruit de la réponse : quantile 84 % (≈ +1σ) des pixels de la zone
                scale = float(percentile(rs, 84.0))
                prev = self._ridge_sigma.get(s)
                scale = scale if prev is None else 0.9 * prev + 0.1 * scale
                self._ridge_sigma[s] = scale
                ridges.append((r, 1.0 / max(scale, 1e-4)))   # seed = max_i cv2.multiply(r_i, c_i)
        self._last = Evidence(hp, sig, ridges, ws, eager=self.eager_hint)
        return self._last


def _scaled32(x, c: float) -> np.float32:
    """float32(double(x) × c) : ce que calcule cv2.multiply(image float32, c) pour chaque pixel
    (le facteur est un double ; produit en double puis arrondi en float32)."""
    return np.float32(np.float64(x) * c)


def gt_threshold(c: float, k: float) -> float | None:
    """Plus grand float32 x tel que float32(x·c) ≤ float32(k), pour c > 0.

    La multiplication par c > 0 suivie d'arrondis est croissante : pour une image float32,
    (cv2.multiply(img, c) > k) ⟺ (img > seuil). NumPy comme OpenCV comparent une image float32
    à float32(k) (vérifié). None si le seuil n'est pas trouvé (l'appelant calcule alors l'image)."""
    if not (c > 0.0 and math.isfinite(c)):
        return None
    K = np.float32(k)
    if not np.isfinite(K):
        return None
    with np.errstate(all="ignore"):
        x = np.float32(float(K) / c)
    if not np.isfinite(x):
        return None
    down, up = np.float32(-np.inf), np.float32(np.inf)
    for _ in range(256):
        if _scaled32(x, c) > K:
            x = np.nextafter(x, down)
        else:
            break
    else:
        return None
    for _ in range(256):
        nx = np.nextafter(x, up)
        if _scaled32(nx, c) <= K:
            x = nx
        else:
            return float(x)
    return None


class Evidence:
    """Évidence d'une image, évaluée à la demande (résultats identiques au calcul complet).

    ext = max(snr, seed) avec snr = cv2.multiply(hp, 1/σ) et seed = max_i cv2.multiply(r_i, c_i).
    Ces images complètes ne sont calculées que sur demande (diagnostic) :
    - les masques « ext > k » et « seed > k » s'obtiennent par des seuils équivalents sur hp et
      r_i (voir ``gt_threshold``) ;
    - les VALEURS d'ext ne sont calculées que dans les cadres des détections (``prepare`` puis
      ``ev[y0:y1, x0:x1]``), avec les mêmes opérations élément par élément.
    Valable jusqu'au prochain ``EvidenceBuilder.compute`` (tampons réutilisés)."""

    def __init__(self, hp: np.ndarray, sig: float, ridges: list, ws: Workspace, eager: bool = False):
        self.hp, self.sig, self.ridges, self.ws = hp, sig, ridges, ws
        # eager : images complètes calculées d'emblée (image très bruitée : nombreuses détections)
        self.eager = bool(eager)
        self.c0 = 1.0 / sig
        self.shape = hp.shape
        self.ndim = 2
        self._ready_rects: list[tuple] = []
        self._ext_full_buf: np.ndarray | None = None
        self._seed_full_buf: np.ndarray | None = None
        self._buf: np.ndarray | None = None
        self._ready: np.ndarray | None = None

    # ------------------------------------------------------------------ masques --
    def _mask(self, terms: list, k: float, dst: np.ndarray) -> np.ndarray | None:
        first = True
        for img, c in terms:
            t = gt_threshold(c, k)
            if t is None:
                return None
            if first:
                cv2.compare(img, t, cv2.CMP_GT, dst=dst)
                first = False
            else:
                tmp = cv2.compare(img, t, cv2.CMP_GT, dst=self.ws.get("ev_cmp", self.shape, np.uint8))
                cv2.bitwise_or(dst, tmp, dst=dst)
        return dst

    def ext_mask(self, k: float, dst: np.ndarray) -> np.ndarray:
        """uint8 (0/255) : ext > k  (≡ cv2.compare(ext, k, CMP_GT))."""
        m = None if self.eager else self._mask([(self.hp, self.c0)] + list(self.ridges), k, dst)
        if m is None:
            m = cv2.compare(self.ext_full(), float(k), cv2.CMP_GT, dst=dst)
        return m

    def seed_mask(self, k: float, dst: np.ndarray) -> np.ndarray:
        """uint8 (0/255) : seed > k  (≡ seed > k de NumPy, comparaison en float32)."""
        terms = list(self.ridges) if self.ridges else [(self.hp, self.c0)]
        m = None if self.eager else self._mask(terms, k, dst)
        if m is None:
            m = cv2.compare(self.seed_full(), float(k), cv2.CMP_GT, dst=dst)
        return m

    # ------------------------------------------------------------------ valeurs --
    def _compute(self, ys, xs, out: np.ndarray | None = None) -> np.ndarray:
        """ext sur une zone : mêmes opérations (et même ordre) que le calcul complet."""
        snr = cv2.multiply(self.hp[ys, xs], self.c0)
        if not self.ridges:
            if out is not None:
                np.copyto(out, snr)
                return out
            return snr
        seed = None
        for r, c in self.ridges:
            rn = cv2.multiply(r[ys, xs], c)
            seed = rn if seed is None else cv2.max(seed, rn)
        if out is not None:
            return cv2.max(snr, seed, dst=out)
        return cv2.max(snr, seed)

    def prepare(self, rects) -> None:
        """Calcule ext dans les rectangles (x0, y0, x1, y1) ; ev[...] les relira sans recalcul.
        Nombreux rectangles (image bruitée) : l'image complète est calculée une fois."""
        if self._ext_full_buf is not None:
            return
        rects = [r for r in rects if r[2] > r[0] and r[3] > r[1]]
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
        if len(rects) > 24 or area > 0.3 * self.shape[0] * self.shape[1]:
            self.ext_full()
            return
        if self._buf is None:
            self._buf = self.ws.get("ev_ext", self.shape, np.float32)
            self._ready = self.ws.zeros("ev_ready", self.shape, np.uint8)
        buf, ready = self._buf, self._ready
        for x0, y0, x1, y1 in rects:
            ys, xs = slice(y0, y1), slice(x0, x1)
            if ready[ys, xs].all():
                continue
            self._compute(ys, xs, out=buf[ys, xs])
            ready[ys, xs] = 1
            self._ready_rects.append((y0, y1, x0, x1))

    def __getitem__(self, key):
        if self._ext_full_buf is not None:
            return self._ext_full_buf[key]
        if (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], slice)
                and key[0].step in (None, 1) and key[1].step in (None, 1)):
            ys, xs = key
            if self._ready_rects:
                rv = self._ready[ys, xs]
                if rv.size and rv.all():
                    return self._buf[ys, xs]
            return self._compute(ys, xs)
        return self.ext_full()[key]

    def ext_full(self) -> np.ndarray:
        if self._ext_full_buf is None:
            out = self.ws.get("ev_ext_full", self.shape, np.float32)
            if self.ridges:
                # ext = max(snr, seed) : mêmes opérations que le calcul d'origine (seed réutilisé)
                seed = self.seed_full()
                snr = cv2.multiply(self.hp, self.c0, dst=self.ws.get("ev_snr", self.shape, np.float32))
                self._ext_full_buf = cv2.max(snr, seed, dst=out)
            else:
                self._ext_full_buf = cv2.multiply(self.hp, self.c0, dst=out)
        return self._ext_full_buf

    def seed_full(self) -> np.ndarray:
        if not self.ridges:
            return self.ext_full()
        if self._seed_full_buf is None:
            seed = None
            for r, c in self.ridges:
                if seed is None:
                    seed = cv2.multiply(r, c, dst=self.ws.get("ev_seed_full", self.shape, np.float32))
                else:
                    rn = cv2.multiply(r, c, dst=self.ws.get("ev_ridge_n", self.shape, np.float32))
                    seed = cv2.max(seed, rn, dst=seed)
            self._seed_full_buf = seed
        return self._seed_full_buf

    def _release(self) -> None:
        """Image suivante : la carte des zones déjà calculées est remise à zéro."""
        if self._ready_rects:
            ready = self._ready
            for y0, y1, x0, x1 in self._ready_rects:
                ready[y0:y1, x0:x1] = 0
            self._ready_rects = []
        self._ext_full_buf = self._seed_full_buf = None
