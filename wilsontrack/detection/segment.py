"""Segmentation : seuillage par hystérésis, composantes connexes, descripteurs rapides
et fusion des fragments colinéaires (traces en pointillés).

Performance (version 2.0), résultats identiques au bit près :
- les composantes « gardées » (celles qui contiennent au moins un pixel graine) sont extraites
  par remplissage (``cv2.floodFill``) à partir des graines au lieu d'étiqueter toute l'image
  (des milliers de petites composantes de bruit) ; elles sont ensuite rangées dans l'ordre
  exact de ``cv2.connectedComponentsWithStats`` (algorithme par blocs 2×2 : ordre du premier
  bloc 2×2 rencontré en balayage ligne par ligne). Repli automatique sur l'étiquetage complet
  si les graines sont trop nombreuses ;
- tampons réutilisés (``Workspace``) ; image d'étiquettes construite seulement sur demande ;
- tests de fusion des fragments vectorisés (mêmes opérations flottantes, même ordre).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import Workspace, ellipse_kernel

# Au-delà, l'étiquetage complet d'OpenCV est plus rapide que les remplissages successifs
MAX_SEED_POINTS = 60000
MAX_FILL_COMPONENTS = 254          # valeurs distinctes disponibles dans le masque de remplissage


@dataclass
class Detection:
    """Composante détectée dans UNE image (coordonnées à l'échelle de traitement)."""
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray          # bool, taille (y1-y0, x1-x0)
    area: int
    cx: float
    cy: float
    theta: float              # orientation de l'axe principal (rad, -pi/2..pi/2)
    l1: float                 # variances principales (px²)
    l2: float
    sum_e: float              # somme de l'évidence (SNR) sur le masque
    mean_e: float

    @property
    def length_q(self) -> float:
        """Longueur rapide ≈ √(12·λ₁) (exacte pour un segment uniforme)."""
        return math.sqrt(12.0 * max(self.l1, 0.0))

    @property
    def width_q(self) -> float:
        return math.sqrt(12.0 * max(self.l2, 1.0 / 12.0))

    @property
    def elongation(self) -> float:
        return self.length_q / max(self.width_q, 1.0)

    @property
    def score(self) -> float:
        """Qualité d'une observation : trace complète ET nette (avant diffusion)."""
        return self.length_q * self.mean_e


def hysteresis_mask(seed: np.ndarray, ext: np.ndarray, k_low: float, k_high: float,
                    roi_bool: np.ndarray) -> np.ndarray:
    """Seuillage par hystérésis : régions ext > k_low contenant au moins un pixel seed > k_high."""
    low = (ext > k_low) & roi_bool
    if not low.any():
        return low
    n, lab = cv2.connectedComponents(low.view(np.uint8), connectivity=8)
    high = (seed > k_high) & low
    keep = np.zeros(n, bool)
    keep[np.unique(lab[high])] = True
    keep[0] = False
    return keep[lab]


_TWELFTH = 1.0 / 12.0


def _moments(mask: np.ndarray) -> tuple[float, float, float, float, float]:
    """Centre, orientation et variances principales des pixels du masque.

    Identique au calcul « np.mean » d'origine : les sommes de coordonnées (entières) sont
    exactes dans n'importe quel ordre ; les sommes de carrés passent par np.add.reduce
    (la même sommation par paires que ndarray.mean), sans le surcoût Python de mean()."""
    ys, xs = np.nonzero(mask)
    n = xs.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    cx = int(xs.sum()) / n
    cy = int(ys.sum()) / n
    dx = xs - cx
    dy = ys - cy
    sxx = float(np.add.reduce(dx * dx)) / n + _TWELFTH
    syy = float(np.add.reduce(dy * dy)) / n + _TWELFTH
    sxy = float(np.add.reduce(dx * dy)) / n
    tr = (sxx + syy) / 2
    disc = math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    l1, l2 = tr + disc, max(tr - disc, 1e-6)
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    return float(cx), float(cy), theta, l1, l2


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _angle_diff(a: float, b: float) -> float:
    """Différence entre deux orientations (mod π), en radians, dans [0, π/2]."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _angle_diff_vec(a: float, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % math.pi
    return np.minimum(d, math.pi - d)


def _roi_info(roi_bool: np.ndarray, ws: Workspace):
    """(masque 0/255 de la zone, zone = image entière ?, aire) — mis en cache dans ws."""
    meta = ws.get_meta("roi")
    if meta is not None and meta[0] is roi_bool:
        return meta[1], meta[2], meta[3]
    roi255 = roi_bool.view(np.uint8) * np.uint8(255)
    full = bool(roi_bool.all())
    area = max(1, int(roi_bool.sum()))
    ws.set_meta("roi", (roi_bool, roi255, full, area))
    return roi255, full, area


def _kept_by_fill(low: np.ndarray, pts: np.ndarray, ws: Workspace):
    """Composantes 8-connexes de ``low`` contenant un point de ``pts``, dans l'ordre de
    cv2.connectedComponentsWithStats. Renvoie [(x, y, w, h, aire, masque bool)] ou None."""
    H, W = low.shape
    fm = ws.zeros("seg_fill", (H + 2, W + 2), np.uint8)       # intérieur toujours remis à 0
    filled = []
    v = 0
    base = 8 | cv2.FLOODFILL_MASK_ONLY
    ok = True
    for x, y in pts.reshape(-1, 2).tolist():
        if fm[y + 1, x + 1]:
            continue
        v += 1
        if v > MAX_FILL_COMPONENTS:
            ok = False
            break
        area, _im, _m, rect = cv2.floodFill(low, fm, (x, y), 0, 0, 0, base | (v << 8))
        filled.append((rect, int(area), v))
    out = []
    if ok:
        for (rx, ry, rw, rh), area, val in filled:
            m = fm[ry + 1:ry + 1 + rh, rx + 1:rx + 1 + rw] == val
            # Ordre d'OpenCV : premier bloc 2×2 (aligné sur l'origine) en balayage ligne par ligne
            top = m[0:2] if ry % 2 == 0 else m[0:1]
            first_x = int(np.flatnonzero(top.any(0))[0])
            out.append(((ry // 2, (rx + first_x) // 2), rx, ry, rw, rh, area, m))
    for (rx, ry, rw, rh), _a, _v in filled:
        fm[ry + 1:ry + 1 + rh, rx + 1:rx + 1 + rw] = 0
    if not ok:
        return None
    out.sort(key=lambda t: t[0])
    return [t[1:] for t in out]


def _kept_by_labels(low: np.ndarray, seeds: np.ndarray, ws: Workspace):
    """Version d'origine : étiquetage complet puis sélection des composantes à graines."""
    H, W = low.shape
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(low, labels=ws.get("seg_cc", (H, W), np.int32),
                                                            connectivity=8)
    keep = np.zeros(n, bool)
    if seeds.any():
        keep[np.unique(lab[seeds])] = True
    keep[0] = False
    out = []
    for i in np.nonzero(keep)[0]:
        i = int(i)
        x, y, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        area = int(stats[i, cv2.CC_STAT_AREA])
        out.append((x, y, w, h, area, lab[y:y + h, x:x + w] == i))
    return out


def segment(seed: np.ndarray, evidence: np.ndarray, roi_bool: np.ndarray, k_low: float, k_high: float,
            close_px: int, min_area: int, merge_gap_px: float, merge_angle_deg: float,
            ws: Workspace | None = None, want_labels: bool = True
            ) -> tuple[list[Detection], np.ndarray | None, float]:
    """Renvoie (détections, image d'étiquettes, fraction « allumée » de la zone).

    Un seul étiquetage : fermeture morphologique du masque bas (relie les gouttelettes
    d'une trace en pointillés), puis on ne garde que les composantes contenant au
    moins un pixel « graine » (hystérésis).

    ``ws`` : tampons réutilisés (sinon alloués pour cet appel). ``want_labels=False`` : l'image
    d'étiquettes n'est pas construite (None) — le détecteur n'en a besoin qu'en diagnostic.
    """
    ws = ws if ws is not None else Workspace()
    H, W = evidence.shape
    roi255, full, roi_area = _roi_info(roi_bool, ws)
    low = cv2.compare(evidence, float(k_low), cv2.CMP_GT, dst=ws.get("seg_low", (H, W), np.uint8))
    low = _close_low(low, roi255, full, close_px, ws)
    # Graines : même comparaison NumPy (float32) que la version d'origine, restreinte au masque bas
    seeds = np.greater(seed, k_high, out=ws.get("seg_seeds", (H, W), bool))
    seeds = np.logical_and(seeds, low, out=seeds)
    return _segment_core(low, seeds.view(np.uint8), evidence, roi_area, min_area, merge_gap_px, merge_angle_deg,
                         ws, want_labels, None)


def segment_evidence(ev, roi_bool: np.ndarray, k_low: float, k_high: float, close_px: int, min_area: int,
                     merge_gap_px: float, merge_angle_deg: float, ws: Workspace | None = None,
                     want_labels: bool = True) -> tuple[list[Detection], np.ndarray | None, float]:
    """``segment`` à partir d'une évidence évaluée à la demande (``preprocess.Evidence``) :
    mêmes masques (seuils équivalents), valeurs d'évidence calculées seulement dans les cadres
    des composantes gardées. Résultat identique à ``segment(ev.seed_full(), ev.ext_full(), …)``."""
    ws = ws if ws is not None else Workspace()
    H, W = ev.shape
    roi255, full, roi_area = _roi_info(roi_bool, ws)
    low = ev.ext_mask(float(k_low), ws.get("seg_low", (H, W), np.uint8))
    low = _close_low(low, roi255, full, close_px, ws)
    seeds = ev.seed_mask(float(k_high), ws.get("seg_seed_u8", (H, W), np.uint8))
    seeds = cv2.bitwise_and(seeds, low, dst=seeds)
    return _segment_core(low, seeds, ev, roi_area, min_area, merge_gap_px, merge_angle_deg, ws, want_labels,
                         ev.prepare)


def _close_low(low: np.ndarray, roi255: np.ndarray, full: bool, close_px, ws: Workspace) -> np.ndarray:
    if not full:
        low = cv2.bitwise_and(low, roi255, dst=low)
    if close_px and close_px > 1:
        k = ellipse_kernel(int(close_px))
        low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, k, dst=ws.get("seg_closed", low.shape, np.uint8))
        if not full:
            low = cv2.bitwise_and(low, roi255, dst=low)
    return low


def _segment_core(low: np.ndarray, seeds_u8: np.ndarray, evidence, roi_area: int, min_area: int,
                  merge_gap_px: float, merge_angle_deg: float, ws: Workspace, want_labels: bool, prepare):
    H, W = low.shape
    pts = cv2.findNonZero(seeds_u8)
    kept = None
    if pts is None:
        kept = []
    elif len(pts) <= MAX_SEED_POINTS:
        kept = _kept_by_fill(low, pts, ws)
    if kept is None:
        kept = _kept_by_labels(low, seeds_u8 > 0, ws)
    lit_frac = float(sum(k[4] for k in kept)) / roi_area if kept else 0.0
    comps = []
    for x, y, w, h, area, m in kept:
        if area < min_area:
            continue
        cx, cy, th, l1, l2 = _moments(m)
        comps.append([0, x, y, x + w, y + h, m, area, cx + x, cy + y, th, l1, l2])
    if prepare is not None and comps:
        prepare([(c[1], c[2], c[3], c[4]) for c in comps])

    # ---- Fusion des fragments colinéaires ----------------------------------
    lab = None
    merged_done = False
    if len(comps) > 1 and merge_gap_px > 0:
        nc = len(comps)
        uf = _UnionFind(nc)
        ang_tol = math.radians(merge_angle_deg)
        Ls = [math.sqrt(12 * c[10]) for c in comps]
        Ws = [math.sqrt(12 * c[11]) for c in comps]
        L = np.array(Ls)
        Wd = np.array(Ws)
        el = np.array([Ls[i] / max(Ws[i], 1.0) for i in range(nc)])
        cxs = np.array([c[7] for c in comps])
        cys = np.array([c[8] for c in comps])
        ths = np.array([c[9] for c in comps])
        areas = np.array([c[6] for c in comps], np.float64)
        big_e = el >= 2.0
        long6 = L >= 6
        for a in range(nc):
            La, Wa, ea = Ls[a], Ws[a], float(el[a])
            if ea < 2.5 or La < 8:
                continue  # seule une composante allongée sert d'« ancre »
            tha = comps[a][9]
            ux, uy = math.cos(tha), math.sin(tha)
            dx = cxs - comps[a][7]
            dy = cys - comps[a][8]
            along = dx * ux + dy * uy
            lateral = np.abs(-dx * uy + dy * ux)
            gap = np.abs(along) - La / 2 - L / 2
            ok = ~((gap > merge_gap_px) | (gap < -0.5 * np.minimum(La, L) - 2))
            ok &= ~(lateral > np.maximum(3.0, 0.75 * (Wa + Wd)))
            ok &= ~(big_e & long6 & (_angle_diff_vec(tha, ths) > ang_tol))
            ok &= ~(~big_e & (areas > 4 * max(Wa, 2) ** 2))   # grosse tache non allongée : ne pas l'absorber
            ok[a] = False
            for b in np.flatnonzero(ok).tolist():
                uf.union(a, b)
        groups: dict[int, list[int]] = {}
        for i in range(nc):
            groups.setdefault(uf.find(i), []).append(i)
        if any(len(g) > 1 for g in groups.values()):
            merged_done = True
            new_lab = np.zeros((H, W), np.int32) if want_labels else None
            merged = []
            for gi, (root, members) in enumerate(groups.items(), start=1):
                if len(members) == 1:
                    c = comps[members[0]]
                    x0, y0, x1, y1, mm = c[1], c[2], c[3], c[4], c[5]
                    if new_lab is not None:
                        new_lab[y0:y1, x0:x1][mm] = gi
                    merged.append([gi, x0, y0, x1, y1, mm, c[6], c[7], c[8], c[9], c[10], c[11]])
                    continue
                x0 = min(comps[m][1] for m in members)
                y0 = min(comps[m][2] for m in members)
                x1 = max(comps[m][3] for m in members)
                y1 = max(comps[m][4] for m in members)
                mm = np.zeros((y1 - y0, x1 - x0), bool)
                for m in members:
                    c = comps[m]
                    mm[c[2] - y0:c[4] - y0, c[1] - x0:c[3] - x0] |= c[5]
                if new_lab is not None:
                    new_lab[y0:y1, x0:x1][mm] = gi
                cx, cy, th, l1, l2 = _moments(mm)
                merged.append([gi, x0, y0, x1, y1, mm, int(mm.sum()), cx + x0, cy + y0, th, l1, l2])
            comps = merged
            lab = new_lab
    if not merged_done:
        if want_labels:
            lab = np.zeros((H, W), np.int32)
        for gi, c in enumerate(comps, start=1):
            if lab is not None:
                lab[c[2]:c[4], c[1]:c[3]][c[5]] = gi
            c[0] = gi

    dets = []
    for c in comps:
        _i, x0, y0, x1, y1, m, area, cx, cy, th, l1, l2 = c
        e = evidence[y0:y1, x0:x1][m]
        dets.append(Detection(x0, y0, x1, y1, m, area, cx, cy, th, l1, l2,
                              float(e.sum()), float(e.mean()) if e.size else 0.0))
    return dets, lab, lit_frac
