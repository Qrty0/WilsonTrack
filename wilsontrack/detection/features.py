"""Mesures morphologiques et physiques d'une trace.

À partir du masque binaire et de l'image de contraste (trace − fond), on calcule :
- le squelette et son plus long chemin (longueur géodésique réelle, même courbée) ;
- la corde (distance entre extrémités) → rectitude = corde / longueur ;
- l'épaisseur (transformée de distance ET largeur effective d'intensité) ;
- la densité linéaire d'intensité (≈ pouvoir ionisant dE/dx) ;
- la tortuosité (somme des changements de direction par mm) ;
- l'ajustement d'un cercle (rayon de courbure → quantité de mouvement si champ B).
"""
from __future__ import annotations

import heapq
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

from .preprocess import ellipse_kernel, median_sorted, rect_kernel
from .preprocess import percentile as _percentile

SQRT2 = math.sqrt(2.0)


def _sssp(adj: list, n: int, src: int) -> list:
    """Distances géodésiques depuis src (Dijkstra). Mêmes additions flottantes que SciPy
    (distance du nœud extrait + poids de l'arête) : les distances finales sont uniques."""
    inf = math.inf
    dist = [inf] * n
    dist[src] = 0.0
    done = [False] * n
    heap = [(0.0, src)]
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        d, v = pop(heap)
        if done[v]:
            continue
        done[v] = True
        for u, wt in adj[v]:
            if not done[u]:
                nd = d + wt
                if nd < dist[u]:
                    dist[u] = nd
                    push(heap, (nd, u))
    return dist


def _argmax_first(dist: list) -> int:
    """np.argmax(d) avec d[~isfinite] = -1 (premier maximum)."""
    best, bi = -2.0, 0
    for i, v in enumerate(dist):
        if v == math.inf:
            v = -1.0
        if v > best:
            best, bi = v, i
    return bi


def _longest_path_exact(n: int, r: list, c: list, wv: list) -> list | None:
    """Même chemin que la version SciPy (double Dijkstra) ou None si un choix de prédécesseur
    dépendrait de l'ordre interne du tas de SciPy (égalité parfaite de deux distances).

    SciPy garde comme prédécesseur le PREMIER nœud extrait qui donne la distance finale ; les
    nœuds sont extraits par distance croissante : c'est donc le candidat de plus petite distance,
    déterminé sans ambiguïté sauf égalité exacte (repli sur SciPy dans ce cas)."""
    adj: list[list] = [[] for _ in range(n)]
    for a, b, wt in zip(r, c, wv):
        adj[a].append((b, wt))
        adj[b].append((a, wt))
    # Composantes connexes numérotées dans l'ordre du plus petit indice (connected_components)
    comp = [-1] * n
    sizes = []
    for s in range(n):
        if comp[s] >= 0:
            continue
        lab = len(sizes)
        comp[s] = lab
        stack = [s]
        size = 0
        while stack:
            v = stack.pop()
            size += 1
            for u, _w in adj[v]:
                if comp[u] < 0:
                    comp[u] = lab
                    stack.append(u)
        sizes.append(size)
    start = 0
    if len(sizes) > 1:
        big = sizes.index(max(sizes))
        start = comp.index(big)
    a = _argmax_first(_sssp(adj, n, start))
    da = _sssp(adj, n, a)
    b = _argmax_first(da)
    path = [b]
    while path[-1] != a:
        w = path[-1]
        dw = da[w]
        best, best_d, tie = -1, 0.0, False
        for v, wt in adj[w]:
            dv = da[v]
            if dv + wt == dw:
                if best < 0 or dv < best_d:
                    best, best_d, tie = v, dv, False
                elif dv == best_d:
                    tie = True
        if best < 0:
            break
        if tie:
            return None
        path.append(best)
    return path[::-1]


def longest_skeleton_path(skel: np.ndarray) -> np.ndarray:
    """Plus long chemin géodésique d'un squelette (double Dijkstra). Renvoie (N, 2) en (y, x).

    Calcul direct en Python (petits graphes : bien plus rapide que SciPy), identique au calcul
    SciPy d'origine ; repli sur SciPy si un prédécesseur est ambigu (égalité exacte)."""
    ys, xs = np.nonzero(skel)
    n = ys.size
    if n == 0:
        return np.zeros((0, 2), np.float32)
    if n == 1:
        return np.array([[ys[0], xs[0]]], np.float32)
    h, w = skel.shape
    idx = -np.ones((h + 2, w + 2), np.int64)
    idx[ys + 1, xs + 1] = np.arange(n)
    rows, cols, wts = [], [], []
    for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
        nb = idx[ys + 1 + dy, xs + 1 + dx]
        ok = nb >= 0
        rows.append(np.arange(n)[ok])
        cols.append(nb[ok])
        wts.append(np.full(int(ok.sum()), wt))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    wv = np.concatenate(wts)
    path = _longest_path_exact(n, r.tolist(), c.tolist(), wv.tolist())
    if path is not None:
        path = np.array(path)
        return np.column_stack([ys[path], xs[path]]).astype(np.float32)
    return _longest_path_scipy(ys, xs, n, r, c, wv)


def _longest_path_scipy(ys, xs, n, r, c, wv) -> np.ndarray:
    """Version d'origine (SciPy)."""
    g = coo_matrix((wv, (r, c)), shape=(n, n)).tocsr()
    # Départ dans la plus grande composante du squelette (et non au premier pixel)
    ncomp, comp = connected_components(g, directed=False)
    start = 0
    if ncomp > 1:
        big = int(np.argmax(np.bincount(comp)))
        start = int(np.nonzero(comp == big)[0][0])
    d0 = dijkstra(g, directed=False, indices=start)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    da, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
    da[~np.isfinite(da)] = -1
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    path = np.array(path[::-1])
    return np.column_stack([ys[path], xs[path]]).astype(np.float32)


def resample_path(path: np.ndarray, step: float = 1.0) -> np.ndarray:
    if len(path) < 2:
        return path.copy()
    seg = np.sqrt((np.diff(path, axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return path[:1].copy()
    t = np.arange(0, s[-1] + 1e-6, step)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def fit_circle_taubin(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Ajustement algébrique d'un cercle (méthode de Taubin, version SVD de N. Chernov).

    Renvoie (xc, yc, R, rms). R = inf si les points sont alignés.
    """
    if x.size < 5:
        return 0.0, 0.0, float("inf"), 0.0
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    z = u * u + v * v
    zm = z.mean()
    if zm <= 1e-9:
        return float(xm), float(ym), float("inf"), 0.0
    z0 = (z - zm) / (2.0 * math.sqrt(zm))
    try:
        _u, _s, vt = np.linalg.svd(np.column_stack([z0, u, v]), full_matrices=False)
    except np.linalg.LinAlgError:
        return float(xm), float(ym), float("inf"), 0.0
    a = vt[2].copy()
    a0 = a[0] / (2.0 * math.sqrt(zm))
    if abs(a0) < 1e-12:
        return float(xm), float(ym), float("inf"), 0.0
    a3 = -zm * a0
    xc = -a[1] / a0 / 2.0 + xm
    yc = -a[2] / a0 / 2.0 + ym
    r2 = a[1] ** 2 + a[2] ** 2 - 4 * a0 * a3
    if r2 <= 0:
        return float(xc), float(yc), float("inf"), 0.0
    r = math.sqrt(r2) / abs(a0) / 2.0
    rms = float(np.sqrt(np.mean((np.hypot(x - xc, y - yc) - r) ** 2)))
    return float(xc), float(yc), float(r), rms


def _arclen(p: np.ndarray) -> np.ndarray:
    if len(p) < 2:
        return np.zeros(len(p))
    return np.concatenate([[0.0], np.cumsum(np.sqrt((np.diff(p, axis=0) ** 2).sum(1)))])


def _sample(img: np.ndarray, pts_yx: np.ndarray) -> np.ndarray:
    """Échantillonnage bilinéaire (0 hors de l'image)."""
    return map_coordinates(img, [pts_yx[:, 0], pts_yx[:, 1]], order=1, mode="constant", cval=0.0)


def _smooth_path(path: np.ndarray, sigma: float) -> np.ndarray:
    rp = resample_path(path, 1.0)
    if len(rp) > 4 and sigma > 0:
        return np.column_stack([gaussian_filter1d(rp[:, 0], sigma, mode="nearest"),
                                gaussian_filter1d(rp[:, 1], sigma, mode="nearest")])
    return rp


def _extend(rps: np.ndarray, ext: float, look: int) -> tuple[np.ndarray, int]:
    """Prolonge le chemin à ses deux extrémités selon la tangente locale."""
    look = max(1, min(look, len(rps) - 1))
    d0 = rps[0] - rps[look]
    d1 = rps[-1] - rps[-1 - look]
    n0, n1 = np.hypot(*d0), np.hypot(*d1)
    s = np.arange(1.0, ext + 1e-6, 1.0)
    pre = rps[0] + np.outer(s, d0 / n0) if n0 > 1e-6 else np.zeros((0, 2))
    post = rps[-1] + np.outer(s, d1 / n1) if n1 > 1e-6 else np.zeros((0, 2))
    return np.vstack([pre[::-1], rps, post]), len(pre)


def bridge_fragments(mask: np.ndarray) -> np.ndarray:
    """Relie les fragments d'une trace en pointillés (masque à plusieurs composantes).

    Les fragments sont ordonnés le long de l'axe principal ; l'extrémité de chacun est
    reliée par un segment à l'extrémité en regard du suivant. Sans cela, le squelette
    ne décrirait qu'un seul fragment (longueur et forme fausses).
    """
    n, lab = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.vstack([xs - cx, ys - cy])) if xs.size > 2 else np.eye(2)
    w, v = np.linalg.eigh(cov)
    ux, uy = v[:, int(np.argmax(w))]
    frags = []
    for i in range(1, n):
        fy, fx = np.nonzero(lab == i)
        proj = (fx - cx) * ux + (fy - cy) * uy
        a, b = int(np.argmin(proj)), int(np.argmax(proj))
        frags.append((float(proj.mean()), (int(fx[a]), int(fy[a])), (int(fx[b]), int(fy[b]))))
    frags.sort(key=lambda t: t[0])
    out = mask.view(np.uint8).copy()
    for (_c0, _s0, e0), (_c1, s1, _e1) in zip(frags[:-1], frags[1:]):
        cv2.line(out, e0, s1, 1, 1)
    return out.astype(bool)


def _fwhm(profile: np.ndarray, step: float) -> float:
    """Largeur à mi-hauteur d'un profil centré (interpolation linéaire, ligne de base retirée)."""
    if profile.size < 3:
        return 0.0
    k = max(1, profile.size // 6)
    base = float(median_sorted(np.concatenate([profile[:k], profile[-k:]]))[0])
    profile = profile - base
    c = profile.size // 2
    lo = max(0, c - 2)
    pk_i = lo + int(np.argmax(profile[lo:c + 3]))
    pk = float(profile[pk_i])
    if pk <= 0:
        return 0.0
    half = 0.5 * pk
    i = pk_i
    while i > 0 and profile[i] > half:
        i -= 1
    left = i + (half - profile[i]) / max(profile[i + 1] - profile[i], 1e-9) if profile[i] <= half else 0.0
    j = pk_i
    n = profile.size
    while j < n - 1 and profile[j] > half:
        j += 1
    right = j - 1 + (profile[j - 1] - half) / max(profile[j - 1] - profile[j], 1e-9) if profile[j] <= half else n - 1
    return float(max(0.0, right - left) * step)


def compute_features(mask: np.ndarray, diff: np.ndarray, px_per_mm: float,
                     noise_sigma: float = 1.0) -> dict:
    """Calcule toutes les mesures d'une trace.

    mask : bool (crop) ; diff : contraste trace − fond (même crop, float32).
    px_per_mm : pixels (échelle de traitement) par millimètre.

    Les longueurs et largeurs sont mesurées sur le PROFIL D'INTENSITÉ (à mi-hauteur),
    et non sur le masque binaire, pour être indépendantes du seuil de détection.
    """
    pad = 3
    # (≡ np.pad(mask, 3).astype(bool) et np.pad(diff.astype(np.float32), 3) : bord de zéros)
    h0, w0 = mask.shape
    m = np.zeros((h0 + 2 * pad, w0 + 2 * pad), bool)
    m[pad:pad + h0, pad:pad + w0] = mask
    d = np.zeros((h0 + 2 * pad, w0 + 2 * pad), np.float32)
    d[pad:pad + h0, pad:pad + w0] = diff
    # Ligne de base locale (brouillard, éclairage) : médiane des pixels hors de la trace
    around = ~cv2.dilate(m.view(np.uint8), rect_kernel(7)).astype(bool)
    around[:pad, :] = around[-pad:, :] = False
    around[:, :pad] = around[:, -pad:] = False
    if int(around.sum()) >= 20:
        d = d - float(median_sorted(d[around])[0])
    dpos = np.maximum(d, 0)
    dsm = cv2.GaussianBlur(dpos, (0, 0), 0.8)
    area = int(m.sum())
    mm = 1.0 / max(px_per_mm, 1e-6)  # mm par pixel

    # Le chemin central est cherché sur la partie BRILLANTE de la trace : les appendices pâles
    # (gouttelettes accolées, brume) ne doivent pas détourner le squelette.
    m_core = m
    if area >= 12:
        vals = dsm[m]
        thr = 0.3 * float(_percentile(vals, 90))
        mb = m & (dsm >= thr)
        if thr > 0 and int(mb.sum()) >= max(6, int(0.25 * area)):
            m_core = mb
    # Trace en pointillés : on relie les fragments avant de squelettiser
    m_skel = bridge_fragments(m_core)
    skel = skeletonize(m_skel)
    path = longest_skeleton_path(skel)

    # --- Épaisseur du masque (transformée de distance) ----------------------
    dt = cv2.distanceTransform(m.view(np.uint8), cv2.DIST_L2, 5)
    sk_in = skel & m
    if sk_in.any():
        w_mask = max(1.0, 2.0 * float(median_sorted(dt[sk_in])[0]) - 1.0)
    else:
        w_mask = max(1.0, 2.0 * float(dt.max()) - 1.0)

    # --- Chemin central lissé (sous-pixel) ----------------------------------
    if len(path) >= 2:
        rps = _smooth_path(path, max(1.5, 0.35 * w_mask))
    else:
        rps = path.astype(np.float64)
    s_rps = _arclen(rps)
    path_len = float(s_rps[-1]) if len(rps) >= 2 else 0.0

    # --- Longueur par le profil d'intensité à mi-hauteur --------------------
    total_region = cv2.dilate(m.view(np.uint8), ellipse_kernel(5)).astype(bool)
    total = float(dpos[total_region].sum())
    if len(rps) >= 3:
        ext = float(np.clip(1.5 * w_mask + 4.0, 5.0, 25.0))
        full, n_pre = _extend(rps, ext, int(max(3, 1.5 * w_mask)))
        prof = _sample(dsm, full)
        prof_s = gaussian_filter1d(prof, 2.0, mode="nearest")
        core = prof_s[n_pre:n_pre + len(rps)]
        core_s = np.sort(core) if core.size else None       # même tri pour les deux centiles
        if core_s is not None and np.isnan(core_s[-1]):
            core_s = None
        # Niveau de référence = intensité du cœur de la trace (70e centile, robuste aux
        # portions pâles) ; les extrémités sont là où le profil tombe à la moitié.
        ref = float(_percentile(core, 70, core_s)) if core.size else 0.0
        s_full = _arclen(full)
        if ref > 0:
            above = np.nonzero(prof_s >= 0.5 * ref)[0]
            i0, i1 = int(above[0]), int(above[-1])
            # interpolation sous-pixel des extrémités
            s0 = s_full[i0]
            if i0 > 0:
                a, b = prof_s[i0 - 1], prof_s[i0]
                s0 -= (b - 0.5 * ref) / max(b - a, 1e-9)
            s1 = s_full[i1]
            if i1 < len(prof_s) - 1:
                a, b = prof_s[i1], prof_s[i1 + 1]
                s1 += (a - 0.5 * ref) / max(a - b, 1e-9)
            length_px = float(max(s1 - s0, path_len * 0.5, 1.0))
        else:
            length_px = path_len + w_mask
        peak = float(_percentile(core, 60, core_s)) if core.size else ref
        prof_along = core
    else:
        length_px = max(path_len + w_mask, 1.0)
        peak = float(dsm[m].max()) if area else 1.0
        prof_along = np.array([peak])
    peak = max(peak, 1e-3)

    # --- Rectitude (sur le chemin lissé) ------------------------------------
    if len(rps) >= 2 and path_len > 0:
        chord_raw = float(np.hypot(*(rps[-1] - rps[0])))
        straightness = min(1.0, chord_raw / path_len)
    else:
        chord_raw = 0.0
        straightness = 1.0
    chord_px = chord_raw + max(0.0, length_px - path_len)

    # --- Largeur à mi-hauteur du profil transversal moyen -------------------
    w_fwhm = 0.0
    if len(rps) >= 5:
        k = max(1, len(rps) // 24)
        idx = np.arange(2, len(rps) - 2, k)
        tang = rps[np.minimum(idx + 2, len(rps) - 1)] - rps[np.maximum(idx - 2, 0)]
        tn = np.hypot(tang[:, 0], tang[:, 1])
        ok = tn > 1e-6
        idx, tang, tn = idx[ok], tang[ok], tn[ok]
        if idx.size:
            nrm = np.column_stack([-tang[:, 1], tang[:, 0]]) / tn[:, None]
            r = float(np.clip(2.5 * w_mask + 4, 6, 40))
            offs = np.arange(-r, r + 1e-6, 0.5)
            pts = rps[idx][:, None, :] + offs[None, :, None] * nrm[:, None, :]
            vals = _sample(dpos, pts.reshape(-1, 2)).reshape(len(idx), len(offs))
            w_fwhm = _fwhm(vals.mean(0), 0.5)
    w_eff = float(np.clip(total / max(length_px * peak, 1e-3), 0.5, 4.0 * w_mask + 4.0))
    width_px = w_fwhm if w_fwhm > 0.5 else w_eff

    # --- Étalement latéral excédentaire (courbure), indépendant du squelette ---
    # Variance perpendiculaire (ACP pondérée par l'intensité) moins celle due à l'épaisseur.
    # ≈ 0 pour une trace droite, ≈ 0.3 × flèche pour un arc : robuste pour les traces courtes.
    lateral_px = 0.0
    ry, rx = np.nonzero(total_region)
    wts = dpos[ry, rx].astype(np.float64)
    if wts.sum() > 0 and rx.size >= 5:
        # np.average(a, weights=wts) = np.multiply(a, wts, dtype=f8).sum() / wts.sum(dtype=f8)
        # (mêmes opérations que NumPy, sans son surcoût de vérifications)
        wsum = wts.sum(dtype=np.float64)

        def _wavg(a):
            return np.multiply(a, wts, dtype=np.float64).sum() / wsum
        mx, my = _wavg(rx), _wavg(ry)
        dx, dy = rx - mx, ry - my
        cxx, cyy, cxy = _wavg(dx * dx), _wavg(dy * dy), _wavg(dx * dy)
        l2 = 0.5 * (cxx + cyy) - math.sqrt(0.25 * (cxx - cyy) ** 2 + cxy ** 2)
        sig_w = width_px / 2.355
        lateral_px = math.sqrt(max(0.0, l2 - sig_w ** 2 - 1.0 / 12.0))

    # Densité linéaire d'intensité (proxy de l'ionisation par unité de longueur)
    density = total / max(length_px, 1.0) / max(noise_sigma, 1e-3)
    peak_snr = peak / max(noise_sigma, 1e-3)

    # --- Profil le long de la trace : asymétrie (pic de Bragg / fin de parcours)
    asym = 1.0
    uniform = 0.0
    if prof_along.size >= 9:
        k = prof_along.size // 3
        a0, a2 = float(prof_along[:k].mean()), float(prof_along[-k:].mean())
        asym = max(a0, a2) / max(min(a0, a2), 1e-3)
        uniform = float(np.std(prof_along) / max(np.mean(prof_along), 1e-3))

    # --- Tortuosité, écart à la droite et courbure ----------------------------
    turn_total = 0.0
    turn_net = 0.0
    sagitta_ratio = 0.0
    line_rms = 0.0
    radius_px = float("inf")
    circle_rms = 0.0
    if len(rps) >= 6:
        step = int(max(4, round(2.0 * px_per_mm), round(1.5 * w_mask)))
        if len(rps) > 2 * step:
            sub = rps[::step]
            if len(sub) >= 3:
                ang = np.arctan2(np.diff(sub[:, 0]), np.diff(sub[:, 1]))
                dth = np.diff(np.unwrap(ang))
                turn_total = float(np.abs(dth).sum())
                turn_net = float(abs(dth.sum()))
        p0, p1 = rps[0], rps[-1]
        ch = p1 - p0
        chl = float(np.hypot(*ch))
        if chl > 1e-6:
            nrm = np.array([-ch[1], ch[0]]) / chl
            dev = (rps - p0) @ nrm
            sagitta_ratio = float(np.max(np.abs(dev)) / max(path_len, 1.0))
        cpts = rps - rps.mean(0)
        _u, sv, _vt = np.linalg.svd(cpts, full_matrices=False)
        line_rms = float(sv[-1] / math.sqrt(len(rps)))
        xc, yc, radius_px, circle_rms = fit_circle_taubin(rps[:, 1], rps[:, 0])

    # --- Embranchements (jonctions du squelette) -----------------------------
    branches = 0
    if skel.sum() > 3:
        nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32)) - 1
        junc = (nb >= 3) & skel
        if junc.any():
            branches = int(cv2.connectedComponents(junc.view(np.uint8), connectivity=8)[0] - 1)

    # Orientation de la corde (0–180°, 0 = horizontale)
    if len(rps) >= 2:
        vy, vx = (rps[-1] - rps[0])
        orient = (math.degrees(math.atan2(-vy, vx)) + 180.0) % 180.0
    else:
        orient = 0.0

    length_mm = length_px * mm
    return {
        "area_px": area,
        "length_px": length_px,
        "chord_px": chord_px,
        "length_mm": length_mm,
        "chord_mm": chord_px * mm,
        "width_mask_px": w_mask,
        "width_eff_px": w_eff,
        "width_fwhm_px": w_fwhm,
        "width_px": width_px,
        "width_mm": width_px * mm,
        "width_mask_mm": w_mask * mm,
        "straightness": straightness,
        "line_rms_mm": line_rms * mm,
        "lateral_mm": lateral_px * mm,
        "elongation": length_px / max(width_px, 1.0),
        "turn_total_rad": turn_total,
        "tortuosity": turn_total / max(length_mm, 1e-3),      # rad / mm
        "curl_consistency": turn_net / turn_total if turn_total > 1e-6 else 1.0,
        "sagitta_ratio": sagitta_ratio,
        "radius_mm": radius_px * mm if math.isfinite(radius_px) else float("inf"),
        "circle_rms_mm": circle_rms * mm,
        "density": density,
        "peak_snr": peak_snr,
        "total_signal": total / max(noise_sigma, 1e-3),
        "asymmetry": asym,
        "profile_cv": uniform,
        "branches": branches,
        "orientation_deg": orient,
        "_path": (rps - pad).astype(np.float32),     # (y, x) lissé, dans le crop d'origine
    }
