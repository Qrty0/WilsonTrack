"""Graphiques d'analyse des traces (matplotlib).

Toutes les figures sont construites avec ``matplotlib.figure.Figure`` (jamais
l'état global de ``pyplot``) : elles peuvent donc être créées dans un fil de
calcul secondaire et intégrées telles quelles dans l'interface Qt via
``FigureCanvasQTAgg(fig)``, ou converties en PNG avec ``fig_to_png_bytes``.

Entrées : ``rows`` = liste de dictionnaires comme renvoyés par
``RunResult.table()`` (colonnes de ``detection.events.EXPORT_COLUMNS``). Les
valeurs numériques manquantes peuvent valoir ``""`` ou ``None`` : elles sont
converties prudemment en NaN et ignorées.

Chaque fonction accepte une liste vide et affiche alors « Aucune donnée ».

Les figures sont des ``FrFigure`` : au moment du dessin, les graduations des axes sont
écrites avec une virgule décimale (« 0,5 ») ; les textes des légendes le sont aussi.

Figures de l'analyse approfondie (entrée : le dictionnaire ``stats.deep_analysis(run)``) :
``fig_time_rates``, ``fig_noise_timeline``, ``fig_cumulative``, ``fig_confidence``,
``fig_unfolded``.
"""
from __future__ import annotations

import io
import math
from typing import Iterable, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, Formatter, FuncFormatter, MaxNLocator, NullFormatter

from .. import physics

# --------------------------------------------------------------------------- #
# Style commun (encres neutres ; la couleur est réservée à l'identité des séries)
# --------------------------------------------------------------------------- #
INK = "#0b0b0b"          # texte principal
INK2 = "#52514e"         # texte secondaire (axes, légendes)
MUTED = "#898781"        # annotations discrètes
GRID = "#e1e0d9"         # grille (trait fin)
AXIS = "#c3c2b7"         # axes / ligne de base
SURFACE = "#ffffff"      # fond des figures (lisible à l'impression)

CLASS_ORDER = list(physics.COUNTED_CLASSES)

# Codage secondaire (forme / trait) : l'identité d'une classe ne repose jamais
# uniquement sur la couleur (daltonisme, impression en noir et blanc).
MARKERS = {"alpha": "o", "beta": "^", "muon": "s", "proton": "D", "fine": "v", "incertain": "X", "artefact": "x"}
LINESTYLES = {"alpha": "-", "beta": (0, (5, 2)), "muon": (0, (6, 2, 1.5, 2)),
              "proton": (0, (1.5, 1.5)), "incertain": (0, (3, 1.5, 1, 1.5, 1, 1.5))}
SHORT_NAMES = {"alpha": "Alpha (α)", "beta": "Bêta (β)", "muon": "Muon (μ)",
               "proton": "Proton (p)", "fine": "Trace fine (β/μ)", "incertain": "Incertaine"}

# Couleurs des enregistrements comparés (ordre fixe, jamais recyclé)
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RUN_OVERFLOW_COLOR = MUTED

EMPTY_TEXT = "Aucune donnée"
DEAD_COLOR = "#d9d8d2"   # périodes inactives (fond grisé)


# --------------------------------------------------------------------------- #
# Virgule décimale
# --------------------------------------------------------------------------- #
def _fr(x, d: int = 2, sign: bool = False) -> str:
    """Nombre avec virgule décimale (« 3,14 ») ; « — » si non fini."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if s.strip("+-0.") == "":
        s = s.replace("-", "+") if sign else s.lstrip("-")
    return s.replace(".", ",")


def _fr_p(p) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    if v < 1e-3:
        e = int(math.floor(math.log10(v))) if v > 0 else 0
        return f"{_fr(v / 10 ** e, 1)}e{e}" if v > 0 else "0"
    return _fr(v, 3)


def _comma(s: str) -> str:
    if not s or "." not in s:
        return s
    return s.replace(".", "{,}") if "$" in s else s.replace(".", ",")


class _DecimalComma(Formatter):
    """Enveloppe un formateur matplotlib : même texte, virgule décimale."""
    _wt_comma = True

    def __init__(self, base: Formatter):
        self.base = base

    def set_axis(self, axis):
        self.axis = axis
        self.base.set_axis(axis)

    def __call__(self, x, pos=None):
        return _comma(self.base(x, pos))

    def format_ticks(self, values):
        return [_comma(s) for s in self.base.format_ticks(values)]

    def set_locs(self, locs):
        self.base.set_locs(locs)

    def get_offset(self):
        return _comma(self.base.get_offset())

    def format_data(self, value):
        return _comma(self.base.format_data(value))

    def format_data_short(self, value):
        return _comma(self.base.format_data_short(value))

    def fix_minus(self, s):
        return self.base.fix_minus(s)


def _apply_decimal_comma(fig: Figure) -> None:
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for get, put in ((axis.get_major_formatter, axis.set_major_formatter),
                             (axis.get_minor_formatter, axis.set_minor_formatter)):
                try:
                    f = get()
                    if getattr(f, "_wt_comma", False) or isinstance(f, NullFormatter):
                        continue
                    put(_DecimalComma(f))
                except Exception:
                    pass


class FrFigure(Figure):
    """Figure matplotlib dont les graduations utilisent la virgule décimale (appliqué au dessin)."""

    def draw(self, renderer):
        _apply_decimal_comma(self)
        return super().draw(renderer)


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def class_color(cls: str) -> str:
    return physics.CLASS_INFO.get(cls, physics.CLASS_INFO["incertain"])["hex"]


def class_name(cls: str) -> str:
    info = physics.CLASS_INFO.get(cls)
    return info["nom"] if info else str(cls)


def short_name(cls: str) -> str:
    return SHORT_NAMES.get(cls, str(cls))


def to_float(v) -> float:
    """Conversion tolérante en float (``""``, ``None``, texte invalide → NaN)."""
    if v is None:
        return float("nan")
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", ".")
        if not s:
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return float("nan")


def _col(rows: Sequence[dict], key: str) -> np.ndarray:
    return np.array([to_float(r.get(key)) for r in rows], dtype=float)


def _cls_of(r: dict) -> str:
    c = r.get("classe")
    return str(c) if c not in (None, "") else "incertain"


def _classes(rows: Sequence[dict]) -> np.ndarray:
    return np.array([_cls_of(r) for r in rows], dtype=object)


def event_times(rows: Sequence[dict]) -> np.ndarray:
    """Instants d'apparition des traces (s).

    Horloge d'analyse ``t_analyse_s`` (0 = début de l'analyse effective, hors perturbations ;
    cohérente avec la durée analysée) si toutes les traces l'ont, sinon ``t_debut_s``
    (résultats enregistrés par une version antérieure).
    """
    ta = _col(rows, "t_analyse_s")
    if ta.size and np.isfinite(ta).all():
        return ta
    return _col(rows, "t_debut_s")


def _classes_in(rows: Sequence[dict]) -> list[str]:
    present = {_cls_of(r) for r in rows}
    out = [c for c in CLASS_ORDER if c in present]
    out += sorted(present - set(out))
    return out


def _rows(rows: Iterable[dict] | None) -> list[dict]:
    return [r for r in (rows or []) if isinstance(r, dict)]


def _new_fig(w: float = 7.2, h: float = 4.4, polar: bool = False):
    fig = FrFigure(figsize=(w, h), dpi=100, facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot(projection="polar") if polar else fig.add_subplot()
    ax.set_facecolor(SURFACE)
    return fig, ax


def _style(ax, grid_axis: str = "both") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=10)


def _legend(ax, **kw):
    kw.setdefault("fontsize", 8.5)
    kw.setdefault("frameon", False)
    kw.setdefault("labelcolor", INK2)
    return ax.legend(**kw)


def _fig_legend(fig, handles=None, ncols: int | None = None, max_cols: int = 3):
    """Légende sous la zone de tracé (ne masque jamais les données)."""
    if handles is None:
        handles = []
        for a in fig.axes:
            handles += [h for h, lab in zip(*a.get_legend_handles_labels()) if not lab.startswith("_")]
    if not handles:
        return None
    n = len(handles)
    ncols = ncols or min(max_cols, n)
    return fig.legend(handles=handles, loc="outside lower center", ncols=ncols, fontsize=8.5,
                      frameon=False, labelcolor=INK2)


def _empty(fig, ax, title: str = "", msg: str = EMPTY_TEXT) -> Figure:
    ax.set_axis_off()
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=14, color=MUTED)
    if title:
        _title(ax, title)
    return fig


def _minor_log_label(v: float, _pos=None) -> str:
    """Étiquettes intermédiaires 2 et 5 sur un axe logarithmique (…, 2, 5, 20, 50, …)."""
    if v <= 0:
        return ""
    m = v / 10 ** math.floor(math.log10(v) + 1e-9)
    return f"{v:g}" if (abs(m - 2) < 1e-6 or abs(m - 5) < 1e-6) else ""


def _plain_log_formatter(ax_axis, minor_labels: bool = True) -> None:
    ax_axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax_axis.set_minor_formatter(FuncFormatter(_minor_log_label) if minor_labels else NullFormatter())
    ax_axis.set_tick_params(which="minor", labelsize=8, labelcolor=MUTED, color=AXIS)


_LOG_LEVELS = [(1, 2, 5), (1, 1.5, 2, 3, 5, 7), (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8),
               (1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 2.2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9)]


def _nice_log_ticks(lo: float, hi: float) -> list[float]:
    """Valeurs « rondes » dans [lo, hi] : la grille la plus fine donnant au plus 7 graduations."""
    e0, e1 = math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1
    best: list[float] = []
    for subs in _LOG_LEVELS:
        ticks = sorted({round(m * 10.0 ** e, 9) for e in range(e0, e1 + 1) for m in subs})
        ticks = [v for v in ticks if lo <= v <= hi]
        if len(ticks) > 7:
            break
        best = ticks
        if len(ticks) >= 4:
            break
    return best


def _log_axis_ticks(ax, which: str = "x") -> None:
    """Graduations lisibles sur un axe logarithmique, y compris pour une plage étroite (< 1 décade)."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
    lo, hi = min(lo, hi), max(lo, hi)
    if not (lo > 0 and hi > lo) or hi / lo >= 8:
        _plain_log_formatter(axis)
        return
    ticks = _nice_log_ticks(lo, hi)
    if len(ticks) < 2:
        _plain_log_formatter(axis)
        return
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _fmt_duration(s: float) -> str:
    if s >= 3600 and abs(s / 3600 - round(s / 3600)) < 1e-9:
        return f"{s / 3600:g} h"
    if s >= 60 and abs(s / 60 - round(s / 60)) < 1e-9:
        return _comma(f"{s / 60:g} min")
    return _comma(f"{s:g} s")


# --------------------------------------------------------------------------- #
# 1. Longueur / épaisseur
# --------------------------------------------------------------------------- #
def fig_length_width(rows, title: str = "") -> Figure:
    """Nuage longueur (échelle log) / épaisseur, coloré par classe."""
    rows = _rows(rows)
    fig, ax = _new_fig()
    ttl = title or "Longueur et épaisseur des traces"
    L, W, C = _col(rows, "longueur_mm"), _col(rows, "epaisseur_mm"), _classes(rows)
    ok = np.isfinite(L) & np.isfinite(W) & (L > 0)
    if not rows or not ok.any():
        return _empty(fig, ax, ttl)
    _style(ax)
    for c in _classes_in(rows):
        m = ok & (C == c)
        n = int(m.sum())
        if not n:
            continue
        ax.scatter(L[m], W[m], s=28, c=class_color(c), marker=MARKERS.get(c, "o"), alpha=0.85,
                   edgecolors=SURFACE, linewidths=0.7, label=f"{class_name(c)} (n = {n})", zorder=3)
    ax.set_xscale("log")
    _log_axis_ticks(ax, "x")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Longueur de la trace (mm, échelle logarithmique)")
    ax.set_ylabel("Épaisseur (mm)")
    _fig_legend(fig)
    _title(ax, ttl)
    return fig


# --------------------------------------------------------------------------- #
# 2. Histogramme des longueurs
# --------------------------------------------------------------------------- #
def fig_length_hist(rows) -> Figure:
    """Histogrammes (en escalier) des longueurs par classe, intervalles logarithmiques."""
    rows = _rows(rows)
    fig, ax = _new_fig()
    ttl = "Distribution des longueurs par type de trace"
    L, C = _col(rows, "longueur_mm"), _classes(rows)
    ok = np.isfinite(L) & (L > 0)
    if not rows or not ok.any():
        return _empty(fig, ax, ttl)
    _style(ax)
    lo, hi = float(L[ok].min()), float(L[ok].max())
    if hi / lo < 1.5:
        lo, hi = lo / 1.3, hi * 1.3
    nb = int(np.clip(round(2 * math.sqrt(ok.sum())), 6, 30))
    bins = np.geomspace(lo * 0.95, hi * 1.05, nb + 1)
    for c in _classes_in(rows):
        m = ok & (C == c)
        n = int(m.sum())
        if not n:
            continue
        ax.hist(L[m], bins=bins, histtype="step", linewidth=2.0, color=class_color(c),
                linestyle=LINESTYLES.get(c, "-"), label=f"{class_name(c)} (n = {n})")
    ax.set_xscale("log")
    _log_axis_ticks(ax, "x")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Longueur (mm, échelle logarithmique)")
    ax.set_ylabel("Nombre de traces")
    _fig_legend(fig)
    _title(ax, ttl)
    return fig


# --------------------------------------------------------------------------- #
# 3. Taux en fonction du temps
# --------------------------------------------------------------------------- #
_NICE_BINS = [0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 7200]


def auto_bin_s(span_s: float) -> float:
    """Largeur d'intervalle « ronde » donnant environ 10 à 30 intervalles (~20 visés)."""
    if not (span_s > 0):
        return 10.0
    target = span_s / 20.0
    ranked = sorted(_NICE_BINS, key=lambda b: abs(math.log(b / target)))
    for b in ranked:
        if 10 <= math.ceil(span_s / b - 1e-9) <= 30:
            return float(b)
    return float(ranked[0])


def fig_rate_timeseries(rows, duration_s: float, bin_s: float | None = None) -> Figure:
    """Comptages empilés par intervalle de temps et par classe + taux total (/min).

    L'axe de droite (taux par minute) est aligné sur l'axe de gauche : c'est la même
    grandeur exprimée dans une autre unité (comptage × 60 / largeur de l'intervalle).
    """
    rows = _rows(rows)
    fig, ax = _new_fig(7.6, 4.6)
    ttl = "Nombre de traces en fonction du temps"
    t, C = event_times(rows), _classes(rows)
    ok = np.isfinite(t) & (t >= 0)
    dur = to_float(duration_s)
    if not rows or not ok.any():
        return _empty(fig, ax, ttl)
    tmax = float(t[ok].max())
    if not (dur > 0):
        dur = max(tmax, 1e-3)
    span = max(dur, tmax)
    b = to_float(bin_s) if bin_s is not None else float("nan")
    if not (b > 0):
        b = auto_bin_s(span)
    nb = max(1, int(math.ceil(span / b - 1e-9)))
    edges = np.arange(nb + 1, dtype=float) * b
    widths = np.minimum(edges[1:], span) - edges[:-1]
    widths = np.where(widths > 1e-9, widths, b)
    u = 60.0 if span >= 300 else 1.0
    _style(ax, grid_axis="y")

    bottom = np.zeros(nb)
    handles = []
    for c in _classes_in(rows):
        m = ok & (C == c)
        n = int(m.sum())
        if not n:
            continue
        counts = np.histogram(t[m], bins=edges)[0].astype(float)
        h = ax.bar(edges[:-1] / u, counts, width=widths / u * 0.94, align="edge", bottom=bottom,
                   color=class_color(c), edgecolor=SURFACE, linewidth=0.8, label=f"{short_name(c)} : {n}",
                   zorder=2)
        handles.append(h)
        bottom += counts
    total = bottom
    k = 60.0 / b                                   # comptage → taux (/min) pour un intervalle complet
    rate = total / widths * 60.0
    err = np.sqrt(total) / widths * 60.0
    n_tot = int(ok.sum())
    mean = n_tot / dur * 60.0
    mean_err = math.sqrt(n_tot) / dur * 60.0
    top = max(float(total.max()), float(np.max((rate + err) / k)), mean / k, 1.0) * 1.18

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    centers = (edges[:-1] + widths / 2) / u
    eb = ax2.errorbar(centers, rate, yerr=err, fmt="o-", color=INK, ms=3.5, lw=1.4, elinewidth=0.9,
                      capsize=2, label="Taux total (/min) ± √N", zorder=4)
    hl = ax2.axhline(mean, color=INK2, ls="--", lw=1.2,
                     label=f"Taux moyen : {_fr(mean, 1)} ± {_fr(mean_err, 1)} /min")
    ax.set_ylim(0, top)
    ax2.set_ylim(0, top * k)
    ax.set_xlim(0, span / u)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(AXIS)
    ax2.tick_params(colors=INK2, labelsize=9, length=3)
    ax2.set_ylabel("Taux total (traces / min)", color=INK2)
    ax.set_xlabel("Temps (min)" if u == 60 else "Temps (s)")
    ax.set_ylabel(f"Traces par intervalle de {_fmt_duration(b)}")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _fig_legend(fig, handles=handles + [eb, hl], max_cols=4 if len(handles) <= 2 else 3)
    _title(ax, ttl)
    return fig


# --------------------------------------------------------------------------- #
# 4. Énergie des alphas
# --------------------------------------------------------------------------- #
def _emitter_lines(ax, lo: float, hi: float, fontsize: float = 7.5) -> None:
    """Raies verticales des émetteurs alpha naturels, étiquettes verticales sans chevauchement.

    Les raies trop proches pour deux étiquettes (ex. Po-218 6,00 et Bi-212 6,05 MeV) partagent une
    étiquette sur plusieurs lignes ; chaque étiquette se place à gauche de sa raie si la place est
    libre, sinon à droite.
    """
    em = sorted((e, n) for n, e in physics.ALPHA_EMITTERS if lo <= e <= hi)
    if not em or hi <= lo:
        return
    fig = ax.figure
    px_per_unit = fig.get_figwidth() * fig.dpi * ax.get_position().width / (hi - lo)
    thick = fontsize * fig.dpi / 72.0 * 1.3 / px_per_unit          # épaisseur d'une ligne de texte (MeV)
    pad = 0.35 * thick
    groups: list[list[tuple[float, str]]] = []
    for e, n in em:
        ax.axvline(e, color=MUTED, ls=":", lw=1.1, zorder=1)
        if groups and e - groups[-1][-1][0] < 1.05 * thick:
            groups[-1].append((e, n))
        else:
            groups.append([(e, n)])
    trans = ax.get_xaxis_transform()
    occupied = -math.inf
    for g in groups:
        width = thick * len(g)
        left_start = g[0][0] - pad - width
        if left_start >= max(occupied + 0.1 * thick, lo):
            x, ha, end = g[0][0] - pad, "right", g[0][0] - pad
        else:
            x, ha, end = g[-1][0] + pad, "left", g[-1][0] + pad + width
        ax.text(x, 0.98, "\n".join(f"{n} · {_fr(e)} MeV" for e, n in g), transform=trans, rotation=90,
                ha=ha, va="top", fontsize=fontsize, color=INK2, zorder=5, linespacing=1.1,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8, "alpha": 0.85})
        occupied = end


def fig_alpha_energy(rows) -> Figure:
    """Histogramme de l'énergie estimée des alphas + raies des émetteurs naturels."""
    rows = _rows(rows)
    fig, ax = _new_fig(7.2, 4.6)
    ttl = "Énergie des particules alpha (relation portée-énergie)"
    al = [r for r in rows if _cls_of(r) == "alpha"]
    E = _col(al, "energie_MeV")
    E = E[np.isfinite(E) & (E > 0)]
    if E.size == 0:
        msg = EMPTY_TEXT if not al else f"{EMPTY_TEXT}\n(aucune alpha avec énergie estimée)"
        return _empty(fig, ax, ttl, msg)
    _style(ax, grid_axis="y")
    lo = min(4.0, math.floor(E.min() * 2) / 2 - 0.5)
    hi = max(9.5, math.ceil(E.max() * 2) / 2 + 0.5)
    bins = np.arange(lo, hi + 0.2501, 0.25)
    counts, _e, _p = ax.hist(E, bins=bins, color=class_color("alpha"), edgecolor=SURFACE, linewidth=0.8,
                             label=f"Alphas mesurées (n = {E.size})", zorder=2)
    ax.set_ylim(0, max(float(np.max(counts)), 1.0) * 1.75)
    ax.set_xlim(lo, hi)
    _emitter_lines(ax, lo, hi)
    mean = float(E.mean())
    sem = float(E.std(ddof=1) / math.sqrt(E.size)) if E.size > 1 else float("nan")
    lab = f"Moyenne : {_fr(mean)}" + (f" ± {_fr(sem)}" if math.isfinite(sem) else "") + " MeV"
    ax.axvline(mean, ymax=0.68, color=INK, ls="--", lw=1.4, label=lab, zorder=3)
    if E.size >= 10:
        # Vue de dessus, les traces sont projetées / tronquées : l'énergie déduite est un minimum.
        # Le bord supérieur (traces entièrement contenues) estime le mieux l'énergie réelle.
        edge = float(np.percentile(E, 90))
        ax.axvline(edge, ymax=0.68, color=INK2, ls="-.", lw=1.4, label=f"90e centile : {_fr(edge)} MeV", zorder=3)
    ax.set_xlim(lo, hi)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Énergie déduite de la longueur (MeV) — valeur minimale : traces projetées")
    ax.set_ylabel("Nombre d'alphas")
    _fig_legend(fig, max_cols=2)
    _title(ax, ttl)
    return fig


# --------------------------------------------------------------------------- #
# 5. Orientation (rose des directions)
# --------------------------------------------------------------------------- #
def fig_orientation(rows, cls: str | None = "muon") -> Figure:
    """Rose des orientations (données axiales 0–180° reproduites sur 180–360°)."""
    rows = _rows(rows)
    all_cls = cls in (None, "", "toutes", "tout", "all")
    sel = rows if all_cls else [r for r in rows if _cls_of(r) == cls]
    label = "toutes les traces" if all_cls else class_name(cls)
    ttl = f"Orientation des traces — {label}"
    ang = _col(sel, "orientation_deg")
    ang = ang[np.isfinite(ang)] % 180.0
    if ang.size == 0:
        fig, ax = _new_fig(5.8, 5.4)
        return _empty(fig, ax, ttl)
    fig, ax = _new_fig(5.8, 5.6, polar=True)
    step = 15.0
    edges = np.arange(0.0, 180.0 + step, step)
    counts = np.histogram(ang, bins=edges)[0]
    centers = edges[:-1] + step / 2
    theta = np.radians(np.concatenate([centers, centers + 180.0]))
    r = np.concatenate([counts, counts]).astype(float)
    color = physics.CLASS_INFO["muon"]["hex"] if all_cls else class_color(cls)
    ax.bar(theta, r, width=np.radians(step) * 0.92, color=color, edgecolor=SURFACE, linewidth=0.8,
           alpha=0.9, zorder=2)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(range(0, 360, 45), labels=[f"{d}°" for d in range(0, 360, 45)], color=INK2, fontsize=8.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(axis="y", colors=MUTED, labelsize=7.5)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(AXIS)
    ax.set_rlabel_position(100)
    ax.set_title(f"{ttl} (n = {ang.size})", loc="center", fontsize=11, color=INK, fontweight="bold", pad=14)
    from .stats import rayleigh_axial  # import local (scipy) — évite un coût à l'import du module
    note = "Données axiales : chaque orientation θ est aussi tracée à θ + 180°. 0° = horizontale."
    ray = rayleigh_axial(ang) if ang.size >= 8 else None
    if ray is not None:
        R, p = ray
        verdict = "compatible avec l'isotropie" if p > 0.05 else "orientations privilégiées (anisotrope)"
        note = f"Test de Rayleigh (axial) : R = {_fr(R)}, p = {_fr_p(p)} → {verdict}.\n" + note
    else:
        note = "Trop peu de traces (< 8) pour tester l'isotropie.\n" + note
    fig.supxlabel(note, fontsize=8, color=INK2)
    return fig


# --------------------------------------------------------------------------- #
# 6. Densité d'ionisation / épaisseur relatives
# --------------------------------------------------------------------------- #
def fig_density_width(rows) -> Figure:
    """Densité d'ionisation relative en fonction de l'épaisseur relative (log-log)."""
    rows = _rows(rows)
    fig, ax = _new_fig()
    ttl = "Densité d'ionisation et épaisseur (relatives aux traces fines)"
    X, Y, C = _col(rows, "epaisseur_rel"), _col(rows, "densite_rel"), _classes(rows)
    ok = np.isfinite(X) & np.isfinite(Y) & (X > 0) & (Y > 0)
    if not rows or not ok.any():
        return _empty(fig, ax, ttl)
    _style(ax)
    for c in _classes_in(rows):
        m = ok & (C == c)
        n = int(m.sum())
        if not n:
            continue
        ax.scatter(X[m], Y[m], s=28, c=class_color(c), marker=MARKERS.get(c, "o"), alpha=0.85,
                   edgecolors=SURFACE, linewidths=0.7, label=f"{class_name(c)} (n = {n})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Référence (= 1) toujours visible : les limites l'incluent avant le calcul des graduations
    ax.axvline(1.0, color=MUTED, ls=":", lw=1.0, zorder=1)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.0, zorder=1)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, 0.9), max(x1, 1.1))
    ax.set_ylim(min(y0, 0.9), max(y1, 1.1))
    _log_axis_ticks(ax, "x")
    _log_axis_ticks(ax, "y")
    ax.set_xlabel("Épaisseur relative (× traces fines de référence)")
    ax.set_ylabel("Densité d'ionisation relative (× référence)")
    _fig_legend(fig)
    _title(ax, ttl)
    return fig


# --------------------------------------------------------------------------- #
# 7. Comparaison de plusieurs enregistrements
# --------------------------------------------------------------------------- #
def run_label(run) -> str:
    cond = (getattr(run, "condition", "") or "Enregistrement").strip()
    return f"{cond} — essai {getattr(run, 'trial', 1)}"


def fig_comparison(runs) -> Figure:
    """Barres groupées : taux (/min) par classe pour chaque enregistrement (barres d'erreur √N / T)."""
    runs = [r for r in (runs or []) if r is not None]
    fig, ax = _new_fig(7.8, 4.6)
    ttl = "Comparaison des taux de traces par type"
    if not runs:
        return _empty(fig, ax, ttl)
    tables = [r.table() for r in runs]
    present = set()
    for rows in tables:
        present |= {_cls_of(x) for x in rows}
    groups = [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))
    if not groups:
        groups = CLASS_ORDER[:4]
    # Traces fines : famille « fine » (incertaines fines comprises) si TOUS les enregistrements
    # l'ont, sinon β + μ (même définition que conclusions.conclude_comparison).
    nonempty = [rows for rows in tables if rows]
    use_family = bool(nonempty) and all(
        all(str(r.get("famille") or "").strip().lower() in ("fine", "lourde") for r in rows) for rows in nonempty)

    def _is_fine(r) -> bool:                     # = stats.is_fine (β, μ, incertaines de famille fine)
        c = _cls_of(r)
        if c in ("beta", "muon"):
            return True
        if c in ("alpha", "proton"):
            return False
        return use_family and str(r.get("famille") or "").strip().lower() == "fine"

    has_fines = any(_is_fine(r) for rows in tables for r in rows)
    keys = groups + (["fines"] if has_fines else [])
    fl = "Traces fines\n(β, μ, incertaines)" if use_family else "Traces fines\n(β + μ)"
    labels = [short_name(c) for c in groups] + ([fl] if has_fines else [])
    _style(ax, grid_axis="y")
    x = np.arange(len(keys), dtype=float)
    nr = len(runs)
    w = 0.8 / nr
    for j, (run, rows) in enumerate(zip(runs, tables)):
        t_min = max(to_float(getattr(run, "duration_s", 0.0)), 0.0) / 60.0
        cl = [_cls_of(r) for r in rows]
        n_fines = sum(1 for r in rows if _is_fine(r))
        rates, errs = [], []
        for k in keys:
            n = n_fines if k == "fines" else sum(1 for c in cl if c == k)
            rates.append(n / t_min if t_min > 0 else 0.0)
            errs.append(math.sqrt(n) / t_min if t_min > 0 else 0.0)
        col = RUN_COLORS[j] if j < len(RUN_COLORS) else RUN_OVERFLOW_COLOR
        ax.bar(x + (j - (nr - 1) / 2) * w, rates, width=w * 0.92, yerr=errs, color=col, edgecolor=SURFACE,
               linewidth=0.8, capsize=3, error_kw={"ecolor": INK2, "elinewidth": 1.0},
               label=f"{run_label(run)} ({_fr(t_min, 1)} min)", zorder=2)
    if has_fines:
        ax.axvline(len(groups) - 0.5, color=AXIS, lw=1.0, ls="--", zorder=1)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Taux (traces / min)")
    _fig_legend(fig, max_cols=2)
    _title(ax, ttl)
    return fig


# --------------------------------------------------------------------------- #
# 8. Analyse approfondie (longs enregistrements)
# --------------------------------------------------------------------------- #
def _time_unit(span_s: float) -> tuple[float, str]:
    if span_s >= 2 * 3600:
        return 3600.0, "h"
    if span_s >= 300:
        return 60.0, "min"
    return 1.0, "s"


def _dead_periods(deep) -> list:
    return list(((deep or {}).get("dead") or {}).get("periods") or [])


def _shade_dead(ax, deep, u: float) -> list:
    """Grise les périodes inactives ; renvoie la poignée de légende (liste vide s'il n'y en a pas)."""
    per = _dead_periods(deep)
    for p in per:
        ax.axvspan(p.t0 / u, p.t1 / u, color=DEAD_COLOR, alpha=0.7, lw=0, zorder=0)
    if not per:
        return []
    return [Patch(facecolor=DEAD_COLOR, edgecolor="none", label="Période inactive (exclue des taux actifs)")]


def fig_time_rates(deep) -> Figure:
    """Taux (/min) par tranche de temps (1 h, 10 min…), empilés par type, avec l'incertitude √N du total.

    ``deep`` : résultat de ``stats.deep_analysis(run)``.
    """
    fig, ax = _new_fig(7.8, 4.6)
    bins = list((deep or {}).get("bins") or [])
    ttl = "Taux par tranche de temps et par type"
    if not bins or sum(b.total for b in bins) == 0:
        return _empty(fig, ax, ttl)
    span = bins[-1].t1
    u, unit = _time_unit(span)
    _style(ax, grid_axis="y")
    x0 = np.array([b.t0 for b in bins]) / u
    wid = np.array([b.t1 - b.t0 for b in bins]) / u
    expo_min = np.array([max(b.exposure_s, 1e-9) for b in bins]) / 60.0
    present = [c for c in CLASS_ORDER if any(b.counts.get(c, 0) for b in bins)]
    bottom = np.zeros(len(bins))
    handles = []
    for c in present:
        r = np.array([b.counts.get(c, 0) for b in bins], float) / expo_min
        h = ax.bar(x0, r, width=wid * 0.94, align="edge", bottom=bottom, color=class_color(c), edgecolor=SURFACE,
                   linewidth=0.8, label=short_name(c), zorder=2)
        handles.append(h)
        bottom = bottom + r
    tot = np.array([b.total for b in bins], float)
    err = np.sqrt(tot) / expo_min
    eb = ax.errorbar(x0 + wid / 2, bottom, yerr=err, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5,
                     label="Total ± √N", zorder=4)
    handles.append(eb)
    live = (((deep.get("live") or {}).get("rates") or {}).get("total"))
    if live is not None and math.isfinite(live.per_min):
        hl = ax.axhline(live.per_min, color=INK2, ls="--", lw=1.2, zorder=3,
                        label=f"Taux moyen (temps actif) : {_fr(live.per_min)} /min")
        handles.append(hl)
    handles += _shade_dead(ax, deep, u)
    ax.set_xlim(0, span / u)
    ax.set_ylim(0, max(float(np.max(bottom + err)), 1e-3) * 1.18)
    ax.set_xlabel(f"Temps d'analyse écoulé ({unit})")
    ax.set_ylabel("Taux (traces / min)")
    st = ((deep.get("stability") or {}).get("total"))
    if st is not None:
        ax.text(0.995, 0.985, f"Taux constant ? χ² = {_fr(st.chi2, 1)} pour {st.dof} ddl, p = {_fr_p(st.p_value)} → "
                + ("oui" if st.constant_ok else "non"), transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=INK2, zorder=6,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.9})
    _fig_legend(fig, handles=handles, max_cols=4)
    b_s = float(deep.get("bin_s") or (bins[0].t1 - bins[0].t0))
    _title(ax, f"{ttl} (tranches de {_fmt_duration(b_s)})")
    return fig


def fig_noise_timeline(deep) -> Figure:
    """Bruit de l'image (relatif à sa médiane) et fraction d'images perturbées au cours du temps."""
    fig, ax = _new_fig(7.8, 4.4)
    ttl = "Bruit de l'image et perturbations au cours du temps"
    iv = list((deep or {}).get("intervals") or [])
    if not iv:
        return _empty(fig, ax, ttl, f"{EMPTY_TEXT}\n(journal du détecteur absent)")
    span = max(i["t1"] for i in iv)
    u, unit = _time_unit(span)
    _style(ax, grid_axis="y")
    mids = np.array([(i["t0"] + i["t1"]) / 2 for i in iv]) / u
    noise = np.array([i["noise"] for i in iv], float)
    ok = np.isfinite(noise)
    med = float(np.median(noise[ok])) if ok.any() else float("nan")
    handles = []
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    dist = np.array([i["disturbed_frac"] for i in iv], float) * 100.0
    dist = np.where(np.isfinite(dist), dist, 0.0)
    if np.any(dist > 0):
        widths = np.array([i["t1"] - i["t0"] for i in iv]) / u
        hb = ax2.bar(np.array([i["t0"] for i in iv]) / u, dist, width=widths * 0.94, align="edge",
                     color=RUN_COLORS[3], alpha=0.55, edgecolor="none", label="Images perturbées (%)", zorder=1)
        handles.append(hb)
    ax2.set_ylim(0, max(float(dist.max()) * 1.25, 10.0))
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(AXIS)
    ax2.tick_params(colors=INK2, labelsize=9, length=3)
    ax2.set_ylabel("Images perturbées (%)", color=INK2)
    if ok.any() and med > 0:
        rel = noise / med
        fmt = "o-" if ok.sum() <= 60 else "-"
        (ln,) = ax.plot(mids[ok], rel[ok], fmt, color=INK, ms=3, lw=1.3, label="Bruit (× médiane)", zorder=4)
        handles.append(ln)
        ax.axhline(1.0, color=MUTED, ls=":", lw=1.0, zorder=2)
        hs = ax.axhline(2.0, color=class_color("alpha"), ls="--", lw=1.0, zorder=2,
                        label="Seuil des pointes de bruit (× 2)")
        handles.append(hs)
        spk = ok & (rel > 2.0)
        if spk.any():
            ax.plot(mids[spk], rel[spk], "o", color=class_color("alpha"), ms=3.5, zorder=5)
        ax.set_ylim(0, max(float(np.nanmax(rel[ok])) * 1.15, 2.4))
        ax.set_ylabel("Bruit de l'image (× médiane)")
    else:
        ax.set_yticks([])
        ax.set_ylabel("Bruit non mesuré")
    handles += _shade_dead(ax, deep, u)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlim(0, span / u)
    ax.set_xlabel(f"Temps d'analyse écoulé ({unit})")
    _fig_legend(fig, handles=handles, max_cols=3)
    _title(ax, ttl)
    return fig


def fig_cumulative(rows, deep=None, duration_s: float | None = None) -> Figure:
    """Nombre cumulé de traces par type en fonction du temps : une droite = taux constant."""
    rows = _rows(rows)
    fig, ax = _new_fig(7.6, 4.6)
    ttl = "Comptages cumulés au cours du temps"
    dur = to_float(duration_s if duration_s is not None else (deep or {}).get("duration_s"))
    if not rows:
        return _empty(fig, ax, ttl)
    from .stats import times_and_clock      # import local (scipy)
    t, _clock = times_and_clock(rows, dur)
    C = _classes(rows)
    ok = np.isfinite(t)
    if not ok.any():
        return _empty(fig, ax, ttl)
    if not (dur > 0):
        dur = float(t[ok].max())
    span = max(dur, float(t[ok].max()), 1e-6)
    u, unit = _time_unit(span)
    _style(ax)
    handles = []
    for c in _classes_in(rows):
        tc = np.sort(t[ok & (C == c)])
        if not tc.size:
            continue
        xs = np.concatenate([[0.0], tc, [span]]) / u
        ys = np.concatenate([[0], np.arange(1, tc.size + 1), [tc.size]])
        (h,) = ax.step(xs, ys, where="post", color=class_color(c), lw=1.6, ls=LINESTYLES.get(c, "-"),
                       label=f"{short_name(c)} : {tc.size}")
        handles.append(h)
    ta = np.sort(t[ok])
    xs = np.concatenate([[0.0], ta, [span]]) / u
    ys = np.concatenate([[0], np.arange(1, ta.size + 1), [ta.size]])
    (ht,) = ax.step(xs, ys, where="post", color=INK, lw=2.0, label=f"Total : {ta.size}")
    handles.append(ht)
    (hr,) = ax.plot([0, span / u], [0, ta.size], color=INK2, ls="--", lw=1.1, label="Taux constant (droite)")
    handles.append(hr)
    handles += _shade_dead(ax, deep, u)
    ax.set_xlim(0, span / u)
    ax.set_ylim(0, ta.size * 1.05 + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(f"Temps d'analyse écoulé ({unit})")
    ax.set_ylabel("Nombre cumulé de traces")
    _fig_legend(fig, handles=handles, max_cols=4)
    _title(ax, ttl)
    return fig


def fig_confidence(rows, threshold: float = 0.85) -> Figure:
    """Histogrammes de la confiance de classification, un panneau par type (seuil en tirets)."""
    rows = _rows(rows)
    ttl = "Confiance de la classification par type"
    conf, pia, C = _col(rows, "confiance"), _col(rows, "proba_ia"), _classes(rows)
    present = [c for c in _classes_in(rows) if np.isfinite(conf[C == c]).any()]
    if not present:
        fig, ax = _new_fig()
        return _empty(fig, ax, ttl)
    n = len(present)
    fig = FrFigure(figsize=(7.6, 1.45 * n + 1.4), dpi=100, facecolor=SURFACE, layout="constrained")
    axes = np.atleast_1d(fig.subplots(n, 1, sharex=True))
    bins = np.linspace(0.0, 1.0, 21)
    thr = to_float(threshold)
    any_ia = False
    for ax, c in zip(axes, present):
        ax.set_facecolor(SURFACE)
        _style(ax, grid_axis="y")
        v = conf[(C == c) & np.isfinite(conf)]
        ax.hist(np.clip(v, 0, 1), bins=bins, color=class_color(c), edgecolor=SURFACE, linewidth=0.6, zorder=2)
        p = pia[(C == c) & np.isfinite(pia)]
        if p.size:
            any_ia = True
            ax.hist(np.clip(p, 0, 1), bins=bins, histtype="step", color=INK, lw=1.3, zorder=3)
        if math.isfinite(thr):
            ax.axvline(thr, color=INK2, ls="--", lw=1.1, zorder=4)
        frac = float(np.mean(v >= thr)) if (v.size and math.isfinite(thr)) else float("nan")
        lab = f"{class_name(c)} — n = {v.size}"
        if math.isfinite(frac):
            lab += f", {_fr(100 * frac, 0)} % ≥ {_fr(thr)}"
        ax.text(0.01, 0.96, lab, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    axes[-1].set_xlim(0, 1)
    axes[-1].set_xlabel("Confiance (0 = aucune, 1 = certaine)")
    fig.supylabel("Nombre de traces", fontsize=9.5, color=INK2)
    handles = [Patch(facecolor=MUTED, edgecolor="none", label="Confiance finale (barres)")]
    if any_ia:
        handles.append(Patch(facecolor="none", edgecolor=INK, label="Probabilité de l'IA de référence (contour)"))
    if math.isfinite(thr):
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color=INK2, ls="--", label=f"Seuil {_fr(thr)}"))
    _fig_legend(fig, handles=handles, max_cols=3)
    _title(axes[0], ttl)
    return fig


def fig_unfolded(unfold) -> Figure:
    """Comptages bruts et estimation corrigée des erreurs de classification (validation sur simulation)."""
    fig, ax = _new_fig(7.6, 4.6)
    ttl = "Comptages bruts et estimation corrigée"
    if unfold is None or not getattr(unfold, "classes", None):
        return _empty(fig, ax, ttl, f"{EMPTY_TEXT}\n(matrice de confusion indisponible)")
    classes = list(unfold.classes)
    raw = {c: unfold.observed.get(c, 0) + unfold.manual.get(c, 0) for c in classes}
    keys = classes + (["incertain"] if unfold.observed.get("incertain") else [])
    _style(ax, grid_axis="y")
    x = np.arange(len(keys), dtype=float)
    w = 0.38
    top = 1.0
    for k, c in enumerate(keys):
        col = class_color(c)
        r = raw.get(c, unfold.observed.get(c, 0))
        ax.bar(x[k] - w / 2, r, width=w * 0.95, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.text(x[k] - w / 2, r, f"{int(r)}", ha="center", va="bottom", fontsize=8, color=INK2)
        top = max(top, r)
        if c in unfold.estimate:
            est = unfold.estimate[c]
            lo, hi = max(est - unfold.lo[c], 0.0), max(unfold.hi[c] - est, 0.0)
            ax.bar(x[k] + w / 2, est, width=w * 0.95, color=SURFACE, edgecolor=col, hatch="///", linewidth=1.2,
                   zorder=2)
            ax.errorbar(x[k] + w / 2, est, yerr=[[lo], [hi]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=3,
                        zorder=3)
            ax.text(x[k] + w / 2, est + hi, f"≈ {_fr(est, 0)}", ha="center", va="bottom", fontsize=8, color=INK2)
            top = max(top, est + hi)
    ax.set_xticks(x, [short_name(c) for c in keys])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, top * 1.18)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Nombre de traces")
    handles = [Patch(facecolor=MUTED, edgecolor="none", label="Comptage brut (classe attribuée)"),
               Patch(facecolor=SURFACE, edgecolor=INK2, hatch="///",
                     label="Estimation corrigée ± IC 68 % (dépend du modèle)")]
    _fig_legend(fig, handles=handles, max_cols=2)
    n_sim = f"{int(unfold.n_matrix):,}".replace(",", " ")
    ax.text(0.995, 0.985, f"Validation sur {n_sim} traces simulées :\nne remplace pas les comptages bruts",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2, linespacing=1.3,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.9})
    _title(ax, ttl)
    return fig


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def fig_to_png_bytes(fig: Figure, dpi: int = 110) -> bytes:
    """Rendu PNG en mémoire (sans pyplot ; fonctionne hors du fil principal)."""
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return buf.getvalue()


LONG_RUN_S = 1800.0      # au-delà : taux par tranche (10 min / 1 h) en plus du taux en fonction du temps


def deep_figures(run, deep=None) -> list[tuple[str, str, Figure]]:
    """Figures de l'analyse approfondie qui ont des données : [(clé, titre, figure)].

    ``deep`` : résultat de ``stats.deep_analysis(run)`` (calculé ici s'il est absent).
    Clés : taux_tranches (enregistrement ≥ 30 min), cumul, bruit_temps (journal du détecteur
    présent), confiance, comptages_corriges (IA de référence disponible, mode hybride).
    """
    rows = run.table()
    if deep is None:
        try:
            from .stats import deep_analysis
            deep = deep_analysis(run)
        except Exception:
            return []
    out: list[tuple[str, str, Figure]] = []
    bins = deep.get("bins") or []
    if bins and sum(b.total for b in bins) and to_float(deep.get("duration_s")) >= LONG_RUN_S:
        out.append(("taux_tranches", "Taux par tranche de temps", fig_time_rates(deep)))
    if len(rows) >= 2:
        out.append(("cumul", "Comptages cumulés", fig_cumulative(rows, deep)))
    if deep.get("has_timeline"):
        out.append(("bruit_temps", "Bruit et perturbations", fig_noise_timeline(deep)))
    if np.isfinite(_col(rows, "confiance")).any():
        thr = (deep.get("confidence") or {}).get("threshold", 0.85)
        out.append(("confiance", "Confiance de la classification", fig_confidence(rows, thr)))
    if deep.get("unfold") is not None:
        out.append(("comptages_corriges", "Comptages bruts et corrigés", fig_unfolded(deep["unfold"])))
    return out


def run_figures(run, deep=None) -> list[tuple[str, str, Figure]]:
    """Toutes les figures d'un enregistrement : [(clé, titre, figure)] (pour l'interface et les rapports).

    Les six figures de base, puis celles de l'analyse approfondie qui ont des données
    (voir ``deep_figures``). ``deep`` évite de refaire ``stats.deep_analysis(run)``.
    """
    rows = run.table()
    figs = [
        ("longueur_epaisseur", "Longueur et épaisseur", fig_length_width(rows, title="")),
        ("longueurs", "Distribution des longueurs", fig_length_hist(rows)),
        ("taux_temps", "Taux en fonction du temps", fig_rate_timeseries(rows, run.duration_s)),
        ("energie_alpha", "Énergie des alphas", fig_alpha_energy(rows)),
        ("orientation_muons", "Orientation des muons", fig_orientation(rows, "muon")),
        ("densite_epaisseur", "Densité et épaisseur relatives", fig_density_width(rows)),
    ]
    try:
        figs += deep_figures(run, deep)
    except Exception:
        pass
    return figs
