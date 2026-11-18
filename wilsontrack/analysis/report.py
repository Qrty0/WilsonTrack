"""Rapports HTML autonomes et résumés texte (Markdown) des analyses.

Les rapports HTML sont des fichiers uniques : feuille de style intégrée, graphiques
et vignettes intégrés en PNG (URI ``data:image/png;base64``). Ils s'ouvrent hors
ligne dans n'importe quel navigateur et s'impriment proprement en PDF
(règles ``@media print``).

Fonctions publiques :
    report_text(run)                          → résumé Markdown d'un enregistrement (pour l'assistant IA)
    comparison_text(sample, background)       → résumé Markdown d'une comparaison
    markdown_to_html(md)                      → petit convertisseur Markdown → HTML sûr
    build_run_report(run, out_html, ai_text)  → rapport HTML d'un enregistrement
    build_comparison_report(sample, background, out_html, ai_text, extra_runs) → rapport de comparaison
"""
from __future__ import annotations

import base64
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import __version__, physics
from ..config import AppConfig
from . import plots
from .conclusions import Conclusion, activity_text, conclude_comparison, conclude_run, expected_sample
from .stats import CLASSES, fr_clock, fr_duration, fr_g, fr_int, fr_num, fr_p, fr_p_eq

MAX_THUMBS = 120
MAX_TABLE_ROWS = 300
THUMB_SIDE = 160

GROUP_LABELS = {c: physics.CLASS_INFO[c]["nom"] for c in CLASSES}
GROUP_LABELS.update({"fines": "Traces fines (β + μ)", "total": "Total (toutes les traces)"})

LEVELS = {"succes": ("Résultat", "✓"), "attention": ("Attention", "⚠"), "info": ("Information", "ℹ")}

FIG_CAPTIONS = {
    "longueur_epaisseur": "Chaque point est une trace. Les alphas (épaisses et courtes) se séparent des traces "
                          "fines (β, μ) ; les muons sont en général les plus longs. Échelle horizontale logarithmique.",
    "longueurs": "Distribution des longueurs mesurées pour chaque type (intervalles logarithmiques).",
    "taux_temps": "Barres : nombre de traces par intervalle de temps, empilées par type (échelle de gauche). "
                  "Points : taux total par minute avec son incertitude √N (échelle de droite, même grandeur "
                  "dans une autre unité). Tirets : taux moyen. Un taux stable indique une chambre stable.",
    "energie_alpha": "Énergie déduite de la longueur (relation portée-énergie de Geiger corrigée de la densité de "
                     "l'air froid). Vue de dessus, une trace est projetée et peut sortir de la couche sensible : "
                     "c'est une énergie MINIMALE. Le bord droit de la distribution (90e centile, à partir de 10 "
                     "alphas) estime le mieux l'énergie réelle. Les lignes pointillées repèrent les émetteurs alpha "
                     "naturels (chaînes du radon et du thoron).",
    "orientation_muons": "Rose des directions des muons dans le plan de la chambre (0° = horizontale sur l'image). "
                         "Des barres de même longueur dans toutes les directions indiquent des arrivées isotropes.",
    "densite_epaisseur": "Densité d'ionisation et épaisseur relatives aux traces fines de référence (= 1). "
                         "Les particules très ionisantes (alphas, protons) se placent en haut à droite.",
    "taux_tranches": "Taux (traces par minute) de chaque type par tranche de temps (1 h pour une nuit, 10 min pour "
                     "un enregistrement plus court), empilés ; barre d'erreur : incertitude √N du total. Tirets : taux "
                     "moyen sur le temps actif. Zones grises : périodes où la chambre était inactive (exclues des "
                     "taux actifs, jamais des comptages bruts). En haut à droite : test du χ² d'un taux constant.",
    "cumul": "Nombre cumulé de traces de chaque type en fonction du temps d'analyse. Un taux constant donne une "
             "droite (tirets : droite attendue pour le total) ; un palier horizontal signale une période sans "
             "trace, une cassure un changement des conditions de la chambre.",
    "bruit_temps": "Bruit de l'image (écart-type du fond, relatif à sa valeur médiane) et pourcentage d'images "
                   "perturbées (lumière, mouvement) au cours du temps, d'après le journal du détecteur. Un bruit qui "
                   "augmente rend les traces fines plus difficiles à détecter.",
    "confiance": "Distribution de la confiance de classification pour chaque type (barres) et, si disponible, de la "
                 "probabilité donnée par l'IA de référence (contour). Tirets : seuil de confiance. Ces valeurs sont "
                 "calibrées sur des traces simulées : la fiabilité réelle se vérifie à la main sur un échantillon.",
    "comptages_corriges": "Barres pleines : comptages bruts (classe attribuée). Barres hachurées : estimation des "
                          "vrais nombres de traces de chaque type, corrigée des erreurs de classification mesurées "
                          "sur simulation (déconvolution bayésienne itérative, incertitudes par bootstrap). "
                          "Estimation dépendante du modèle : elle ne remplace pas les comptages bruts.",
}

TRACK_COLUMNS = [
    ("id", "N°", 0), ("classe", "Type", None), ("confiance", "Confiance", 2), ("t_debut_s", "Début (s)", 2),
    ("duree_visible_s", "Visible (s)", 2), ("longueur_mm", "Longueur (mm)", 1), ("epaisseur_mm", "Épaisseur (mm)", 2),
    ("rectitude", "Rectitude", 3), ("densite_rel", "Densité rel.", 2), ("epaisseur_rel", "Épaisseur rel.", 2),
    ("orientation_deg", "Orientation (°)", 0), ("energie_MeV", "Énergie (MeV)", 2), ("touche_bord", "Bord", "bool"),
    ("pourquoi", "Pourquoi ce type ?", "text"),
]


# =========================================================================== #
# Utilitaires de formatage
# =========================================================================== #
def _t(s) -> str:
    """Échappement pour du texte (contenu d'élément)."""
    return html.escape("" if s is None else str(s), quote=False)


def _a(s) -> str:
    """Échappement pour une valeur d'attribut."""
    return html.escape("" if s is None else str(s), quote=True)


def _num(v) -> float:
    return plots.to_float(v)


def _f(v, d: int = 2, dash: str = "—") -> str:
    return fr_num(v, d, dash)


def _pm(v, e, d: int = 2) -> str:
    x, s = _num(v), _num(e)
    if not math.isfinite(x):
        return "—"
    return f"{fr_num(x, d)} ± {fr_num(s, d)}" if math.isfinite(s) else fr_num(x, d)


def _g(v, digits: int = 3) -> str:
    return fr_g(v, digits)


def _gn(v) -> str:
    """Valeur de configuration au format ``:g`` avec virgule décimale (« 4,5 »)."""
    x = _num(v)
    return f"{x:g}".replace(".", ",") if math.isfinite(x) else "—"


def _fmt_p(p) -> str:
    return fr_p(p)


def _fmt_ratio(r) -> str:
    r = _num(r)
    if math.isinf(r):
        return "∞"
    return f"×{fr_num(r)}" if math.isfinite(r) else "—"


def _rate_txt(r, d: int = 2) -> str:
    """« 12,34 ± 1,23 » pour un objet Rate."""
    return f"{fr_num(r.per_min, d)} ± {fr_num(r.err, d)}"


def _cmp_ratio(cp) -> str:
    """Rapport des taux d'une comparaison (« — » si aucune trace des deux côtés)."""
    return "—" if (cp.a.n == 0 and cp.b.n == 0) else _fmt_ratio(cp.ratio)


def _fmt_date(s) -> str:
    try:
        return datetime.fromisoformat(str(s)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(s or "—")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _png_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _fig_uri(fig, dpi: int = 110) -> str:
    return _png_uri(plots.fig_to_png_bytes(fig, dpi=dpi))


def _bgr_uri(img: np.ndarray | None, max_side: int = THUMB_SIDE) -> str | None:
    if img is None or not hasattr(img, "shape") or img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    f = max_side / max(h, w)
    if f < 1.0:
        img = cv2.resize(img, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return _png_uri(buf.tobytes()) if ok else None


def _imread(path: Path) -> np.ndarray | None:
    """Lecture d'image compatible avec les chemins accentués (Windows)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _swatch(cls: str) -> str:
    return f'<span class="swatch" style="background:{_a(plots.class_color(cls))}"></span>'


def _run_title(run) -> str:
    return plots.run_label(run)


def _duration_txt(dur_s) -> str:
    d = _num(dur_s)
    if not math.isfinite(d):
        return "—"
    if d >= 3600:
        return f"{fr_duration(d)} ({fr_num(d / 60, 1)} min)"
    return f"{fr_num(d, 1)} s ({fr_num(d / 60)} min)"


def _calib_txt(run) -> str:
    ppm = _num(run.px_per_mm)
    if run.calibrated:
        return f"oui — {fr_num(ppm, 3)} px/mm"
    return f"non — échelle supposée ({fr_num(ppm, 3)} px/mm)"


def _sample_txt(run) -> str:
    s = run.sample_type or "Aucun"
    m = _num(run.sample_mass_g)
    return f"{s} — {_gn(m)} g" if math.isfinite(m) and m > 0 else s


def _run_meta(run) -> list[tuple[str, str]]:
    items = [
        ("Nom", run.name), ("Condition", run.condition or "—"), ("Échantillon", _sample_txt(run)),
        ("Essai", str(run.trial)), ("Date", _fmt_date(run.date)),
        ("Durée analysée", _duration_txt(run.duration_s)),
        ("Images analysées", f"{fr_int(run.n_frames)} à {_gn(run.fps)} images/s"),
        ("Calibration", _calib_txt(run)),
        ("Surface analysée", f"{fr_num(_num(run.roi_area_mm2) / 100, 1)} cm²"),
        ("Nombre de traces", str(len(run.table()))),
        ("Fichier source", Path(run.source).name if run.source else "—"),
    ]
    op = (run.config or {}).get("run", {}).get("operator", "") if isinstance(run.config, dict) else ""
    if op:
        items.append(("Opérateur", op))
    if run.notes:
        items.append(("Notes", run.notes))
    return items


def _stat_items(s: dict, calibrated: bool = True) -> list[tuple[str, str, str]]:
    """(test, résultat, interprétation) pour les tests statistiques d'un enregistrement."""
    out = []
    d = s.get("dispersion")
    if d is not None:
        out.append(("Indice de dispersion (loi de Poisson)",
                    f"variance/moyenne = {fr_num(d.index)} ({d.n_bins} tranches de {fr_num(d.bin_s, 0)} s), "
                    f"{fr_p_eq(d.p_value)}",
                    "compatible avec un processus de Poisson" if d.poisson_ok else "écart significatif à la loi de Poisson"))
    else:
        out.append(("Indice de dispersion (loi de Poisson)", "non calculable",
                    "trop peu de traces ou durée trop courte"))
    ia = s.get("interarrival")
    if ia is not None:
        mu, p = ia
        out.append(("Temps entre traces (Kolmogorov-Smirnov / Lilliefors)", f"moyenne {fr_num(mu, 1)} s, {fr_p_eq(p)}",
                    "compatible avec une loi exponentielle" if p > 0.01 else "non exponentiel"))
    else:
        out.append(("Temps entre traces (Kolmogorov-Smirnov / Lilliefors)", "non calculable", "moins de 10 traces"))
    iso = s.get("muon_isotropy")
    if iso is not None:
        R, p = iso
        out.append(("Isotropie des muons (Rayleigh, axial)", f"R = {fr_num(R)}, {fr_p_eq(p)}",
                    "directions isotropes" if p > 0.05 else "directions privilégiées"))
    else:
        out.append(("Isotropie des muons (Rayleigh, axial)", "non calculable", "moins de 8 muons"))
    em, ee, ne = s.get("alpha_energy", (float("nan"), float("nan"), 0))
    label = "Énergie des alphas (minimale, d'après la longueur projetée)"
    if not calibrated:
        out.append((label, "non estimée", "échelle non calibrée : calibrez pour estimer l'énergie"))
    elif ne and math.isfinite(em):
        edge, n_e = s.get("alpha_energy_edge", (float("nan"), ne))
        res = f"moyenne {_pm(em, ee, 2)} MeV (n = {ne})"
        if n_e >= 10 and math.isfinite(_num(edge)):
            name, e_ref = physics.closest_alpha_emitter(edge)
            res += f" ; 90e centile {fr_num(edge)} MeV"
            interp = (f"bord supérieur compatible avec {name} ({fr_num(e_ref)} MeV)" if abs(edge - e_ref) < 0.5
                      else "aucun émetteur identifiable avec certitude")
        else:
            interp = "énergie minimale (traces projetées) ; moins de 10 alphas : émetteur non identifiable"
        out.append((label, res, interp))
    else:
        out.append((label, "non disponible", "aucune alpha avec énergie estimée"))
    return out


# =========================================================================== #
# Résumés Markdown (entrée de l'assistant IA)
# =========================================================================== #
def _md_cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def report_text(run) -> str:
    """Résumé Markdown concis d'un enregistrement (métadonnées, taux, statistiques, analyse approfondie,
    conclusions)."""
    s, concl = conclude_run(run)
    lines = [f"# Enregistrement « {run.name} »", ""]
    for k, v in _run_meta(run):
        if k in ("Nom",):
            continue
        lines.append(f"- **{k}** : {v}")
    exp = expected_sample(run)
    if exp is not None:
        lines.append(f"- **Activité attendue de l'échantillon (K-40)** : {activity_text(exp)}")
    lines += ["", "## Comptages et taux (incertitude poissonnienne √N/T, IC 68 % de Garwood)", "",
              "| Type | N | Taux (/min) | IC 68 % (/min) | /cm²/min | Longueur moy. (mm) | Épaisseur moy. (mm) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in CLASSES:
        d = s["classes"][c]
        r = d["rate"]
        lines.append(f"| {_md_cell(physics.CLASS_INFO[c]['nom'])} | {r.n} | {_rate_txt(r)} | "
                     f"{fr_num(r.lo)} – {fr_num(r.hi)} | {_g(d['per_cm2_min'])} | "
                     f"{_pm(d['length'][0], d['length'][1], 1)} | {_pm(d['width'][0], d['width'][1], 2)} |")
    t = s["total"]
    area = max(_num(s["area_cm2"]), 1e-9)
    lines.append(f"| **Total** | {t.n} | {_rate_txt(t)} | {fr_num(t.lo)} – {fr_num(t.hi)} | "
                 f"{_g(t.per_min / area)} | | |")
    lines += ["", "## Tests statistiques", ""]
    for name, res, interp in _stat_items(s, bool(run.calibrated)):
        lines.append(f"- **{name}** : {res} → {interp}")
    lines += _deep_markdown(run, s)
    lines += ["", "## Conclusions automatiques", ""]
    for c in concl:
        lines.append(f"- **{c.title}** ({LEVELS.get(c.level, LEVELS['info'])[0].lower()}) — {c.text}")
    return "\n".join(lines) + "\n"


def comparison_text(sample, background) -> str:
    """Résumé Markdown d'une comparaison échantillon / bruit de fond."""
    comps, concl = conclude_comparison(sample, background)
    lines = [f"# Comparaison : « {sample.condition} » (échantillon) contre « {background.condition} » (fond)", "",
             "## Enregistrements", "", "| | Échantillon | Fond |", "|---|---|---|"]
    ms, mb = dict(_run_meta(sample)), dict(_run_meta(background))
    for k in ms:
        if k in mb:
            lines.append(f"| {k} | {_md_cell(ms[k])} | {_md_cell(mb[k])} |")
    exp = expected_sample(sample)
    if exp is not None:
        lines += ["", f"**Activité attendue de l'échantillon (K-40)** : {activity_text(exp)}"]
    lines += ["", "## Taux comparés (test binomial exact unilatéral : excès de l'échantillon)", "",
              "| Type | Échantillon (/min) | Fond (/min) | Différence nette (/min) | Rapport | p | Signification | Significatif (p < 0,01) |",
              "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for key in list(CLASSES) + ["fines", "total"]:
        cp = comps[key]
        lines.append(f"| {_md_cell(_group_label(key, cp))} | {_rate_txt(cp.a)} (N = {cp.a.n}) | "
                     f"{_rate_txt(cp.b)} (N = {cp.b.n}) | {fr_num(cp.diff, sign=True)} ± {_f(cp.diff_err)} | "
                     f"{_cmp_ratio(cp)} | {_fmt_p(cp.p_value)} | {fr_num(cp.sigma, 1)} σ | "
                     f"{'oui' if cp.significant else 'non'} |")
    note = area_note(sample, background)
    if note:
        lines += ["", note]
    al2 = comps.get("alpha_bilateral")
    if al2 is not None:
        lines += ["", f"Contrôle des alphas (test bilatéral) : {fr_p_eq(al2.p_value)}."]
    lines += ["", "## Conclusions automatiques", ""]
    for c in concl:
        lines.append(f"- **{c.title}** ({LEVELS.get(c.level, LEVELS['info'])[0].lower()}) — {c.text}")
    return "\n".join(lines) + "\n"


def area_ratio(sample, background) -> float:
    """Rapport des surfaces analysées A_échantillon / A_fond (1 si inconnu), comme dans conclude_comparison."""
    As, Ab = _num(getattr(sample, "roi_area_mm2", 0)), _num(getattr(background, "roi_area_mm2", 0))
    k = As / Ab if (As > 0 and Ab > 0) else 1.0
    return k if math.isfinite(k) else 1.0


def area_note(sample, background) -> str:
    """Phrase expliquant la normalisation du fond par la surface analysée ('' si les surfaces sont égales)."""
    k = area_ratio(sample, background)
    if abs(k - 1.0) < 1e-3:
        return ""
    return (f"Les taux du fond sont ramenés à la surface analysée de l'échantillon (× {fr_num(k, 3)} : "
            f"{fr_num(_num(sample.roi_area_mm2) / 100, 0)} cm² contre {fr_num(_num(background.roi_area_mm2) / 100, 0)} "
            "cm²) ; les nombres N indiqués sont les comptages bruts.")


def _group_label(key: str, cp=None) -> str:
    """Libellé d'un groupe de comparaison (définition des traces fines selon les données)."""
    lab = getattr(cp, "label", "") if cp is not None else ""
    return lab or GROUP_LABELS.get(key, key)


# =========================================================================== #
# Markdown → HTML (sûr : tout le texte est échappé avant la mise en forme)
# =========================================================================== #
_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_HR = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_MD_ITEM = re.compile(r"^(\s*)([-*+•]|\d{1,9}[.)])\s+(.*)$")
_MD_FENCE = re.compile(r"^\s*(```|~~~)")
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")

_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__")
_ITAL = re.compile(r"(?<![*\w])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![*\w])|(?<![_\w])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![_\w])")
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")


def _md_inline(text: str) -> str:
    t = html.escape(text, quote=True)
    codes: list[str] = []

    def keep(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    t = _CODE.sub(keep, t)
    t = _LINK.sub(lambda m: f'<a href="{m.group(2)}" rel="noopener noreferrer">{m.group(1)}</a>', t)
    t = _BOLD.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", t)
    t = _ITAL.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", t)
    t = _STRIKE.sub(lambda m: f"<del>{m.group(1)}</del>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", t)


def _md_split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


def _md_align(cell: str) -> str:
    c = cell.strip()
    left, right = c.startswith(":"), c.endswith(":")
    return "center" if left and right else "right" if right else "left" if left else ""


def _md_table(header: list[str], aligns: list[str], body: list[list[str]]) -> str:
    n = len(header)

    def cell(tag: str, txt: str, k: int) -> str:
        al = aligns[k] if k < len(aligns) else ""
        st = f' style="text-align:{al}"' if al else ""
        return f"<{tag}{st}>{_md_inline(txt)}</{tag}>"

    parts = ['<div class="table-wrap"><table class="md-table"><thead><tr>']
    parts += [cell("th", h, k) for k, h in enumerate(header)]
    parts.append("</tr></thead><tbody>")
    for row in body:
        row = (row + [""] * n)[:n]
        parts.append("<tr>" + "".join(cell("td", c, k) for k, c in enumerate(row)) + "</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def markdown_to_html(md: str, heading_offset: int = 0) -> str:
    """Convertit un petit sous-ensemble de Markdown en HTML sûr.

    Pris en charge : titres (# à ######), **gras**, *italique*, `code`, ~~barré~~, listes à puces
    et numérotées (imbriquées par indentation), paragraphes, citations (>), blocs de code (```),
    tableaux à barres verticales, liens http(s) et lignes horizontales. Tout le texte est échappé
    avant la mise en forme : aucune balise HTML du texte source n'est interprétée.
    ``heading_offset`` décale le niveau des titres (ex. 2 : « # » → <h3>).
    """
    if not md:
        return ""
    lines = str(md).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []
    stack: list[tuple[int, str]] = []          # listes ouvertes : (indentation, "ul" | "ol")
    prev_blank = False

    def flush_para():
        if para:
            out.append("<p>" + "\n".join(_md_inline(p.strip()) + ("<br>" if p.endswith("  ") else "")
                                         for p in para) + "</p>")
            para.clear()

    def close_top():
        out.append(f"</li></{stack.pop()[1]}>")

    def close_lists():
        while stack:
            close_top()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].expandtabs(4)
        stripped = line.strip()
        if not stripped:
            flush_para()
            prev_blank = True
            i += 1
            continue
        m = _MD_FENCE.match(line)
        if m:
            flush_para()
            close_lists()
            fence, buf = m.group(1), []
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf), quote=False) + "</code></pre>")
            prev_blank = False
            continue
        m = _MD_HEADING.match(line)
        if m:
            flush_para()
            close_lists()
            lvl = max(1, min(6, len(m.group(1)) + heading_offset))
            out.append(f"<h{lvl}>{_md_inline(m.group(2))}</h{lvl}>")
            prev_blank = False
            i += 1
            continue
        if _MD_HR.match(line):
            flush_para()
            close_lists()
            out.append("<hr>")
            prev_blank = False
            i += 1
            continue
        if "|" in line and i + 1 < n and "|" in lines[i + 1] and _MD_TABLE_SEP.match(lines[i + 1]):
            flush_para()
            close_lists()
            header = _md_split_row(line)
            aligns = [_md_align(c) for c in _md_split_row(lines[i + 1])]
            i += 2
            body = []
            while i < n and "|" in lines[i] and lines[i].strip():
                body.append(_md_split_row(lines[i]))
                i += 1
            out.append(_md_table(header, aligns, body))
            prev_blank = False
            continue
        m = _MD_QUOTE.match(line)
        if m:
            flush_para()
            close_lists()
            buf = []
            while i < n:
                mq = _MD_QUOTE.match(lines[i])
                if not mq:
                    break
                buf.append(mq.group(1))
                i += 1
            out.append("<blockquote>" + markdown_to_html("\n".join(buf), heading_offset) + "</blockquote>")
            prev_blank = False
            continue
        m = _MD_ITEM.match(line)
        if m:
            flush_para()
            indent, marker, content = len(m.group(1)), m.group(2), m.group(3)
            tag = "ol" if marker[0].isdigit() else "ul"
            while stack and indent < stack[-1][0] - 1:
                close_top()
            if stack and abs(indent - stack[-1][0]) <= 1:
                if stack[-1][1] == tag:
                    out.append("</li>")
                else:
                    close_top()
                    stack.append((indent, tag))
                    out.append(_md_list_open(tag, marker))
            else:                                   # nouvelle liste (ou sous-liste dans l'élément courant)
                stack.append((indent, tag))
                out.append(_md_list_open(tag, marker))
            out.append("<li>" + _md_inline(content))
            prev_blank = False
            i += 1
            continue
        if stack:
            if line[:1] == " " or not prev_blank:  # suite du dernier élément de liste
                out.append(" " + _md_inline(stripped))
                prev_blank = False
                i += 1
                continue
            close_lists()
        para.append(line)
        prev_blank = False
        i += 1
    flush_para()
    close_lists()
    return "\n".join(out)


def _md_list_open(tag: str, marker: str) -> str:
    if tag == "ol":
        try:
            start = int(marker[:-1])
        except ValueError:
            start = 1
        return f'<ol start="{start}">' if start != 1 else "<ol>"
    return "<ul>"


# =========================================================================== #
# Gabarit HTML
# =========================================================================== #
CSS = """
:root{--ink:#1c1c1c;--ink2:#4f5d55;--muted:#7d8a82;--line:#dbe5dd;--page:#f1f5f1;--card:#ffffff;
--forest:#154a34;--green:#247f59;--bright:#26b076;--mint:#98f29d;--mint-soft:#e3f8e4;
--accent:#247f59;--ok:#1a7a50;--ok-bg:#eaf6ee;--warn:#e07b12;--warn-bg:#fff4e6;--info:#2a78d6;--info-bg:#edf3fc;
color-scheme:light}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif}
.page{max-width:1120px;margin:0 auto;padding:28px 16px 48px}
header.hero{position:relative;overflow:hidden;background:var(--forest);
background-image:linear-gradient(180deg,#1a5a40 0%,var(--forest) 70%);color:#fff;border:0;border-radius:18px;
padding:24px 28px 26px;margin-bottom:18px;-webkit-print-color-adjust:exact;print-color-adjust:exact}
header.hero .cmv-logo{display:block;height:40px;margin:0 0 22px;line-height:0}
header.hero .cmv-logo svg{display:block;height:40px;width:auto;max-width:100%}
header.hero .cmv-logo .cmv-name{display:inline-block;line-height:40px;font-weight:700;font-size:1.05rem;color:#fff}
header.hero .brand{display:flex;align-items:center;gap:10px;font-size:.78rem;font-weight:700;letter-spacing:.1em;
text-transform:uppercase;color:var(--mint)}
header.hero .brand::before{content:"";display:inline-block;width:22px;height:3px;border-radius:2px;background:var(--mint)}
header.hero h1{color:#fff;font-size:1.85rem;line-height:1.2;margin:8px 0 6px;letter-spacing:-.01em}
header.hero .sub{margin:0;color:rgba(255,255,255,.82);font-size:1.02rem}
header.hero .tiles{margin-top:20px}
header.hero .tile{background:rgba(255,255,255,.07);border:1px solid rgba(152,242,157,.28);color:#fff}
header.hero .tile .v{color:#fff}
header.hero .tile .k{color:var(--mint)}
h1{font-size:1.75rem;line-height:1.2;margin:6px 0 4px}
.sub{margin:0;color:var(--ink2);font-size:1.02rem}
h2{font-size:1.28rem;margin:34px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--line);color:var(--forest)}
h2::before{content:"";display:inline-block;width:6px;height:.95em;border-radius:3px;background:var(--bright);
margin-right:10px;vertical-align:-2px}
h3{font-size:1.05rem;margin:20px 0 8px;color:var(--forest)}
h4,h5,h6{font-size:.98rem;margin:16px 0 6px}
p{margin:8px 0}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 8px;margin:0 0 8px}
nav.toc a{font-size:.84rem;color:var(--ink2);text-decoration:none;border:1px solid var(--line);background:var(--card);
border-radius:999px;padding:3px 10px}
nav.toc a:hover{border-color:var(--green);color:var(--forest);background:var(--mint-soft)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,190px),1fr));gap:10px;margin-top:16px}
.tile{border:1px solid var(--line);border-radius:10px;padding:10px 14px;background:#fbfdfb}
.tile .v{font-size:1.45rem;font-weight:700;line-height:1.2}
.tile .k{font-size:.8rem;color:var(--ink2)}
dl.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:8px;margin:0}
dl.meta div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 12px}
dl.meta dt{font-size:.76rem;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
dl.meta dd{margin:2px 0 0;font-weight:600;overflow-wrap:anywhere}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table.data,table.md-table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.9rem}
table.data th,table.data td,table.md-table th,table.md-table td{padding:6px 10px;border-bottom:1px solid var(--line);
text-align:right;white-space:nowrap}
table.data th,table.md-table th{background:#eef4ef;color:var(--ink2);font-weight:600;font-size:.8rem;position:sticky;top:0}
table.data td:first-child,table.data th:first-child,table.md-table td:first-child,table.md-table th:first-child{text-align:left}
table.data tr:last-child td{border-bottom:none}
table.data tr.total td{font-weight:700;background:#f5f9f5}
table.data td.wrap{white-space:normal;text-align:left;min-width:220px}
table.tracks{font-size:.8rem}
.why{display:inline-block;white-space:normal;min-width:260px;max-width:440px;text-align:left;color:var(--ink2)}
.why-block table.data td,.why-block table.data th{white-space:normal;text-align:left;vertical-align:top}
.why-block h3{margin:4px 0 6px}.why-block h4{margin:12px 0 4px;font-size:.95rem}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:0}
.yes{font-weight:700}
.callout{border:1px solid var(--line);border-left:5px solid var(--info);background:var(--info-bg);border-radius:10px;
padding:10px 14px;margin:10px 0}
.callout.succes{border-left-color:var(--ok);background:var(--ok-bg)}
.callout.attention{border-left-color:var(--warn);background:var(--warn-bg)}
.callout .head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px}
.badge{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--ink2);
border:1px solid currentColor;border-radius:999px;padding:0 8px;white-space:nowrap}
.callout p{margin:4px 0 0}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:14px}
figure.fig{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
figure.fig img{width:100%;height:auto;display:block}
figure.fig figcaption{font-size:.84rem;color:var(--ink2);margin-top:6px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(122px,1fr));gap:8px}
figure.thumb{margin:0;border:2px solid var(--muted);border-radius:8px;overflow:hidden;background:#000;
break-inside:avoid}
figure.thumb img{display:block;width:100%;height:112px;object-fit:contain;background:#000}
figure.thumb figcaption{background:var(--card);color:var(--ink);font-size:.74rem;padding:3px 6px;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.note{color:var(--ink2);font-size:.86rem}
.ai{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px 20px}
.ai blockquote,.md blockquote{border-left:3px solid var(--line);margin:8px 0;padding:2px 12px;color:var(--ink2)}
code{font-family:ui-monospace,Consolas,"Cascadia Mono",monospace;font-size:.88em;background:#f1f1ee;border-radius:4px;
padding:1px 4px}
pre{background:#f1f1ee;border-radius:8px;padding:12px;overflow:auto;font-size:.8rem;line-height:1.45}
pre code{background:none;padding:0}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin-top:12px}
details summary{cursor:pointer;font-weight:600;color:var(--forest)}
details pre{max-height:520px}
.method ul{margin:6px 0 6px 0;padding-left:22px}
.score-card{display:flex;flex-wrap:wrap;align-items:center;gap:8px 22px;background:var(--card);
border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:8px 0 12px}
.score-num{font-size:2.6rem;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.score-num span{font-size:1rem;font-weight:600;color:var(--ink2);margin-left:2px}
.score-grade{font-size:1.1rem;font-weight:700}
.meter{display:inline-block;width:64px;height:7px;border-radius:4px;background:var(--line);margin-right:8px;
vertical-align:1px;overflow:hidden}
.meter span{display:block;height:100%;background:var(--accent)}
tr.dead td{background:#f1f0ec;color:var(--ink2)}
.deep h3{margin-top:26px}
ul.advice li{margin:3px 0}
footer{margin-top:40px;color:var(--muted);font-size:.8rem;border-top:2px solid var(--forest);padding-top:10px}
@media (max-width:600px){h1,header.hero h1{font-size:1.4rem}header.hero{padding:18px 16px;border-radius:16px}header.hero .cmv-logo,header.hero .cmv-logo svg{height:32px}.figs{grid-template-columns:1fr}}
@page{size:A4;margin:14mm 12mm}
@media print{
body{background:#fff;font-size:10pt}
.page{max-width:none;padding:0}
nav.toc{display:none}
.table-wrap,figure.fig,details,.ai,dl.meta div,.tile{border-color:#ccc}
header.hero{break-inside:avoid;page-break-inside:avoid}
header.hero .tile{border-color:rgba(152,242,157,.35)}
h2{break-after:avoid;page-break-after:avoid}
h3{break-after:avoid}
figure,.callout,tr,dl.meta div,.tile{break-inside:avoid;page-break-inside:avoid}
.figs{grid-template-columns:1fr 1fr}
.gallery{grid-template-columns:repeat(6,1fr)}
figure.thumb img{height:88px}
.table-wrap{overflow:visible}
table.data th{position:static}
table.tracks{font-size:7pt}
table.tracks th,table.tracks td{padding:2px 4px}
details pre{max-height:none;white-space:pre-wrap}
a{color:inherit;text-decoration:none}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""


_CMV_LOGO = Path(__file__).resolve().parents[1] / "assets" / "cmv" / "logo_cmv_fond_fonce.svg"
_CMV_LABEL = "Cégep Marie-Victorin"


def _cmv_logo_html() -> str:
    """Logo du Cégep (version fond foncé) intégré en SVG en ligne ; texte de repli si le fichier manque."""
    svg = ""
    try:
        txt = _CMV_LOGO.read_text(encoding="utf-8")
        m = re.search(r"<svg\b.*?</svg>", txt, re.S | re.I)
        if m and not re.search(r"<script|<foreignObject|\son\w+\s*=", m.group(0), re.I):
            svg = m.group(0)
    except OSError:
        svg = ""
    if not svg:
        return f'<div class="cmv-logo"><span class="cmv-name">{_t(_CMV_LABEL)}</span></div>'
    open_tag = re.match(r"<svg\b[^>]*>", svg).group(0)
    tag = re.sub(r'\s(?:width|height|role|aria-label|aria-hidden|focusable|preserveAspectRatio)="[^"]*"', "",
                 open_tag)
    size = ""
    vb = re.search(r'viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"', tag)
    if vb and float(vb.group(2)) > 0:
        size = f' width="{round(40 * float(vb.group(1)) / float(vb.group(2)))}" height="40"'
    tag = (tag[:-1].rstrip() + f'{size} role="img" aria-label="{_a(_CMV_LABEL)}" focusable="false"'
           ' preserveAspectRatio="xMinYMid meet">')
    return f'<div class="cmv-logo">{tag}{svg[len(open_tag):]}</div>'


def _page(title: str, subtitle: str, body: str, toc: list[tuple[str, str]], tiles: str = "") -> str:
    nav = "".join(f'<a href="#{_a(k)}">{_t(v)}</a>' for k, v in toc)
    return (
        "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<meta name=\"generator\" content=\"WilsonTrack {_a(__version__)}\">\n"
        f"<title>{_t(title)} — {_t(subtitle)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<div class=\"page\">\n"
        f"<header class=\"hero\">{_cmv_logo_html()}<div class=\"brand\">WilsonTrack · Chambre de Wilson</div>"
        f"<h1>{_t(title)}</h1><p class=\"sub\">{_t(subtitle)}</p>{tiles}</header>\n"
        f"<nav class=\"toc\" aria-label=\"Sommaire\">{nav}</nav>\n{body}\n"
        f"<footer>Rapport généré le {_t(_now())} par WilsonTrack {_t(__version__)} — Cégep Marie-Victorin. "
        "Fichier autonome : graphiques et images sont intégrés (consultable hors ligne, imprimable en PDF).</footer>\n"
        "</div>\n</body>\n</html>\n"
    )


def _section(sid: str, title: str, content: str) -> str:
    return f'<section id="{_a(sid)}">\n<h2>{_t(title)}</h2>\n{content}\n</section>\n'


def _tiles(items: list[tuple[str, str]]) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{_t(v)}</div><div class="k">{_t(k)}</div></div>' for v, k in items) + "</div>"


def _meta_dl(items: list[tuple[str, str]]) -> str:
    return '<dl class="meta">' + "".join(f"<div><dt>{_t(k)}</dt><dd>{_t(v)}</dd></div>" for k, v in items) + "</dl>"


def _callouts(items: list[Conclusion]) -> str:
    if not items:
        return '<p class="note">Aucune conclusion.</p>'
    parts = []
    for c in items:
        lvl = c.level if c.level in LEVELS else "info"
        label, icon = LEVELS[lvl]
        parts.append(f'<div class="callout {lvl}"><div class="head"><span class="badge">{icon} {_t(label)}</span>'
                     f"<strong>{_t(c.title)}</strong></div><p>{_t(c.text)}</p></div>")
    return "\n".join(parts)


def _figure(uri: str, alt: str, caption: str = "") -> str:
    cap = f"<figcaption>{_t(caption)}</figcaption>" if caption else ""
    return f'<figure class="fig"><img src="{uri}" alt="{_a(alt)}">{cap}</figure>'


def _ai_section(ai_text: str | None) -> str:
    if not ai_text or not str(ai_text).strip():
        return ""
    body = ('<p class="note">Texte rédigé par un modèle d\'IA (Claude, Anthropic) à partir des mesures de ce '
            "rapport. Il peut contenir des erreurs : vérifiez-le avant de l'utiliser.</p>"
            f'<div class="ai md">{markdown_to_html(str(ai_text), heading_offset=2)}</div>')
    return _section("ia", "Interprétation de l'assistant IA (Claude)", body)


# =========================================================================== #
# Rapport d'un enregistrement
# =========================================================================== #
def _counts_table(s: dict) -> str:
    head = ("<tr><th>Type de trace</th><th>N</th><th>Taux (/min)</th><th>IC 68 % Garwood (/min)</th>"
            "<th>Par surface (/cm²/min)</th><th>Longueur moy. (mm)</th><th>Épaisseur moy. (mm)</th><th>Part</th></tr>")
    rows = []
    for c in CLASSES:
        d = s["classes"][c]
        r = d["rate"]
        rows.append(
            f"<tr><td>{_swatch(c)}{_t(physics.CLASS_INFO[c]['nom'])}</td><td>{r.n}</td>"
            f"<td>{_rate_txt(r)}</td><td>{fr_num(r.lo)} – {fr_num(r.hi)}</td>"
            f"<td>{_g(d['per_cm2_min'])}</td><td>{_pm(d['length'][0], d['length'][1], 1)}</td>"
            f"<td>{_pm(d['width'][0], d['width'][1], 2)}</td><td>{fr_num(100 * d['fraction'], 0)} %</td></tr>")
    t = s["total"]
    area = max(_num(s["area_cm2"]), 1e-9)
    rows.append(f'<tr class="total"><td>Total</td><td>{t.n}</td><td>{_rate_txt(t)}</td>'
                f"<td>{fr_num(t.lo)} – {fr_num(t.hi)}</td><td>{_g(t.per_min / area)}</td><td></td><td></td>"
                f"<td>{'100 %' if t.n else '—'}</td></tr>")
    return (f'<div class="table-wrap"><table class="data"><thead>{head}</thead><tbody>{"".join(rows)}'
            "</tbody></table></div>"
            '<p class="note">Taux = N / T ; incertitude poissonnienne √N / T ; IC 68 % : intervalle exact de Garwood '
            "(adapté aux petits comptages). Moyennes ± erreur-type.</p>")


def _stats_table(s: dict, calibrated: bool = True) -> str:
    rows = "".join(f"<tr><td>{_t(a)}</td><td class=\"wrap\">{_t(b)}</td><td class=\"wrap\">{_t(c)}</td></tr>"
                   for a, b, c in _stat_items(s, calibrated))
    return ('<div class="table-wrap"><table class="data"><thead><tr><th>Test</th><th>Résultat</th>'
            f"<th>Interprétation</th></tr></thead><tbody>{rows}</tbody></table></div>")


def _gallery_items(run) -> list[dict]:
    items: list[dict] = []
    if run.events:
        for e in run.events:
            if getattr(e, "valid", True) and getattr(e, "thumb", None) is not None:
                items.append({"id": e.id, "cls": e.final_cls, "L": e.features.get("length_mm"), "img": e.thumb})
        return items
    folder = Path(run.folder) / "vignettes" if run.folder else None
    if folder is None or not folder.is_dir():
        return items
    by_id = {}
    for r in run.table():
        rid = _num(r.get("id"))
        if math.isfinite(rid):
            by_id[int(rid)] = r
    for p in sorted(folder.glob("*.png")):
        m = re.match(r"trace_(\d+)_(.+)\.png$", p.name)
        tid = int(m.group(1)) if m else None
        row = by_id.get(tid) if tid is not None else None
        cls = (row.get("classe") if row else None) or (m.group(2) if m else "incertain")
        items.append({"id": tid, "cls": cls, "L": row.get("longueur_mm") if row else None, "path": p})
    return items


def _select_balanced(items: list[dict], max_n: int) -> list[dict]:
    """Sélection équilibrée entre les types (les types rares restent représentés)."""
    if len(items) <= max_n:
        return list(items)
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["cls"], []).append(it)
    order = [c for c in plots.CLASS_ORDER if c in by] + sorted(k for k in by if k not in plots.CLASS_ORDER)
    sel: list[dict] = []
    while len(sel) < max_n and any(by[c] for c in order):
        for c in order:
            if by[c] and len(sel) < max_n:
                sel.append(by[c].pop(0))
    sel.sort(key=lambda it: (it["id"] is None, it["id"] or 0))
    return sel


def _gallery(run) -> str:
    items = _gallery_items(run)
    if not items:
        return '<p class="note">Aucune vignette disponible pour cet enregistrement.</p>'
    sel = _select_balanced(items, MAX_THUMBS)
    tiles = []
    for it in sel:
        img = it.get("img")
        if img is None and it.get("path") is not None:
            img = _imread(it["path"])
        uri = _bgr_uri(img)
        if uri is None:
            continue
        cls = it["cls"]
        col = plots.class_color(cls)
        L = _num(it.get("L"))
        num = f"#{it['id']}" if it["id"] is not None else "#?"
        cap = f"{num} · {plots.short_name(cls)}" + (f" · {L:.0f} mm" if math.isfinite(L) else "")
        tiles.append(f'<figure class="thumb" style="border-color:{_a(col)}"><img src="{uri}" '
                     f'alt="{_a(f"Trace {num}, {plots.class_name(cls)}")}" loading="lazy">'
                     f"<figcaption>{_swatch(cls)}{_t(cap)}</figcaption></figure>")
    note = (f'<p class="note">{len(tiles)} vignettes affichées sur {len(items)}'
            + (" (sélection équilibrée entre les types de traces)" if len(items) > len(sel) else "")
            + ". Chaque vignette montre la trace à l'instant où elle était la plus nette ; "
              "la couleur du cadre indique le type attribué.</p>")
    return note + '<div class="gallery">' + "".join(tiles) + "</div>"


def _cell(row: dict, key: str, fmt) -> str:
    v = row.get(key)
    if fmt is None:
        cls = str(v or "incertain")
        manual = row.get("classe_manuelle")
        return f"{_swatch(cls)}{_t(plots.short_name(cls))}" + (" <em>(manuelle)</em>" if manual else "")
    if fmt == "bool":
        x = _num(v)
        return "oui" if math.isfinite(x) and x > 0 else "non"
    if fmt == "text":
        return f'<span class="why">{_t(v or row.get("raison") or "")}</span>'
    return _f(v, fmt)


_VERDICT_ICONS = {"pour": ("✔", "#30a46c"), "contre": ("✘", "#e5484d"), "neutre": ("•", "#8b8d98")}


def _explanation_block(ex, thumb=None) -> str:
    """Rendu « rapport » (HTML autonome) d'une explication « pourquoi cette classe ? »."""
    col = physics.CLASS_INFO.get(ex.cls, physics.CLASS_INFO["incertain"])["hex"]
    rows = []
    for c in ex.criteria:
        icon, icol = _VERDICT_ICONS.get(c.verdict, ("•", "#8b8d98"))
        extra = "" if c.used_for_decision else ' <em>(indice complémentaire, non utilisé pour décider)</em>'
        rows.append(f'<tr><td style="color:{icol};font-weight:700">{icon}</td><td>{_t(c.label)}{extra}</td>'
                    f"<td>{_t(c.measured)}</td><td>{_t(c.expected)}</td><td>{_t(c.explanation)}</td></tr>")
    crit = ('<div class="table-wrap"><table class="data"><thead><tr><th></th><th>Critère</th><th>Mesure</th>'
            "<th>Attendu</th><th>Signification physique</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    why_not = "".join(f"<li>{_t(s)}</li>" for _k, s in ex.why_not.items())
    caveats = "".join(f"<li>{_t(s)}</li>" for s in ex.caveats)
    img = ""
    uri = _bgr_uri(thumb) if thumb is not None else None
    if uri:
        img = (f'<figure class="thumb" style="float:right;width:170px;margin:0 0 8px 12px;border-color:{_a(col)}">'
               f'<img src="{uri}" alt="vignette"></figure>')
    return (f'<div class="why-block" style="border-left:5px solid {_a(col)};padding:6px 12px;margin:14px 0">{img}'
            f'<h3 style="color:{_a(col)}">{_t(ex.title)}</h3><p><strong>{_t(ex.summary)}</strong></p>'
            f"<p class=\"note\">Méthode : {_t(ex.method)} — confiance {100 * float(ex.confidence):.0f} %</p>"
            f"{crit}<h4>Pourquoi pas les autres ?</h4><ul>{why_not}</ul>"
            f"<h4>Physique</h4><p>{_t(ex.physics)}</p>"
            + (f"<h4>Limites</h4><ul>{caveats}</ul>" if caveats else "")
            + '<div style="clear:both"></div></div>')


def _why_section(run) -> str:
    """Un exemple expliqué par type de particule (la trace la plus sûre de chaque type)."""
    events = [e for e in (getattr(run, "events", None) or []) if getattr(e, "valid", True)]
    if not events:
        return ""
    try:
        from ..detection.explain import explain_event
        cfg = AppConfig.from_dict(run.config or {})
    except Exception:
        return ""
    blocks = []
    for c in ("alpha", "beta", "muon", "proton", "incertain"):
        cand = [e for e in events if e.final_cls == c]
        if not cand:
            continue
        e = max(cand, key=lambda x: float(x.confidence))
        try:
            ex = explain_event(e, cfg.classification, cfg.physics)
        except Exception:
            continue
        blocks.append(_explanation_block(ex, getattr(e, "thumb", None)))
    if not blocks:
        return ""
    intro = ('<p class="note">Pour chaque type de particule détecté, voici la trace la plus sûre et le raisonnement '
             "complet qui a conduit à sa classification : critères mesurés, comparaison avec les autres particules, "
             "physique en jeu et limites. Chaque trace du tableau ci-dessous a une explication courte "
             "(colonne « Pourquoi ce type ? ») ; l'explication complète est visible dans l'application.</p>")
    return intro + "".join(blocks)


def _tracks_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="note">Aucune trace détectée.</p>'
    shown = rows[:MAX_TABLE_ROWS]
    head = "".join(f"<th>{_t(h)}</th>" for _k, h, _f_ in TRACK_COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{_cell(r, k, f)}</td>" for k, _h, f in TRACK_COLUMNS) + "</tr>"
                   for r in shown)
    note = (f'<p class="note">{len(shown)} premières traces sur {len(rows)}. Le tableau complet (toutes les '
            "colonnes) est dans les fichiers traces.csv et traces.xlsx du dossier de l'enregistrement.</p>")
    return (note + f'<div class="table-wrap"><table class="data tracks"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def _method_html(cfg_dict: dict | None) -> str:
    try:
        cfg = AppConfig.from_dict(cfg_dict or {})
    except Exception:
        cfg = AppConfig()
    d, c, ph = cfg.detection, cfg.classification, cfg.physics
    ridge = ", renforcé par un filtre de crêtes de Hessian (traces fines ou en pointillés)" if d.use_ridge else ""
    if c.mode == "ia":
        mode = "modèle d'apprentissage automatique entraîné sur des traces étiquetées par l'équipe"
    elif c.mode == "hybride":
        mode = "hybride : règles physiques + IA de référence"
    else:
        mode = "règles physiques"
    hybrid_txt = ""
    if c.mode == "hybride":
        n_ref = ""
        try:
            from ..detection.reference_model import ReferenceModel
            m = ReferenceModel.load_default()
            if m is not None:
                n = int((m.info or {}).get("n_tracks_total") or 0)
                n_ref = f" sur {n:,} traces simulées".replace(",", " ") if n else ""
        except Exception:
            pass
        hybrid_txt = ("<p>Mode <strong>hybride</strong> : les scores des règles physiques sont combinés aux probabilités "
                      f"d'une <strong>IA de référence</strong> (gradient boosting) entraînée{_t(n_ref)} dont la vraie "
                      "nature est connue (simulation physique, conditions d'éclairage, de bruit, d'échelle et de "
                      "température très variées). Quand les deux avis divergent, la confiance baisse et l'explication "
                      "de la trace le signale. L'IA de référence n'a jamais vu votre chambre : ses avis sont un "
                      "complément aux critères physiques, pas une vérité absolue.</p>")
    exp = cfg.expansion
    exp_txt = (f"<p>Mode chambre à expansion : seules les traces apparues entre {exp.delay_ms:.0f} ms et "
               f"{exp.delay_ms + exp.window_ms:.0f} ms après chaque expansion (déclenchement « {_t(exp.trigger)} ») "
               "sont comptées.</p>") if exp.enabled else ""
    cfg_json = json.dumps(cfg_dict or {}, indent=2, ensure_ascii=False, default=str)
    return f"""<div class="method">
<h3>Acquisition et prétraitement</h3>
<p>La chambre est filmée par le dessus (fond noir, éclairage latéral). Chaque image est réduite à
{d.process_width} px de large, lissée (σ = {_gn(d.blur_sigma)} px) et moyennée sur {d.temporal_avg} images pour réduire
le bruit du capteur. Un fond adaptatif (constante de temps {_gn(d.bg_time_s)} s) est soustrait : il ne reste que les
gouttelettes nouvellement condensées. Les images perturbées (lumière allumée, caméra bougée) sont exclues de la
durée analysée.</p>
<h3>Détection et suivi</h3>
<p>Les pixels dépassant le fond de {_gn(d.k_high)} écarts-types du bruit servent de germes, étendus jusqu'à
{_gn(d.k_low)} σ (seuillage par hystérésis){ridge}. Les fragments alignés (écart ≤ {_gn(d.merge_gap_mm)} mm, angle ≤
{_gn(d.merge_angle_deg)}°) sont fusionnés. Chaque trace est suivie d'image en image et <strong>comptée une seule
fois</strong> ; ses mesures sont prises à l'instant où elle est la plus nette. Sont rejetées les détections visibles
moins de {_gn(d.min_visible_s)} s, plus courtes que {_gn(d.min_length_mm)} mm, peu allongées (longueur/largeur &lt;
{_gn(d.min_elongation)}) ou peu contrastées (&lt; {_gn(d.min_peak_snr)} σ).</p>
{exp_txt}
<h3>Classification ({_t(mode)})</h3>
<p>L'épaisseur et la densité d'ionisation de chaque trace sont mesurées <em>relativement</em> aux traces fines du même
enregistrement (percentile {_gn(c.ref_percentile)}), ce qui rend la classification indépendante de l'éclairage :</p>
<ul>
<li><strong>Alpha</strong> : trace lourde (densité ≳ {_gn(c.heavy_density_rel)}× la référence, épaisseur ≳
{_gn(c.heavy_width_rel)}× ou ≥ {_gn(c.alpha_width_mm)} mm), droite et courte (≤ {_gn(c.alpha_max_length_mm)} mm).</li>
<li><strong>Proton / particule lourde</strong> : lourde et droite, mais plus longue qu'une alpha naturelle.</li>
<li><strong>Muon</strong> : fine, rectiligne (rectitude ≳ {_gn(c.straight_min)}) et longue (≳ {_gn(c.muon_min_length_mm)} mm).</li>
<li><strong>Bêta (électron)</strong> : fine et sinueuse (rectitude ≲ {_gn(c.curly_max)}) ou fine et courte.</li>
<li><strong>Incertaine</strong> : score de confiance inférieur à {_gn(c.min_confidence)}.</li>
</ul>
{hybrid_txt}
<h3>Physique et statistiques</h3>
<p>Le nombre de traces suit une loi de Poisson : taux r = N/T, incertitude √N/T, intervalle exact de Garwood
pour les petits comptages. Le caractère aléatoire est vérifié par l'indice de dispersion (variance/moyenne ≈ 1) et
par la distribution exponentielle des temps entre traces (Kolmogorov-Smirnov / Lilliefors, p-valeur par
Monte-Carlo) ; l'isotropie des muons par le test de Rayleigh (données axiales). L'énergie des alphas est estimée
par la relation portée-énergie de Geiger (R ≈ 0,318·E<sup>1,5</sup> cm dans l'air à 15 °C), corrigée de la
densité de l'air à {_gn(ph.chamber_temp_c)} °C et {_gn(ph.pressure_kpa)} kPa. Vue de dessus, la trace est projetée et
peut sortir de la couche sensible (≈ {_gn(ph.sensitive_depth_mm)} mm) : c'est une énergie <em>minimale</em>, et
seul le bord supérieur de la distribution (90<sup>e</sup> centile) sert à identifier l'émetteur. Les comparaisons
échantillon/fond utilisent un test binomial exact conditionnel (unilatéral), le fond étant ramené à la surface
analysée de l'échantillon ; un excès est jugé significatif si p &lt; 0,01.</p>
<h3>Analyse approfondie (longs enregistrements)</h3>
<p>Le temps analysé est découpé en tranches (1 h au-delà de 3 h, 10 min au-delà de 30 min, 1 min au-delà de
5 min). La stabilité du taux est testée par un χ² de Pearson contre un taux constant (valeurs attendues
proportionnelles à la durée de chaque tranche ; p-valeur exacte par Monte-Carlo si moins de 5 traces sont attendues
par tranche) ; la dérive est estimée par moindres carrés pondérés. Une période est dite <em>inactive</em> si son taux
tombe sous 25 % du taux médian pendant au moins deux tranches consécutives d'environ 30 traces attendues (ou sous
60 % avec un bruit d'image au moins doublé) et si un si petit nombre de traces a une probabilité de Poisson
inférieure à 0,001 divisé par le nombre de tranches (correction de Bonferroni pour les tests multiples) : elle est
retirée des taux « sur le temps actif », jamais des comptages bruts. Les <em>traces fines</em> regroupent toutes les traces au minimum d'ionisation (électrons et
muons), y compris les fines classées incertaines. Les comptages corrigés des erreurs de classification sont obtenus
par déconvolution bayésienne itérative (D'Agostini, 4 itérations) avec la matrice de confusion de l'IA de référence
mesurée sur simulation ; incertitudes par rééchantillonnage (bootstrap de Poisson et de la matrice). La note de
qualité (0–100) combine huit critères pondérés : durée (tous les points à 2 h), nombre de traces, stabilité, bruit,
temps perdu, calibration, part de traces incertaines et nombre d'alphas ; elle est plafonnée à 59 sous 10 min
analysées.</p>
<details><summary>Configuration complète utilisée (JSON)</summary><pre><code>{_t(cfg_json)}</code></pre></details>
</div>"""


# =========================================================================== #
# Analyse approfondie et session de nuit
# =========================================================================== #
_SYMB = {"alpha": "α", "beta": "β", "muon": "μ", "proton": "p", "fine": "β/μ", "incertain": "?", "artefact": "×"}


def _session(run) -> dict:
    s = getattr(run, "session", None)
    return s if isinstance(s, dict) else {}


def _parse_dt(v) -> datetime | None:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None


def _session_get(sess: dict, *keys):
    for k in keys:
        v = sess.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


_SESSION_START = ("started_at", "start", "debut", "start_time", "date_debut")
_SESSION_END = ("ended_at", "end", "fin", "stop_time", "date_fin")
_SESSION_PLANNED_END = ("planned_end", "fin_prevue")
_SESSION_PLANNED = ("planned_s", "planned_duration_s", "duree_prevue_s", "planned_duration")
_SESSION_ACTUAL = ("elapsed_s", "actual_s", "actual_duration_s", "duree_reelle_s", "wall_s")
_SESSION_CHECKPOINTS = ("checkpoints", "points_de_controle", "n_checkpoints")
_SESSION_STOP = ("stop_reason", "raison_arret", "status", "etat")
_SESSION_SEGMENTS = ("video_segments", "segments", "files", "fichiers", "videos")
_SESSION_WARN = ("warnings", "avertissements")
# Réglages connus de la session de nuit (longrun.NightSessionConfig et session_info) → libellés
_SESSION_LABELS = {
    "camera_index": "Caméra n°", "duration_h": "Durée demandée (h)", "end_time": "Heure de fin demandée",
    "segment_min": "Durée d'un segment vidéo (min)", "record_video": "Enregistrement vidéo",
    "record_scale": "Échelle de la vidéo", "record_fps": "Cadence de la vidéo (images/s)",
    "snapshot_every_min": "Photo toutes les (min)", "autosave_every_min": "Sauvegarde toutes les (min)",
    "keep_awake": "Empêcher la mise en veille", "dead_after_min": "Alerte sans trace après (min)",
    "condition": "Condition", "sample": "Échantillon", "mass_g": "Masse (g)", "px_per_mm": "Calibration (px/mm)",
    "output_dir": "Dossier de sortie", "keep_display_on": "Garder l'écran allumé", "video_codec": "Codec vidéo",
    "photos": "Photos", "reconnections": "Reconnexions de la caméra", "simulated_clock": "Horloge simulée (essai)",
    "folder": "Dossier de la session", "journal": "Journal", "partial": "Résultats provisoires",
}


def _session_value(v) -> str:
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _gn(v) if isinstance(v, float) else str(v)
    if isinstance(v, (list, tuple)):
        return str(len(v))
    return str(v)


def _warning_txt(w) -> str:
    """Avertissement de session : texte, ou dict {heure, temps_s, message}."""
    if isinstance(w, str):
        return w
    if isinstance(w, dict):
        msg = str(_session_get(w, "message", "texte", "msg", "text") or "")
        dt = _parse_dt(_session_get(w, "heure", "time", "date"))
        when = dt.strftime("%H:%M") if dt is not None else ""
        if not when and _session_get(w, "temps_s", "t_s") is not None:
            when = fr_clock(_session_get(w, "temps_s", "t_s"))
        return f"{when} — {msg}" if when and msg else (msg or json.dumps(w, ensure_ascii=False, default=str))
    return str(w)


def _session_start_dt(run) -> datetime | None:
    return _parse_dt(_session_get(_session(run), *_SESSION_START))


def _quality_html(q) -> str:
    if q is None:
        return ""
    col = {"excellente": "var(--ok)", "bonne": "var(--ok)", "passable": "var(--warn)"}.get(q.grade, "#e5484d")
    rows = []
    for c in q.components:
        if c.value is None:
            pts = '<span class="note">non évalué</span>'
        else:
            pts = (f'<span class="meter"><span style="width:{100 * c.value:.0f}%"></span></span>'
                   f"{fr_num(c.points, 1)} / {fr_num(c.weight, 0)}")
        rows.append(f"<tr><td>{_t(c.label)}</td><td class=\"wrap\">{_t(c.measured)}</td><td>{pts}</td>"
                    f"<td class=\"wrap\">{_t(c.comment)}</td></tr>")
    table = ('<div class="table-wrap"><table class="data"><thead><tr><th>Critère</th><th>Mesure</th>'
             f"<th>Points</th><th>Commentaire</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    advice = "".join(f"<li>{_t(a)}</li>" for a in q.advice)
    n_ev = q.n_evaluated
    return (f'<div class="score-card"><div class="score-num" style="color:{col}">{q.score}<span>/100</span></div>'
            f'<div><div class="score-grade">Qualité {_t(q.grade)}</div><p class="note">Note calculée sur {n_ev} '
            f"critère{'s' if n_ev > 1 else ''} évalué{'s' if n_ev > 1 else ''} sur {len(q.components)} ; un critère "
            "sans donnée (ancien enregistrement, durée trop courte) ne compte pas."
            + (f" {_t(q.note)}" if q.note else "") + "</p></div></div>" + table
            + (f'<h4>Conseils pour le prochain enregistrement</h4><ul class="advice">{advice}</ul>' if advice else ""))


def _bins_table(deep: dict, run) -> str:
    bins = deep.get("bins") or []
    if not bins:
        return '<p class="note">Durée analysée nulle : aucun découpage en tranches.</p>'
    start = _session_start_dt(run)
    has_wall = any(math.isfinite(b.wall0) for b in bins)
    nz = [i["noise"] for i in deep.get("intervals") or [] if math.isfinite(i["noise"])]
    med = float(np.median(nz)) if nz else float("nan")
    present = [c for c in CLASSES if any(b.counts.get(c, 0) for b in bins)] or ["alpha", "beta", "muon"]
    fl = "Fines (/min)"
    head = "<tr><th>Tranche (temps analysé)</th>"
    if has_wall:
        head += "<th>Vidéo / caméra</th>"
    if start is not None and has_wall:
        head += "<th>Heure</th>"
    head += "<th>Durée analysée</th>" + "".join(f"<th>{_SYMB[c]}</th>" for c in present)
    head += f"<th>Total</th><th>Taux total (/min)</th><th>{fl}</th>"
    if med > 0:
        head += "<th>Bruit (× méd.)</th>"
    head += "<th>État</th></tr>"
    body = []
    from datetime import timedelta
    for b in bins[:240]:
        r, rf = b.rate(None), b.rate("fines")
        state = "normale"
        if b.dead:
            state = "inactive"
        elif b.dead_s > 0:
            state = f"inactive {fr_num(100 * b.dead_s / max(b.exposure_s, 1e-9), 0)} %"
        elif math.isfinite(b.disturbed_frac) and b.disturbed_frac > 0.2:
            state = f"perturbée ({fr_num(100 * b.disturbed_frac, 0)} %)"
        if b.partial:
            state += " (partielle)"
        cls = ' class="dead"' if b.dead else ""
        tr = f"<tr{cls}><td>{_t(fr_clock(b.t0))} – {_t(fr_clock(b.t1))}</td>"
        if has_wall:
            tr += f"<td>{_t(fr_clock(b.wall0))} – {_t(fr_clock(b.wall1))}</td>"
        if start is not None and has_wall:
            try:
                h0 = (start + timedelta(seconds=float(b.wall0))).strftime("%H:%M")
                h1 = (start + timedelta(seconds=float(b.wall1))).strftime("%H:%M")
                tr += f"<td>{h0} – {h1}</td>"
            except (OverflowError, ValueError):
                tr += "<td>—</td>"
        tr += f"<td>{_t(fr_duration(b.exposure_s))}</td>"
        tr += "".join(f"<td>{b.counts.get(c, 0)}</td>" for c in present)
        tr += f"<td>{b.total}</td><td>{_rate_txt(r)}</td><td>{_rate_txt(rf)}</td>"
        if med > 0:
            tr += f"<td>{fr_num(b.noise / med) if math.isfinite(b.noise) else '—'}</td>"
        tr += f"<td>{_t(state)}</td></tr>"
        body.append(tr)
    more = (f'<p class="note">{len(bins) - 240} tranches supplémentaires non affichées.</p>' if len(bins) > 240 else "")
    clock = ("Temps analysé : horloge d'analyse (hors perturbations). "
             if deep.get("clock") == "analyse" else
             "Ancien enregistrement : instants mesurés sur l'horloge de la vidéo (perturbations comprises). ")
    return (f'<div class="table-wrap"><table class="data"><thead>{head}</thead><tbody>{"".join(body)}'
            "</tbody></table></div>" + more +
            f'<p class="note">{_t(clock)}Taux = N / durée analysée de la tranche, incertitude √N / T. Traces fines : '
            f"{_t(deep['families']['label'].lower())}. Une tranche « inactive » chevauche une période où la chambre ne "
            "détectait presque plus rien (voir ci-dessous).</p>")


def _stability_html(deep: dict) -> str:
    st = deep.get("stability") or {}
    labels = {"total": "Toutes les traces", "alpha": "Alphas", "fines": deep["families"]["label"]}
    rows = []
    for k in ("total", "alpha", "fines"):
        x = st.get(k)
        if x is None:
            rows.append(f"<tr><td>{_t(labels[k])}</td><td colspan=\"6\" class=\"wrap\">non calculable (moins de 3 "
                        "tranches ou de 10 traces)</td></tr>")
            continue
        dur = float(deep.get("duration_s") or 0.0)
        if math.isfinite(x.slope_rel_per_h) and math.isfinite(x.slope_err):
            if dur >= 7200:
                drift = f"{fr_num(100 * x.slope_rel_per_h, 1, sign=True)} ± {fr_num(100 * x.slope_err, 1)} %/h"
            else:
                k_ = dur / 3600.0
                drift = (f"{fr_num(100 * k_ * x.slope_rel_per_h, 1, sign=True)} ± {fr_num(100 * k_ * x.slope_err, 1)} "
                         "% sur la durée")
        else:
            drift = "—"
        verdict = ('<span class="yes">✓ taux constant</span>' if x.constant_ok else "⚠ taux variable")
        rows.append(f"<tr><td>{_t(labels[k])}</td><td>{fr_int(x.n_total)}</td><td>{x.n_bins}</td>"
                    f"<td>{fr_num(x.chi2, 1)} / {x.dof}</td><td>{_fmt_p(x.p_value)} <span class=\"note\">"
                    f"({_t(x.method)})</span></td><td>{_t(drift)}</td><td>{verdict}</td></tr>")
    return ('<div class="table-wrap"><table class="data"><thead><tr><th>Groupe</th><th>N</th><th>Tranches</th>'
            "<th>χ² / ddl</th><th>p</th><th>Dérive</th><th>Verdict</th></tr></thead><tbody>" + "".join(rows)
            + "</tbody></table></div>"
            '<p class="note">Test du χ² : les comptages des tranches sont-ils compatibles avec un taux constant '
            "(fluctuations de Poisson seulement) ? Verdict « taux constant » si p &gt; 0,01. Si seules les alphas "
            "varient, la cause peut être le radon de la pièce (sa concentration change naturellement) ; si les traces "
            "fines varient, la sensibilité de la chambre a probablement changé.</p>")


def _dead_html(deep: dict) -> str:
    dead = deep.get("dead") or {}
    per = dead.get("periods") or []
    if not dead.get("possible"):
        return f'<p class="note">Recherche des périodes inactives impossible : {_t(dead.get("reason", ""))}.</p>'
    head = (f"<p>Taux de référence (médiane des tranches de {_t(fr_duration(dead['bin_s']))}) : "
            f"{fr_num(dead['ref_rate_per_min'])} /min.</p>")
    if not per:
        return head + ('<p class="note">Aucune période inactive : le taux n\'est jamais tombé sous 25 % du taux de '
                       "référence de façon significative (au moins deux tranches consécutives).</p>")
    rows = []
    for p in per:
        wall = (f"{fr_clock(p.wall0)} – {fr_clock(p.wall1)}" if math.isfinite(p.wall0) and math.isfinite(p.wall1)
                else "—")
        rows.append(f"<tr><td>{_t(fr_clock(p.t0))} – {_t(fr_clock(p.t1))}</td><td>{_t(wall)}</td>"
                    f"<td>{_t(fr_duration(p.duration_s))}</td><td>{p.n_obs}</td><td>{fr_num(p.n_expected, 0)}</td>"
                    f"<td>{_fmt_p(p.p_value)}</td><td>{_t(p.kind)}"
                    + (f" (bruit ×{fr_num(p.noise_ratio, 1)})" if math.isfinite(p.noise_ratio) else "")
                    + "</td></tr>")
    lr = deep["live"]["rates"]["total"]
    return (head + '<div class="table-wrap"><table class="data"><thead><tr><th>Période (temps analysé)</th>'
            "<th>Vidéo / caméra</th><th>Durée</th><th>Traces observées</th><th>Attendues</th><th>p (Poisson)</th>"
            f"<th>Type</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f'<p class="note">Total retiré : {_t(fr_duration(dead["dead_s"]))}. Taux total sur le temps actif : '
            f"{_t(lr.fmt())} ({_t(fr_duration(deep['live']['duration_s']))}). Les comptages bruts restent inchangés. "
            "Vérifiez la vidéo à ces moments : retirer une période parce qu'elle compte peu de traces n'est justifié "
            "que si la chambre était réellement inactive.</p>")


def _noise_html(deep: dict) -> str:
    nz = deep.get("noise")
    if not nz:
        return ('<p class="note">Journal du détecteur absent (enregistrement d\'une version antérieure) : bruit et '
                "perturbations au cours du temps non disponibles.</p>")
    items = [
        ("Intervalles du journal", str(nz["n_intervals"])),
        ("Temps analysé", fr_duration(nz["analysed_s"])),
        ("Durée de la source (vidéo / caméra)", fr_duration(nz["wall_s"])),
        ("Temps non analysé", f"{fr_num(100 * nz['unanalysed_frac'], 1)} %" if math.isfinite(nz["unanalysed_frac"]) else "—"),
        ("Images perturbées", f"{fr_num(100 * nz['disturbed_frac'], 1)} %" if math.isfinite(nz["disturbed_frac"]) else "—"),
        ("Bruit médian (niveaux de gris)", fr_num(nz["median"])),
        ("Dérive du bruit (fin / début)", f"×{fr_num(nz['drift_ratio'])}" if math.isfinite(nz["drift_ratio"]) else "—"),
        ("Pointes de bruit (> 2× la médiane)", str(nz["n_spikes"])),
    ]
    return _meta_dl(items)


def _families_html(deep: dict, run) -> str:
    fam = deep["families"]
    dur = float(deep.get("duration_s") or 0.0)
    from .stats import rate as _rate
    rf, rl = _rate(fam["fines"], dur), _rate(fam["lourdes"], dur)

    def comp(d: dict) -> str:
        return ", ".join(f"{_SYMB.get(k, k)} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "—"

    mode = ("traces classées β ou μ (resp. α ou proton), plus les traces incertaines rangées dans la famille « fine » "
            "(resp. « lourde ») d'après l'épaisseur et la densité d'ionisation mesurées par les règles"
            if fam["mode"] == "famille" else "ancien enregistrement : fines = β + μ, lourdes = α + protons")
    if fam.get("famille_desaccord"):
        mode += (f" ; pour {fam['famille_desaccord']} trace(s), la famille mesurée contredit la classe finale "
                 "(ex. muon de famille « lourde ») : la classe finale est retenue")
    return ('<div class="table-wrap"><table class="data"><thead><tr><th>Famille</th><th>N</th><th>Taux (/min)</th>'
            "<th>Dont incertaines</th><th>Composition</th></tr></thead><tbody>"
            f"<tr><td>{_t(fam['label'])}</td><td>{fam['fines']}</td><td>{_t(_rate_txt(rf))}</td>"
            f"<td>{fam['fines_incertaines']}</td><td class=\"wrap\">{_t(comp(fam['fines_par_classe']))}</td></tr>"
            f"<tr><td>Traces lourdes (α, protons)</td><td>{fam['lourdes']}</td><td>{_t(_rate_txt(rl))}</td>"
            f"<td>{fam['lourdes_incertaines']}</td><td class=\"wrap\">{_t(comp(fam['lourdes_par_classe']))}</td></tr>"
            "</tbody></table></div>"
            f'<p class="note">Définition : {_t(mode)}. Les traces fines sont dues aux particules au minimum '
            "d'ionisation (électrons, muons) ; une trace fine, courte et droite peut être un muon presque vertical ou "
            "un électron énergétique : cette ambiguïté β/μ est physique. Regrouper toutes les traces fines conserve "
            "toute la statistique pour la recherche de radioactivité β.</p>")


def _unfold_html(deep: dict) -> str:
    uf = deep.get("unfold")
    if uf is None:
        return f'<p class="note">Estimation corrigée non disponible : {_t(deep.get("unfold_reason") or "—")}.</p>'
    rows = []
    for c in uf.classes:
        raw = uf.observed.get(c, 0) + uf.manual.get(c, 0)
        rows.append(f"<tr><td>{_swatch(c)}{_t(physics.CLASS_INFO[c]['nom'])}</td><td>{raw}</td>"
                    f"<td>{fr_num(uf.estimate[c], 0)} ± {fr_num(uf.err[c], 0)}</td>"
                    f"<td>{fr_num(uf.lo[c], 0)} – {fr_num(uf.hi[c], 0)}</td>"
                    f"<td>{fr_num(100 * uf.recall[c], 0)} %</td><td>{fr_num(100 * uf.efficiency[c], 0)} %</td></tr>")
    n_inc = uf.observed.get("incertain", 0)
    if n_inc:
        rows.append(f"<tr><td>{_swatch('incertain')}Incertaines</td><td>{n_inc}</td><td colspan=\"4\" class=\"wrap\">"
                    "réparties entre les types selon la matrice de confusion</td></tr>")
    tot_raw = sum(uf.observed.values()) + sum(uf.manual.values())
    rows.append(f'<tr class="total"><td>Total</td><td>{tot_raw}</td><td>{fr_num(uf.total_estimate, 0)}</td>'
                "<td></td><td></td><td></td></tr>")
    leak = (f" Sur simulation, {fr_num(100 * uf.artefact_leak, 1)} % des fausses détections (artefacts) sont "
            "acceptées comme traces ; l'estimation suppose qu'il n'en reste pas." if math.isfinite(uf.artefact_leak) else "")
    man = sum(uf.manual.values())
    man_txt = f" Les {man} traces corrigées à la main sont comptées telles quelles." if man else ""
    if uf.total_estimate > tot_raw + 0.5:
        man_txt += (" Le total corrigé dépasse le total brut parce qu'une partie des vraies traces est rejetée à tort "
                    "comme fausse détection (colonne « Acceptées »).")
    return ('<div class="table-wrap"><table class="data"><thead><tr><th>Type</th><th>Comptage brut</th>'
            "<th>Estimation corrigée</th><th>IC 68 %</th><th>Bien reconnues (simulation)</th>"
            f"<th>Acceptées (simulation)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            '<p class="note"><strong>Estimation corrigée (d\'après la validation sur simulation).</strong> La matrice de '
            f"confusion de l'IA de référence, mesurée sur {fr_int(uf.n_matrix)} traces simulées dont la nature est "
            "connue, indique quelle fraction de chaque type est reconnue, confondue avec un autre type, laissée "
            "incertaine ou rejetée. La déconvolution bayésienne itérative (D'Agostini, "
            f"{uf.n_iter} itérations) remonte des comptages observés aux nombres les plus probables de traces de chaque "
            f"type ; incertitudes par {uf.n_boot} rééchantillonnages.{leak}{man_txt} Cette estimation dépend du "
            "modèle : une simulation ne reproduit jamais parfaitement une vraie chambre. Elle ne remplace pas les "
            "comptages bruts.</p>")


def _confidence_html(deep: dict) -> str:
    cf = deep.get("confidence") or {}
    if not cf.get("n"):
        return '<p class="note">Aucune trace.</p>'
    thr = cf["threshold"]
    ts = fr_num(100 * thr, 0)
    has_ia = cf.get("has_ia")
    head = (f"<tr><th>Type</th><th>N</th><th>Confiance médiane</th><th>Quartiles</th><th>Confiance ≥ {ts} %</th>"
            + (f"<th>Probabilité IA médiane</th><th>IA ≥ {ts} %</th>" if has_ia else "") + "</tr>")
    rows = []
    for c in CLASSES:
        d = cf["classes"][c]
        if not d["n"]:
            continue
        tr = (f"<tr><td>{_swatch(c)}{_t(physics.CLASS_INFO[c]['nom'])}</td><td>{d['n']}</td>"
              f"<td>{fr_num(d['median'])}</td><td>{fr_num(d['q1'])} – {fr_num(d['q3'])}</td>"
              f"<td>{fr_num(100 * d['frac_ge'], 0) + ' %' if math.isfinite(d['frac_ge']) else '—'}</td>")
        if has_ia:
            tr += (f"<td>{fr_num(d['median_ia'])}</td>"
                   f"<td>{fr_num(100 * d['frac_ia_ge'], 0) + ' %' if math.isfinite(d['frac_ia_ge']) else '—'}</td>")
        rows.append(tr + "</tr>")
    notes = []
    if math.isfinite(cf.get("frac_confident", float("nan"))):
        notes.append(f"{fr_num(100 * cf['frac_confident'], 0)} % des traces classées ont une confiance d'au moins {ts} %.")
    inc = cf.get("incertain") or {}
    if inc.get("n"):
        bf = inc.get("by_family") or {}
        t_ = f"{inc['n']} traces incertaines ({fr_num(100 * inc['frac'], 0)} %)"
        if bf:
            t_ += " : " + ", ".join(f"{v} {('fines' if k == 'fine' else 'lourdes' if k == 'lourde' else 'de famille inconnue')}"
                                    for k, v in bf.items())
        if inc.get("capped") is not None:
            t_ += (f" ; {inc['capped']} plafonnées pour l'ambiguïté « courte et droite » (β/μ), {inc['disagree']} "
                   f"en désaccord règles/IA, {inc['low_conf']} à score trop faible")
        comp = inc.get("competitors") or {}
        if comp:
            t_ += " ; avis de l'IA de référence : " + ", ".join(
                f"{_SYMB.get(k, k)} {v}" for k, v in sorted(comp.items(), key=lambda kv: -kv[1]))
        notes.append(t_ + ".")
    ag = cf.get("agreement")
    if ag and ag.get("n"):
        who = "règles physiques / IA de référence" if ag["kind"] == "regles_ia" else "classification finale / IA de référence"
        notes.append(f"Accord {who} : {fr_num(100 * ag['frac'], 0)} % ({ag['n_agree']} sur {ag['n']}).")
    man = cf.get("manual")
    if man and man.get("n"):
        notes.append(f"Traces corrigées à la main : {man['n']}, dont {man['n_same']} où la classification automatique "
                     "était déjà juste (échantillon non aléatoire).")
    notes.append("Les probabilités de l'IA de référence sont calibrées sur des traces simulées ; avec les règles seules, "
                 "la confiance est un score flou. La fiabilité réelle sur votre chambre se mesure en vérifiant à la "
                 "main un échantillon de traces tirées au hasard.")
    return (f'<div class="table-wrap"><table class="data"><thead>{head}</thead><tbody>{"".join(rows)}</tbody>'
            f"</table></div><p class=\"note\">{_t(' '.join(notes))}</p>")


def _deep_section_html(run, s: dict) -> str:
    """Contenu de la section « Analyse approfondie » ('' si l'analyse n'a pas été calculée)."""
    deep, q = s.get("deep"), s.get("quality")
    if not deep:
        return ""
    parts = ['<div class="deep">', "<h3>Note de qualité de l'enregistrement</h3>", _quality_html(q),
             f"<h3>Taux par tranche de {_t(fr_duration(deep['bin_s']))}</h3>", _bins_table(deep, run),
             "<h3>Stabilité du taux</h3>", _stability_html(deep),
             "<h3>Périodes inactives de la chambre</h3>", _dead_html(deep),
             "<h3>Bruit de l'image et perturbations</h3>", _noise_html(deep),
             "<h3>Familles de traces : fines et lourdes</h3>", _families_html(deep, run),
             "<h3>Comptages corrigés des erreurs de classification</h3>", _unfold_html(deep),
             "<h3>Confiance de la classification</h3>", _confidence_html(deep), "</div>"]
    return "\n".join(parts)


def _session_html(run) -> str:
    """Résumé d'une session de nuit (métadonnées ``run.session``) ; '' si absente."""
    sess = _session(run)
    if not sess:
        return ""
    used = set()

    def get(keys):
        for k in keys:
            v = sess.get(k)
            if v not in (None, "", [], {}):
                used.add(k)
                return v
        return None

    start, end = get(_SESSION_START), get(_SESSION_END)
    planned_end = get(_SESSION_PLANNED_END)
    planned, actual = _num(get(_SESSION_PLANNED)), _num(get(_SESSION_ACTUAL))
    if math.isfinite(planned) and planned < 0:                  # -1 : jusqu'à l'arrêt manuel
        planned = float("nan")
    sd, ed = _parse_dt(start), _parse_dt(end)
    if not math.isfinite(actual) and sd is not None and ed is not None:
        actual = (ed - sd).total_seconds()
    cps = get(_SESSION_CHECKPOINTS)
    stop = get(_SESSION_STOP)
    segs = get(_SESSION_SEGMENTS)
    warns = get(_SESSION_WARN) or []
    if isinstance(warns, (str, dict)):
        warns = [warns]
    items = []
    if start is not None:
        items.append(("Début", _fmt_date(start)))
    if end is not None:
        items.append(("Fin", _fmt_date(end)))
    if planned_end is not None:
        items.append(("Fin prévue", _fmt_date(planned_end)))
    if math.isfinite(planned):
        items.append(("Durée prévue", fr_duration(planned)))
    elif "planned_s" in sess:
        items.append(("Durée prévue", "jusqu'à l'arrêt manuel"))
        used.add("planned_s")
    if math.isfinite(actual):
        items.append(("Durée réelle", fr_duration(actual)))
    items.append(("Durée analysée", fr_duration(_num(run.duration_s))))
    if cps is not None:
        if isinstance(cps, (list, tuple)):
            last = cps[-1] if cps else None
            txt = f"{len(cps)}"
            if isinstance(last, dict):
                lt = _session_get(last, "time", "t", "date", "heure", "wall_s", "t_s")
                if lt is not None:
                    txt += f" (dernier : {_fmt_date(lt) if _parse_dt(lt) else fr_clock(_num(lt))})"
            items.append(("Points de sauvegarde", txt))
        else:
            items.append(("Points de sauvegarde", str(cps)))
    if segs is not None:
        items.append(("Fichiers vidéo", str(len(segs)) if isinstance(segs, (list, tuple)) else str(segs)))
    if stop is not None:
        items.append(("Fin de session", str(stop)))
    for k in ("reconnections", "photos"):
        if k in sess and sess[k] not in (None, ""):
            items.append((_SESSION_LABELS[k], _session_value(sess[k])))
            used.add(k)
    used.update(_SESSION_WARN)
    other = [(_SESSION_LABELS.get(str(k), str(k)), _session_value(v)) for k, v in sess.items()
             if k not in used and isinstance(v, (str, int, float, bool)) and str(v).strip()][:30]
    html_ = _meta_dl(items)
    if sess.get("partial") is True:
        html_ += _callouts([Conclusion(
            "Résultats provisoires",
            "Ces résultats viennent d'un point de sauvegarde pendant la session (session non terminée ou "
            "interrompue) : les traces et la classification finale peuvent encore changer.", "attention")])
    if math.isfinite(planned) and math.isfinite(actual) and planned > 0 and actual < 0.95 * planned:
        html_ += _callouts([Conclusion(
            "Session écourtée",
            f"La session a duré {fr_duration(actual)} au lieu des {fr_duration(planned)} prévues "
            f"({fr_num(100 * actual / planned, 0)} %). Les résultats portent sur la partie enregistrée.", "attention")])
    if warns:
        html_ += ("<h3>Avertissements de la session</h3><ul>"
                  + "".join(f"<li>{_t(_warning_txt(w))}</li>" for w in warns[:50]) + "</ul>")
        if len(warns) > 50:
            html_ += f'<p class="note">{len(warns) - 50} avertissements supplémentaires non affichés.</p>'
    else:
        html_ += '<p class="note">Aucun avertissement pendant la session.</p>'
    if other:
        html_ += "<details><summary>Réglages et autres informations de la session</summary>" + _meta_dl(other) \
                 + "</details>"
    return html_


def _deep_markdown(run, s: dict) -> list[str]:
    """Lignes Markdown de l'analyse approfondie (résumé pour l'assistant IA)."""
    deep, q = s.get("deep"), s.get("quality")
    if not deep:
        return []
    out = ["", "## Analyse approfondie", ""]
    if q is not None:
        out.append(f"- **Qualité de l'enregistrement** : {q.score}/100 ({q.grade}) — {q.summary()}")
    st = (deep.get("stability") or {}).get("total")
    if st is not None:
        out.append(f"- **Stabilité du taux** (tranches de {fr_duration(deep['bin_s'])}) : χ² = {fr_num(st.chi2, 1)} "
                   f"pour {st.dof} ddl, {fr_p_eq(st.p_value)} → {st.verdict()}")
    dead = deep.get("dead") or {}
    if dead.get("periods"):
        out.append(f"- **Périodes inactives** : {len(dead['periods'])} ({fr_duration(dead['dead_s'])}) ; taux total "
                   f"sur le temps actif {deep['live']['rates']['total'].fmt()}")
    elif dead.get("possible"):
        out.append("- **Périodes inactives** : aucune")
    fam = deep["families"]
    out.append(f"- **{fam['label']}** : {fam['fines']} (dont {fam['fines_incertaines']} incertaines) ; "
               f"traces lourdes : {fam['lourdes']}")
    nz = deep.get("noise")
    if nz and math.isfinite(nz.get("drift_ratio", float("nan"))):
        out.append(f"- **Bruit de l'image** : dérive ×{fr_num(nz['drift_ratio'])}, {nz['n_spikes']} pointe(s), "
                   f"images perturbées {fr_num(100 * nz['disturbed_frac'], 1) if math.isfinite(nz['disturbed_frac']) else '—'} %")
    uf = deep.get("unfold")
    if uf is not None:
        out.append("- **Comptages corrigés (d'après la validation sur simulation, dépend du modèle)** : "
                   + ", ".join(f"{_SYMB[c]} ≈ {fr_num(uf.estimate[c], 0)} ± {fr_num(uf.err[c], 0)}" for c in uf.classes))
    cf = deep.get("confidence") or {}
    if math.isfinite(cf.get("frac_confident", float("nan"))):
        out.append(f"- **Confiance** : {fr_num(100 * cf['frac_confident'], 0)} % des traces classées ≥ "
                   f"{fr_num(100 * cf['threshold'], 0)} %")
    bins = deep.get("bins") or []
    if len(bins) >= 2:
        out += ["", f"### Taux par tranche de {fr_duration(deep['bin_s'])}", "",
                "| Tranche | α | β | μ | p | ? | Total | Taux total (/min) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for b in bins[:48]:
            out.append(f"| {fr_clock(b.t0)} – {fr_clock(b.t1)}{' (inactive)' if b.dead else ''} | "
                       + " | ".join(str(b.counts.get(c, 0)) for c in CLASSES)
                       + f" | {b.total} | {_rate_txt(b.rate(None))} |")
    sess = _session(run)
    if sess:
        w = _session_get(sess, *_SESSION_WARN) or []
        w = w if isinstance(w, list) else [w]
        out.append("")
        out.append(f"- **Session de nuit** : début {_fmt_date(_session_get(sess, *_SESSION_START))}, fin "
                   f"{_fmt_date(_session_get(sess, *_SESSION_END))}, {len(w)} avertissement(s)"
                   + (" : " + " ; ".join(_warning_txt(x) for x in w[:10]) if w else ""))
    return out


def build_run_report(run, out_html=None, ai_text: str | None = None) -> Path:
    """Écrit le rapport HTML d'un enregistrement et renvoie son chemin.

    Par défaut : <dossier de l'enregistrement>/rapport.html (ValueError si aucun dossier n'est connu).
    Sections : enregistrement, [session de nuit si ``run.session``], comptages, conclusions, analyse
    approfondie, graphiques, galerie, pourquoi ces types, tableau des traces, [IA], méthode.
    """
    if out_html is None:
        if not run.folder:
            raise ValueError("Aucun dossier d'enregistrement : indiquez le chemin du rapport (out_html).")
        out_html = Path(run.folder) / "rapport.html"
    out = Path(out_html)
    rows = run.table()
    s, concl = conclude_run(run)
    t = s["total"]

    tile_items = [
        (str(t.n), "traces détectées"),
        (fr_duration(_num(run.duration_s)), "durée analysée"),
        (f"{fr_num(t.per_min, 1)} ± {fr_num(t.err, 1)} /min", "taux global"),
        (f"{fr_num(_num(s['area_cm2']), 0)} cm²", "surface analysée"),
    ]
    q = s.get("quality")
    if q is not None:
        tile_items.append((f"{q.score}/100", f"qualité {q.grade}"))
    tiles = _tiles(tile_items)
    sections = []
    toc = [("infos", "Enregistrement"), ("comptages", "Comptages et taux"), ("conclusions", "Conclusions"),
           ("graphiques", "Graphiques"), ("galerie", "Galerie"), ("traces", "Tableau des traces")]
    sections.append(_section("infos", "Enregistrement", _meta_dl(_run_meta(run))))
    sess_html = _session_html(run)
    if sess_html:
        sections.append(_section("session", "Session de nuit", sess_html))
        toc.insert(1, ("session", "Session de nuit"))
    exp = expected_sample(run)
    counts_html = _counts_table(s)
    if exp is not None:
        counts_html += f'<p class="note">Activité attendue de l\'échantillon (K-40) : {_t(activity_text(exp))}.</p>'
    counts_html += "<h3>Tests statistiques</h3>" + _stats_table(s, bool(run.calibrated))
    sections.append(_section("comptages", "Comptages et taux", counts_html))
    sections.append(_section("conclusions", "Conclusions", _callouts(concl)))
    deep_html = _deep_section_html(run, s)
    if deep_html:
        sections.append(_section("approfondie", "Analyse approfondie", deep_html))
        toc.insert(toc.index(("graphiques", "Graphiques")), ("approfondie", "Analyse approfondie"))

    figs = []
    for key, title, fig in plots.run_figures(run, deep=s.get("deep")):
        figs.append(_figure(_fig_uri(fig), title, FIG_CAPTIONS.get(key, "")))
    sections.append(_section("graphiques", "Graphiques", '<div class="figs">' + "".join(figs) + "</div>"))
    sections.append(_section("galerie", "Galerie des traces", _gallery(run)))
    why = _why_section(run)
    if why:
        sections.append(_section("pourquoi", "Pourquoi ces types ? (exemples expliqués)", why))
        toc.insert(toc.index(("traces", "Tableau des traces")), ("pourquoi", "Pourquoi ces types ?"))
    sections.append(_section("traces", "Tableau des traces", _tracks_table(rows)))
    ai = _ai_section(ai_text)
    if ai:
        sections.append(ai)
        toc.append(("ia", "Interprétation IA"))
    sections.append(_section("methode", "Méthode et paramètres", _method_html(run.config)))
    toc.append(("methode", "Méthode"))

    subtitle = f"{run.name} — {_sample_txt(run)}"
    page = _page("WilsonTrack — Rapport d'analyse", subtitle, "\n".join(sections), toc, tiles)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


# =========================================================================== #
# Rapport de comparaison
# =========================================================================== #
def _comparison_table(comps: dict, note: str = "") -> str:
    head = ("<tr><th>Type de trace</th><th>Échantillon (/min)</th><th>Fond (/min)</th>"
            "<th>Différence nette (/min)</th><th>Rapport</th><th>p (unilatéral)</th><th>Signification</th>"
            "<th>Significatif ?</th></tr>")
    rows = []
    for key in list(CLASSES) + ["fines", "total"]:
        cp = comps.get(key)
        if cp is None:
            continue
        sw = _swatch(key) if key in physics.CLASS_INFO else ""
        sig = '<span class="yes">✓ oui</span>' if cp.significant else "non"
        cls = ' class="total"' if key in ("fines", "total") else ""
        rows.append(
            f"<tr{cls}><td>{sw}{_t(_group_label(key, cp))}</td>"
            f"<td>{_rate_txt(cp.a)} <span class=\"note\">(N = {cp.a.n})</span></td>"
            f"<td>{_rate_txt(cp.b)} <span class=\"note\">(N = {cp.b.n})</span></td>"
            f"<td>{fr_num(cp.diff, sign=True)} ± {_f(cp.diff_err)}</td><td>{_cmp_ratio(cp)}</td>"
            f"<td>{_fmt_p(cp.p_value)}</td><td>{fr_num(cp.sigma, 1)} σ</td><td>{sig}</td></tr>")
    fp = comps.get("fines")
    fam = bool(fp is not None and "incertaines" in (getattr(fp, "label", "") or ""))
    fines_def = ("Traces fines = β + μ + traces fines classées incertaines (famille « fine » : ambiguïté β/μ)"
                 if fam else "Traces fines = β + μ")
    return (f'<div class="table-wrap"><table class="data"><thead>{head}</thead><tbody>{"".join(rows)}'
            "</tbody></table></div>"
            '<p class="note">Test binomial exact conditionnel : sachant le nombre total de traces des deux '
            "enregistrements, on teste si l'échantillon en compte plus que ne le prévoient les durées (et les "
            f"surfaces analysées). Significatif = excès avec p &lt; 0,01. {_t(fines_def)} : les électrons "
            "énergétiques du K-40, presque rectilignes, peuvent être classés comme muons."
            + (f" {_t(note)}" if note else "") + "</p>")


def _runs_meta_table(runs: list, heads: list[str]) -> str:
    metas = [dict(_run_meta(r)) for r in runs]
    keys = [k for k in metas[0] if k not in ("Notes", "Opérateur")]
    head = "<tr><th></th>" + "".join(f"<th>{_t(h)}</th>" for h in heads) + "</tr>"
    body = "".join("<tr><td>" + _t(k) + "</td>" + "".join(f'<td class="wrap">{_t(m.get(k, "—"))}</td>' for m in metas)
                   + "</tr>" for k in keys)
    return f'<div class="table-wrap"><table class="data"><thead>{head}</thead><tbody>{body}</tbody></table></div>'


def build_comparison_report(sample, background, out_html=None, ai_text: str | None = None,
                            extra_runs=()) -> Path:
    """Écrit le rapport HTML comparant un échantillon au bruit de fond et renvoie son chemin.

    Par défaut (out_html=None) : <dossier de l'échantillon>/comparaison.html.
    """
    if out_html is None:
        if not sample.folder:
            raise ValueError("Aucun dossier d'enregistrement : indiquez le chemin du rapport (out_html).")
        out_html = Path(sample.folder) / "comparaison.html"
    out = Path(out_html)
    extra = [r for r in (extra_runs or ()) if r is not None]
    comps, concl = conclude_comparison(sample, background)
    f, b = comps["fines"], comps["beta"]
    al = comps.get("alpha_bilateral", comps["alpha"])

    flab = _group_label("fines", f)
    tiles = _tiles([
        (f"{fr_num(f.diff, 1, sign=True)} ± {_f(f.diff_err, 1)} /min",
         f"excès de {flab[0].lower() + flab[1:]}, {fr_num(f.sigma, 1)} σ"),
        (f"{fr_num(b.diff, 1, sign=True)} ± {_f(b.diff_err, 1)} /min", f"excès de traces β, {fr_num(b.sigma, 1)} σ"),
        (f"{fr_p_eq(al.p_value)}", "contrôle : variation des alphas (test bilatéral)"),
        (f"{fr_num(_num(sample.duration_s) / 60, 1)} / {fr_num(_num(background.duration_s) / 60, 1)} min",
         "durées échantillon / fond"),
    ])
    runs = [sample, background] + extra
    heads = ["Échantillon", "Fond"] + [f"Autre {i + 1}" for i in range(len(extra))]
    sections = [
        _section("infos", "Enregistrements comparés", _runs_meta_table(runs, heads)),
        _section("comparaison", "Taux comparés", _comparison_table(comps, area_note(sample, background))),
        _section("conclusions", "Conclusions", _callouts(concl)),
    ]
    exp = expected_sample(sample)
    if exp is not None:
        sections[-1] = sections[-1].replace(
            "</section>", f'<p class="note">Activité attendue de l\'échantillon (K-40) : {_t(activity_text(exp))}.</p>\n'
                          "</section>", 1)

    order = [background, sample] + extra
    figs = [_figure(_fig_uri(plots.fig_comparison(order)), "Comparaison des taux",
                    "Taux par type de trace pour chaque enregistrement (barres d'erreur : √N / T). À droite : "
                    f"l'ensemble des {flab[0].lower() + flab[1:]}, l'indicateur le plus fiable de la radioactivité β.")]
    sections.append(_section("graphiques", "Graphiques", '<div class="figs">' + "".join(figs) + "</div>"))
    scat = [_figure(_fig_uri(plots.fig_length_width(r.table(), title=plots.run_label(r))),
                    f"Longueur et épaisseur — {plots.run_label(r)}") for r in order]
    ts = [_figure(_fig_uri(plots.fig_rate_timeseries(r.table(), r.duration_s)),
                  f"Taux en fonction du temps — {plots.run_label(r)}", plots.run_label(r)) for r in order]
    sections.append(_section(
        "formes", "Longueur et épaisseur de chaque enregistrement",
        '<p class="note">Mêmes axes de lecture pour chaque condition : un excès de traces fines et sinueuses '
        "(β) avec l'échantillon est la signature attendue du potassium-40.</p>"
        '<div class="figs">' + "".join(scat) + "</div>"))
    sections.append(_section(
        "stabilite", "Stabilité des taux pendant les mesures",
        '<p class="note">Une comparaison n\'est valable que si la chambre est restée stable pendant chaque '
        "enregistrement (taux sans dérive).</p>" + '<div class="figs">' + "".join(ts) + "</div>"))
    toc = [("infos", "Enregistrements"), ("comparaison", "Taux comparés"), ("conclusions", "Conclusions"),
           ("graphiques", "Graphiques"), ("formes", "Longueur et épaisseur"), ("stabilite", "Stabilité")]
    ai = _ai_section(ai_text)
    if ai:
        sections.append(ai)
        toc.append(("ia", "Interprétation IA"))
    method = ("<p>Chaque enregistrement est analysé avec la même chaîne de traitement (détection, suivi, "
              "classification ; voir le rapport de chaque enregistrement). Pour chaque type de trace, on compare "
              "le nombre N<sub>é</sub> de traces de l'échantillon (durée T<sub>é</sub>, surface analysée "
              "A<sub>é</sub>) à celui du fond N<sub>f</sub> (durée T<sub>f</sub>, surface A<sub>f</sub>). Sous "
              "l'hypothèse « même taux par unité de surface », N<sub>é</sub> suit une loi binomiale "
              "B(N<sub>é</sub> + N<sub>f</sub>, T<sub>é</sub>A<sub>é</sub> / (T<sub>é</sub>A<sub>é</sub> + "
              "T<sub>f</sub>A<sub>f</sub>)) : la p-valeur unilatérale mesure la probabilité d'observer un excès au "
              "moins aussi grand par hasard. La signification est exprimée en écarts-types (σ) d'une loi normale "
              "équivalente. Le contrôle des alphas (que le K-40 n'émet pas) utilise un test bilatéral : une hausse "
              "comme une baisse signale un changement des conditions de la chambre.</p>"
              "<p>Pour une conclusion solide : enregistrer au moins 10 à 20 min par condition, dans les mêmes "
              "conditions (glace sèche, alcool, éclairage, zone analysée), et alterner fond / échantillon / fond.</p>")
    sections.append(_section("methode", "Méthode de comparaison", method))
    toc.append(("methode", "Méthode"))

    subtitle = f"{_run_title(sample)} comparé à {_run_title(background)}"
    page = _page("WilsonTrack — Rapport de comparaison", subtitle, "\n".join(sections), toc, tiles)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
