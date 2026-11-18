"""Explication « pourquoi cette classe ? » de chaque trace.

« Pourquoi c'est une alpha / un électron (bêta) / un muon / un proton / un artefact, ou
pourquoi la trace reste incertaine ? »

Règles d'honnêteté scientifique suivies ici :

- l'explication ne s'appuie que sur ce que la décision a réellement utilisé : les valeurs
  floues intermédiaires des règles physiques (clés ``ev_*`` écrites par
  ``classifier.classify_rules``), leurs seuils (``ClassificationParams``) et, en mode hybride,
  les probabilités de l'IA de référence (entraînée sur des traces SIMULÉES) ;
- les indices que la décision n'utilise pas (durée de visibilité, épaississement de Bragg,
  embranchements) sont marqués ``used_for_decision=False`` : « indice complémentaire (non
  utilisé pour décider) » ;
- chaque affirmation physique s'appuie sur une source publiée (NIST ASTAR / ESTAR / PSTAR,
  PDG, NNDC, CERN S'Cool LAB…), citée dans ``Explanation.sources`` ;
- les nombres sont affichés avec une virgule décimale, jamais « nan », « inf » ou « None ».

API :
    explain_features(f, cls, confidence, scores, ...) -> Explanation
    explain_event(e, cp=None, php=None) -> Explanation
    Explanation.to_html(compact=False) / .to_text() / .short()
"""
from __future__ import annotations

import functools
import html as _html
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass

from .. import physics
from ..config import ClassificationParams, DetectionParams, PhysicsParams

__all__ = ["Criterion", "Explanation", "explain_features", "explain_event", "clear_cache",
           "BASE_CLASSES", "VERDICTS", "COMPLEMENTARY_LABEL"]

BASE_CLASSES = ("alpha", "beta", "muon", "proton")
VERDICTS = ("pour", "contre", "neutre")
COMPLEMENTARY_LABEL = "indice complémentaire (non utilisé pour décider)"

# Couleurs (panneau de détail BLANC, charte du Cégep Marie-Victorin : encre #1C1C1C, gris #5B6B62,
# accents verts). Les icônes ✔ / ✘ gardent le vert et le rouge « physiques » ; les textes colorés par classe
# utilisent une version assombrie de la même teinte (_text_hex) pour rester lisibles (WCAG AA).
_TXT, _MUTED = "#1C1C1C", "#5B6B62"
_GREEN, _RED, _GREY = "#30a46c", "#e5484d", "#8a968f"
_LINK, _BAR_BG = "#247F59", "#E3ECE5"
_HEAD, _SOFT = "#247F59", "#F1F5F1"
_ICON = {"pour": ("✔", _GREEN), "contre": ("✘", _RED), "neutre": ("•", _GREY)}
NBSP = " "

# Seuils de forme codés en dur dans classifier.classify_rules (copiés ici pour les afficher ;
# les verdicts, eux, viennent des valeurs ev_* réellement calculées par les règles).
_SH = (12.0, 25.0)                 # poids « trace longue » : ramp(L, 12, 25)
_LAT_STRAIGHT = (0.20, 0.45)       # étalement latéral (mm) : droite si < 0,20
_LAT_CURLY = (0.30, 0.60)          # courbée si > 0,30
_LAT_STRAIGHT_A = (0.30, 0.60)
_SAG_STRAIGHT = (0.035, 0.07)      # flèche / longueur
_WIG_STRAIGHT = (0.012, 0.022)     # écart RMS / longueur
_SAG_CURLY = (0.05, 0.10)
_WIG_CURLY = (0.015, 0.03)
_TORT_CURLY = (0.10, 0.20)         # rad/mm
_S_ALPHA = (0.80, 0.90)            # rectitude d'une particule lourde
_REF_WIDTH_MM = 0.6                # largeur type d'une trace fine (référence par défaut)
_AMBIGUITY_CAP = 0.45              # plafond « courte et droite »

# Physique (sources : voir _SOURCES)
_X0_AIR_G_CM2 = 36.62              # longueur de radiation de l'air sec (PDG)
_PO212_MEV = 8.785                 # alpha naturelle la plus énergétique (NNDC)
_K40_EMAX = physics.K40_BETA_EMAX_MEV
_MUON_MEV = 4000.0                 # énergie moyenne des muons au sol (PDG)
_SUSSMAN_ALPHA_S = 1.5
_RANGE_EMITTERS = (("Po-210", 5.304), ("Rn-222", 5.490), ("Po-218", 6.002), ("Po-214", 7.687),
                   ("Po-212", 8.785))

_NAMES = {  # (court, avec article, négation, symbole)
    "alpha": ("alpha", "une alpha", "Pas une alpha", "α"),
    "beta": ("électron", "un électron (bêta)", "Pas un électron (bêta)", "β"),
    "muon": ("muon", "un muon", "Pas un muon", "μ"),
    "proton": ("proton", "un proton", "Pas un proton", "p"),
    "artefact": ("artefact", "un artefact", "Pas un artefact", "×"),
    "incertain": ("incertaine", "une trace incertaine", "Pas incertaine", "?"),
}

_SOURCES = {
    "astar": "NIST ASTAR — portées et pouvoir d'arrêt des alphas dans l'air — "
             "https://physics.nist.gov/PhysRefData/Star/Text/ASTAR.html",
    "estar": "NIST ESTAR — électrons dans l'air — https://physics.nist.gov/PhysRefData/Star/Text/ESTAR-t.html",
    "pstar": "NIST PSTAR — protons dans l'air — https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html",
    "pdg_pass": "PDG 2024, « Passage of particles through matter » (Bethe, diffusion multiple de Highland) — "
                "https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf",
    "pdg_cosmic": "PDG 2020, « Cosmic rays » (flux, énergie et angle des muons, protons, électrons) — "
                  "https://pdg.lbl.gov/2020/reviews/rpp2020-rev-cosmic-rays.pdf",
    "pdg_air": "PDG, propriétés de l'air sec (dE/dx minimal, longueur de radiation) — "
               "https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/air_dry_1_atm.html",
    "nndc_k40": "NNDC NuDat 3, désintégration du K-40 (ENSDF, J. Chen 2017) — "
                "https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc=40K&unc=nds",
    "nndc_po212": "NNDC NuDat 3, Po-212 (alpha de 8,78 MeV) — "
                  "https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc=212Po&unc=nds",
    "nndc_rn222": "NNDC NuDat 3, Rn-222 (alpha de 5,49 MeV) — "
                  "https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc=222Rn&unc=nds",
    "gamma": "NIST, coefficients d'atténuation des gammas dans l'air (Hubbell et Seltzer) — "
             "https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html",
    "scool": "CERN S'Cool LAB, J. Woithe, « Cloud Chamber – DIY manual » — "
             "https://people.iut.ac.ir/sites/default/files/users/samin/page_files/scoollab_cloudchamber_diymanual.pdf",
    "cambridge": "Université de Cambridge, groupe HEP, « Cloud Chamber » — https://www.hep.phy.cam.ac.uk/outreach/cloud",
    "sussman": "Z. Sussman, « Automatic Particle Detection in Cloud Chambers » (rapport d'élève, non révisé) — "
               "https://zacharysussman.com/pdf/cloud-chamber.pdf",
    "toronto": "K. Clark, « The Cloud Chamber Experiment », PHY326, Université de Toronto — "
               "https://www.physics.utoronto.ca/~phy326/cc/cc.pdf",
    "model": "IA de référence WilsonTrack : traces simulées, mesures dans wilsontrack/assets/modele_reference.json",
}


# --------------------------------------------------------------------------- #
# Structures de données
# --------------------------------------------------------------------------- #
@dataclass
class Criterion:
    """Un critère mesuré et son verdict pour la classe expliquée.

    strength (0..1) = poids décisif du critère : 0 = indécis (valeur au milieu de la zone de
    transition des règles), 1 = tranché.
    """
    key: str
    label: str
    measured: str
    expected: str
    verdict: str                  # "pour" | "contre" | "neutre"
    strength: float               # 0..1
    explanation: str              # une phrase de physique
    used_for_decision: bool = True


@dataclass
class Explanation:
    cls: str
    cls_name: str
    confidence: float
    title: str
    summary: str
    criteria: list[Criterion]
    why_not: dict[str, str]
    physics: str
    caveats: list[str]
    scores: dict[str, float]
    method: str
    sources: list[str]

    # ------------------------------------------------------------------ texte --
    def short(self) -> str:
        """Une ligne (≤ 160 caractères) pour les tableaux et les infobulles."""
        s = getattr(self, "_short", "") or ""
        if not s:
            s = f"{_NAMES.get(self.cls, _NAMES['incertain'])[3]} {self.title} : {self.summary}"
        return _clip(" ".join(s.split()), 160)

    def to_text(self) -> str:
        """Texte brut multiligne (export CSV, invite envoyée à Claude)."""
        out = [self.title, self.summary,
               f"{_conf_label(self.method)} : {_pct(self.confidence)} — Méthode : {self.method}"]
        if self.criteria:
            out.append("Ce qui a été mesuré :")
            for c in self.criteria:
                icon = _ICON.get(c.verdict, _ICON["neutre"])[0]
                tag = "" if c.used_for_decision else f" [{COMPLEMENTARY_LABEL}]"
                out.append(f"  {icon} {c.label}{tag} : {c.measured} (attendu : {c.expected}) — {c.explanation}")
        if self.scores:
            out.append(f"{_scores_title(self.method)} : " + " · ".join(
                f"{_NAMES.get(k, (k, k, k, k))[3]} {_pct(v)}" for k, v in _ordered_scores(self.scores)))
        if self.why_not:
            out.append("Pourquoi pas les autres ?")
            out += [f"  - {t}" for t in self.why_not.values()]
        if self.physics:
            out.append(f"Physique : {self.physics}")
        if self.caveats:
            out.append("Limites :")
            out += [f"  - {t}" for t in self.caveats]
        if self.sources:
            out.append("Sources :")
            out += [f"  - {s}" for s in self.sources]
        return "\n".join(out)

    # ------------------------------------------------------------------- HTML --
    def to_html(self, compact: bool = False) -> str:
        """Sous-ensemble HTML de Qt (QTextBrowser) pour un fond clair (blanc)."""
        e = _esc
        col = _hex(self.cls)
        tcol = _text_hex(self.cls)
        h = [f'<div style="color:{_TXT}">', _h4("Verdict"),
             f'<table width="100%" cellspacing="0" cellpadding="8" style="margin-bottom:6px"><tr>'
             f'<td width="5" bgcolor="{col}"></td><td bgcolor="{_SOFT}">'
             f'<span style="font-size:15pt; font-weight:700; color:{tcol}">{e(self.title)}</span><br>'
             f'{e(self.summary)}<br>'
             f'<span style="color:{_MUTED}">{e(_conf_label(self.method))} : '
             f'<b style="color:{tcol}">{e(_pct(self.confidence))}</b> · Méthode : {e(self.method)}</span>'
             f'</td></tr></table>']
        if self.criteria:
            h.append(_h4("Ce qui a été mesuré"))
            h.append('<table width="100%" cellspacing="0" cellpadding="3">')
            h.append(f'<tr style="color:{_MUTED}"><td width="4%"></td><td width="46%"><b>Critère</b></td>'
                     f'<td width="26%"><b>Mesuré</b></td><td width="24%"><b>Attendu</b></td></tr>')
            for c in self.criteria:
                icon, icol = _ICON.get(c.verdict, _ICON["neutre"])
                tag = ("" if c.used_for_decision else
                       f'<br><i style="color:{_MUTED}">{e(COMPLEMENTARY_LABEL)}</i>')
                expl = ("" if compact or not c.explanation else
                        f'<br><span style="color:{_MUTED}; font-size:small">{e(c.explanation)}</span>')
                h.append(f'<tr><td valign="top" style="color:{icol}; font-weight:700">{icon}</td>'
                         f'<td valign="top"><b>{e(c.label)}</b>{tag}{expl}</td>'
                         f'<td valign="top">{e(c.measured)}</td>'
                         f'<td valign="top" style="color:{_MUTED}">{e(c.expected)}</td></tr>')
            h.append("</table>")
        if self.scores:
            h.append(_h4(_scores_title(self.method)))
            h.append('<table width="100%" cellspacing="0" cellpadding="2">')
            for k, v in _ordered_scores(self.scores):
                p = max(0.0, min(1.0, float(v)))
                w = int(round(100 * p))
                cc = _hex(k)
                if w <= 0:
                    bar = f'<td width="100%" bgcolor="{_BAR_BG}"><span style="font-size:6pt">&nbsp;</span></td>'
                elif w >= 100:
                    bar = f'<td width="100%" bgcolor="{cc}"><span style="font-size:6pt">&nbsp;</span></td>'
                else:
                    bar = (f'<td width="{w}%" bgcolor="{cc}"><span style="font-size:6pt">&nbsp;</span></td>'
                           f'<td width="{100 - w}%" bgcolor="{_BAR_BG}"><span style="font-size:6pt">&nbsp;</span></td>')
                name = physics.CLASS_INFO.get(k, {}).get("nom", k)
                strong = k == self.cls
                lab = f"<b>{e(name)}</b>" if strong else e(name)
                h.append(f'<tr><td width="36%" style="color:{_text_hex(k)}">{lab}</td>'
                         f'<td width="50%"><table width="100%" cellspacing="0" cellpadding="0"><tr>{bar}</tr></table></td>'
                         f'<td width="14%" align="right">{"<b>" if strong else ""}{e(_pct(p))}{"</b>" if strong else ""}</td></tr>')
            h.append("</table>")
        if self.why_not:
            h.append(_h4("Pourquoi pas les autres ?"))
            h.append('<ul style="margin-top:0px">')
            for k, t in self.why_not.items():
                head, sep, rest = t.partition(" : ")
                if sep:
                    h.append(f'<li><b style="color:{_text_hex(k)}">{e(head)}</b> : {e(rest)}</li>')
                else:
                    h.append(f"<li>{e(t)}</li>")
            h.append("</ul>")
        if self.physics and not compact:
            h.append(_h4("Physique"))
            h.append(f'<p style="margin-top:0px">{e(self.physics)}</p>')
        if self.caveats:
            h.append(_h4("Limites"))
            h.append('<ul style="margin-top:0px">')
            h += [f'<li style="color:{_MUTED}">{e(t)}</li>' for t in (self.caveats[:3] if compact else self.caveats)]
            h.append("</ul>")
        if self.sources and not compact:
            h.append(_h4("Sources"))
            links = []
            for s in self.sources:
                title, url = _split_source(s)
                if url:
                    links.append(f'<a href="{_esc_attr(url)}" style="color:{_LINK}">{e(title)}</a>')
                else:
                    links.append(e(title))
            h.append(f'<p style="margin-top:0px; font-size:small; color:{_MUTED}">' + "<br>".join(links) + "</p>")
        h.append("</div>")
        return "\n".join(h)


# --------------------------------------------------------------------------- #
# Mise en forme (virgule décimale, jamais nan / inf / None)
# --------------------------------------------------------------------------- #
def _f(x) -> float | None:
    """Nombre fini ou None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _num(x, nd: int = 1, default: str = "—") -> str:
    v = _f(x)
    if v is None:
        return default
    if round(v, nd) == 0:
        v = 0.0
    s = f"{v:,.{nd}f}"
    return s.replace(",", NBSP).replace(".", ",").replace("-", "−")


def _pct(p) -> str:
    v = _f(p)
    if v is None:
        return "—"
    v = min(1.0, max(0.0, v))
    if 0.0 < v < 0.005:
        return f"<{NBSP}1{NBSP}%"
    if 0.995 < v < 1.0:
        return f">{NBSP}99{NBSP}%"
    return f"{int(round(100 * v))}{NBSP}%"


def _pct_fine(p) -> str:
    """Pourcentage avec des décimales pour les petites fractions (0,4 %, 0,02 %)."""
    v = _f(p)
    if v is None:
        return "—"
    v = min(1.0, max(0.0, v))
    if v >= 0.1:
        return f"{int(round(100 * v))}{NBSP}%"
    if v >= 0.01:
        return f"{_num(100 * v, 1)}{NBSP}%"
    if v >= 0.0001:
        return f"{_num(100 * v, 2)}{NBSP}%"
    return f"moins de 0,01{NBSP}%"


def _only(p) -> str:
    """« ne lui donne que 3 % » / « ne lui donne pratiquement aucune chance (< 1 %) »."""
    v = _f(p) or 0.0
    if v < 0.005:
        return f"ne lui donne pratiquement aucune chance (<{NBSP}1{NBSP}%)"
    return f"ne lui donne que {_pct(v)}"


def _rel(x) -> str:
    v = _f(x)
    if v is None:
        return "×—"
    return "×" + _num(v, 2 if abs(v) < 0.1 else 1)


def _mm(x, nd: int | None = None) -> str:
    v = _f(x)
    if v is None:
        return "—"
    if nd is None:
        nd = 0 if abs(v) >= 20 else (1 if abs(v) >= 1 else 2)
    return f"{_num(v, nd)}{NBSP}mm"


def _small_mm(x) -> str:
    v = _f(x)
    if v is None:
        return "—"
    if v < 0.001:
        return f"<{NBSP}0,001{NBSP}mm"
    if v < 0.1:
        return f"{_num(v, 3)}{NBSP}mm"
    return f"{_num(v, 1 if v >= 1 else 2)}{NBSP}mm"


def _temp(t) -> str:
    return f"{_num(t, 0)}{NBSP}°C"


def _join(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " et " + parts[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip(" ,;:·—") + "…"


def _esc(s) -> str:
    """Échappement d'un texte (contenu d'élément)."""
    return _html.escape(str(s), quote=False)


def _esc_attr(s) -> str:
    """Échappement d'une valeur d'attribut (href…)."""
    return _html.escape(str(s), quote=True)


def _h4(t: str) -> str:
    return f'<h4 style="color:{_HEAD}; margin-top:12px; margin-bottom:4px">{_esc(t)}</h4>'


def _hex(c: str) -> str:
    return physics.CLASS_INFO.get(c, physics.CLASS_INFO["incertain"])["hex"]


_TEXT_HEX_CACHE: dict[str, str] = {}


def _text_hex(c: str) -> str:
    """Couleur de classe assombrie (même teinte) pour un texte lisible sur fond blanc (contraste ≥ 4,5:1)."""
    hx = _hex(c)
    if hx not in _TEXT_HEX_CACHE:
        _TEXT_HEX_CACHE[hx] = _darken_for_white(hx)
    return _TEXT_HEX_CACHE[hx]


def _darken_for_white(hx: str, target: float = 4.6) -> str:
    import colorsys

    def lum(rgb):
        def ch(v):
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = (ch(x) for x in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    rgb = tuple(int(hx[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    for _ in range(40):
        if 1.05 / (lum(rgb) + 0.05) >= target:
            break
        v = max(0.0, v - 0.03)
        rgb = colorsys.hsv_to_rgb(h, min(1.0, s * 1.02), v)
    return "#" + "".join(f"{round(x * 255):02x}" for x in rgb)


def _nom(c: str) -> str:
    return physics.CLASS_INFO.get(c, {}).get("nom", _NAMES.get(c, (c,))[0])


def _short(c: str) -> str:
    return _NAMES.get(c, (c, c, c, "?"))[0]


def _un(c: str) -> str:
    return _NAMES.get(c, (c, c, c, "?"))[1]


def _not(c: str) -> str:
    return _NAMES.get(c, (c, c, f"Pas {c}", "?"))[2]


def _sym(c: str) -> str:
    return _NAMES.get(c, (c, c, c, "?"))[3]


def _conf_label(method: str) -> str:
    return "Confiance de l'analyse automatique" if str(method).startswith("Correction manuelle") else "Confiance"


def _scores_title(method: str) -> str:
    m = str(method)
    if "IA locale" in m:
        return "Probabilités de l'IA locale"
    if "IA de référence" in m or "hybride" in m.lower() or "calibr" in m.lower():
        return "Probabilités (règles + IA de référence)"
    return "Scores relatifs des règles (normalisés, pas des probabilités calibrées)"


def _ordered_scores(scores: dict) -> list[tuple[str, float]]:
    order = list(BASE_CLASSES) + ["artefact"]
    out = [(k, float(scores[k])) for k in order if k in scores and _f(scores[k]) is not None]
    out += [(k, float(v)) for k, v in scores.items() if k not in order and k != "incertain" and _f(v) is not None]
    return out


def _split_source(s: str) -> tuple[str, str]:
    title, sep, url = str(s).rpartition(" — ")
    if sep and url.startswith("http"):
        return title, url
    return str(s), ""


def _ramp(x: float, a: float, b: float) -> float:
    if b == a:
        return 1.0 if x >= b else 0.0
    return float(min(1.0, max(0.0, (x - a) / (b - a))))


def _verdict(support: float, hi: float = 0.6, lo: float = 0.4) -> tuple[str, float]:
    s = min(1.0, max(0.0, float(support)))
    v = "pour" if s >= hi else ("contre" if s <= lo else "neutre")
    return v, round(abs(s - 0.5) * 2.0, 3)


# --------------------------------------------------------------------------- #
# Physique utilisée dans les phrases
# --------------------------------------------------------------------------- #
def _air_density(temp_c: float, p_kpa: float) -> float:
    return physics.AIR_DENSITY_15C_G_CM3 * physics.air_density_factor(temp_c, p_kpa)


def _highland_lateral_mm(t_kin_mev: float, mass_mev: float, x_mm: float, temp_c: float, p_kpa: float) -> float:
    """Écart latéral rms (mm) dû à la diffusion multiple (PDG, éq. 34.16 et 34.20 : y = x·θ0/√3)."""
    if not x_mm or x_mm <= 0:
        return 0.0
    try:
        x0_mm = _X0_AIR_G_CM2 / _air_density(temp_c, p_kpa) * 10.0
        e = t_kin_mev + mass_mev
        p = math.sqrt(max(e * e - mass_mev * mass_mev, 1e-12))
        beta = p / e
        xr = x_mm / x0_mm
        th = 13.6 / (beta * p) * math.sqrt(xr) * (1.0 + 0.038 * math.log(xr / beta ** 2))
        return max(0.0, x_mm * th / math.sqrt(3.0))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _alpha_range_mm(e_mev: float, temp_c: float, p_kpa: float) -> float | None:
    try:
        return _f(physics.alpha_range_astar_cm(e_mev, temp_c, p_kpa) * 10.0)
    except Exception:
        return _f(physics.alpha_range_cm(e_mev, temp_c, p_kpa) * 10.0)


def _muon_fraction_longer(x_mm: float, depth_mm: float) -> float:
    """Fraction des muons (cos²θ, couche d'épaisseur d) dont la projection dépasse x : (1 + (x/d)²)⁻²."""
    d = max(float(depth_mm), 0.1)
    return (1.0 + (x_mm / d) ** 2) ** -2


# --------------------------------------------------------------------------- #
# Informations du modèle de référence (chargées une seule fois)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _model_info() -> dict:
    info: dict = {}
    try:
        from .reference_model import ReferenceModel
        m = ReferenceModel.load_default()
        if m is not None:
            info = dict(m.info)
            info["features"] = list(m.features)
            met = info.get("metrics") or {}
            info["bal_acc"] = met.get("test_hybrid_balanced_accuracy")
    except Exception:
        info = {}
    if not info.get("n_tracks_total"):
        try:
            from .reference_model import DEFAULT_PATH
            d = json.loads(DEFAULT_PATH.with_suffix(".json").read_text(encoding="utf-8"))
            info.setdefault("n_tracks_total", d.get("n_tracks_total"))
            info.setdefault("w_ml", d.get("w_ml"))
            info.setdefault("features", d.get("features"))
            hy = ((d.get("metrics") or {}).get("test") or {}).get("hybrid") or {}
            info.setdefault("bal_acc", hy.get("balanced_accuracy"))
        except Exception:
            pass
    return info


def _model_features() -> set:
    feats = _model_info().get("features")
    if feats:
        return set(feats)
    try:
        from .classifier import ML_FEATURES
        from .reference_model import EXTRA_FEATURES
        return set(ML_FEATURES) | set(EXTRA_FEATURES)
    except Exception:
        return set()


def _local_ai_features() -> set:
    try:
        from .classifier import ML_FEATURES
        return set(ML_FEATURES)
    except Exception:
        return set()


def clear_cache() -> None:
    """Oublie les informations du modèle de référence (après un réentraînement)."""
    _model_info.cache_clear()


# --------------------------------------------------------------------------- #
# Paramètres et valeurs intermédiaires des règles
# --------------------------------------------------------------------------- #
def _params(obj, klass):
    """ClassificationParams / PhysicsParams à partir d'un objet, d'un dict ou de None."""
    if isinstance(obj, klass):
        return obj
    out = klass()
    if obj is None:
        return out
    if isinstance(obj, dict):
        src = obj
    elif is_dataclass(obj):
        src = {fl.name: getattr(obj, fl.name) for fl in fields(obj)}
    else:
        src = {fl.name: getattr(obj, fl.name) for fl in fields(klass) if hasattr(obj, fl.name)}
    for fl in fields(klass):
        if fl.name in src:
            try:
                setattr(out, fl.name, src[fl.name])
            except Exception:
                pass
    return out


def _pf(obj, name: str, default: float) -> float:
    v = _f(getattr(obj, name, default))
    return default if v is None else v


_EV_KEYS = ("ev_thick", "ev_dense", "ev_heavy", "ev_straight_m", "ev_straight_a", "ev_curly", "ev_long",
            "ev_short_a", "ev_sh", "ev_sag", "ev_wig", "ev_lat", "ev_refs_ready", "ev_short_straight")


def _rebuild_ev(f: dict, cp, calibrated: bool) -> dict | None:
    """Recalcule les valeurs ev_* d'une trace d'une ancienne version (sur une COPIE des mesures)."""
    try:
        from .classifier import References, classify_rules
    except Exception:
        return None
    g = dict(f)
    L = _f(g.get("length_mm"))
    if L is None:
        return None
    W = _f(g.get("width_mm")) or 0.0
    wpx = _f(g.get("width_px"))
    lpx = _f(g.get("length_px"))
    if wpx and W > 0:
        ppm = wpx / W
    elif lpx and L > 0:
        ppm = lpx / L
    else:
        ppm = 960.0 / 250.0
    if not wpx or wpx <= 0:
        wpx = W * ppm if W > 0 else 1.0
    g.update(length_mm=L, width_mm=W, width_px=wpx,
             straightness=_f(g.get("straightness")) if _f(g.get("straightness")) is not None else 1.0,
             tortuosity=_f(g.get("tortuosity")) or 0.0)
    for k in ("sagitta_ratio", "line_rms_mm", "lateral_mm"):
        g[k] = _f(g.get(k)) or 0.0
    wr, dr, dens = _f(g.get("width_rel")), _f(g.get("density_rel")), _f(g.get("density"))
    fallback_refs = False
    if wr and wr > 0 and dr and dr > 0:
        if not dens or dens <= 0:
            dens = dr
            g["density"] = dens
        refs = References(width_px=wpx / wr, density=dens / dr, n=30, method="reconstruit")
        fallback_refs = abs(dr - 0.9 * wr) < 1e-6           # référence par défaut des règles
    else:
        g["density"] = dens if dens and dens > 0 else 1.0
        refs = References()
    try:
        _cls, _conf, scores, _reason = classify_rules(g, refs, cp, bool(calibrated), ppm)
    except Exception:
        return None
    out = {k: g[k] for k in _EV_KEYS if k in g}
    if fallback_refs:
        out["ev_refs_ready"] = 0.0
    out["width_rel"] = g.get("width_rel")
    out["density_rel"] = g.get("density_rel")
    out["_rules_scores"] = scores
    return out


def _rule_scores_from_ev(ev: dict) -> dict:
    heavy = ev.get("heavy", 0.0)
    light = 1.0 - heavy
    sa, sm = ev.get("straight_a", 0.0), ev.get("straight_m", 0.0)
    lg, sha, curly = ev.get("long", 0.0), ev.get("short_a", 1.0), ev.get("curly", 0.0)
    return {"alpha": heavy * sa * sha, "proton": heavy * sa * (1.0 - sha),
            "muon": light * sm * (0.3 + 0.7 * lg), "beta": light * max(curly, 0.45 * (1.0 - lg) * (1.0 - sm))}


def _clean_scores(d) -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            x = _f(v)
            if x is not None:
                out[str(k)] = max(0.0, x)
    return out


def _normalise(d: dict, keys=None) -> dict:
    keys = [k for k in (keys or d.keys()) if k in d and k != "incertain"]
    s = sum(d[k] for k in keys)
    if s <= 0:
        return {}
    return {k: d[k] / s for k in keys}


def _argmax(d: dict, keys=None) -> str | None:
    keys = [k for k in (keys or d.keys()) if k in d]
    return max(keys, key=lambda k: d[k]) if keys else None


# --------------------------------------------------------------------------- #
# Contexte d'une explication
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, f, cls, confidence, scores, cp, php, duration_s, touches_border, calibrated,
                 manual_cls, method, ml_proba, rules_cls, rules_scores, abstained, threshold, family, capped):
        self.cp = _params(cp, ClassificationParams)
        self.php = _params(php, PhysicsParams)
        self.dp = DetectionParams()
        f = dict(f) if isinstance(f, dict) else {}
        self.calibrated = bool(calibrated)
        self.rebuilt = False
        rebuilt_scores = None
        if "ev_heavy" not in f:
            rb = _rebuild_ev(f, self.cp, self.calibrated)
            if rb is not None:
                rebuilt_scores = rb.pop("_rules_scores", None)
                for k, v in rb.items():
                    if k.startswith("ev_") or f.get(k) is None:
                        f[k] = v
                self.rebuilt = True
        self.F = f
        g = self._g
        self.has_ev = "ev_heavy" in f
        # --- mesures ---
        self.L, self.W = g("length_mm"), g("width_mm")
        self.w_rel, self.d_rel = g("width_rel"), g("density_rel")
        self.S, self.T = g("straightness"), g("tortuosity")
        self.sag, self.rms, self.lat = g("sagitta_ratio"), g("line_rms_mm"), g("lateral_mm")
        self.elong, self.snr = g("elongation"), g("peak_snr")
        self.asym, self.branches = g("asymmetry"), g("branches")
        wpx = g("width_px")
        self.mmpp = (self.W / wpx) if (self.W and wpx and wpx > 0) else None
        self.dur = _f(duration_s)
        self.border = bool(touches_border)
        # --- valeurs floues des règles ---
        e = lambda k, d=None: _f(f.get(k)) if _f(f.get(k)) is not None else d   # noqa: E731
        self.ev = {"thick": e("ev_thick"), "dense": e("ev_dense"), "heavy": e("ev_heavy"),
                   "straight_m": e("ev_straight_m"), "straight_a": e("ev_straight_a"), "curly": e("ev_curly"),
                   "long": e("ev_long"), "short_a": e("ev_short_a"), "sh": e("ev_sh")}
        self.refs_ready = e("ev_refs_ready", 1.0) >= 0.5
        self.short_straight = e("ev_short_straight", 0.0) >= 0.5
        if self.ev["sh"] is None and self.L is not None:
            self.ev["sh"] = _ramp(self.L, *_SH)
        self.ev_ok = all(self.ev[k] is not None for k in ("heavy", "straight_m", "straight_a", "curly", "long",
                                                           "short_a"))
        # --- paramètres ---
        cp, php = self.cp, self.php
        self.hwr = _pf(cp, "heavy_width_rel", 1.7)
        self.hdr = _pf(cp, "heavy_density_rel", 2.2)
        self.awm = _pf(cp, "alpha_width_mm", 1.3)
        self.amax = _pf(cp, "alpha_max_length_mm", 75.0)
        self.mml = _pf(cp, "muon_min_length_mm", 45.0)
        self.smin = _pf(cp, "straight_min", 0.95)
        self.cmax = _pf(cp, "curly_max", 0.86)
        self.min_conf = _pf(cp, "min_confidence", 0.30)
        self.tc = _pf(php, "chamber_temp_c", -25.0)
        self.pk = _pf(php, "pressure_kpa", 101.3)
        self.depth = _pf(php, "sensitive_depth_mm", 8.0)
        self.fov = _pf(php, "default_fov_mm", 250.0)
        self.bfield = _pf(php, "b_field_mt", 0.0)
        self.alpha_max_phys = _alpha_range_mm(_PO212_MEV, self.tc, self.pk) or 74.0
        # --- décision ---
        self.method = str(method).strip() if method and str(method).strip() not in ("None", "nan") else \
            "Règles physiques"
        self.ml = _normalise(_clean_scores(ml_proba)) if ml_proba else {}
        self.mode = "hybride" if self.ml else ("locale" if "IA locale" in self.method else "regles")
        sc = _clean_scores(scores)
        rs = _clean_scores(rules_scores)
        if not rs:
            if self.mode == "regles" and any(k in sc for k in BASE_CLASSES):
                rs = {k: sc.get(k, 0.0) for k in BASE_CLASSES}
            elif rebuilt_scores:
                rs = _clean_scores(rebuilt_scores)
            elif self.ev_ok:
                rs = _rule_scores_from_ev(self.ev)
        self.rs = rs
        self.rules_best = rules_cls if rules_cls in BASE_CLASSES else _argmax(rs, BASE_CLASSES)
        self.conf = _f(confidence) or 0.0
        self.threshold = _f(threshold) if _f(threshold) is not None else 0.85
        self.abstained = bool(abstained)
        self.family = str(family or "").strip()
        cls = str(cls or "incertain").strip() or "incertain"
        self.manual = str(manual_cls or "").strip()
        self.final = self.manual or cls
        if self.manual and cls == self.manual:
            # l'appelant a passé la classe finale : retrouver la classe automatique
            base = _argmax(sc, list(BASE_CLASSES) + ["artefact"])
            self.auto = "incertain" if self.abstained or base is None else base
        else:
            self.auto = cls
        # probabilités affichées (barres)
        keys = list(BASE_CLASSES) + ["artefact"]
        if self.mode == "regles":
            self.probs = _normalise(rs, BASE_CLASSES) or _normalise(sc, keys)
        else:
            self.probs = _normalise(sc, keys) or _normalise(self.ml, keys) or _normalise(rs, BASE_CLASSES)
        self.artefact_considered = "artefact" in self.ml or "artefact" in sc
        if capped is None:
            capped = (self.short_straight and self.final in ("muon", "beta") and not self.manual
                      and abs(self.conf - _AMBIGUITY_CAP) < 0.005)
        self.capped = bool(capped)
        # hypothèses concurrentes (classe incertaine)
        order = sorted(self.probs, key=self.probs.get, reverse=True)
        self.c1 = order[0] if order else (self.rules_best or "muon")
        self.c2 = order[1] if len(order) > 1 else None
        self.y_e13 = _highland_lateral_mm(_K40_EMAX, physics.ELECTRON_MASS_MEV, self.L or 20.0, self.tc, self.pk)
        self.y_e05 = _highland_lateral_mm(0.5, physics.ELECTRON_MASS_MEV, self.L or 20.0, self.tc, self.pk)
        self.y_mu = _highland_lateral_mm(_MUON_MEV, physics.MUON_MASS_MEV, self.L or 20.0, self.tc, self.pk)

    def _g(self, k):
        return _f(self.F.get(k))

    # ------------------------------------------------------------------ outils --
    @property
    def ai_feats(self) -> set:
        if self.mode == "hybride":
            return _model_features()
        if self.mode == "locale":
            return _local_ai_features()
        return set()

    def p(self, c: str) -> float:
        return float(self.probs.get(c, 0.0))

    def length_txt(self) -> str:
        if self.L is None:
            return "longueur non mesurée"
        return _mm(self.L) + (" (minimum : touche le bord)" if self.border else "")

    def is_long(self) -> bool:
        sh = self.ev.get("sh")
        return (sh if sh is not None else 0.0) >= 0.5


# --------------------------------------------------------------------------- #
# Critères (verdict par rapport à la classe expliquée « t »)
# --------------------------------------------------------------------------- #
def _crit_thickness(x: _Ctx, t: str):
    heavy_t = t in ("alpha", "proton")
    thick = x.ev.get("thick")
    if x.w_rel is None or thick is None:
        c = Criterion("thickness", "Épaisseur (par rapport aux traces fines)", "non mesurée", "—", "neutre", 0.0,
                      "Sans mesure d'épaisseur, l'ionisation ne peut pas être jugée.")
        return c, "", ""
    support = thick if heavy_t else 1.0 - thick
    v, s = _verdict(support)
    meas = f"{_rel(x.w_rel)} la référence"
    if x.calibrated and x.W is not None:
        meas += f" ({_num(x.W, 2)}{NBSP}mm)"
    if not x.refs_ready:
        meas += f" ; référence provisoire (largeur type {_num(_REF_WIDTH_MM, 1)}{NBSP}mm)"
    if heavy_t:
        exp = f"≥ {_rel(x.hwr)} la référence" + (f" ou ≥ {_num(x.awm, 1)}{NBSP}mm" if x.calibrated else "")
        if t == "alpha":
            expl = ("Formule de Bethe : l'ionisation par millimètre croît comme z²/β². Chargée deux fois et lente, "
                    "une alpha ionise l'air ≈ 300 à 400 fois plus qu'un muon : davantage de gouttelettes, donc une "
                    "trace plus large.")
        else:
            expl = ("Formule de Bethe (z²/β²) : un proton lent de quelques MeV ionise ≈ 50 fois plus qu'une "
                    "particule au minimum d'ionisation, d'où une trace nettement plus large que celle d'un muon.")
    else:
        exp = f"≤ {_rel(0.8 * x.hwr)} la référence (trace fine)"
        expl = ("Une particule rapide de charge 1 (muon, électron énergétique) est proche du minimum d'ionisation "
                "(≈ 0,26 keV/mm dans l'air froid) : peu de gouttelettes, trace fine.")
    if heavy_t:
        word = ("très épaisse" if thick >= 0.85 else "épaisse") if v == "pour" else "peu épaisse"
    else:
        word = ("fine" if thick <= 0.1 else "assez fine") if v == "pour" else "épaisse"
    c = Criterion("thickness", "Épaisseur (par rapport aux traces fines)", meas, exp, v, s, expl)
    return c, f"{word} ({_rel(x.w_rel)} la référence)", f"{word} {_rel(x.w_rel)}"


def _crit_density(x: _Ctx, t: str):
    heavy_t = t in ("alpha", "proton")
    dense = x.ev.get("dense")
    if x.d_rel is None or dense is None:
        c = Criterion("density", "Densité d'ionisation (gouttelettes par mm)", "non mesurée", "—", "neutre", 0.0,
                      "Sans mesure de densité, l'ionisation ne peut pas être jugée.")
        return c, "", ""
    support = dense if heavy_t else 1.0 - dense
    v, s = _verdict(support)
    meas = f"{_rel(x.d_rel)} la référence"
    if heavy_t:
        exp = f"≥ {_rel(x.hdr)} la référence"
        expl = ("Le nombre de gouttelettes par millimètre mesure directement l'ionisation (énergie déposée par mm) ; "
                "c'est le critère le plus lourd des règles (65 % de l'indice « particule lourde »).")
    else:
        exp = f"≤ {_rel(0.75 * x.hdr)} la référence"
        expl = ("Peu de gouttelettes par millimètre : faible perte d'énergie, propre à une particule rapide de "
                "charge 1 (muon ou électron).")
    if heavy_t:
        word = ("très dense" if dense >= 0.85 else "dense") if v == "pour" else "peu dense"
    else:
        word = ("peu dense" if dense <= 0.1 else "moyennement dense") if v == "pour" else "dense"
    ph = f"{word} en gouttelettes ({_rel(x.d_rel)})" if word.endswith("dense") and heavy_t and v == "pour" \
        else f"{word} ({_rel(x.d_rel)})"
    return (Criterion("density", "Densité d'ionisation (gouttelettes par mm)", meas, exp, v, s, expl), ph,
            f"{word} {_rel(x.d_rel)}")


def _crit_shape(x: _Ctx, t: str):
    label = "Forme (rectitude)"
    long_ = x.is_long() or x.lat is None
    S = x.S
    if t in ("alpha", "proton"):
        sup = x.ev.get("straight_a")
        if sup is None or S is None:
            return Criterion("shape", label, "non mesurée", "—", "neutre", 0.0,
                             "Forme non mesurée."), "", ""
        v, s = _verdict(sup)
        if long_:
            meas, exp = f"rectitude {_num(S, 3)}", f"≥ {_num(_S_ALPHA[1], 2)} (corde / chemin)"
            ph = f"{'droite' if sup >= 0.5 else 'courbée'} (rectitude {_num(S, 3)})"
        else:
            meas = f"étalement latéral {_small_mm(x.lat)} (rectitude {_num(S, 3)})"
            exp = (f"droite sous {_num(_LAT_STRAIGHT_A[0], 2)}{NBSP}mm (courbée au-delà de "
                   f"{_num(_LAT_STRAIGHT_A[1], 2)}{NBSP}mm) ou rectitude ≥ {_num(_S_ALPHA[1], 2)}")
            ph = f"{'droite' if sup >= 0.5 else 'courbée'} (étalement latéral {_small_mm(x.lat)})"
        if t == "alpha":
            expl = ("Une alpha est ≈ 7 300 fois plus lourde qu'un électron : les collisions la dévient à peine "
                    "(NIST ASTAR : parcours projeté ≈ 0,99 × parcours réel). Sa trace est droite.")
        else:
            expl = ("Un proton est 1 836 fois plus lourd qu'un électron : les collisions le dévient très peu, sa "
                    "trace est droite.")
        return Criterion("shape", label, meas, exp, v, s, expl), ph, ph.split(" (")[0]
    if t == "muon":
        sup = x.ev.get("straight_m")
        if sup is None:
            return Criterion("shape", label, "non mesurée", "—", "neutre", 0.0, "Forme non mesurée."), "", ""
        v, s = _verdict(sup)
        if long_ and S is not None:
            meas = (f"rectitude {_num(S, 3)} · flèche {_num(100 * (x.sag or 0.0), 1)}{NBSP}% de la longueur · "
                    f"écart RMS {_small_mm(x.rms)}")
            exp = (f"rectitude ≥ {_num(x.smin, 2)}, flèche ≤ {_num(100 * _SAG_STRAIGHT[0], 1)}{NBSP}%, "
                   f"écart RMS ≤ {_num(100 * _WIG_STRAIGHT[0], 1)}{NBSP}% de la longueur")
            ph = f"{'parfaitement droite' if sup >= 0.5 else 'pas assez droite'} (rectitude {_num(S, 3)})"
        else:
            meas = f"étalement latéral {_small_mm(x.lat)}"
            exp = (f"droite sous {_num(_LAT_STRAIGHT[0], 2)}{NBSP}mm, courbée au-delà de "
                   f"{_num(_LAT_STRAIGHT[1], 2)}{NBSP}mm")
            ph = f"{'droite' if sup >= 0.5 else 'pas assez droite'} (étalement latéral {_small_mm(x.lat)})"
        expl = (f"Un muon de ≈ 4 GeV n'est presque pas dévié : sur {_mm(x.L or 20.0)} d'air, la diffusion multiple "
                f"(formule de Highland, PDG) l'écarte d'environ {_small_mm(x.y_mu)} de la droite, contre "
                f"≈ {_small_mm(x.y_e13)} pour un électron de 1,31 MeV (ordre de grandeur).")
        return Criterion("shape", label, meas, exp, v, s, expl), ph, ph.split(" (")[0]
    # électron
    curly = x.ev.get("curly")
    if curly is None:
        return Criterion("shape", "Forme (sinuosité)", "non mesurée", "—", "neutre", 0.0,
                         "Forme non mesurée."), "", ""
    lg, sm = x.ev.get("long") or 0.0, x.ev.get("straight_m") or 0.0
    floor = 0.45 * (1.0 - lg) * (1.0 - sm)
    if floor > curly and floor >= 0.15:
        sup = min(1.0, floor / 0.45)
        v, s = _verdict(sup)
        meas = f"{x.length_txt()}, pas nettement droite (indice de rectitude {_num(sm, 2)} sur 1)"
        exp = f"courte (< {_mm(x.mml)}) et pas parfaitement droite"
        expl = ("Règle prudente : une trace fine et courte qui n'est pas nettement droite est le plus souvent un "
                "électron ; les muons presque verticaux donnent des traces courtes mais parfaitement droites.")
        ph = "courte et pas nettement droite"
        return Criterion("shape", "Forme (courte, pas nettement droite)", meas, exp, v, s, expl), ph, ph
    v, s = _verdict(curly)
    if long_ and S is not None:
        parts = [(1.0 - _ramp(S, x.cmax, x.smin), f"rectitude {_num(S, 3)}"),
                 (_ramp(x.sag or 0.0, *_SAG_CURLY), f"flèche {_num(100 * (x.sag or 0.0), 1)}{NBSP}%"),
                 (_ramp((x.rms or 0.0) / max(x.L or 1.0, 1.0), *_WIG_CURLY),
                  f"écart RMS {_small_mm(x.rms)}"),
                 (_ramp(x.T or 0.0, *_TORT_CURLY), f"tortuosité {_num(x.T, 2)}{NBSP}rad/mm")]
        parts.sort(key=lambda p: -p[0])
        meas = " · ".join(p[1] for p in parts[:2])
        exp = (f"rectitude < {_num(x.cmax, 2)}, flèche > {_num(100 * _SAG_CURLY[0], 0)}{NBSP}% ou tortuosité > "
               f"{_num(_TORT_CURLY[0], 2)}{NBSP}rad/mm")
        ph = f"{'sinueuse' if curly >= 0.5 else 'peu sinueuse'} ({parts[0][1]})"
    else:
        meas = f"étalement latéral {_small_mm(x.lat)}"
        exp = (f"courbée au-delà de {_num(_LAT_CURLY[1], 2)}{NBSP}mm (droite sous "
               f"{_num(_LAT_CURLY[0], 2)}{NBSP}mm)")
        ph = f"{'courbée' if curly >= 0.5 else 'peu courbée'} (étalement latéral {_small_mm(x.lat)})"
    expl = ("Un électron est léger : chaque collision le dévie (diffusion multiple, formule de Highland, PDG). Sur "
            "1 cm d'air froid, un électron de 50 keV dévie de 35 à 50°, un de 0,5 MeV de 4 à 6° : trace sinueuse, "
            "de plus en plus enroulée en fin de parcours.")
    return Criterion("shape", "Forme (sinuosité)", meas, exp, v, s, expl), ph, ph.split(" (")[0]


def _crit_length(x: _Ctx, t: str):
    label = "Longueur"
    if x.L is None:
        return Criterion("length", label, "non mesurée", "—", "neutre", 0.0, "Longueur non mesurée."), "", ""
    phys = x.alpha_max_phys
    if t in ("alpha", "proton"):
        sa = x.ev.get("short_a")
        if sa is None:
            sa = 1.0 - _ramp(x.L, x.amax, 1.3 * x.amax)
        if t == "alpha":
            v, s = _verdict(sa)
            exp = (f"≤ {_mm(x.amax)} (portée maximale d'une alpha naturelle ≈ {_mm(phys)} à {_temp(x.tc)})")
            expl = (f"Une alpha dépose toute son énergie en quelques centimètres d'air, de plus en plus vite en fin de "
                    f"parcours (pic de Bragg). Même la plus énergétique des alphas naturelles (Po-212, 8,78 MeV) "
                    f"s'arrête après ≈ {_num(phys / 10.0, 1)}{NBSP}cm dans l'air à {_temp(x.tc)} (NIST ASTAR).")
            ph = f"longue de {_mm(x.L)} (≤ {_mm(x.amax)})" if sa >= 0.5 else f"trop longue ({_mm(x.L)})"
            sh = f"{_mm(x.L)} ≤ {_mm(x.amax)}" if sa >= 0.5 else f"{_mm(x.L)} > {_mm(x.amax)}"
        else:
            v, s = _verdict(1.0 - sa)
            exp = f"> {_mm(x.amax)} (plus long que toute alpha naturelle, ≈ {_mm(phys)})"
            expl = ("Un proton lent parcourt bien plus d'air qu'une alpha de même ionisation (≈ 12 cm à 3 MeV dans "
                    "l'air froid, NIST PSTAR) : une trace épaisse plus longue que toute alpha naturelle désigne un "
                    "proton.")
            ph = f"longue de {_mm(x.L)} (> {_mm(x.amax)})" if sa < 0.5 else f"courte pour un proton ({_mm(x.L)})"
            sh = f"{_mm(x.L)} > {_mm(x.amax)}" if sa < 0.5 else f"{_mm(x.L)}"
        return Criterion("length", label, x.length_txt(), exp, v, s, expl), ph, sh
    if t == "muon":
        lg = x.ev.get("long")
        if lg is None:
            lg = _ramp(x.L, 0.65 * x.mml, x.mml)
        v, s = _verdict(lg)
        exp = f"≥ {_mm(x.mml)} pour une longue droite (muon incliné)"
        expl = (f"Les muons arrivent surtout à la verticale (loi en cos²θ) et la couche sensible ne fait qu'environ "
                f"{_mm(x.depth, 0)} : la plupart laissent des traces courtes. Une trace courte et droite peut être un "
                f"muon presque vertical ou un électron énergétique : les règles n'affirment « muon » qu'au-delà "
                f"de {_mm(x.mml)}.")
        ph = f"longue de {_mm(x.L)}" if lg >= 0.5 else f"courte ({_mm(x.L)})"
        return Criterion("length", label, x.length_txt(), exp, v, s, expl), ph, ph
    return None, "", ""


def _crit_ai(x: _Ctx, t: str):
    probs = x.ml
    if not probs:
        return None
    info = _model_info()
    n = _f(info.get("n_tracks_total"))
    label = (f"IA de référence (entraînée sur {_num(n, 0)} traces simulées)" if n and n > 0
             else "IA de référence (entraînée sur des traces simulées)")
    meas = " · ".join(f"{_sym(k)} {_pct(v)}" for k, v in _ordered_scores(probs))
    best = _argmax(probs)
    pt = probs.get(t, 0.0)
    v = "pour" if best == t else ("contre" if pt < 0.25 else "neutre")
    s = round(abs(pt - 0.5) * 2.0, 3)
    w = _f(getattr(x.cp, "ref_model_weight", -1.0))
    if w is None or w < 0:
        w = _f(info.get("w_ml"))
    nfeat = len(info.get("features") or []) or len(_model_features())
    expl = (f"Gradient boosting appris sur des vidéos simulées passées dans le vrai détecteur ; il combine "
            f"{nfeat} mesures (épaisseur, densité, forme, durée…)"
            + (f" et pèse {_pct(w)} dans la décision hybride." if w is not None else "."))
    return Criterion("reference_ai", label, meas, f"{_sym(t)} en tête", v, s, expl)


def _crit_local_ai(x: _Ctx, t: str):
    if x.mode != "locale" or not x.probs:
        return None
    meas = " · ".join(f"{_sym(k)} {_pct(v)}" for k, v in _ordered_scores(x.probs))
    pt = x.p(t)
    best = _argmax(x.probs)
    v = "pour" if best == t else ("contre" if pt < 0.25 else "neutre")
    return Criterion("local_ai", "IA locale (entraînée sur vos corrections)", meas, f"{_sym(t)} en tête", v,
                     round(abs(pt - 0.5) * 2.0, 3),
                     "Forêt aléatoire entraînée sur les traces que vous avez étiquetées ; dans ce mode, c'est elle qui "
                     "décide à partir des mêmes mesures.")


def _used_by_ai(x: _Ctx, feature: str) -> bool:
    return feature in x.ai_feats


def _ai_note(x: _Ctx, feature: str) -> str:
    if _used_by_ai(x, feature):
        who = "l'IA de référence" if x.mode == "hybride" else "l'IA locale"
        return f" Mesure utilisée par {who}, pas par les règles."
    return ""


def _crit_duration(x: _Ctx, t: str):
    if x.dur is None or x.dur <= 0:
        return None
    if t in ("alpha", "proton"):
        sup = _ramp(x.dur, 0.5, 1.1)
        exp = (f"≈ {_num(_SUSSMAN_ALPHA_S, 1)}{NBSP}s pour une alpha (Sussman)" if t == "alpha" else
               f"longue, comme une trace lourde (alphas : ≈ {_num(_SUSSMAN_ALPHA_S, 1)}{NBSP}s, Sussman)")
    elif t in ("muon", "beta"):
        sup = 1.0 - _ramp(x.dur, 0.7, 1.3)
        exp = "≈ 0,3 à 0,4 s pour une trace fine (Sussman)"
    else:
        return None
    v, s = _verdict(sup)
    expl = ("Une trace dense en gouttelettes reste visible plus longtemps : à 30 images/s, Sussman mesure "
            "1,5 ± 0,4 s pour les alphas contre ≈ 0,4 s (médiane 0,27 s) pour les traces fines ; la durée dépend "
            "beaucoup de la chambre." + _ai_note(x, "duration_s"))
    return Criterion("duration", "Durée de visibilité", f"{_num(x.dur, 2)}{NBSP}s", exp, v, s, expl,
                     used_for_decision=_used_by_ai(x, "duration_s"))


def _crit_bragg(x: _Ctx, t: str):
    if t != "alpha" or x.asym is None or (x.L or 0.0) < 10.0:
        return None
    sup = _ramp(x.asym, 1.15, 1.5)
    v = "pour" if sup >= 0.6 else "neutre"
    expl = ("En ralentissant, une alpha perd de plus en plus d'énergie par millimètre (NIST ASTAR : 713 MeV·cm²/g "
            "à 5,5 MeV, ≈ 2 020 vers 0,7 MeV) : la trace s'épaissit vers sa fin. Absent si l'alpha sort de la couche "
            "sensible ou si l'image sature." + _ai_note(x, "asymmetry"))
    return Criterion("bragg", "Épaississement à une extrémité (pic de Bragg)",
                     f"une extrémité {_rel(x.asym)} plus intense que l'autre",
                     "extrémité finale plus dense (pouvoir d'arrêt jusqu'à ≈ ×2,8)", v,
                     round(sup if v == "pour" else 0.0, 3), expl, used_for_decision=_used_by_ai(x, "asymmetry"))


def _crit_branches(x: _Ctx, t: str):
    n = int(x.branches) if x.branches is not None else 0
    if n <= 0:
        return None
    expl = ("Un embranchement peut être un rayon delta (électron arraché par la particule), un V du thoron (deux "
            "alphas partant du même point en moins d'une seconde : Rn-220 → Po-216, 0,14 s), un coude (muon qui se "
            "désintègre) ou deux traces qui se croisent." + _ai_note(x, "branches"))
    return Criterion("branches", "Embranchements", f"{n} embranchement{'s' if n > 1 else ''}",
                     "0 pour une trace simple", "neutre", 0.0, expl, used_for_decision=_used_by_ai(x, "branches"))


def _crits_artefact(x: _Ctx) -> tuple[list[Criterion], list[str]]:
    used = x.artefact_considered and not x.manual
    out, phrases = [], []
    min_snr = _pf(x.dp, "min_peak_snr", 6.0)
    if x.elong is not None:
        sup = 1.0 - _ramp(x.elong, 3.0, 6.0)
        v, s = _verdict(sup)
        out.append(Criterion("elongation", "Allongement (longueur / largeur)", _rel(x.elong),
                             "≥ ×4 pour une vraie traînée de gouttelettes", v, s,
                             "Une particule chargée laisse une traînée fine de gouttelettes le long de sa trajectoire ; "
                             "une bouffée de brume ou un amas de gouttelettes forme une tache presque ronde.", used))
        if sup >= 0.5:
            phrases.append(f"peu allongée ({_rel(x.elong)})")
    if x.snr is not None:
        sup = 1.0 - _ramp(x.snr, 8.0, 14.0)
        v, s = _verdict(sup)
        out.append(Criterion("contrast", "Contraste (en écarts-types du bruit)", f"{_num(x.snr, 0)}{NBSP}σ",
                             f"≥ {_num(2 * min_snr, 0)}{NBSP}σ pour une trace nette (seuil de détection "
                             f"{_num(min_snr, 0)}{NBSP}σ)", v, s,
                             "Une vraie trace naît en une fraction de seconde et se détache nettement du fond ; la "
                             "brume et les reflets diffus restent pâles.", used))
        if sup >= 0.5:
            phrases.append(f"peu contrastée ({_num(x.snr, 0)}{NBSP}σ)")
    if x.L is not None:
        sup = 1.0 - _ramp(x.L, 6.0, 12.0)
        v, s = _verdict(sup)
        out.append(Criterion("length", "Longueur", x.length_txt(), "≥ 10 mm environ pour une trace bien identifiable",
                             v, s, "Les détections très courtes sont le plus souvent des gouttes isolées ou du bruit.",
                             used))
        if sup >= 0.5:
            phrases.append(f"très courte ({_mm(x.L)})")
    if x.dur is not None and x.dur > 0:
        max_s = _pf(x.dp, "max_track_s", 4.0)
        if x.dur >= 0.9 * max_s:
            v, s, ph = "pour", 0.6, f"immobile ({_num(x.dur, 1)}{NBSP}s)"
        elif x.dur < 0.2:
            v, s, ph = "pour", 0.3, f"fugace ({_num(x.dur, 2)}{NBSP}s)"
        else:
            v, s, ph = "neutre", 0.0, ""
        out.append(Criterion("duration", "Durée de visibilité", f"{_num(x.dur, 2)}{NBSP}s",
                             "≈ 0,3 à 1,5 s pour une vraie trace", v, s,
                             "Un reflet ou une goutte sur le couvercle reste visible très longtemps sans bouger ; un "
                             "éclair de bruit ne dure qu'une image ou deux.", used))
        if ph:
            phrases.append(ph)
    ai = _crit_ai(x, "artefact")
    if ai is not None:
        out.append(ai)
    return out, phrases


# --------------------------------------------------------------------------- #
# « Pourquoi pas … ? »
# --------------------------------------------------------------------------- #
def _factors(x: _Ctx, c: str) -> list[tuple[float, str]]:
    """Facteurs du score des règles de la classe c qui sont sous 0,5, avec une phrase chiffrée.

    L'ionisation (critère le plus fiable) vient toujours en premier ; les autres facteurs sont
    triés du plus pénalisant au moins pénalisant.
    """
    ev = x.ev
    first: list[tuple[float, str]] = []
    out: list[tuple[float, str]] = []
    if not x.ev_ok:
        return out
    heavy = ev["heavy"]
    L = x.L if x.L is not None else 0.0
    if c in ("alpha", "proton"):
        if heavy < 0.5:
            parts = []
            if x.w_rel is not None and x.w_rel < x.hwr:
                parts.append(f"trace {_num(x.hwr / max(x.w_rel, 1e-3), 1)}× trop fine ({_rel(x.w_rel)} la référence ; "
                             f"{_un(c)} dépasse {_rel(x.hwr)})")
            if x.d_rel is not None and x.d_rel < x.hdr:
                parts.append(f"pas assez dense en gouttelettes ({_rel(x.d_rel)} ; il faut au moins {_rel(x.hdr)})")
            if not parts:
                parts.append(f"ionisation insuffisante (indice « particule lourde » {_num(heavy, 2)} sur 1)")
            first.append((heavy, " et ".join(parts)))
        sa = ev["straight_a"]
        if sa < 0.5:
            if x.is_long() and x.S is not None:
                out.append((sa, f"trace courbée (rectitude {_num(x.S, 3)}) alors qu'{_un(c)}, très lourde, reste "
                                f"droite (≥ {_num(_S_ALPHA[1], 2)})"))
            else:
                out.append((sa, f"trace courbée (étalement latéral {_small_mm(x.lat)}) alors qu'{_un(c)}, très "
                                f"lourde, va en ligne droite"))
        if c == "alpha" and ev["short_a"] < 0.5:
            out.append((ev["short_a"], f"{_mm(L)} de long alors qu'une alpha naturelle parcourt au plus "
                                       f"≈ {_mm(x.alpha_max_phys)} dans l'air à {_temp(x.tc)} (Po-212, 8,78 MeV, "
                                       f"NIST ASTAR)"))
        if c == "proton" and 1.0 - ev["short_a"] < 0.5:
            if heavy >= 0.5:
                out.append((1.0 - ev["short_a"],
                            f"{_mm(L)} seulement : une trace épaisse aussi courte s'explique par une alpha (portée "
                            f"≤ ≈ {_mm(x.alpha_max_phys)}) ; un proton n'est retenu qu'au-delà de ≈ {_mm(x.amax)}, et "
                            f"les protons cosmiques sont ≈ 100 fois plus rares que les muons"))
            else:
                out.append((1.0 - ev["short_a"], f"{_mm(L)} seulement (un proton lent dépasserait ≈ {_mm(x.amax)})"))
    else:
        light = 1.0 - heavy
        if light < 0.5:
            first.append((light, f"trace trop dense pour une particule au minimum d'ionisation (densité "
                                 f"{_rel(x.d_rel)}, épaisseur {_rel(x.w_rel)} la référence ; {_un(c)} laisse une trace "
                                 f"fine, ≈ 0,26 keV/mm)"))
        if c == "muon":
            sm = ev["straight_m"]
            if sm < 0.5:
                if x.is_long() and x.S is not None:
                    out.append((sm, f"trace pas assez droite (rectitude {_num(x.S, 3)}, flèche "
                                    f"{_num(100 * (x.sag or 0.0), 1)}{NBSP}% de la longueur) alors qu'un muon de "
                                    f"≈ 4 GeV ne s'écarte que d'environ {_small_mm(x.y_mu)} sur {_mm(L)} (Highland)"))
                else:
                    out.append((sm, f"étalement latéral {_small_mm(x.lat)} alors qu'un muon trace une droite "
                                    f"(courbée au-delà de {_num(_LAT_STRAIGHT[1], 2)}{NBSP}mm selon les règles)"))
            lg = ev["long"]
            if lg < 0.5 and not first and not out:
                out.append((0.3 + 0.7 * lg, f"trace courte ({_mm(L)} < {_mm(x.mml)}) : un muon presque vertical "
                                            f"reste possible, mais rien ne le distingue alors d'un électron énergétique"))
        if c == "beta":
            lg, sm = ev["long"], ev["straight_m"]
            shape = max(ev["curly"], 0.45 * (1.0 - lg) * (1.0 - sm))
            if shape < 0.5:
                visible = x.mmpp is None or x.y_e13 > x.mmpp
                if x.is_long() and x.S is not None and visible:
                    out.append((shape, f"trace droite (rectitude {_num(x.S, 3)}, flèche "
                                       f"{_num(100 * (x.sag or 0.0), 1)}{NBSP}%) sur {_mm(L)} : même un électron de "
                                       f"1,31 MeV, le plus énergétique du K-40, s'en écarterait d'environ "
                                       f"{_small_mm(x.y_e13)} (diffusion multiple, Highland)"))
                elif x.is_long() and x.S is not None:
                    out.append((shape, f"trace droite (rectitude {_num(x.S, 3)}) sur {_mm(L)} : aucun signe de "
                                       f"diffusion multiple"))
                else:
                    out.append((shape, f"trace courte mais droite (étalement latéral {_small_mm(x.lat)}) : aucun signe "
                                       f"de la diffusion multiple d'un électron lent"))
    out.sort(key=lambda p: p[0])
    return first + out


def _rule_score(x: _Ctx, c: str) -> float | None:
    return _f(x.rs.get(c)) if c in x.rs else None


def _ai_suffix(x: _Ctx, c: str) -> str:
    if x.ml:
        return f" (IA : {_pct(x.ml.get(c, 0.0))})"
    return ""


def _why_not_artefact(x: _Ctx) -> str:
    pa = x.ml.get("artefact", x.p("artefact"))
    parts = [f"l'IA de référence {_only(pa)}"]
    good_elong = x.elong is not None and x.elong >= 4
    good_snr = x.snr is not None and x.snr >= 12
    feats = []
    if x.elong is not None:
        feats.append(f"allongée ({_rel(x.elong)} sa largeur)" if good_elong else
                     f"peu allongée ({_rel(x.elong)} sa largeur)")
    if x.snr is not None:
        feats.append(f"contrastée ({_num(x.snr, 0)}{NBSP}σ du bruit)" if good_snr else
                     f"d'un contraste modeste ({_num(x.snr, 0)}{NBSP}σ du bruit)")
    if feats:
        tail = ", comme une vraie traînée de gouttelettes" if good_elong and good_snr else ""
        parts.append("trace " + " et ".join(feats) + tail)
    return f"{_not('artefact')} : " + " ; ".join(parts) + "."


def _why_not(x: _Ctx, c: str, t: str) -> str:
    """Phrase « pourquoi pas c » quand la classe retenue (ou expliquée) est t."""
    if c == "artefact":
        return _why_not_artefact(x)
    reasons = _factors(x, c)
    rc, rt = _rule_score(x, c), _rule_score(x, t)
    if t == "artefact":
        head = f"{_not(c)} : "
        bits = []
        if x.ml:
            bits.append(f"l'IA de référence {_only(x.ml.get(c, 0.0))}")
        if reasons:
            bits.append(reasons[0][1])
        elif x.manual and c == x.rules_best and rc is not None:
            bits.append(f"les règles la classaient {_short(c)} (score {_num(rc, 2)}), mais vous l'avez marquée comme "
                        f"artefact")
        if not bits:
            bits.append("les mesures ressemblent davantage à un artefact qu'à une trace de particule")
        return head + " ; ".join(bits) + "."
    if x.capped and {c, t} == {"muon", "beta"} and not x.manual:
        _txt, pair_short = _pair(x, t, c)
        return (f"{_cap(_short(c))} possible ({_pct(x.p(c))}) : {pair_short} ; d'où la confiance plafonnée à "
                f"{_pct(_AMBIGUITY_CAP)}.")
    rules_favoured = (c == x.rules_best and c != t and rc is not None and rc >= 0.3)
    if rules_favoured:
        txt = f"{_cap(_short(c))} possible selon les règles (score {_num(rc, 2)})"
        if x.manual:
            txt += f", mais vous avez choisi : {_short(t)}"
        elif x.mode == "hybride":
            txt += (f", mais l'IA de référence {_only(x.ml.get(c, 0.0))} (probabilité combinée "
                    f"{_pct(x.p(c))} contre {_pct(x.p(t))} pour {_short(t)})")
        elif x.mode == "locale":
            txt += f", mais l'IA locale lui donne {_pct(x.p(c))} contre {_pct(x.p(t))} pour {_short(t)}"
        if reasons:
            txt += f" ; point faible : {reasons[0][1]}"
        return txt + "."
    ai_best = _argmax(x.ml) if x.ml else None
    if ai_best == c and c != t and not x.manual:
        txt = (f"{_cap(_short(c))} selon l'IA de référence ({_pct(x.ml.get(c, 0.0))}), mais pas selon les règles"
               + (f" : {reasons[0][1]}" if reasons else "")
               + f" ; probabilité combinée {_pct(x.p(c))} contre {_pct(x.p(t))} pour {_short(t)}")
        return txt + "."
    if reasons:
        return f"{_not(c)} : " + " ; ".join(r for _v, r in reasons[:2]) + _ai_suffix(x, c) + "."
    if rc is not None and rt is not None:
        return (f"{_not(c)} : les mesures lui conviennent moins bien (score des règles {_num(rc, 2)} contre "
                f"{_num(rt, 2)} pour {_short(t)})" + _ai_suffix(x, c) + ".")
    return f"{_not(c)} : les mesures disponibles ne la désignent pas" + _ai_suffix(x, c) + "."


# --------------------------------------------------------------------------- #
# Ambiguïté entre deux hypothèses (trace incertaine)
# --------------------------------------------------------------------------- #
def _pair(x: _Ctx, a: str, b: str | None) -> tuple[str, str]:
    """(explication détaillée, raison courte) de l'ambiguïté entre a et b."""
    pair = {a, b}
    heavy_set, light_set = {"alpha", "proton"}, {"muon", "beta"}
    if pair == light_set:
        if (x.L or 0.0) < x.mml and (x.ev.get("straight_m") or 0.0) >= 0.5:
            px = (f", pour {_num(x.mmpp, 2)}{NBSP}mm par pixel" if x.mmpp else "")
            txt = (f"Les deux sont des particules au minimum d'ionisation : même trace fine ({_rel(x.w_rel)} la "
                   f"référence). Sur {_mm(x.L)}, la diffusion multiple (Highland) n'écarterait un électron de 1,31 MeV "
                   f"que d'environ {_small_mm(x.y_e13)} de la droite (≈ {_small_mm(x.y_e05)} à 0,5 MeV){px} : la "
                   f"différence avec la droite parfaite d'un muon est à peine mesurable. Seule une trace plus longue, "
                   f"où la déviation devient visible, permettrait de trancher. C'est une ambiguïté physique réelle, "
                   f"pas un défaut du logiciel.")
            return txt, f"fine, courte ({_mm(x.L)}) et droite : muon presque vertical ou électron énergétique"
        txt = (f"Les deux sont des particules au minimum d'ionisation (trace fine, {_rel(x.w_rel)} la référence) ; la "
               f"forme est intermédiaire (rectitude {_num(x.S, 3)}, étalement latéral {_small_mm(x.lat)}) : ni "
               f"nettement droite comme un muon, ni nettement sinueuse comme un électron lent.")
        return txt, "fine, forme intermédiaire : muon ou électron"
    if pair == heavy_set:
        txt = (f"Les deux sont des particules lourdes (trace dense, {_rel(x.d_rel)} la référence) et droites : seule la "
               f"longueur les départage. {_cap(_mm(x.L))} est proche de la limite des règles ({_mm(x.amax)} ; portée "
               f"maximale d'une alpha naturelle ≈ {_mm(x.alpha_max_phys)} à {_temp(x.tc)})"
               + (" et la trace touche le bord (longueur minimale)." if x.border else "."))
        return txt, f"lourde, {_mm(x.L)} : limite alpha/proton"
    if a in heavy_set | light_set and b in heavy_set | light_set:
        txt = (f"L'ionisation est intermédiaire (épaisseur {_rel(x.w_rel)}, densité {_rel(x.d_rel)} la référence) : "
               f"entre une trace fine (×1) et le seuil des particules lourdes ({_rel(x.hdr)}). Causes possibles : trace "
               f"floue ou peu contrastée, alpha qui ne traverse qu'un bord de la couche sensible, ou référence "
               f"« trace fine » imprécise.")
        return txt, f"ionisation intermédiaire ({_rel(x.d_rel)})"
    if "artefact" in pair:
        txt = (f"Contraste {_num(x.snr, 0)}{NBSP}σ et allongement {_rel(x.elong)} : à la limite entre une vraie trace "
               f"pâle et une tache de brume ou un amas de gouttelettes.")
        return txt, "pâle ou peu allongée : trace ou artefact ?"
    return "Les mesures sont compatibles avec plusieurs hypothèses.", "aucune signature nette"


# --------------------------------------------------------------------------- #
# Paragraphes de physique
# --------------------------------------------------------------------------- #
def _physics_alpha(x: _Ctx) -> str:
    rng = [(n, _alpha_range_mm(e, x.tc, x.pk)) for n, e in _RANGE_EMITTERS]
    rtxt = ", ".join(f"{_num(r / 10.0, 1)}{NBSP}cm ({n})" for n, r in rng if r)
    s = ("Une particule alpha est un noyau d'hélium (2 protons + 2 neutrons, charge +2), émise surtout par le radon "
         "222 de l'air (5,49 MeV) et ses descendants. Selon la formule de Bethe, l'ionisation par millimètre varie "
         "comme z²/β² : chargée deux fois et lente, l'alpha ionise ≈ 300 à 400 fois plus qu'une particule au minimum "
         "d'ionisation, d'où une trace épaisse et brillante ; ≈ 7 300 fois plus lourde qu'un électron, elle n'est "
         f"presque pas déviée. Elle s'arrête après quelques centimètres : à {_temp(x.tc)}, NIST ASTAR donne {rtxt}.")
    L = x.L
    if L:
        e = _f(x.F.get("energy_mev"))
        if e:
            s += (f" Ici, {_mm(L)} correspondent à une énergie d'au moins ≈ {_num(e, 1)} MeV (règle de Geiger ; "
                  f"minimum, car la trace peut être inclinée ou sortir de la couche).")
        ok = [(n, r) for n, r in rng if r and r >= 0.95 * L]
        r_min = min((r for _n, r in rng if r), default=0.0)
        if not ok:
            s += (" Elle est plus longue que toute alpha naturelle : trace mal mesurée, deux traces alignées ou "
                  "particule plus pénétrante ?")
        elif r_min and L < 0.6 * r_min:
            s += (f" Cette trace est bien plus courte que la portée de toute alpha naturelle (au moins ≈ {_mm(r_min)}) : "
                  f"l'alpha est sans doute très inclinée, sort de la couche sensible ou n'est vue qu'en partie.")
        else:
            s += " Émetteurs dont la portée atteint cette longueur : " + ", ".join(
                f"{n} ({_mm(r)})" for n, r in ok) + "."
            close = [(n, r) for n, r in rng if r and abs(r - L) <= 0.08 * r]
            if close and not x.border:
                s += (f" La longueur est proche de la portée du {close[0][0]} : trace peut-être complète et à plat "
                      f"(compatible, pas une identification).")
    return s


def _physics_beta(x: _Ctx) -> str:
    s = ("Un électron (bêta⁻) porte une charge −1 et pèse 1 836 fois moins qu'un proton. Rapide, il ionise peu "
         "(trace fine) ; léger, il est dévié à chaque collision : c'est la diffusion multiple (formule de Highland, "
         "PDG), d'autant plus forte que l'électron est lent. Les électrons lents (< 0,1 MeV) dessinent des traces "
         "courtes très tortueuses ; ceux d'environ 1 MeV, des traces longues presque droites qui sortent de la couche "
         "sensible (un électron de 0,5 MeV peut parcourir ≈ 1,4 m d'air froid, NIST ESTAR). Le K-40 (KCl, banane) se "
         "désintègre à 89 % par bêta⁻ (énergie maximale 1,311 MeV, moyenne 0,56 MeV) ; son gamma de 1,46 MeV est "
         "invisible lui-même et ne se manifeste que par les électrons Compton qu'il arrache, surtout aux parois, à la "
         "plaque ou à l'échantillon.")
    return s + _momentum_txt(x, "beta")


def _physics_muon(x: _Ctx) -> str:
    d = x.depth
    p10, p30 = _muon_fraction_longer(10.0, d), _muon_fraction_longer(30.0, d)
    return ("Le muon cosmique est une sorte d'électron lourd (207 fois sa masse), produit vers 15 km d'altitude ; il "
            "arrive au sol avec ≈ 4 GeV en moyenne, à raison d'environ 1 par cm² et par minute (PDG). Si rapide, il "
            "est au minimum d'ionisation (≈ 0,26 keV/mm dans l'air froid) : trace fine, parfois pointillée. Sa grande "
            "impulsion le rend presque insensible à la diffusion multiple (≈ 0,001° sur 1 cm) : trace parfaitement "
            "droite. Le flux varie comme cos²θ : la plupart des muons arrivent presque verticalement et ne traversent "
            f"la couche sensible (≈ {_mm(d, 0)} ici) que sur une courte longueur projetée ; avec cette épaisseur, "
            f"environ {_pct_fine(p10)} dépassent 1 cm et {_pct_fine(p30)} dépassent 3 cm. Une longue droite fine est donc un muon "
            "très incliné, ou un électron cosmique de plus de 10 MeV, presque aussi droit (≈ 30 contre 70 m⁻²·s⁻¹·sr⁻¹ "
            "pour les muons) : « muon » signifie ici « muon probable ».") + _momentum_txt(x, "muon")


def _physics_proton(x: _Ctx) -> str:
    return ("Un proton (charge +1, 1 836 fois la masse de l'électron) lent, de quelques MeV à quelques dizaines de "
            "MeV, ionise bien plus qu'un muon (≈ 56 fois à 3 MeV, NIST PSTAR) mais nettement moins qu'une alpha : "
            "épaisseur intermédiaire, trace droite, et une portée qui peut dépasser celle de toute alpha naturelle "
            f"(≈ {_mm(x.alpha_max_phys)} à {_temp(x.tc)} pour le Po-212, contre ≈ 12 cm d'air froid pour un proton de "
            "3 MeV). Un proton du GeV ionise presque comme un muon et ne se distingue pas d'une trace fine. Les protons "
            "cosmiques sont rares au niveau de la mer (≈ 1 % des muons, PDG) : une trace classée « proton » peut aussi "
            "être une alpha longue mal mesurée ou deux traces alignées ; à vérifier sur la vignette.")


def _physics_artefact(x: _Ctx) -> str:
    return ("Une vraie trace naît en une fraction de seconde : les ions laissés par la particule servent de germes aux "
            "gouttelettes d'alcool, alignées le long de la trajectoire (traînée fine et nette). Une bouffée de brume, "
            "un amas de gouttelettes qui tombent vers la plaque froide (Clark, Toronto), un reflet ou une poussière "
            "donnent au contraire une tache diffuse, peu allongée, peu contrastée ou immobile. Les artefacts sont "
            "exclus des comptages de particules.")


def _physics_uncertain(x: _Ctx, thr: float) -> str:
    pair = {x.c1, x.c2}
    if pair == {"muon", "beta"}:
        core = ("Muons et électrons énergétiques sont tous deux au minimum d'ionisation (formule de Bethe) : leurs "
                "traces ont la même épaisseur. Seule la diffusion multiple les distingue : elle courbe la trajectoire "
                "d'un électron (formule de Highland, PDG) mais pas celle d'un muon de ≈ 4 GeV. Sur une trace courte, "
                "cette courbure reste sous la résolution de l'image ; de plus, des électrons cosmiques de plus de "
                "10 MeV laissent des traces presque aussi droites qu'un muon.")
    elif pair == {"alpha", "proton"}:
        core = (f"Alphas et protons lents sont tous deux fortement ionisants et droits. Une alpha naturelle ne dépasse "
                f"pas ≈ {_mm(x.alpha_max_phys)} à {_temp(x.tc)} (Po-212, NIST ASTAR) ; au voisinage de cette longueur, "
                "ou si la trace touche le bord, la longueur ne suffit plus à trancher.")
    elif "artefact" in pair:
        core = ("Une trace pâle ou peu allongée peut être une vraie particule mal vue (hors du plan de netteté, en "
                "bordure de la couche sensible) ou une tache de brume : le contraste et l'allongement ne suffisent "
                "pas à trancher.")
    else:
        core = ("L'ionisation (épaisseur et densité des gouttelettes) est l'indice le plus fiable ; quand elle est "
                "intermédiaire, la trace peut être floue, peu contrastée ou ne traverser qu'un bord de la couche "
                "sensible.")
    return core + (f" Cette prudence est voulue : l'application ne nomme une particule que si elle en est sûre à au "
                   f"moins {_pct(thr)} ; une étiquette fausse fausserait les taux de comptage, alors qu'une trace "
                   f"« incertaine » reste comptée comme trace détectée.")


def _momentum_txt(x: _Ctx, c: str) -> str:
    p = _f(x.F.get("momentum_mev"))
    if not p or x.bfield <= 0:
        return ""
    e = _f(x.F.get("energy_mev"))
    s = f" Courbure dans le champ de {_num(x.bfield, 0)}{NBSP}mT : p ≈ {_num(p, 2)} MeV/c"
    if e:
        s += f", soit une énergie cinétique ≈ {_num(e, 2)} MeV si c'est {_un(c)}"
    return s + "."


# --------------------------------------------------------------------------- #
# Limites, sources
# --------------------------------------------------------------------------- #
def _caveats(x: _Ctx, t: str) -> list[str]:
    out = []
    if not x.calibrated:
        out.append(f"Échelle non calibrée : les longueurs en mm sont estimées (champ supposé de {_mm(x.fov)} de large) ; "
                   f"les seuils en mm (alpha ≤ {_mm(x.amax)}, muon ≥ {_mm(x.mml)}) sont donc approximatifs. Calibrez "
                   f"l'échelle avec une règle filmée.")
    if x.border and x.L is not None:
        out.append(f"La trace touche le bord de la zone analysée : sa longueur ({_mm(x.L)}) est un minimum"
                   + (", tout comme l'énergie estimée." if t == "alpha" else "."))
    named = x.final in BASE_CLASSES
    if named and not x.manual:
        if x.mode == "regles" and x.conf < 0.5:
            out.append(f"Confiance modérée ({_pct(x.conf)}) : vérifiez la trace sur la vignette.")
        elif x.mode != "regles" and x.conf < x.threshold:
            out.append(f"Confiance ({_pct(x.conf)}) sous le seuil de {_pct(x.threshold)} : à vérifier sur la vignette.")
    if x.capped and x.final in ("muon", "beta"):
        out.append(f"Confiance volontairement plafonnée à {_pct(_AMBIGUITY_CAP)} : trace fine, courte et droite, "
                   f"compatible avec un muon presque vertical ou un électron énergétique.")
    if t in ("alpha", "proton"):
        out.append("Vue de dessus (projection 2D) : une particule inclinée ou qui sort de la couche sensible paraît plus "
                   "courte ; la longueur et l'énergie estimée sont des minima.")
    if t == "alpha":
        out.append(f"La température de l'air de la couche sensible (réglée à {_temp(x.tc)}) est mal connue : ±15 °C "
                   f"change la portée d'environ ±6 % et l'énergie estimée d'environ ±4 %.")
    if t == "muon":
        out.append("Vue de dessus : la longueur d'une trace de muon dépend de son inclinaison, pas de son énergie. "
                   "« Muon » = muon probable : un électron cosmique de plus de 10 MeV laisse une trace presque identique.")
    if x.has_ev and not x.refs_ready:
        out.append(f"Référence « trace fine » pas encore établie (il faut au moins 6 traces bien mesurées) : épaisseur et "
                   f"densité relatives sont estimées à partir d'une largeur type de {_num(_REF_WIDTH_MM, 1)}{NBSP}mm, "
                   f"moins fiable.")
    ml_best = _argmax(x.ml) if x.ml else None
    if (ml_best in BASE_CLASSES and t != "artefact" and not x.manual and x.rules_best
            and x.rules_best != ml_best):
        out.append("Règles physiques et IA de référence en désaccord : l'IA est un modèle statistique qui a pu "
                   "s'appuyer sur des mesures que les règles n'utilisent pas (durée, continuité de la trace…) ; "
                   "vérifiez la trace sur la vignette.")
    if x.ml:
        info = _model_info()
        n = _f(info.get("n_tracks_total"))
        acc = _f(info.get("bal_acc"))
        s = ("L'IA de référence a appris sur " + (f"{_num(n, 0)} traces simulées" if n else "des traces simulées")
             + " et n'a jamais vu votre chambre : son avis est un indice, pas une preuve")
        if acc:
            s += (f" (exactitude équilibrée de {_pct(acc)} mesurée sur des vidéos simulées jamais vues, pas sur des "
                  f"vidéos réelles)")
        out.append(s + ".")
    if x.mode == "locale":
        out.append("En mode « IA locale », la décision vient du modèle entraîné sur vos corrections ; les seuils "
                   "« attendu » sont ceux des règles physiques, donnés pour comparaison.")
    if x.rebuilt:
        out.append("Valeurs intermédiaires des règles recalculées à partir des mesures enregistrées (trace analysée "
                   "par une version antérieure).")
    elif not x.has_ev:
        out.append("Valeurs intermédiaires des règles indisponibles : explication limitée aux mesures brutes.")
    if x.manual and x.auto != x.manual:
        out.append(f"Votre étiquette ({_short(x.manual)}) diffère de l'analyse automatique ({_short(x.auto)}) : elle "
                   f"prime pour les comptages et sert à entraîner l'IA locale ; assurez-vous qu'elle est juste.")
    return out


def _sources(x: _Ctx, t: str, keys_extra=()) -> list[str]:
    keys = {
        "alpha": ["astar", "pdg_pass", "nndc_rn222", "nndc_po212", "scool"],
        "beta": ["estar", "pdg_pass", "nndc_k40", "gamma", "scool"],
        "muon": ["pdg_cosmic", "pdg_pass", "pdg_air", "cambridge"],
        "proton": ["pstar", "pdg_cosmic", "astar"],
        "artefact": ["toronto", "scool"],
    }.get(t, ["pdg_pass"])
    keys = list(keys) + [k for k in keys_extra if k not in keys]
    return [_SOURCES[k] for k in keys if k in _SOURCES]


# --------------------------------------------------------------------------- #
# Fonctions publiques
# --------------------------------------------------------------------------- #
_CONCLUSION = {
    "alpha": "signature d'une particule lourde, très ionisante, qui s'arrête en quelques centimètres d'air",
    "proton": "particule lourde et ionisante, mais plus longue que toute alpha naturelle",
    "muon": "signature d'une particule rapide au minimum d'ionisation que rien ne dévie",
    "beta": "signature d'un électron léger, dévié à chaque collision (diffusion multiple)",
}


def _criteria_for(x: _Ctx, t: str) -> tuple[list[Criterion], list[str], list[str], list[str]]:
    """Critères pour la classe t ; renvoie (critères, phrases « pour », phrases « contre », phrases courtes)."""
    crits, pros, cons, shorts = [], [], [], []
    for fn in (_crit_thickness, _crit_density, _crit_shape, _crit_length):
        c, ph, sh = fn(x, t)
        if c is None:
            continue
        if x.mode == "locale":
            c.used_for_decision = True
        crits.append(c)
        if c.verdict == "pour" and ph:
            pros.append(ph)
            if sh:
                shorts.append(sh)
        elif c.verdict == "contre" and ph:
            cons.append(ph)
    for fn in (_crit_ai, _crit_local_ai):
        c = fn(x, t)
        if c is not None:
            crits.append(c)
    for fn in (_crit_duration, _crit_bragg, _crit_branches):
        c = fn(x, t)
        if c is not None:
            crits.append(c)
    return crits, pros, cons, shorts


def _method_sentence(x: _Ctx, t: str) -> str:
    if x.mode == "hybride":
        mb = _argmax(x.ml)
        if x.rules_best == mb:
            return f"Règles physiques et IA de référence d'accord (IA : {_pct(x.ml.get(mb, 0.0))})."
        return (f"Règles ({_short(x.rules_best or '?')}) et IA de référence ({_short(mb or '?')}, "
                f"{_pct(x.ml.get(mb, 0.0) if mb else 0.0)}) en désaccord : la combinaison pondérée retient "
                f"{_short(t)} ({_pct(x.p(t))}).")
    if x.mode == "locale":
        return (f"Décision de l'IA locale entraînée sur vos corrections ({_pct(x.p(t))}) ; les règles indiquaient "
                f"{_short(x.rules_best or '?')}.")
    return ""


def explain_features(f: dict, cls: str, confidence: float, scores: dict, *, cp=None, php=None,
                     duration_s: float | None = None, touches_border: bool = False, calibrated: bool = True,
                     manual_cls: str = "", method: str = "Règles physiques", ml_proba: dict | None = None,
                     rules_cls: str | None = None, rules_scores: dict | None = None, abstained: bool = False,
                     threshold: float = 0.85, family: str = "", capped: bool | None = None) -> Explanation:
    """Explique la classe d'une trace à partir de ses mesures et de la décision prise.

    f : mesures (features) de la trace, de préférence passées par classifier.classify_rules (clés ev_*) ;
        sans clés ev_*, elles sont recalculées sur une copie (f n'est jamais modifié).
    cls : classe AUTOMATIQUE (TrackEvent.cls) ; manual_cls : étiquette de l'utilisateur (prioritaire).
    confidence, scores : confiance et scores/probabilités de la décision (TrackEvent.confidence / .scores).
    ml_proba : probabilités de l'IA de référence (mode hybride) ; rules_cls / rules_scores : avis des règles.
    abstained : True si la probabilité la plus forte était sous ``threshold`` (classe « incertain »).
    family : « fine » / « lourde » si l'application a déterminé la famille de la trace.
    capped : plafond d'ambiguïté « courte et droite » appliqué (None = déduit des mesures).
    """
    x = _Ctx(f, cls, confidence, scores, cp, php, duration_s, touches_border, calibrated, manual_cls, method,
             ml_proba, rules_cls, rules_scores, abstained, threshold, family, capped)
    final = x.final
    probs = {k: round(float(v), 4) for k, v in x.probs.items()}
    srcs_extra: list[str] = []
    short = ""
    why_not: dict[str, str] = {}
    others_base = list(BASE_CLASSES) + (["artefact"] if x.artefact_considered else [])

    if final == "artefact":
        crits, phrases = _crits_artefact(x)
        t = "artefact"
        if x.manual:
            title = f"{_nom('artefact')} — marquée par l'utilisateur"
        else:
            title = f"{_nom('artefact')} — probablement pas une particule"
        summ = ("Détection probablement due à autre chose qu'une particule"
                + (f" : {_join(phrases)}" if phrases else "") + ". Elle est exclue des comptages de particules.")
        if x.manual:
            summ = (f"Vous avez marqué cette détection comme artefact. Analyse automatique : {_short(x.auto)} "
                    f"({_pct(x.conf)}). " + summ)
        elif x.ml:
            summ += f" IA de référence : artefact à {_pct(x.ml.get('artefact', 0.0))}."
        why_not = {c: _why_not(x, c, "artefact") for c in BASE_CLASSES}
        phys = _physics_artefact(x)
        short = f"× artefact : {_join(phrases) or 'pas une trace de particule'} ; exclue des comptages"
        caveats = _caveats(x, t)
        if x.ml:
            srcs_extra.append("model")
    elif final in BASE_CLASSES:
        t = final
        crits, pros, cons, shorts = _criteria_for(x, t)
        if x.capped and t in ("muon", "beta") and not x.manual:
            crits.append(Criterion(
                "ambiguity", "Ambiguïté « courte et droite »", f"{x.length_txt()}, trace fine et droite",
                f"≥ {_mm(x.mml)} pour lever l'ambiguïté (ou IA d'accord à ≥ 90 %)", "contre", 0.55,
                "Vue de dessus, une trace fine, courte et droite peut être un muon presque vertical ou un électron "
                "énergétique : la confiance est volontairement plafonnée à 45 %."))
        if pros:
            summ = f"Trace {_join(pros)} : {_CONCLUSION[t]}."
        else:
            summ = f"Les mesures penchent pour {_un(t)}, sans critère tranché : {_CONCLUSION[t]}."
        if cons:
            summ += f" Point faible : {_join(cons)}."
        ms = _method_sentence(x, t)
        if ms:
            summ += " " + ms
        if x.capped and t in ("muon", "beta") and not x.manual:
            summ += (f" Confiance plafonnée à {_pct(_AMBIGUITY_CAP)} : trace courte et droite (muon presque vertical "
                     f"ou électron énergétique).")
        if x.manual:
            title = f"{_nom(t)} — classée par l'utilisateur"
            agree = x.auto == t
            pre = (f"Vous avez classé cette trace : {_short(t)}. Analyse automatique : {_short(x.auto)} "
                   f"({_pct(x.conf)}, {x.method})")
            if x.ml:
                mb = _argmax(x.ml)
                pre += f", IA de référence : {_short(mb)} ({_pct(x.ml.get(mb, 0.0))})"
            pre += (". Les mesures et l'analyse automatique sont d'accord avec votre choix. " if agree else
                    ". Désaccord avec votre choix : vérifiez la vignette (traces superposées, flou, bord ?). ")
            parts = []
            if pros:
                parts.append(f"en faveur de « {_short(t)} » : {_join(pros)}")
            if cons:
                parts.append(f"contre : {_join(cons)}")
            summ = pre + ("Ce que disent les mesures, " + " ; ".join(parts) + "." if parts else
                          "Les mesures ne tranchent pas nettement.")
        else:
            title = _nom(t)
        why_not = {c: _why_not(x, c, t) for c in others_base if c != t}
        phys = {"alpha": _physics_alpha, "beta": _physics_beta, "muon": _physics_muon,
                "proton": _physics_proton}[t](x)
        caveats = _caveats(x, t)
        tail = f" · IA {_pct(x.ml.get(t, 0.0))}" if x.ml else ""
        if x.manual:
            short = (f"{_sym(t)} {_short(t)} (manuel ; auto : {_short(x.auto)} {_pct(x.conf)}) : "
                     + ", ".join(shorts))
        else:
            short = f"{_sym(t)} {_short(t)} : " + (", ".join(shorts) or "mesures intermédiaires")
            if cons:
                short += f" ; mais {cons[0]}"
            short += tail
            if x.capped and t in ("muon", "beta"):
                short += " · courte et droite : ambiguë"
        if x.dur is not None and t in ("alpha", "proton", "muon", "beta"):
            srcs_extra.append("sussman")
        if x.ml:
            srcs_extra.append("model")
    else:
        # ----- trace incertaine (abstention ou score insuffisant) -----
        t = x.c1 if x.c1 in BASE_CLASSES else (x.rules_best or "muon")
        crits, pros, cons, shorts = _criteria_for(x, t)
        thr = x.threshold if (x.abstained or x.mode != "regles") else x.min_conf
        p1, p2 = x.p(x.c1), (x.p(x.c2) if x.c2 else 0.0)
        pair_txt, pair_short = _pair(x, x.c1, x.c2)
        if x.mode == "regles" and not x.abstained:
            meas, exp = f"{_num(x.conf, 2)} (score de confiance des règles)", f"≥ {_num(x.min_conf, 2)}"
        else:
            meas = f"{_short(x.c1)} {_pct(p1)}" + (f" · {_short(x.c2)} {_pct(p2)}" if x.c2 else "")
            exp = f"≥ {_pct(thr)} pour nommer une particule"
        crits.insert(0, Criterion("ambiguity", "Écart entre les deux hypothèses", meas, exp, "contre",
                                  round(min(1.0, max(0.0, 1.0 - p1 / max(thr, 1e-6))), 3) if x.abstained else 0.6,
                                  pair_txt.split(". ")[0].rstrip(".") + "."))
        if x.manual:
            title = f"{_nom('incertain')} — choisie par l'utilisateur"
        elif x.c2:
            title = f"{_nom('incertain')} : {_short(x.c1)} ({_pct(p1)}) ou {_short(x.c2)} ({_pct(p2)}) ?"
        else:
            title = f"{_nom('incertain')} (aucune signature nette)"
        if x.abstained or x.mode != "regles":
            summ = (f"L'application ne nomme une particule que si elle en est sûre à au moins {_pct(thr)} ; ici, "
                    f"l'hypothèse la plus probable, {_un(x.c1)} à {_pct(p1)}, ne l'atteint pas"
                    + (f", car {_un(x.c2)}, à {_pct(p2)}, reste plausible. " if x.c2 else ". ") + pair_txt)
        else:
            summ = (f"Aucune signature nette : le score de confiance des règles ({_num(x.conf, 2)}) est sous le seuil "
                    f"minimal ({_num(x.min_conf, 2)}). Hypothèses les plus proches : {_short(x.c1)} ({_pct(p1)} des "
                    f"scores)" + (f" et {_short(x.c2)} ({_pct(p2)})" if x.c2 else "") + ". " + pair_txt)
        fam_txt = _family_sentence(x, thr)
        if fam_txt:
            summ += " " + fam_txt
        if x.manual:
            summ = f"Vous avez laissé cette trace « incertaine ». Analyse automatique : {_short(x.auto)}. " + summ
        why_not = {}
        for c in others_base:
            if c in (x.c1, x.c2) and x.p(c) > 0:
                why_not[c] = (f"{_cap(_short(c))} : {'hypothèse la plus probable' if c == x.c1 else 'deuxième hypothèse'} "
                              f"({_pct(x.p(c))}), sous le seuil de {_pct(thr)} — {pair_short}.")
            else:
                why_not[c] = _why_not(x, c, t)
        phys = _physics_uncertain(x, thr)
        caveats = _caveats(x, t if t in ("alpha", "proton", "muon") else "incertain")
        short = (f"? {_short(x.c1)} {_pct(p1)}" + (f" / {_short(x.c2)} {_pct(p2)}" if x.c2 else "")
                 + f" : {pair_short}")
        if x.ml:
            srcs_extra.append("model")
        t_src = "muon" if {x.c1, x.c2} == {"muon", "beta"} else t
        srcs = _sources(x, t_src, ["pdg_pass"] + srcs_extra)
        ex = Explanation(cls="incertain", cls_name=_nom("incertain"), confidence=float(x.conf), title=title,
                         summary=" ".join(summ.split()), criteria=crits, why_not=why_not, physics=phys,
                         caveats=caveats, scores=probs, method=_method_label(x), sources=srcs)
        object.__setattr__(ex, "_short", _clip(" ".join(short.split()), 160))
        return ex

    ex = Explanation(cls=final, cls_name=_nom(final), confidence=float(x.conf), title=title,
                     summary=" ".join(summ.split()), criteria=crits, why_not=why_not, physics=phys,
                     caveats=caveats, scores=probs, method=_method_label(x), sources=_sources(x, t, srcs_extra))
    object.__setattr__(ex, "_short", _clip(" ".join(short.split()), 160))
    return ex


def _family_sentence(x: _Ctx, thr: float) -> str:
    fam = x.family.lower()
    p_fine = x.p("beta") + x.p("muon")
    p_heavy = x.p("alpha") + x.p("proton")
    if not fam:
        if p_fine >= thr:
            fam = "fine"
        elif p_heavy >= thr:
            fam = "lourde"
        else:
            return ""
    if fam.startswith("fine"):
        txt, p = "trace fine (particule au minimum d'ionisation : électron ou muon)", p_fine
    elif fam.startswith("lourde"):
        txt, p = "trace lourde (particule fortement ionisante : alpha ou proton)", p_heavy
    else:
        return f"Famille déterminée : {x.family}."
    if p >= thr:
        return f"La famille, elle, est sûre à {_pct(p)} : {txt}."
    if p > 0:
        return f"Famille la plus probable : {txt} ({_pct(p)})."
    return f"Famille déterminée : {txt}."


def _method_label(x: _Ctx) -> str:
    if x.manual:
        return f"Correction manuelle (analyse automatique : {x.method})"
    return x.method


def _fine_explanation(e, ex: dict, cp, php, manual: str, thr) -> Explanation:
    """Classe « fine » : trace au minimum d'ionisation (électron ou muon) que la physique ne permet pas de
    départager, mais dont la FAMILLE est sûre. On réutilise l'explication d'abstention, retitrée."""
    try:
        dur = e.duration_s
    except Exception:
        dur = None
    thr_v = thr if _f(thr) is not None else 0.85
    base = explain_features(
        getattr(e, "features", None) or {}, "incertain", getattr(e, "confidence", 0.0),
        getattr(e, "scores", None) or {}, cp=cp, php=php, duration_s=dur,
        touches_border=bool(getattr(e, "touches_border", False)), calibrated=bool(ex.get("calibrated", True)),
        manual_cls="", method=ex.get("method") or "Hybride", ml_proba=ex.get("ml_proba"),
        rules_cls=ex.get("rules_cls"), rules_scores=ex.get("rules_scores"), abstained=True,
        threshold=thr_v, family="fine", capped=ex.get("capped"))
    fc = _f(ex.get("family_confidence"))
    sure = f" — famille sûre à {_pct(fc)}" if fc is not None else ""
    base.cls = "fine"
    base.cls_name = _nom("fine") if "fine" in physics.CLASS_INFO else "Trace fine (électron ou muon)"
    base.title = f"{base.cls_name}{sure}"
    base.summary = ("Trace fine : la particule ionise au minimum (électron rapide ou muon cosmique), c'est sûr. "
                    "Mais sur cette longueur, un électron énergétique et un muon laissent la même trace droite et "
                    "fine : la physique ne permet pas de les départager, donc l'application ne force pas de choix. "
                    "Ces traces comptent dans le groupe « traces fines » utilisé pour comparer KCl et fond. "
                    + base.summary)
    if manual:
        base.title += " — classe choisie par l'utilisateur"
    old = getattr(base, "_short", "") or ""
    object.__setattr__(base, "_short", _clip("f trace fine (β ou μ)" + sure + " · " + old.lstrip("? "), 160))
    return base


def explain_event(e, cp=None, php=None) -> Explanation:
    """Explication d'un TrackEvent (lit e.features, e.cls / e.final_cls, e.confidence, e.scores, e.extra…)."""
    ex = getattr(e, "extra", None) or {}
    if cp is None:
        cp = ex.get("cls_params")
    if php is None:
        php = ex.get("phys_params")
    manual = getattr(e, "manual_cls", "") or ""
    cls = getattr(e, "cls", None) or getattr(e, "final_cls", "incertain")
    try:
        dur = e.duration_s
    except Exception:
        dur = None
    thr = ex.get("threshold")
    if _f(thr) is not None and _f(thr) > 1.0:   # classe « jamais affirmée » (seuil 101 %) : afficher le seuil de base
        thr = getattr(cp, "ml_threshold", None) if cp is not None and not isinstance(cp, dict) else 0.85
        thr = thr if _f(thr) is not None else 0.85
    if cls == "fine":
        return _fine_explanation(e, ex, cp, php, manual, thr)
    return explain_features(
        getattr(e, "features", None) or {}, cls, getattr(e, "confidence", 0.0), getattr(e, "scores", None) or {},
        cp=cp, php=php, duration_s=dur, touches_border=bool(getattr(e, "touches_border", False)),
        calibrated=bool(ex.get("calibrated", True)), manual_cls=manual,
        method=ex.get("method") or "Règles physiques", ml_proba=ex.get("ml_proba"),
        rules_cls=ex.get("rules_cls"), rules_scores=ex.get("rules_scores"),
        abstained=bool(ex.get("abstained", False)), threshold=thr if _f(thr) is not None else 0.85,
        family=ex.get("family") or "", capped=ex.get("capped"))
