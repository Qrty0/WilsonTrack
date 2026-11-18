"""Assistant IA optionnel : Claude (API Anthropic) comme second avis scientifique.

L'application fonctionne ENTIÈREMENT sans cet assistant (détection, classification,
statistiques et conclusions sont calculées localement). Si une clé API Anthropic
est fournie, Claude peut en plus :
  1. examiner une image annotée de la chambre et donner son interprétation de
     chaque trace numérotée (vision) ;
  2. donner un second avis « Pourquoi ? » sur UNE trace : classe, justification
     physique, raisons d'exclure les autres classes, limites (vision + mesures) ;
  3. rédiger une interprétation scientifique d'un enregistrement ou d'une
     comparaison à partir des mesures ;
  4. répondre aux questions de l'équipe sur les résultats.

Modèles : Claude Opus 5.5 (par défaut), Sonnet 5.5 ou Haiku 5.5. La réflexion est
toujours adaptative sur ces modèles ; sa profondeur se règle avec l'« effort ».
Pour Opus 5.5 / Sonnet 5.5, le repli côté serveur (« server-side fallback ») est
activé : si le filtre de sécurité décline par erreur, un autre modèle reprend.

Seules les images et mesures que l'utilisateur choisit d'envoyer quittent l'ordinateur.
"""
from __future__ import annotations

import base64
import json
import math
import re
import threading
from dataclasses import dataclass

import cv2
import numpy as np

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_EFFORT = "medium"

# (identifiant API, libellé affiché) — le premier est le modèle recommandé.
MODEL_CHOICES = [
    ("claude-opus-5-5", "Claude Opus 5.5 — recommandé"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5 — plus rapide"),
    ("claude-haiku-5-5", "Claude Haiku 5.5 — économique"),
]

# (valeur API, libellé affiché)
EFFORT_CHOICES = [
    ("low", "Faible — rapide et économique"),
    ("medium", "Moyen — équilibré"),
    ("high", "Élevé — analyse approfondie"),
    ("xhigh", "Très élevé — le plus rigoureux (plus lent)"),
]

# Modèles qui acceptent le repli côté serveur (fallbacks="default", API Claude).
# Haiku 5.5 n'en a pas : requête ordinaire, comme pour tout identifiant inconnu.
FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"})

_EFFORT_VALUES = ("low", "medium", "high", "xhigh", "max")
_EFFORT_WORDS = {
    "faible": "low", "bas": "low", "basse": "low", "rapide": "low",
    "moyen": "medium", "moyenne": "medium", "normal": "medium", "équilibré": "medium",
    "élevé": "high", "eleve": "high", "élevée": "high", "haut": "high",
    "très élevé": "xhigh", "tres eleve": "xhigh", "très élevée": "xhigh",
    "maximum": "max", "maximal": "max",
}

SYSTEM = (
    "Tu es un physicien expert des détecteurs de particules et des chambres de Wilson (chambres à "
    "brouillard à diffusion et à expansion). Tu aides une équipe d'étudiants du cégep Marie-Victorin "
    "(Québec) qui étudie la radioactivité naturelle : rayons cosmiques (muons), radon (alphas), "
    "chlorure de potassium et bananes (potassium-40, émetteur β⁻, Emax 1,31 MeV). Tu agis comme SECOND "
    "AVIS indépendant du logiciel WilsonTrack, qui classe déjà les traces localement : tu peux être en "
    "désaccord avec lui, à condition de le justifier. Chaque classification doit être justifiée "
    "physiquement à partir de ce qui est visible et mesuré : ionisation (formule de Bethe–Bloch, "
    "dE/dx ∝ z²/β²), diffusion multiple (forte pour les électrons légers, négligeable pour les "
    "particules lourdes ou très énergétiques), relation portée–énergie (alphas de 4 à 8 MeV : 3 à 7 cm "
    "dans l'air) et muon au minimum d'ionisation (trace fine et très droite). Tu réponds en français "
    "(Québec), de façon rigoureuse et pédagogique, avec la virgule décimale. Tu distingues clairement "
    "ce qui est mesuré de ce qui est une hypothèse, et tu signales les limites (projection 2D, "
    "calibration, qualité d'image, statistique)."
)

FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "traces": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "classe": {"type": "string", "enum": ["alpha", "beta", "muon", "proton", "incertain",
                                                          "artefact"]},
                    "confiance": {"type": "number"},
                    "justification": {"type": "string"},
                },
                "required": ["id", "classe", "confiance", "justification"],
                "additionalProperties": False,
            },
        },
        "traces_non_detectees": {"type": "string"},
        "qualite_image": {"type": "string"},
        "conseils_montage": {"type": "array", "items": {"type": "string"}},
        "resume": {"type": "string"},
    },
    "required": ["traces", "traces_non_detectees", "qualite_image", "conseils_montage", "resume"],
    "additionalProperties": False,
}

TRACK_CLASSES = ["alpha", "beta", "muon", "proton", "incertain", "artefact"]
PARTICLE_CLASSES = ["alpha", "beta", "muon", "proton"]

# Les contraintes numériques (minimum / maximum) ne sont pas acceptées par les sorties
# structurées : l'intervalle de « confiance » est donné en description et vérifié ici.
TRACK_SCHEMA = {
    "type": "object",
    "properties": {
        "indices_visuels": {
            "type": "array", "items": {"type": "string"},
            "description": "Observations faites sur l'image elle-même (épaisseur, gouttelettes, coudes…).",
        },
        "classe": {"type": "string", "enum": TRACK_CLASSES},
        "confiance": {"type": "number", "description": "Confiance entre 0 et 1."},
        "pourquoi": {
            "type": "array", "items": {"type": "string"},
            "description": "Raisons physiques tirées de ce qui est VISIBLE et MESURÉ.",
        },
        "pourquoi_pas": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in PARTICLE_CLASSES},
            "required": list(PARTICLE_CLASSES),
            "additionalProperties": False,
        },
        "accord_avec_logiciel": {"type": "boolean"},
        "limites": {"type": "array", "items": {"type": "string"}},
        "resume": {"type": "string"},
    },
    "required": ["indices_visuels", "classe", "confiance", "pourquoi", "pourquoi_pas",
                 "accord_avec_logiciel", "limites", "resume"],
    "additionalProperties": False,
}

TRACK_GUIDE = (
    "Repères physiques (chambre à diffusion : air + vapeur d'alcool, couche sensible d'environ 1 cm "
    "au-dessus de la plaque froide, vue de dessus) :\n"
    "- Ionisation (Bethe–Bloch) : la perte d'énergie par unité de longueur croît comme z²/β². Une "
    "particule lente et doublement chargée (alpha) laisse un chapelet de gouttelettes très dense, donc "
    "une trace épaisse et brillante ; une particule au minimum d'ionisation (muon, électron rapide) "
    "laisse une trace fine et pâle.\n"
    "- Alpha (noyau d'hélium, 4 à 8 MeV : radon-222 5,5 MeV, polonium-218 6,0 MeV, polonium-214 "
    "7,7 MeV) : portée dans l'air d'environ 3 à 7 cm (relation portée–énergie), trace droite (particule "
    "lourde, diffusion multiple négligeable), épaisse, parfois plus large en fin de parcours (pic de "
    "Bragg) ; un petit crochet final est possible (collision avec un noyau). En projection 2D, la "
    "longueur observée est un minimum.\n"
    "- Bêta (électron) : masse environ 7300 fois plus faible que celle d'une alpha, donc forte "
    "diffusion multiple (trajet sinueux, coudes brusques, zigzags) ; ionisation faible sauf en fin de "
    "parcours, où l'électron ralenti s'épaissit et se tortille. Le potassium-40 (Emax 1,31 MeV) donne "
    "surtout des électrons de quelques centaines de keV : traces fines et courbées de quelques "
    "centimètres. Les électrons de très basse énergie (moins de 50 keV) font des traces courtes et "
    "tortueuses.\n"
    "- Muon cosmique (environ 4 GeV en moyenne au sol, environ 1 par cm² par minute) : particule au "
    "minimum d'ionisation, trace fine, très droite (diffusion multiple négligeable à cette énergie), "
    "souvent longue et traversant le champ (touche les bords). Un muon très incliné ne traverse la "
    "couche sensible que sur une courte distance : sa projection est COURTE, fine et droite.\n"
    "- Proton (secondaire cosmique, rare) : ionisation plus forte qu'un muon, trace droite, plus "
    "épaisse qu'un muon, souvent plus longue qu'une alpha naturelle.\n"
    "- Artefact (pas une particule) : reflet, bouffée de brume, goutte ou buée sur la vitre, poussière "
    "éclairée, turbulence : pas de chapelet de gouttelettes aligné, contours en tache, apparition ou "
    "disparition anormale.\n"
    "- Ambiguïté classique : une trace COURTE, DROITE et FINE peut être un muon très incliné OU un "
    "électron énergétique (peu dévié sur une courte distance). Indices utiles : densité de "
    "gouttelettes, légère courbure ou coude, épaississement final, contact avec les bords, durée de "
    "visibilité. Si tu ne peux pas trancher, dis-le et baisse ta confiance.\n"
    "- Les traces vieillissent : les gouttelettes diffusent, la trace s'élargit et pâlit ; une trace "
    "large mais pâle peut être une trace fine plus ancienne."
)

THUMB_TARGET = 768        # côté long visé (px) pour la vignette envoyée à Claude
_MAX_SIDE = 1568           # côté long maximal utile pour la vision
_MIN_SHORT = 160           # côté court minimal (bande noire ajoutée au besoin)

_DEFAULT_NAMES = {
    "alpha": "Alpha (α)",
    "beta": "Bêta / électron (β⁻)",
    "muon": "Muon cosmique (μ)",
    "proton": "Proton / particule lourde",
    "incertain": "Incertaine",
    "artefact": "Artefact (pas une particule)",
}


class AssistantError(Exception):
    """Erreur présentable à l'utilisateur (message en français)."""


@dataclass
class FrameVerdict:
    traces: list
    missed: str
    quality: str
    advice: list
    summary: str
    raw: dict


@dataclass
class TrackOpinion:
    """Second avis de Claude sur une trace (réponse « Pourquoi ? »)."""
    cls: str                      # alpha / beta / muon / proton / incertain / artefact
    confidence: float             # 0..1
    agrees: bool                  # d'accord avec la classe du logiciel ?
    why: list[str]                # raisons physiques (visible + mesuré)
    why_not: dict[str, str]       # alpha/beta/muon/proton → pourquoi pas
    visual_clues: list[str]
    limits: list[str]
    summary: str
    raw: dict
    model: str = ""               # modèle qui a réellement répondu (peut différer en cas de repli)
    local_cls: str = ""           # classe donnée par le logiciel (si connue)


# ============================================================ utilitaires ====
def resolve_model(text: str | None) -> str:
    """Identifiant API à partir d'un identifiant ou d'un libellé de MODEL_CHOICES.

    Un identifiant inconnu (saisi à la main) est conservé tel quel."""
    t = (text or "").strip()
    if not t:
        return MODEL
    low = t.lower()
    for mid, label in MODEL_CHOICES:
        if low in (mid, label.lower(), label.split(" — ")[0].strip().lower()):
            return mid
    return t


def resolve_effort(text: str | None) -> str:
    """Valeur d'effort API (low/medium/high/xhigh/max) à partir d'une valeur ou d'un libellé."""
    t = (text or "").strip().lower()
    if not t:
        return DEFAULT_EFFORT
    if t in _EFFORT_VALUES:
        return t
    for val, label in EFFORT_CHOICES:
        if t == label.lower() or t == label.split(" — ")[0].strip().lower():
            return val
    head = t.split("—")[0].strip()
    return _EFFORT_WORDS.get(head, _EFFORT_WORDS.get(t, DEFAULT_EFFORT))


def model_label(model_id: str | None) -> str:
    for mid, label in MODEL_CHOICES:
        if mid == model_id:
            return label.split(" — ")[0]
    return model_id or "Claude"


def supports_fallback(model: str) -> bool:
    return resolve_model(model) in FALLBACK_MODELS


def _effort_for(model: str, effort: str) -> str | None:
    """Effort réellement envoyé selon le modèle (None : paramètre non pris en charge)."""
    m = model.lower()
    if m.startswith(("claude-3", "claude-2", "claude-instant", "claude-haiku-4", "claude-sonnet-4-5",
                     "claude-sonnet-4-0", "claude-opus-4-0", "claude-opus-4-1")) \
            or re.match(r"claude-(sonnet|opus)-4-\d{8}", m):
        return None
    if m.startswith("claude-opus-4-5"):
        return effort if effort in ("low", "medium", "high") else "high"
    if m.startswith(("claude-opus-4-6", "claude-sonnet-4-6")) and effort == "xhigh":
        return "high"
    return effort


def _encode_jpeg(img: np.ndarray, max_side: int = _MAX_SIDE, quality: int = 88) -> str:
    h, w = img.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise AssistantError("Impossible d'encoder l'image.")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _prepare_thumb(img, target: int = THUMB_TARGET) -> tuple[np.ndarray, float] | None:
    """Vignette en BGR uint8 dont le côté long vaut ≈ `target` px (agrandie en bicubique).

    Retourne (image, facteur d'agrandissement) ou None si la vignette est vide."""
    if img is None:
        return None
    a = np.asarray(img)
    if a.size == 0 or a.ndim not in (2, 3) or min(a.shape[:2]) == 0:
        return None
    if a.dtype != np.uint8:
        a = np.nan_to_num(a.astype(np.float32))
        if float(a.max(initial=0.0)) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2 or a.shape[2] == 1:
        a = cv2.cvtColor(a.reshape(a.shape[0], a.shape[1]), cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    elif a.shape[2] != 3:
        return None
    h, w = a.shape[:2]
    long_side = max(h, w)
    zoom = 1.0
    if long_side < target:
        zoom = target / long_side
        a = cv2.resize(a, (max(1, round(w * zoom)), max(1, round(h * zoom))), interpolation=cv2.INTER_CUBIC)
    elif long_side > _MAX_SIDE:
        zoom = _MAX_SIDE / long_side
        a = cv2.resize(a, (max(1, round(w * zoom)), max(1, round(h * zoom))), interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]
    if min(h, w) < _MIN_SHORT:      # bande trop fine : on complète avec du noir (fond de la chambre)
        pad = _MIN_SHORT - min(h, w)
        p0, p1 = pad // 2, pad - pad // 2
        if h < w:
            a = cv2.copyMakeBorder(a, p0, p1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            a = cv2.copyMakeBorder(a, 0, 0, p0, p1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.ascontiguousarray(a), zoom


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fr(v, nd: int = 2) -> str:
    """Nombre formaté avec la virgule décimale."""
    x = _num(v)
    if x is None:
        return str(v)
    if not math.isfinite(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:.{nd}f}".replace(".", ",")


def _fr_auto(v) -> str:
    x = _num(v)
    if x is None:
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{x:.4g}".replace(".", ",") if math.isfinite(x) else _fr(x)


def _pct(c: float) -> str:
    return f"{round(100 * max(0.0, min(1.0, float(c))))} %"


def _f_num(unit: str, nd: int):
    return lambda v: f"{_fr(v, nd)} {unit}".strip()


def _f_rel(v) -> str:
    return f"×{_fr(v, 2)} la référence des traces fines"


def _f_radius(v) -> str:
    x = _num(v)
    if x is not None and not math.isfinite(x):
        return "infini (aucune courbure mesurable)"
    return f"{_fr(v, 0)} mm"


def _f_border(v) -> str:
    return "oui — la trace sort du champ, la longueur mesurée est donc un minimum" if _truthy(v) else "non"


def _f_calib(v) -> str:
    if isinstance(v, str) and v.strip().lower() not in ("oui", "non", "true", "false", "vrai", "faux", "1", "0"):
        return v.strip()
    if _truthy(v):
        return "calibrée (règle filmée dans la chambre)"
    return ("NON calibrée — les millimètres sont estimés d'après un champ de vision supposé, les longueurs "
            "absolues peuvent donc être fausses ; les grandeurs relatives sont plus fiables")


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "oui", "true", "vrai", "yes")
    return bool(v)


# (alias acceptés, libellé, formateur) — clés de TrackEvent.to_row() ou de compute_features().
_MEASURES = [
    (("longueur_mm", "length_mm"), "Longueur (profil à mi-hauteur)", _f_num("mm", 1)),
    (("corde_mm", "chord_mm"), "Corde (distance entre les extrémités)", _f_num("mm", 1)),
    (("epaisseur_mm", "width_mm"), "Épaisseur (largeur à mi-hauteur)", _f_num("mm", 2)),
    (("epaisseur_rel", "width_rel"), "Épaisseur relative", _f_rel),
    (("densite_rel", "density_rel"), "Densité d'ionisation relative", _f_rel),
    (("rectitude", "straightness"), "Rectitude (corde / chemin ; 1 = parfaitement droite)", _f_num("", 3)),
    (("fleche_rel", "sagitta_ratio"), "Flèche relative (écart maximal à la corde / longueur)", _f_num("", 3)),
    (("fleche_mm", "sagitta_mm"), "Flèche (écart maximal à la corde)", _f_num("mm", 2)),
    (("ecart_rms_mm", "line_rms_mm"), "Écart RMS à une droite", _f_num("mm", 2)),
    (("tortuosite_rad_mm", "tortuosity"), "Tortuosité (virage cumulé par mm)", _f_num("rad/mm", 3)),
    (("rayon_courbure_mm", "radius_mm"), "Rayon de courbure", _f_radius),
    (("duree_visible_s", "duration_s"), "Durée visible", _f_num("s", 2)),
    (("images_vues", "n_seen"), "Nombre d'images où la trace est vue", _f_num("", 0)),
    (("snr_crete", "peak_snr"), "Contraste crête", _f_num("σ du bruit", 1)),
    (("orientation_deg",), "Orientation", lambda v: f"{_fr(v, 0)}°"),
    (("embranchements", "branches"), "Embranchements du squelette", _f_num("", 0)),
    (("energie_MeV", "energy_mev"), "Énergie estimée par le logiciel", _f_num("MeV", 3)),
    (("impulsion_MeV_c", "momentum_mev"), "Impulsion estimée (courbure)", _f_num("MeV/c", 3)),
    (("touche_bord", "touches_border"), "Touche le bord du champ", _f_border),
    (("calibre", "calibrated", "calibration"), "Échelle", _f_calib),
    (("px_par_mm", "px_per_mm"), "Calibration", _f_num("px/mm", 3)),
    (("temperature_c", "temperature_chambre_c", "temperature"), "Température de la chambre", _f_num("°C", 1)),
]
_LOCAL_KEYS = {
    "cls": ("classe_locale", "classe", "cls", "final_cls"),
    "conf": ("confiance_locale", "confiance", "confidence"),
    "scores": ("scores_locaux", "scores"),
    "manual": ("classe_manuelle", "manual_cls"),
}
_THUMB_SCALE_KEYS = ("vignette_px_par_mm", "thumb_px_per_mm")
_SKIP_KEYS = {"id", "raison", "reason"}


def _present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    x = _num(v)
    return not (x is not None and math.isnan(x))


def _pick(d: dict, keys: tuple, used: set):
    for k in keys:
        if k in d and _present(d[k]):
            used.add(k)
            return d[k]
    for k in keys:
        if k in d:
            used.add(k)
    return None


def _measurement_lines(measurements: dict) -> tuple[list[str], dict, set]:
    """Lignes « - libellé : valeur » (virgule décimale), décision locale et clés utilisées."""
    m = dict(measurements or {})
    used: set = set()
    local = {name: _pick(m, keys, used) for name, keys in _LOCAL_KEYS.items()}
    lines = []
    for keys, label, fmt in _MEASURES:
        v = _pick(m, keys, used)
        if v is None:
            continue
        try:
            lines.append(f"- {label} : {fmt(v)}")
        except Exception:
            lines.append(f"- {label} : {v}")
    _pick(m, _THUMB_SCALE_KEYS, used)
    extra = []
    for k, v in m.items():
        if k in used or k in _SKIP_KEYS or str(k).startswith("_") or not _present(v):
            continue
        if isinstance(v, (bool, np.bool_)):
            extra.append(f"- {k} : {'oui' if v else 'non'}")
        elif isinstance(v, (int, float, np.integer, np.floating, str)):
            extra.append(f"- {k} : {_fr_auto(v)}")
    if extra:
        lines.append("- Autres mesures :")
        lines.extend("  " + e for e in extra)
    return lines, local, used


def _norm_cls(c) -> str:
    t = str(c or "").strip().lower()
    t = (t.replace("ê", "e").replace("è", "e").replace("é", "e")
         .replace("β", "beta").replace("α", "alpha").replace("μ", "muon"))
    if t in TRACK_CLASSES:
        return t
    for k in TRACK_CLASSES:
        if t.startswith(k):
            return k
    if t.startswith(("electron", "électron")):
        return "beta"
    return "incertain"


def _str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    out = []
    for x in v if isinstance(v, (list, tuple)) else [v]:
        s = " ".join(str(x).split())
        if s:
            out.append(s)
    return out


def _parse_json(text: str) -> dict:
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        try:
            d = json.loads(text[i:j + 1]) if 0 <= i < j else None
        except json.JSONDecodeError:
            d = None
    if not isinstance(d, dict):
        raise AssistantError("Réponse illisible du modèle.")
    return d


def _opinion_from_dict(d: dict, model: str = "", local_cls: str = "") -> TrackOpinion:
    c = _num(d.get("confiance"))
    if c is None or not math.isfinite(c):
        c = 0.0
    if 1.0 < c <= 100.0:            # pourcentage donné au lieu d'une fraction
        c /= 100.0
    c = max(0.0, min(1.0, c))
    agrees = d.get("accord_avec_logiciel", False)
    agrees = _truthy(agrees) if isinstance(agrees, str) else bool(agrees)
    wn = d.get("pourquoi_pas") if isinstance(d.get("pourquoi_pas"), dict) else {}
    why_not = {k: " ".join(str(wn.get(k, "") or "").split()) for k in PARTICLE_CLASSES}
    return TrackOpinion(
        cls=_norm_cls(d.get("classe")),
        confidence=c,
        agrees=agrees,
        why=_str_list(d.get("pourquoi")),
        why_not=why_not,
        visual_clues=_str_list(d.get("indices_visuels")),
        limits=_str_list(d.get("limites")),
        summary=" ".join(str(d.get("resume", "") or "").split()),
        raw=d,
        model=model or "",
        local_cls=_norm_cls(local_cls) if local_cls else "",
    )


def _class_names(names: dict | None) -> dict:
    out = dict(_DEFAULT_NAMES)
    if names is None:
        try:
            from . import physics
            for k, v in physics.CLASS_INFO.items():
                if isinstance(v, dict) and v.get("nom"):
                    out[k] = v["nom"]
        except Exception:
            pass
    else:
        for k, v in names.items():
            if isinstance(v, dict):
                v = v.get("nom")
            if v:
                out[k] = str(v)
    return out


def _md(text: str) -> str:
    return " ".join(str(text).split())


def opinion_markdown(op: TrackOpinion, names: dict | None = None) -> str:
    """Mise en forme Markdown (français) du second avis, pour QTextBrowser.setMarkdown().

    `names` : {classe: nom affiché} (ex. {k: v["nom"] for k, v in physics.CLASS_INFO.items()})."""
    nm = _class_names(names)

    def n(k: str) -> str:
        return nm.get(k, k)

    out = [f"### Second avis de Claude : {n(op.cls)}", "", f"- **Confiance de Claude :** {_pct(op.confidence)}"]
    accord = "Claude est **d'accord**" if op.agrees else "Claude est **en désaccord**"
    if op.local_cls:
        out.append(f"- **Classe du logiciel :** {n(op.local_cls)} — {accord}")
    else:
        out.append(f"- **Accord avec le logiciel :** {'oui' if op.agrees else 'non'}")
    if op.summary:
        out += ["", f"> {_md(op.summary)}"]
    if op.why:
        if op.cls == "incertain":
            title = "Pourquoi la trace reste incertaine"
        elif op.cls == "artefact":
            title = "Pourquoi ce n'est probablement pas une particule"
        else:
            title = f"Pourquoi c'est : {n(op.cls)}"
        out += ["", f"#### {title}", ""] + [f"- {_md(w)}" for w in op.why]
    others = [(k, t) for k, t in op.why_not.items() if k != op.cls and t]
    if others:
        out += ["", "#### Pourquoi pas les autres classes", ""]
        out += [f"- **{n(k)}** : {_md(t)}" for k, t in others]
    if op.visual_clues:
        out += ["", "#### Indices visuels", ""] + [f"- {_md(v)}" for v in op.visual_clues]
    if op.limits:
        out += ["", "#### Limites et incertitudes", ""] + [f"- {_md(v)}" for v in op.limits]
    who = model_label(op.model) if op.model else "Claude"
    out += ["", f"*Avis indépendant généré par {who} à partir de la vignette et des mesures : "
                f"c'est un second avis à confronter aux mesures, pas une vérité absolue.*"]
    return "\n".join(out) + "\n"


def track_measurements(ev, calibrated: bool | None = None, temperature_c: float | None = None) -> dict:
    """Dictionnaire de mesures pour explain_track() à partir d'un TrackEvent (ou objet équivalent)."""
    f = dict(getattr(ev, "features", None) or {})
    extra = dict(getattr(ev, "extra", None) or {})
    L = _num(f.get("length_mm"))
    sag = _num(f.get("sagitta_ratio"))
    dur = getattr(ev, "duration_s", None)
    if dur is None and hasattr(ev, "t_last") and hasattr(ev, "t_first"):
        dur = max(0.0, float(ev.t_last) - float(ev.t_first))
    m = {
        "classe_locale": getattr(ev, "cls", None),
        "confiance_locale": _num(getattr(ev, "confidence", None)),
        "scores_locaux": dict(getattr(ev, "scores", None) or {}) or None,
        "classe_manuelle": getattr(ev, "manual_cls", "") or None,
        "longueur_mm": L,
        "corde_mm": f.get("chord_mm"),
        "epaisseur_mm": f.get("width_mm"),
        "epaisseur_rel": f.get("width_rel"),
        "densite_rel": f.get("density_rel"),
        "rectitude": f.get("straightness"),
        "fleche_rel": sag,
        "fleche_mm": sag * L if (sag is not None and L is not None) else None,
        "ecart_rms_mm": f.get("line_rms_mm"),
        "tortuosite_rad_mm": f.get("tortuosity"),
        "rayon_courbure_mm": f.get("radius_mm"),
        "snr_crete": f.get("peak_snr"),
        "orientation_deg": f.get("orientation_deg"),
        "embranchements": f.get("branches"),
        "energie_MeV": f.get("energy_mev") or None,
        "impulsion_MeV_c": f.get("momentum_mev") or None,
        "duree_visible_s": dur,
        "images_vues": getattr(ev, "n_seen", None),
        "touche_bord": bool(getattr(ev, "touches_border", False)),
    }
    cal = calibrated if calibrated is not None else extra.get("calibrated")
    if cal is not None:
        m["calibre"] = bool(cal)
        if cal and _num(getattr(ev, "px_per_mm_full", None)):
            m["px_par_mm"] = float(ev.px_per_mm_full)
    if temperature_c is not None:
        m["temperature_c"] = temperature_c
    return {k: v for k, v in m.items() if v is not None}


# ============================================================== assistant ====
class ClaudeAssistant:
    def __init__(self, api_key: str = "", model: str = MODEL, effort: str = DEFAULT_EFFORT,
                 timeout: float = 300.0):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise AssistantError("Le module « anthropic » n'est pas installé.") from e
        self._anthropic = anthropic
        kwargs = {"timeout": timeout, "max_retries": 2}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            self.client = anthropic.Anthropic(**kwargs)
        except Exception as e:
            raise AssistantError("Aucune clé API Anthropic trouvée. Entrez votre clé dans Paramètres › "
                                 "Assistant IA (ou définissez la variable ANTHROPIC_API_KEY).") from e
        self.model = resolve_model(model)
        self.effort = resolve_effort(effort)
        self.history: list[dict] = []
        self.last_model = ""            # modèle ayant réellement répondu au dernier appel
        self._chat_lock = threading.Lock()

    # ------------------------------------------------------------------ appel --
    def _call(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 16000,
              effort: str | None = None) -> str:
        a = self._anthropic
        model = resolve_model(self.model)
        eff = _effort_for(model, resolve_effort(effort or self.effort))
        output_config: dict = {}
        if eff:
            output_config["effort"] = eff
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kw: dict = {"model": model, "max_tokens": max_tokens, "system": SYSTEM, "messages": messages}
        if output_config:
            kw["output_config"] = output_config
        try:
            if model in FALLBACK_MODELS:
                resp = self.client.beta.messages.create(**kw, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                resp = self.client.messages.create(**kw)
        except a.AuthenticationError as e:
            raise AssistantError("Clé API refusée (invalide ou révoquée).") from e
        except a.PermissionDeniedError as e:
            raise AssistantError("Cette clé API n'a pas accès à ce modèle.") from e
        except a.NotFoundError as e:
            raise AssistantError(f"Modèle introuvable : {model}.") from e
        except a.RateLimitError as e:
            raise AssistantError("Limite de requêtes atteinte : réessayez dans une minute.") from e
        except a.BadRequestError as e:
            raise AssistantError(f"Requête refusée par l'API : {e.message}") from e
        except a.APIStatusError as e:
            extra = " : service surchargé" if e.status_code == 529 else ""
            raise AssistantError(f"Erreur du service Anthropic ({e.status_code}){extra}. "
                                 "Réessayez plus tard.") from e
        except a.APITimeoutError as e:
            raise AssistantError("Délai d'attente dépassé : la connexion au service Anthropic est trop lente "
                                 "ou interrompue. Réessayez, éventuellement avec un effort plus faible. "
                                 "Le reste de l'application fonctionne hors ligne.") from e
        except a.APIConnectionError as e:
            raise AssistantError("Pas de connexion Internet (ou pare-feu) : l'assistant IA est indisponible. "
                                 "Le reste de l'application fonctionne hors ligne.") from e
        served = getattr(resp, "model", None)
        self.last_model = served if isinstance(served, str) and served else model
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details is not None else None
            raise AssistantError(
                "Le modèle a décliné cette demande (filtre de sécurité automatique"
                + (f", catégorie « {cat} »" if isinstance(cat, str) and cat else "")
                + "). Reformulez la question ou réessayez ; le reste de l'application n'est pas affecté.")
        if stop == "max_tokens":
            raise AssistantError("La réponse de Claude a été coupée : la limite de longueur a été atteinte "
                                 "avant la fin. Réessayez (un effort plus faible ou une question plus ciblée "
                                 "aide).")
        text = "".join(getattr(b, "text", "") or "" for b in (getattr(resp, "content", None) or [])
                       if getattr(b, "type", "") == "text")
        if not text.strip():
            raise AssistantError("Réponse vide du modèle.")
        return text

    # ------------------------------------------------------------ fonctions --
    def test_connection(self) -> str:
        return self._call([{"role": "user", "content": "Réponds seulement : « Connexion établie. »"}],
                          max_tokens=4000, effort="low")

    def analyze_frame(self, annotated_bgr: np.ndarray, tracks: list[dict], context: str = "") -> FrameVerdict:
        """Demande à Claude d'interpréter chaque trace numérotée d'une image annotée."""
        lines = []
        for t in tracks:
            lines.append(f"#{t['id']}: classe locale={t['classe']}, L={t['longueur_mm']} mm, "
                         f"épaisseur={t['epaisseur_mm']} mm, rectitude={t['rectitude']}, "
                         f"densité rel.={t['densite_rel']}, durée visible={t['duree_visible_s']} s")
        prompt = (
            "Voici une image d'une chambre de Wilson filmée par le dessus (fond noir, éclairage latéral). "
            "Les traces détectées par notre logiciel sont entourées et numérotées (#id). Mesures locales :\n"
            + ("\n".join(lines) if lines else "(aucune trace détectée sur cette image)") + "\n\n"
            + (f"Contexte : {context}\n\n" if context else "")
            + "Pour chaque trace numérotée, donne ta classification (alpha, beta, muon, proton, incertain ou "
              "artefact si ce n'est pas une trace de particule : reflet, bouffée de brume, goutte) avec une "
              "confiance entre 0 et 1 et une justification courte fondée sur l'aspect visuel et la physique "
              "(ionisation, diffusion multiple, portée). Signale les traces visibles que le logiciel a "
              "manquées, évalue la qualité de l'image et donne des conseils concrets pour améliorer le "
              "montage (éclairage, sursaturation, mise au point).")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": _encode_jpeg(annotated_bgr)}},
            {"type": "text", "text": prompt},
        ]
        txt = self._call([{"role": "user", "content": content}], schema=FRAME_SCHEMA)
        try:
            d = json.loads(txt)
        except json.JSONDecodeError as e:
            raise AssistantError("Réponse illisible du modèle.") from e
        return FrameVerdict(d.get("traces", []), d.get("traces_non_detectees", ""), d.get("qualite_image", ""),
                            d.get("conseils_montage", []), d.get("resume", ""), d)

    def explain_track(self, thumb_bgr: np.ndarray | None, measurements: dict, local_explanation: str,
                      context: str = "") -> TrackOpinion:
        """Second avis « Pourquoi ? » sur UNE trace : vignette agrandie + mesures + explication locale.

        `measurements` : clés de TrackEvent.to_row() (longueur_mm, epaisseur_mm, densite_rel…) ou de
        compute_features() (length_mm, width_mm…), plus classe_locale / confiance_locale / scores_locaux,
        calibre (bool), temperature_c… — voir track_measurements(). Lève AssistantError en cas d'échec."""
        lines, local, _ = _measurement_lines(measurements)
        prep = _prepare_thumb(thumb_bgr)
        content: list[dict] = []
        if prep is not None:
            img, zoom = prep
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": _encode_jpeg(img, quality=92)}})
            if zoom > 1.0:
                img_desc = (f"la vignette ci-dessus montre la trace au centre ; elle a été agrandie "
                            f"×{_fr(zoom, 1)} (interpolation bicubique) pour la lisibilité. La pixellisation "
                            "et le flou viennent de l'agrandissement, pas de la trace.")
            else:
                img_desc = "la vignette ci-dessus montre la trace au centre (résolution d'origine)."
            tpm = None
            for k in _THUMB_SCALE_KEYS:
                tpm = _num((measurements or {}).get(k))
                if tpm:
                    break
            if tpm and tpm > 0:
                img_desc += f" Sur l'image envoyée, 1 mm ≈ {_fr(tpm * zoom, 1)} px."
            img_desc += (" Si d'autres traces apparaissent en bordure, juge uniquement celle du centre "
                         "(la plus grande).")
        else:
            img_desc = ("aucune image n'est disponible : juge uniquement d'après les mesures et baisse ta "
                        "confiance en conséquence.")

        loc_cls = _norm_cls(local["cls"]) if local["cls"] else ""
        manual = _norm_cls(local["manual"]) if local["manual"] else ""
        dec = []
        if loc_cls:
            s = f"Classe du logiciel : {loc_cls} — {_DEFAULT_NAMES.get(loc_cls, loc_cls)}"
            if local["conf"] is not None and _num(local["conf"]) is not None:
                s += f", confiance {_pct(_num(local['conf']))}"
            dec.append(s)
        else:
            dec.append("Classe du logiciel : non fournie")
        if isinstance(local["scores"], dict) and local["scores"]:
            sc = sorted(((k, _num(v)) for k, v in local["scores"].items() if _num(v) is not None),
                        key=lambda t: -t[1])
            if sc:
                dec.append("Scores des règles physiques : " + " ; ".join(f"{k} {_fr(v, 2)}" for k, v in sc))
        if manual and manual != loc_cls:
            dec.append(f"Classe corrigée manuellement par l'équipe : {manual}")
        expl = (local_explanation or "").strip() or "(aucune explication fournie)"
        dec.append(f"Explication du logiciel : {expl}")

        prompt = (
            "Tu donnes un SECOND AVIS indépendant sur UNE trace détectée par notre logiciel WilsonTrack "
            "dans une chambre de Wilson à diffusion filmée par le dessus (fond noir, éclairage latéral).\n\n"
            f"Image : {img_desc}\n\n"
            "Mesures du logiciel :\n" + ("\n".join(lines) if lines else "- (aucune mesure fournie)") + "\n\n"
            + (f"Contexte de l'expérience : {context.strip()}\n\n" if context and context.strip() else "")
            + TRACK_GUIDE + "\n\n"
            "Méthode demandée :\n"
            "1. Jugement INDÉPENDANT d'abord : à partir de l'image et de la physique, décris ce que tu VOIS "
            "(épaisseur, densité de gouttelettes, rectitude, coudes, extrémités, netteté) et ce que disent "
            "les MESURES, puis déduis la classe la plus probable (alpha, beta, muon, proton, incertain, ou "
            "artefact si ce n'est pas une particule) sans te laisser guider par la décision du logiciel.\n"
            "2. Ensuite seulement, compare avec la décision du logiciel (plus bas) et indique si tu es "
            "d'accord (accord_avec_logiciel).\n"
            "3. « pourquoi » : 2 à 5 raisons physiques précises, chacune reliant une observation VISIBLE ou "
            "une valeur MESURÉE à un mécanisme (ionisation de Bethe–Bloch, diffusion multiple, relation "
            "portée–énergie, minimum d'ionisation du muon…). Cite les valeurs quand elles servent "
            "l'argument.\n"
            "4. « pourquoi_pas » : pour CHACUNE des classes alpha, beta, muon et proton, une phrase qui dit "
            "ce qui l'exclut ou la rend moins probable ; pour la classe retenue, ce qui pourrait encore la "
            "remettre en cause.\n"
            "5. « indices_visuels » : uniquement des observations faites sur l'image elle-même.\n"
            "6. « limites » : sois honnête sur l'incertitude : projection 2D (longueur et courbure "
            "apparentes), calibration de l'échelle, qualité et résolution de l'image, agrandissement, âge "
            "de la trace, un seul événement. Mentionne l'ambiguïté muon incliné / électron énergétique si "
            "elle s'applique.\n"
            "7. « confiance » : nombre entre 0 et 1, calibré honnêtement (0,5 ou moins si c'est ambigu ; "
            "0,9 ou plus seulement si plusieurs indices indépendants concordent).\n"
            "8. « resume » : une ou deux phrases simples pour des étudiants de cégep.\n"
            "N'invente aucune mesure absente. Écris en français avec la virgule décimale.\n\n"
            "Décision du logiciel (à comparer APRÈS ton propre jugement) :\n" + "\n".join(dec)
        )
        content.append({"type": "text", "text": prompt})
        txt = self._call([{"role": "user", "content": content}], schema=TRACK_SCHEMA)
        d = _parse_json(txt)
        return _opinion_from_dict(d, model=self.last_model, local_cls=manual or loc_cls)

    def interpret(self, report_text: str, question: str = "") -> str:
        """Interprétation scientifique rédigée (Markdown) d'un ou plusieurs enregistrements."""
        q = question or (
            "Rédige une interprétation scientifique de ces résultats pour notre rapport de projet : ce que "
            "montrent les mesures, leur signification physique, les incertitudes et biais possibles, et "
            "les prochaines expériences à faire. Structure avec des titres courts. Reste factuel.")
        msgs = [{"role": "user", "content": f"Résultats de l'analyse automatique :\n\n{report_text}\n\n{q}"}]
        return self._call(msgs)

    def chat(self, question: str, context: str = "") -> str:
        """Conversation suivie sur les résultats (le contexte est joint au premier message).

        Sûr en multi-fil : une seule question à la fois ; l'historique n'est modifié
        qu'en cas de succès (question + réponse ajoutées ensemble). L'historique est
        strictement ajouté en fin (jamais réécrit), comme l'exigent les modèles récents.
        """
        with self._chat_lock:
            if not self.history and context:
                question = f"Contexte (résultats actuels) :\n{context}\n\nQuestion : {question}"
            msg = {"role": "user", "content": question}
            ans = self._call(self.history + [msg])
            self.history.extend([msg, {"role": "assistant", "content": ans}])
            return ans

    def reset(self):
        with self._chat_lock:
            self.history.clear()
