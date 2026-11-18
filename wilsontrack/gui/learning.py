"""Centre d'apprentissage de l'IA : comment l'IA de référence a appris, et ce que valent vraiment ses résultats.

Page pédagogique (enseignant et élèves) :
- « comme un bébé » : l'IA a vu beaucoup de traces SIMULÉES dont on connaissait la vraie particule ;
- chiffres réels lus dans le modèle installé (``ReferenceModel.load_default().info``) : nombre de traces,
  date, type de modèle, précision et rappel par classe (examen et conditions extrêmes), couverture,
  matrice de confusion, exemples de traces ;
- encadré d'honnêteté : données simulées, l'IA n'a jamais vu VOTRE chambre, « incertaine » plutôt que deviner ;
- mesure de la précision sur les traces corrigées à la main (signal ``validate_requested``, résultat affiché
  par ``show_validation``).
La page fonctionne (avec des messages clairs) si le modèle ou certaines clés manquent.
"""
from __future__ import annotations

import datetime as _dt
import html
import math
import re
from pathlib import Path

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from . import brand as B
from .home import (ACCENT, BORDER, CARD, FAINT, MUTED, NBSP, SUCCESS, TEXT, WINDOW, FlowLayout, _discard,
                   _tinted_icon, class_hex, fmt_num)

__all__ = ["LearningCenter", "ConfusionMatrix", "summarize_model", "normalize_metrics", "normalize_confusion",
           "wilson_interval", "EXAMPLES_DIR"]

TINT = B.MINT                  # icône du bandeau (fond vert forêt)
ARTEFACT_HEX = "#a0785a"
WARN = B.WARN_TEXT             # texte d'avertissement lisible sur fond blanc
DANGER = "#e5484d"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "assets" / "ia_exemples"
CLASS_ORDER = ["alpha", "beta", "muon", "proton", "artefact", "incertain"]
CLASS_NAMES = {"alpha": "Alpha (α)", "beta": "Bêta / électron (β)", "muon": "Muon cosmique (μ)",
               "proton": "Proton (p)", "artefact": "Artefact", "incertain": "Incertaine (?)"}
CLASS_SHORT = {"alpha": "α", "beta": "β", "muon": "μ", "proton": "p", "artefact": "art.", "incertain": "?"}
CLASS_PLURAL = {"alpha": "vraies alphas", "beta": "vrais électrons (β)", "muon": "vrais muons (μ)",
                "proton": "vrais protons", "artefact": "vrais artefacts", "incertain": "traces incertaines"}
MODEL_TYPES = {"HistGradientBoostingClassifier": "des centaines de petits arbres de décision combinés",
               "GradientBoostingClassifier": "des arbres de décision combinés",
               "RandomForestClassifier": "des arbres de décision qui votent",
               "MLPClassifier": "perceptron multicouche"}
MODEL_SHORT = {"HistGradientBoostingClassifier": "Gradient boosting", "GradientBoostingClassifier": "Gradient boosting",
               "RandomForestClassifier": "Forêt aléatoire", "MLPClassifier": "Réseau de neurones"}
# Libellés des familles de traces simulées (clés possibles de info["kinds"]).
KIND_LABELS = {
    "alpha": "α du radon et de ses descendants", "alpha_rn222": "α du radon 222", "alpha_po218": "α du Po-218",
    "alpha_po214": "α du Po-214", "alpha_po210": "α du Po-210", "alpha_thoron": "α du thoron",
    "thoron_v": "paires en « V » du thoron", "alpha_v": "paires en « V » du thoron",
    "beta": "électrons β⁻ (K-40)", "beta_slow": "électrons lents", "beta_fast": "électrons rapides",
    "electron_slow": "électrons lents", "electron_fast": "électrons rapides", "compton": "électrons Compton (γ)",
    "muon": "muons cosmiques", "cosmic_electron": "électrons cosmiques", "proton": "protons",
    "delta": "rayons delta", "delta_ray": "rayons delta", "artefact": "artefacts (gouttes, reflets, bruit)",
}
DEFAULT_KINDS = ["α du radon et de ses descendants", "électrons β⁻ et électrons Compton", "muons cosmiques",
                 "protons"]
_MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
           "novembre", "décembre")


def cls_hex(cls: str) -> str:
    return ARTEFACT_HEX if cls == "artefact" else class_hex(cls)


def cls_text(cls: str) -> str:
    """Couleur de la classe pour du TEXTE sur fond clair (même teinte, assombrie : WCAG AA)."""
    return B.readable_on(cls_hex(cls))


def _pct(x, dec: int = 0) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{fmt_num(100 * v, dec)}{NBSP}%"


def _int(x) -> int | None:
    try:
        v = int(x)
    except (TypeError, ValueError):
        return None
    return v


def _float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float] | None:
    """Intervalle de confiance à 95 % (Wilson) d'une proportion k/n ; None si n = 0."""
    if not n or n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _fmt_created(s: str) -> str:
    try:
        d = _dt.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return str(s or "—")
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


# ============================================================================ lecture du modèle ====
def normalize_confusion(c) -> dict | None:
    """Matrice de confusion au format {"labels_true", "labels_pred", "matrix"} (accepte aussi le format pilote
    {vraie: {prédite: n}}) ; None si inutilisable."""
    if not isinstance(c, dict):
        return None
    if "matrix" in c:
        lt, lp, m = list(c.get("labels_true") or []), list(c.get("labels_pred") or []), c.get("matrix") or []
        try:
            m = [[int(v or 0) for v in row] for row in m]
        except (TypeError, ValueError):
            return None
        if not lt or not lp or len(m) != len(lt) or any(len(row) != len(lp) for row in m):
            return None
        return {"labels_true": [str(x) for x in lt], "labels_pred": [str(x) for x in lp], "matrix": m}
    rows = {str(k): v for k, v in c.items() if isinstance(v, dict)}
    if not rows:
        return None
    lt = [k for k in CLASS_ORDER if k in rows] + [k for k in rows if k not in CLASS_ORDER]
    preds = set()
    for v in rows.values():
        preds.update(str(k) for k in v)
    lp = [k for k in CLASS_ORDER if k in preds or k in lt]
    lp += [k for k in sorted(preds) if k not in lp]
    if "incertain" in lp:                      # « incertaine » toujours en dernière colonne
        lp = [k for k in lp if k != "incertain"] + ["incertain"]
    try:
        m = [[int(rows[t].get(p, 0) or 0) for p in lp] for t in lt]
    except (TypeError, ValueError):
        return None
    return {"labels_true": lt, "labels_pred": lp, "matrix": m}


def _metrics_from_confusion(conf: dict) -> dict:
    lt, lp, m = conf["labels_true"], conf["labels_pred"], conf["matrix"]
    n = sum(sum(r) for r in m)
    inc_col = lp.index("incertain") if "incertain" in lp else None
    n_inc = sum(r[inc_col] for r in m) if inc_col is not None else 0
    correct = sum(m[i][lp.index(c)] for i, c in enumerate(lt) if c in lp)
    n_conf = n - n_inc
    per = {}
    recalls = []
    for i, c in enumerate(lt):
        row_n = sum(m[i])
        hit = m[i][lp.index(c)] if c in lp else 0
        col_n = sum(m[k][lp.index(c)] for k in range(len(lt))) if c in lp else 0
        rec = hit / row_n if row_n else None
        if rec is not None:
            recalls.append(rec)
        per[c] = {"precision": hit / col_n if col_n else None, "recall": rec, "n": row_n}
    return {"n": n, "coverage": n_conf / n if n else None, "accuracy_confident": correct / n_conf if n_conf else None,
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None, "per_class": per,
            "n_confident": n_conf, "n_correct": correct}


def normalize_metrics(m, raw_split=None) -> dict | None:
    """Résultats d'un examen au format du contrat :
    {"n", "coverage", "accuracy_confident", "balanced_accuracy", "per_class": {cls: {"precision","recall","n"}}}.

    ``m`` = info["metrics"][split] (format du contrat) ; ``raw_split`` = model.metrics[split] (format pilote :
    {"rules"|"ml"|"hybrid": {...}}) utilisé en repli. Ajoute "legacy" (bool), "n_confident", "n_correct" et
    "methods" ({"rules","ml","hybrid": exactitude équilibrée}) quand c'est possible. None si rien d'exploitable.
    """
    contract = None
    if isinstance(m, dict) and isinstance(m.get("per_class"), dict):
        contract = {"n": _int(m.get("n")), "coverage": _float(m.get("coverage")),
                    "accuracy_confident": _float(m.get("accuracy_confident")),
                    "balanced_accuracy": _float(m.get("balanced_accuracy")), "per_class": {}, "legacy": False}
        for c, v in m["per_class"].items():
            if isinstance(v, dict):
                contract["per_class"][str(c)] = {"precision": _float(v.get("precision")),
                                                 "recall": _float(v.get("recall")), "n": _int(v.get("n"))}
        if contract["n"] and contract["coverage"] is not None:
            contract["n_confident"] = int(round(contract["n"] * contract["coverage"]))
            if contract["accuracy_confident"] is not None:
                contract["n_correct"] = int(round(contract["n_confident"] * contract["accuracy_confident"]))
    legacy = None
    if isinstance(raw_split, dict):
        best = raw_split.get("hybrid") if isinstance(raw_split.get("hybrid"), dict) else raw_split.get("ml")
        if isinstance(best, dict):
            conf = normalize_confusion(best.get("confusion"))
            if conf is not None:
                legacy = _metrics_from_confusion(conf)
                for c, v in legacy["per_class"].items():   # valeurs publiées par l'entraînement si présentes
                    p = _float((best.get("precision") or {}).get(c))
                    r = _float((best.get("recall") or {}).get(c))
                    v["precision"] = p if p is not None else v["precision"]
                    v["recall"] = r if r is not None else v["recall"]
                ba = _float(best.get("balanced_accuracy"))
                legacy["balanced_accuracy"] = ba if ba is not None else legacy["balanced_accuracy"]
                legacy["legacy"] = True
                legacy["confusion"] = conf
    # Contrat complet → contrat ; contrat vide (modèle pilote relu par la nouvelle version) → détail pilote.
    out = contract if contract is not None and (contract["per_class"] or legacy is None) else (legacy or contract)
    if out is None:
        return None
    methods = {}
    for src in (m if isinstance(m, dict) else {}, raw_split if isinstance(raw_split, dict) else {}):
        for k in ("rules", "ml", "hybrid"):
            v = src.get(k)
            v = v.get("balanced_accuracy") if isinstance(v, dict) else v
            if _float(v) is not None and k not in methods:
                methods[k] = _float(v)
    if methods:
        out["methods"] = methods
    return out


def summarize_model(model, rm_module=None) -> dict:
    """Résumé normalisé et tolérant du modèle de référence (None → available=False)."""
    s = {"available": model is not None, "n_tracks_total": None, "n_train": None, "n_jobs": None, "classes": [],
         "threshold": None, "created": "", "model_type": "", "generator": {}, "kinds": [],
         "splits": {"test": None, "stress": None},
         "confusion": None, "w_ml": None, "sklearn_version": "", "format": None, "legacy": False}
    if model is None:
        return s
    try:
        info = model.info if isinstance(model.info, dict) else {}
    except Exception:
        info = {}
    raw = getattr(model, "metrics", None)
    raw = raw if isinstance(raw, dict) else {}
    for k in ("n_tracks_total", "n_train", "n_jobs"):
        s[k] = _int(info.get(k, getattr(model, k, None)))
    s["classes"] = [str(c) for c in (info.get("classes") or getattr(model, "classes", []) or [])]
    s["threshold"] = _float(info.get("threshold"))
    s["created"] = str(info.get("created") or getattr(model, "created", "") or "")
    mt = info.get("model_type") or type(getattr(model, "model", None)).__name__
    s["model_type"] = "" if mt in ("NoneType", "") else str(mt)
    gen = info.get("generator") if isinstance(info.get("generator"), dict) else getattr(model, "generator", {})
    s["generator"] = gen if isinstance(gen, dict) else {}
    s["w_ml"] = _float(info.get("w_ml", getattr(model, "w_ml", None)))
    s["sklearn_version"] = str(info.get("sklearn_version", "") or "")
    kinds = info.get("kinds") or s["generator"].get("kinds")
    if isinstance(kinds, dict):
        s["kinds"] = [(KIND_LABELS.get(str(k), str(k)), _int(v)) for k, v in kinds.items()]
    elif isinstance(kinds, (list, tuple)):
        s["kinds"] = [(KIND_LABELS.get(str(k), str(k)), None) for k in kinds]
    metrics = info.get("metrics") if isinstance(info.get("metrics"), dict) else {}
    for split in ("test", "stress"):
        s["splits"][split] = normalize_metrics(metrics.get(split), raw.get(split))
    conf = normalize_confusion(info.get("confusion_test"))
    if conf is None and rm_module is not None and hasattr(rm_module, "reference_confusion"):
        try:
            conf = normalize_confusion(rm_module.reference_confusion())
        except Exception:
            conf = None
    if conf is None and s["splits"].get("test"):
        conf = s["splits"]["test"].get("confusion")
    s["confusion"] = conf
    s["format"] = _int(info.get("format"))
    # chiffres du modèle pilote : mesurés avec l'ancienne décision, sans le seuil d'abstention actuel
    s["legacy"] = any(bool((v or {}).get("legacy")) for v in s["splits"].values())
    return s


def load_reference():
    """(modèle ou None, module reference_model ou None, message d'erreur)."""
    try:
        from ..detection import reference_model as rm
    except Exception as e:  # pragma: no cover - module en cours de modification
        return None, None, f"module de l'IA indisponible ({e})"
    try:
        return rm.ReferenceModel.load_default(), rm, ""
    except Exception as e:  # pragma: no cover - load_default ne doit jamais lever
        return None, rm, str(e)


def _generator_lines(gen: dict) -> list[str]:
    """Quelques plages de paramètres tirés au hasard, en mots simples."""
    labels = (("resolution", "résolution"), ("fps", "cadence (img/s)"), ("noise_sigma", "bruit de la caméra"),
              ("fog_amp", "brouillard"), ("brightness", "luminosité"), ("chamber_temp_c", "température (°C)"),
              ("sensitive_depth_mm", "couche sensible (mm)"), ("fov_mm", "champ filmé (mm)"))
    out = []
    for key, lab in labels:
        v = gen.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a, b = v
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                dec = 0 if float(a).is_integer() and float(b).is_integer() else 1
                out.append(f"{lab} : {fmt_num(a, dec)} à {fmt_num(b, dec)}".replace("-", "−"))
            else:
                out.append(f"{lab} : {a} à {b}")
    return out


# ============================================================================ widgets ====
class ConfusionMatrix(QWidget):
    """Matrice de confusion colorée : lignes = vraie particule, colonnes = réponse de l'IA."""

    CELL_W, CELL_H, HEAD_W, HEAD_H = 76, 46, 150, 58

    def __init__(self, confusion: dict | None = None, parent=None):
        super().__init__(parent)
        self.conf = None
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.set_confusion(confusion)

    def set_confusion(self, confusion):
        self.conf = normalize_confusion(confusion)
        self.updateGeometry()
        self.update()
        if self.conf:
            m = self.conf["matrix"]
            self.setAccessibleDescription("; ".join(
                f"{CLASS_NAMES.get(t, t)} : " + ", ".join(f"{CLASS_SHORT.get(p, p)} {m[i][j]}"
                                                         for j, p in enumerate(self.conf["labels_pred"]))
                for i, t in enumerate(self.conf["labels_true"])))

    def sizeHint(self):  # noqa: N802
        if not self.conf:
            return QSize(320, 80)
        return QSize(self.HEAD_W + len(self.conf["labels_pred"]) * self.CELL_W + 8,
                     self.HEAD_H + len(self.conf["labels_true"]) * self.CELL_H + 8)

    def minimumSizeHint(self):  # noqa: N802
        return self.sizeHint()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont(self.font())
        if not self.conf:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Matrice de confusion indisponible")
            return
        lt, lp, m = self.conf["labels_true"], self.conf["labels_pred"], self.conf["matrix"]
        x0, y0 = self.HEAD_W, self.HEAD_H
        cw, ch = self.CELL_W, self.CELL_H
        f.setPixelSize(10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(FAINT))
        p.drawText(QRectF(x0, 0, cw * len(lp), 18), int(Qt.AlignmentFlag.AlignCenter), "RÉPONSE DE L'IA  →")
        p.drawText(QRectF(0, y0 - 22, self.HEAD_W - 12, 18),
                   int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), "VRAIE PARTICULE  ↓")
        f.setPixelSize(15)
        p.setFont(f)
        for j, c in enumerate(lp):       # en-têtes de colonnes
            p.setPen(QColor(cls_text(c)))
            p.drawText(QRectF(x0 + j * cw, 20, cw, y0 - 24), int(Qt.AlignmentFlag.AlignCenter), CLASS_SHORT.get(c, c))
        f.setPixelSize(12)
        for i, c in enumerate(lt):       # en-têtes de lignes
            p.setFont(f)
            p.setPen(QColor(cls_text(c)))
            p.drawText(QRectF(0, y0 + i * ch, self.HEAD_W - 12, ch),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), CLASS_NAMES.get(c, c))
        for i, t in enumerate(lt):
            row_n = sum(m[i]) or 1
            for j, pr in enumerate(lp):
                v = m[i][j]
                frac = v / row_n
                r = QRectF(x0 + j * cw + 2, y0 + i * ch + 2, cw - 4, ch - 4)
                if pr == t:
                    base = QColor(SUCCESS)
                elif pr == "incertain":
                    base = QColor("#8b8d98")
                else:
                    base = QColor(DANGER)
                fill = QColor(base)
                fill.setAlpha(int(18 + 132 * frac) if v else 8)     # reste clair : texte encre lisible (AA)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(fill)
                p.drawRoundedRect(r, 7, 7)
                if not v:
                    p.setPen(QPen(QColor(B.BORDER), 1))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRoundedRect(r, 7, 7)
                if pr == t:
                    p.setPen(QPen(QColor(SUCCESS), 1.6))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRoundedRect(r, 7, 7)
                f2 = QFont(self.font())
                f2.setPixelSize(14)
                f2.setBold(True)
                p.setFont(f2)
                p.setPen(QColor(B.INK if v else FAINT))
                p.drawText(r.adjusted(0, 3, 0, -ch * 0.38), int(Qt.AlignmentFlag.AlignCenter), str(v))
                f2.setPixelSize(10)
                f2.setBold(False)
                p.setFont(f2)
                p.setPen(QColor(B.INK if v else FAINT))
                p.drawText(r.adjusted(0, ch * 0.42, 0, -2), int(Qt.AlignmentFlag.AlignCenter),
                           _pct(frac) if v else "")


class _Tile(QFrame):
    def __init__(self, title: str, value: str, sub: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("lcTile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(1)
        t = QLabel(title.upper())
        t.setObjectName("lcTileTitle")
        self.value = QLabel(value)
        self.value.setObjectName("lcTileValue")
        self.value.setTextFormat(Qt.TextFormat.RichText)
        self.sub = QLabel(sub)
        self.sub.setObjectName("lcTileSub")
        self.sub.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)
        self.setMinimumWidth(170)
        self.setAccessibleName(f"{title} : {re.sub('<[^>]+>', '', value)}")


def _chip(text: str, color: str = B.FOREST) -> QLabel:
    c = QLabel(text)
    c.setObjectName("lcChip")
    c.setStyleSheet(f"QLabel#lcChip {{ color: {color}; }}")
    return c


def _rich(text: str, name: str, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(name)
    lab.setTextFormat(Qt.TextFormat.RichText)
    lab.setWordWrap(wrap)
    return lab


def _metrics_table(metrics: dict, classes_hint: list[str] | None = None) -> QWidget:
    """Tableau Classe | Précision | Rappel | Traces (grille de QLabel, lisible et sans barre de défilement)."""
    box = QWidget()
    g = QGridLayout(box)
    g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(14)
    g.setVerticalSpacing(5)
    heads = ("Classe", "Précision", "Rappel", "Traces")
    tips = ("", "Quand l'IA répond cette classe, à quelle fréquence a-t-elle raison ?",
            "Parmi toutes les vraies traces de cette classe, quelle part l'IA a-t-elle reconnue ?",
            "Nombre de vraies traces de cette classe dans l'examen")
    for j, (h, tip) in enumerate(zip(heads, tips)):
        lab = QLabel(h.upper())
        lab.setObjectName("lcTh")
        lab.setToolTip(tip)
        g.addWidget(lab, 0, j, Qt.AlignmentFlag.AlignLeft if j == 0 else Qt.AlignmentFlag.AlignRight)
    per = metrics.get("per_class") or {}
    order = [c for c in CLASS_ORDER if c in per] + [c for c in per if c not in CLASS_ORDER]
    for i, c in enumerate(order, start=1):
        v = per[c]
        name = QLabel(f"<span style='color:{cls_text(c)}; font-weight:700'>{CLASS_SHORT.get(c, c)}</span>"
                      f"&nbsp;&nbsp;{html.escape(CLASS_NAMES.get(c, c))}")
        name.setObjectName("lcTd")
        name.setTextFormat(Qt.TextFormat.RichText)
        n = v.get("n")
        rec = v.get("recall")
        rec_txt = _pct(rec)
        if rec is not None and n:
            rec_txt += f" <span style='color:{FAINT}'>({int(round(rec * n))}/{n})</span>"
        prec = QLabel(_pct(v.get("precision")))
        recl = QLabel(rec_txt)
        recl.setTextFormat(Qt.TextFormat.RichText)
        nl = QLabel(fmt_num(n, 0) if n is not None else "—")
        for lab in (prec, recl, nl):
            lab.setObjectName("lcTdNum")
        if n is not None and 0 < n < 20:
            nl.setToolTip("Moins de 20 traces : pourcentage très incertain")
            nl.setText(f"{n} <span style='color:{WARN}'>⚠</span>")
            nl.setTextFormat(Qt.TextFormat.RichText)
        g.addWidget(name, i, 0)
        g.addWidget(prec, i, 1, Qt.AlignmentFlag.AlignRight)
        g.addWidget(recl, i, 2, Qt.AlignmentFlag.AlignRight)
        g.addWidget(nl, i, 3, Qt.AlignmentFlag.AlignRight)
    g.setColumnStretch(0, 1)
    return box


def _accuracy_text(m: dict) -> str:
    acc = m.get("accuracy_confident")
    if acc is None:
        return "—"
    txt = _pct(acc)
    k, n = m.get("n_correct"), m.get("n_confident")
    ci = wilson_interval(k, n) if k is not None and n else None
    if ci:
        half = 100 * (ci[1] - ci[0]) / 2
        txt += f"<span style='font-size:9pt; color:{MUTED}'> ±{NBSP}{fmt_num(half, 0)}</span>"
    return txt


class _ResultCard(QFrame):
    """Résultats d'un examen (simulations jamais vues ou conditions extrêmes)."""

    def __init__(self, icon: str, title: str, subtitle: str, metrics: dict | None, threshold: float | None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("lcCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 18)
        lay.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(_tinted_icon(icon, ACCENT, 38, 10, 18), 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        t = QLabel(title)
        t.setObjectName("lcCardTitle")
        s = QLabel(subtitle)
        s.setObjectName("lcMuted")
        s.setWordWrap(True)
        col.addWidget(t)
        col.addWidget(s)
        head.addLayout(col, 1)
        lay.addLayout(head)
        if not metrics:
            e = QLabel("Pas de résultats d'examen enregistrés dans ce modèle.")
            e.setObjectName("lcEmpty")
            e.setWordWrap(True)
            lay.addWidget(e)
            lay.addStretch(1)
            return
        row = QHBoxLayout()
        row.setSpacing(10)
        n = metrics.get("n")
        if metrics.get("legacy"):          # mesuré avant le seuil d'abstention actuel : ne pas l'afficher
            threshold = None
        cov_sub = "des traces reçoivent une classe" + (f" (seuil {_pct(threshold)})" if threshold else "")
        for title_, value, sub, tip in (
                ("Couverture", _pct(metrics.get("coverage")), cov_sub,
                 "Part des traces auxquelles l'IA ose donner une classe ; les autres sont « incertaines »."),
                ("Exactitude", _accuracy_text(metrics), "parmi les traces classées",
                 "Part des réponses correctes parmi les traces auxquelles l'IA a donné une classe "
                 "(± : intervalle de confiance à 95 %)."),
                ("Équilibrée", _pct(metrics.get("balanced_accuracy")), "moyenne des rappels",
                 "Moyenne des rappels de chaque classe : une classe rare (protons) compte autant qu'une "
                 "classe fréquente.")):
            tile = _Tile(title_, value, sub)
            tile.setToolTip(tip)
            tile.setMinimumWidth(120)
            row.addWidget(tile, 1)
        lay.addLayout(row)
        lay.addWidget(_metrics_table(metrics))
        foot = []
        if n:
            foot.append(f"{fmt_num(n, 0)} traces dans cet examen")
        meth = metrics.get("methods") or {}
        if meth:
            names = {"rules": "règles physiques seules", "ml": "IA seule", "hybrid": "combinaison"}
            foot.append("exactitude équilibrée — " + " · ".join(f"{names[k]} : {_pct(v)}" for k, v in meth.items()
                                                               if k in names))
        if metrics.get("legacy"):
            foot.append("modèle pilote : chiffres mesurés avant l'ajout du seuil de confiance (décision hybride "
                        "règles + IA)")
        if foot:
            fl = QLabel(" · ".join(foot))
            fl.setObjectName("lcFoot")
            fl.setWordWrap(True)
            lay.addWidget(fl)
        lay.addStretch(1)


class _StepCard(QFrame):
    def __init__(self, num: str, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("lcStep")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(8)
        n = QLabel(num)
        n.setObjectName("lcStepNum")
        n.setFixedSize(30, 30)
        n.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t = QLabel(title)
        t.setObjectName("lcStepTitle")
        t.setWordWrap(True)
        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(n, 0, Qt.AlignmentFlag.AlignTop)
        head.addWidget(t, 1)
        lay.addLayout(head)
        self.body = _rich(text, "lcStepText")
        lay.addWidget(self.body)
        lay.addStretch(1)


class _Thumb(QLabel):
    SIZE = 92

    def __init__(self, path: Path, cls: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setToolTip(f"{CLASS_NAMES.get(cls, cls)} — {path.name}")
        src = QPixmap(str(path))
        pm = QPixmap(self.SIZE, self.SIZE)
        pm.fill(Qt.GlobalColor.transparent)
        q = QPainter(pm)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0.5, 0.5, self.SIZE - 1, self.SIZE - 1), 10, 10)
        q.fillPath(clip, QColor(10, 11, 14))
        q.setClipPath(clip)
        if not src.isNull():
            s = src.scaled(self.SIZE, self.SIZE, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                           Qt.TransformationMode.SmoothTransformation)
            q.drawPixmap(int((self.SIZE - s.width()) / 2), int((self.SIZE - s.height()) / 2), s)
        q.setClipping(False)
        q.setPen(QPen(QColor(cls_hex(cls)), 1.5))
        q.drawPath(clip)
        q.end()
        self.setPixmap(pm)


LC_QSS = f"""
QWidget#lcContent {{ background: {B.SOFT}; }}
QScrollArea#lcScroll {{ background: {B.SOFT}; border: none; }}
QFrame#lcHero {{ border: none; border-radius: 20px; background: {B.FOREST}; }}
QLabel#lcEyebrow {{ color: {B.MINT}; font-size: 8pt; font-weight: 700; }}
QLabel#lcTitle {{ color: {B.WHITE}; font-size: 24pt; font-weight: 700; }}
QLabel#lcTagline {{ color: {B.WHITE}; font-size: 12pt; }}
QLabel#lcSection {{ color: {B.INK}; font-size: 15pt; font-weight: 700; }}
QLabel#lcSectionSub {{ color: {B.MUTED}; }}
QLabel#lcMuted {{ color: {B.MUTED}; }}
QLabel#lcText {{ color: {B.INK}; }}
QFrame#lcStep {{ background: {B.WHITE}; border: 1px solid {B.BORDER}; border-radius: 16px; }}
QLabel#lcStepNum {{ color: {B.MINT}; background: {B.FOREST}; border-radius: 15px; font-weight: 700; }}
QLabel#lcStepTitle {{ color: {B.INK}; font-size: 11pt; font-weight: 700; }}
QLabel#lcStepText {{ color: {B.MUTED}; }}
QLabel#lcChip {{ background: {B.MINT_SOFT}; border: 1px solid #BFE6C4; border-radius: 12px;
    padding: 4px 11px; font-size: 9pt; font-weight: 600; }}
QFrame#lcTile {{ background: {B.WHITE}; border: 1px solid {B.BORDER}; border-radius: 14px; }}
QFrame#lcCard QFrame#lcTile, QFrame#lcValid QFrame#lcTile {{ background: {B.ALT_ROW}; }}
QLabel#lcTileTitle {{ color: {B.GREEN}; font-size: 8pt; font-weight: 700; }}
QLabel#lcTileValue {{ color: {B.INK}; font-size: 16pt; font-weight: 700; }}
QLabel#lcTileSub {{ color: {B.MUTED}; font-size: 9pt; }}
QFrame#lcCard {{ background: {B.WHITE}; border: 1px solid {B.BORDER}; border-radius: 18px; }}
QLabel#lcCardTitle {{ color: {B.INK}; font-size: 12pt; font-weight: 700; }}
QLabel#lcTh {{ color: {B.FAINT}; font-size: 8pt; font-weight: 700; }}
QLabel#lcTd {{ color: {B.INK}; }}
QLabel#lcTdNum {{ color: {B.INK}; font-weight: 600; }}
QLabel#lcFoot {{ color: {B.FAINT}; font-size: 9pt; }}
QLabel#lcEmpty {{ color: {B.MUTED}; padding: 14px; background: {B.ALT_ROW}; border: 1px dashed {B.BORDER_STRONG};
    border-radius: 12px; }}
QFrame#lcHonest {{ background: {B.WARN_BG}; border: 1px solid {B.WARN_BORDER}; border-radius: 18px; }}
QLabel#lcHonestTitle {{ color: {B.WARN_TEXT}; font-size: 12pt; font-weight: 700; }}
QLabel#lcHonestText {{ color: {B.INK}; }}
QFrame#lcMissing {{ background: {B.DANGER_BG}; border: 1px solid {B.DANGER_BORDER}; border-radius: 18px; }}
QLabel#lcMissingTitle {{ color: {B.DANGER_TEXT}; font-size: 12pt; font-weight: 700; }}
QFrame#lcValid {{ background: {B.WHITE}; border: 1px solid {B.BORDER}; border-radius: 18px; }}
QLabel#lcNote {{ color: {B.INK}; background: {B.HOVER}; border-radius: 10px; padding: 8px 10px; }}
QFrame#lcLine {{ background: {B.BORDER}; border: none; }}
QLabel#lcClassHead {{ font-weight: 700; }}
"""


# ============================================================================ page ====
class LearningCenter(QWidget):
    """« Centre d'apprentissage de l'IA » (onglet ou fenêtre).

    Signal ``validate_requested()`` : l'utilisateur veut mesurer la précision sur SES traces corrigées ;
    la fenêtre principale calcule puis appelle ``show_validation(result)``.
    """
    validate_requested = Signal()

    def __init__(self, parent=None, examples_dir: str | Path | None = None):
        super().__init__(parent)
        self.setObjectName("LearningCenter")
        self.setStyleSheet(LC_QSS)
        self.examples_dir = Path(examples_dir) if examples_dir else EXAMPLES_DIR
        self.summary: dict = summarize_model(None)
        self.load_error = ""
        self.validation: dict | None = None
        self._step_cols = 0
        self._res_cols = 0
        self._tile_cols = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("lcScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        content = QWidget()
        content.setObjectName("lcContent")
        self.scroll.setWidget(content)
        hl = QHBoxLayout(content)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addStretch(1)
        center = QWidget()
        center.setMaximumWidth(1240)
        center.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        hl.addWidget(center, 100)
        hl.addStretch(1)
        self._center = center
        center.installEventFilter(self)
        v = QVBoxLayout(center)
        v.setContentsMargins(28, 24, 28, 28)
        v.setSpacing(16)
        self._v = v

        v.addWidget(self._hero())
        self.missing_box = QFrame()
        self.missing_box.setObjectName("lcMissing")
        ml = QVBoxLayout(self.missing_box)
        ml.setContentsMargins(20, 16, 20, 16)
        ml.setSpacing(6)
        mt = QLabel("🧠  Aucune IA de référence n'est installée")
        mt.setObjectName("lcMissingTitle")
        self.missing_text = _rich("", "lcText")
        ml.addWidget(mt)
        ml.addWidget(self.missing_text)
        v.addWidget(self.missing_box)

        v.addLayout(self._section("Comme un bébé : apprendre par l'exemple",
                                  "Un bébé apprend à reconnaître un chat en voyant beaucoup de chats, avec quelqu'un qui "
                                  "lui dit « c'est un chat ». L'IA de référence a appris de la même façon : on lui a "
                                  "montré beaucoup de traces en lui donnant chaque fois la bonne réponse "
                                  "(apprentissage supervisé)."))
        self._steps_host = QWidget()
        self._steps_grid = QGridLayout(self._steps_host)
        self._steps_grid.setContentsMargins(0, 0, 0, 0)
        self._steps_grid.setHorizontalSpacing(14)
        self._steps_grid.setVerticalSpacing(14)
        v.addWidget(self._steps_host)
        self.steps: list[_StepCard] = []
        kinds_row = QHBoxLayout()
        kinds_row.setSpacing(10)
        kl = QLabel("Familles de traces simulées :")
        kl.setObjectName("lcMuted")
        kinds_row.addWidget(kl, 0, Qt.AlignmentFlag.AlignTop)
        self._kinds_host = QWidget()
        self._kinds_flow = FlowLayout(self._kinds_host, 6, 6)
        sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self._kinds_host.setSizePolicy(sp)
        kinds_row.addWidget(self._kinds_host, 1)
        v.addLayout(kinds_row)

        v.addSpacing(6)
        v.addLayout(self._section("Ce qu'elle a vu", "Chiffres lus dans le modèle installé avec l'application."))
        self._tiles_host = QWidget()
        self._tiles_grid = QGridLayout(self._tiles_host)
        self._tiles_grid.setContentsMargins(0, 0, 0, 0)
        self._tiles_grid.setHorizontalSpacing(12)
        self._tiles_grid.setVerticalSpacing(12)
        self.tiles: list[_Tile] = []
        v.addWidget(self._tiles_host)
        self.generator_label = _rich("", "lcMuted")
        v.addWidget(self.generator_label)

        v.addSpacing(6)
        v.addLayout(self._section(
            "Son examen final",
            "<b>Précision</b> : quand l'IA répond « alpha », à quelle fréquence a-t-elle raison ? "
            "<b>Rappel</b> : parmi toutes les vraies alphas, combien en a-t-elle reconnues ? "
            "Les examens portent sur des enregistrements simulés que l'IA n'a jamais vus pendant son apprentissage."))
        self._results_host = QWidget()
        self._results_grid = QGridLayout(self._results_host)
        self._results_grid.setContentsMargins(0, 0, 0, 0)
        self._results_grid.setHorizontalSpacing(14)
        self._results_grid.setVerticalSpacing(14)
        v.addWidget(self._results_host)
        self.result_cards: list[_ResultCard] = []

        v.addSpacing(6)
        v.addLayout(self._section("Où se trompe-t-elle ?",
                                  "Matrice de confusion de l'examen : chaque ligne est la vraie particule, chaque "
                                  "colonne la réponse de l'IA. Les cases vertes (diagonale) sont les bonnes réponses ; "
                                  "les rouges, les confusions ; la colonne « ? », les traces déclarées incertaines."))
        conf_card = QFrame()
        conf_card.setObjectName("lcCard")
        cl = QHBoxLayout(conf_card)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(24)
        self.confusion = ConfusionMatrix(None)
        cl.addWidget(self.confusion, 0, Qt.AlignmentFlag.AlignTop)
        self.confusion_note = _rich("", "lcText")
        self.confusion_note.setMinimumWidth(220)
        cl.addWidget(self.confusion_note, 1, Qt.AlignmentFlag.AlignTop)
        v.addWidget(conf_card)

        self.examples_title = self._section("Exemples de traces vues pendant l'apprentissage",
                                            "Vignettes de traces simulées, telles que le détecteur les a découpées.")
        self._examples_header = QWidget()
        self.examples_title.setContentsMargins(0, 0, 0, 0)
        self._examples_header.setLayout(self.examples_title)
        v.addWidget(self._examples_header)
        self._examples_host = QWidget()
        self._examples_lay = QVBoxLayout(self._examples_host)
        self._examples_lay.setContentsMargins(0, 0, 0, 0)
        self._examples_lay.setSpacing(10)
        v.addWidget(self._examples_host)

        v.addSpacing(6)
        self.honest_box = QFrame()
        self.honest_box.setObjectName("lcHonest")
        hb = QVBoxLayout(self.honest_box)
        hb.setContentsMargins(22, 18, 22, 18)
        hb.setSpacing(8)
        ht = QLabel("🔎  Ce que ces chiffres veulent dire — et ce qu'ils ne veulent pas dire")
        ht.setObjectName("lcHonestTitle")
        ht.setWordWrap(True)
        self.honest_text = _rich("", "lcHonestText")
        hb.addWidget(ht)
        hb.addWidget(self.honest_text)
        v.addWidget(self.honest_box)

        v.addSpacing(6)
        self.validate_btn = QPushButton("✔  Mesurer la précision sur mes traces corrigées")
        self.validate_btn.setProperty("success", True)
        self.validate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.validate_btn.setToolTip("Compare la réponse de l'IA aux classes que vous avez corrigées à la main")
        self.validate_btn.clicked.connect(self.validate_requested)
        v.addLayout(self._section("Et sur VOTRE chambre ?",
                                  "La seule façon de connaître la vraie précision : comparer l'IA à des traces de vos "
                                  "vidéos que vous avez vérifiées vous-mêmes.", self.validate_btn))
        self.valid_box = QFrame()
        self.valid_box.setObjectName("lcValid")
        self._valid_lay = QVBoxLayout(self.valid_box)
        self._valid_lay.setContentsMargins(20, 16, 20, 16)
        self._valid_lay.setSpacing(10)
        v.addWidget(self.valid_box)

        line = QFrame()
        line.setObjectName("lcLine")
        line.setFixedHeight(1)
        v.addSpacing(6)
        v.addWidget(line)
        self.source_label = _rich("", "lcFoot")
        v.addWidget(self.source_label)
        v.addStretch(1)

        self.refresh()
        self.show_validation(None)

    # ------------------------------------------------------------------ construction --
    def _hero(self) -> QFrame:
        hero = QFrame()
        hero.setObjectName("lcHero")
        lay = QHBoxLayout(hero)
        lay.setContentsMargins(30, 24, 30, 24)
        lay.setSpacing(20)
        lay.addWidget(_tinted_icon("🧠", TINT, 64, 18, 32), 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(4)
        e = QLabel("INTELLIGENCE ARTIFICIELLE  ·  APPRENTISSAGE SUR SIMULATIONS")
        e.setObjectName("lcEyebrow")
        t = QLabel("Centre d'apprentissage de l'IA")
        t.setObjectName("lcTitle")
        g = QLabel("Comment l'IA de référence a appris à reconnaître les particules — et ce que valent vraiment "
                   "ses résultats.")
        g.setObjectName("lcTagline")
        g.setWordWrap(True)
        col.addWidget(e)
        col.addWidget(t)
        col.addWidget(g)
        lay.addLayout(col, 1)
        self.refresh_btn = QPushButton("↻  Relire le modèle")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.setToolTip("Relire le fichier du modèle (après un nouvel entraînement)")
        self.refresh_btn.clicked.connect(self.reload)
        lay.addWidget(self.refresh_btn, 0, Qt.AlignmentFlag.AlignTop)
        return hero

    def _section(self, title: str, subtitle: str = "", right: QWidget | None = None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(3)
        t = QLabel(title)
        t.setObjectName("lcSection")
        col.addWidget(t)
        if subtitle:
            col.addWidget(_rich(subtitle, "lcSectionSub"))
        row.addLayout(col, 1)
        if right is not None:
            row.addWidget(right, 0, Qt.AlignmentFlag.AlignBottom)
        return row

    # ------------------------------------------------------------------ données --
    def reload(self):
        """Vide le cache du modèle puis relit le fichier."""
        try:
            from ..detection import reference_model as rm
            if hasattr(rm, "clear_cache"):
                rm.clear_cache()
        except Exception:
            pass
        self.refresh()

    def refresh(self):
        """Relit le modèle de référence (ReferenceModel.load_default()) et met la page à jour."""
        model, rm, err = load_reference()
        self.load_error = err
        self.summary = summarize_model(model, rm)
        s = self.summary
        self.missing_box.setVisible(not s["available"])
        self.missing_text.setText(
            "L'application fonctionne normalement avec les <b>règles physiques seules</b> (classification "
            "explicable trace par trace). Le fichier du modèle (<i>wilsontrack/assets/modele_reference.joblib</i>) "
            "est absent, illisible ou a été créé avec une autre version de scikit-learn."
            + (f"<br><span style='color:{FAINT}'>Détail : {html.escape(err)}</span>" if err else "")
            + "<br>Pour le recréer : <code>python tools/train_reference_model.py</code>.")
        self._fill_steps(s)
        self._fill_kinds(s)
        self._fill_tiles(s)
        self._fill_results(s)
        self._fill_confusion(s)
        self._fill_examples()
        self._fill_honest(s)
        src = []
        if s["available"]:
            src.append("Chiffres lus dans <i>wilsontrack/assets/modele_reference.joblib</i>")
            if s["created"]:
                src.append(f"modèle créé le {_fmt_created(s['created'])}")
            if s["sklearn_version"]:
                src.append(f"scikit-learn {html.escape(s['sklearn_version'])}")
        else:
            src.append("Aucun modèle : aucun chiffre d'examen n'est affiché.")
        src.append("Toutes ces mesures portent sur des données SIMULÉES.")
        self.source_label.setText(" · ".join(src))
        self._step_cols = self._res_cols = self._tile_cols = 0
        self._reflow()

    def _clear(self, layout):
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget()
            if w is not None:
                _discard(w)
            elif it.layout() is not None:
                self._clear(it.layout())

    def _fill_steps(self, s: dict):
        self._clear(self._steps_grid)
        n_tr, n_jobs = s["n_tracks_total"], s["n_jobs"]
        many = (f"<b>{fmt_num(n_tr, 0)} traces simulées</b>" if n_tr else "des traces simulées")
        runs = (f" tirées de <b>{fmt_num(n_jobs, 0)} enregistrements</b> de chambre simulés" if n_jobs else "")
        gen = _generator_lines(s["generator"])
        varied = ("Résolution, cadence, bruit de la caméra, brouillard, luminosité, épaisseur et durée de vie des "
                  "traces, température, épaisseur de la couche sensible, calibration présente ou non : tout est tiré "
                  "au hasard pour chaque enregistrement.")
        if gen:
            varied += f"<br><span style='color:{FAINT}'>Par exemple {html.escape(gen[0])}.</span>"
        texts = [
            ("1", "On lui montre des exemples",
             f"Le simulateur physique de WilsonTrack a produit {many}{runs}. Pour chacune, on connaît la vraie "
             "particule : c'est la « bonne réponse » montrée à l'IA."),
            ("2", "Dans des conditions très variées", varied),
            ("3", "Elle regarde comme l'application",
             "Chaque vidéo simulée passe par le <b>vrai détecteur</b> de l'application : l'IA apprend à partir des "
             "mêmes mesures que pour vos vidéos (longueur, épaisseur, densité, rectitude, courbure…)."),
            ("4", "Examen sur des traces jamais vues",
             "On la note ensuite sur des enregistrements simulés qu'elle n'a <b>jamais vus</b>, puis sur un examen "
             "« conditions extrêmes » (image très sombre, très claire ou très bruitée)."),
        ]
        self.steps = [_StepCard(*t) for t in texts]

    def _fill_kinds(self, s: dict):
        while self._kinds_flow.count():
            it = self._kinds_flow.takeAt(0)
            if it.widget() is not None:
                _discard(it.widget())
        kinds = s["kinds"] or [(k, None) for k in DEFAULT_KINDS]
        for label, count in kinds:
            txt = label + (f" · {fmt_num(count, 0)}" if count else "")
            self._kinds_flow.addWidget(_chip(txt))
        self._kinds_host.updateGeometry()

    def _fill_tiles(self, s: dict):
        self._clear(self._tiles_grid)
        self.tiles = []
        avail = s["available"]
        mt = s["model_type"]
        tiles = [
            ("Traces étiquetées", fmt_num(s["n_tracks_total"], 0) if s["n_tracks_total"] else "—",
             "traces simulées dont on connaît la vraie nature"),
            ("Enregistrements", fmt_num(s["n_jobs"], 0) if s["n_jobs"] else "—", "chambres simulées différentes"),
            ("Pour apprendre", fmt_num(s["n_train"], 0) if s["n_train"] else "—",
             "traces d'entraînement (le reste sert à l'examen)"),
            ("Créé le", _fmt_created(s["created"]) if s["created"] else "—", "date de l'entraînement"),
            ("Type de modèle", html.escape(MODEL_SHORT.get(mt, mt.replace("Classifier", ""))) if mt else "—",
             MODEL_TYPES.get(mt, "méthode d'apprentissage automatique") if mt else ""),
            ("Seuil de confiance", _pct(s["threshold"]) if s["threshold"] else "—",
             "en dessous : trace « incertaine »" if s["threshold"] else "non indiqué dans ce modèle"),
        ]
        for title, value, sub in tiles:
            t = _Tile(title, value if avail or title == "Seuil de confiance" else "—", sub)
            t.setMinimumWidth(140)
            if title in ("Type de modèle", "Créé le"):       # textes plus longs que des nombres
                t.value.setStyleSheet("font-size: 12pt;")
                t.value.setWordWrap(True)
            if title == "Type de modèle" and mt:
                t.setToolTip(f"scikit-learn : {mt}")
            self.tiles.append(t)
        gen = _generator_lines(s["generator"])
        self.generator_label.setText(("Paramètres tirés au hasard : " + " ; ".join(html.escape(g) for g in gen) + ".")
                                     if gen else "")
        self.generator_label.setVisible(bool(gen))

    def _fill_results(self, s: dict):
        self._clear(self._results_grid)
        sp = s["splits"]
        self.result_cards = [
            _ResultCard("📝", "Examen : simulations jamais vues",
                        "Enregistrements simulés mis de côté avant l'apprentissage.", sp.get("test"), s["threshold"]),
            _ResultCard("🌪", "Conditions extrêmes",
                        "Images très sombres, très claires ou très bruitées : plus difficile que la réalité "
                        "habituelle.", sp.get("stress"), s["threshold"]),
        ]

    def _fill_confusion(self, s: dict):
        conf = s["confusion"]
        self.confusion.set_confusion(conf)
        self.confusion.setVisible(bool(conf))
        if not conf:
            self.confusion_note.setText("Aucune matrice de confusion n'est enregistrée dans ce modèle.")
            return
        m, lt, lp = conf["matrix"], conf["labels_true"], conf["labels_pred"]
        worst = None
        for i, t in enumerate(lt):
            for j, p in enumerate(lp):
                if p != t and p != "incertain" and m[i][j] and (worst is None or m[i][j] > worst[0]):
                    worst = (m[i][j], t, p, sum(m[i]))
        lines = ["<b>Comment la lire :</b> prenez une ligne (la vraie particule) et regardez où sont partis ses "
                 "exemples. Une IA parfaite n'aurait que la diagonale verte."]
        if worst:
            k, t, p, row_n = worst
            lines.append(f"<b>Confusion la plus fréquente :</b> {CLASS_PLURAL.get(t, t)} → réponse "
                         f"« {CLASS_NAMES.get(p, p)} » : {k} cas sur {row_n} ({_pct(k / row_n)}).")
        if {"beta", "muon"} <= set(lt):
            lines.append("La confusion <b>électron ↔ muon</b> est attendue : vus de dessus, un électron rapide et un "
                         "muon presque vertical laissent tous deux une trace courte, fine et droite.")
        self.confusion_note.setText("<br><br>".join(lines))

    def _fill_examples(self):
        self._clear(self._examples_lay)
        groups: dict[str, list[Path]] = {}
        try:
            if self.examples_dir.is_dir():
                for f in sorted(self.examples_dir.glob("*.png")):
                    m = re.match(r"^([a-zA-Z]+)_(\d+)$", f.stem)
                    if m:
                        groups.setdefault(m.group(1).lower(), []).append(f)
        except OSError:
            groups = {}
        self._examples_header.setVisible(bool(groups))
        self._examples_host.setVisible(bool(groups))
        order = [c for c in CLASS_ORDER if c in groups] + [c for c in groups if c not in CLASS_ORDER]
        for cls in order:
            row = QHBoxLayout()
            row.setSpacing(10)
            lab = QLabel(f"<span style='color:{cls_text(cls)}; font-weight:700; font-size:13pt'>"
                         f"{CLASS_SHORT.get(cls, cls)}</span><br><span style='color:{MUTED}'>"
                         f"{html.escape(CLASS_NAMES.get(cls, cls))}</span>")
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setFixedWidth(150)
            row.addWidget(lab)
            files = sorted(groups[cls], key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))[:8]
            for f in files:
                row.addWidget(_Thumb(f, cls))
            row.addStretch(1)
            self._examples_lay.addLayout(row)

    def _fill_honest(self, s: dict):
        test = s["splits"].get("test") or {}
        items = [
            "Ils sont mesurés sur des traces <b>SIMULÉES</b>, dont on connaît la vraie nature. Ce n'est pas une mesure "
            "faite sur une vraie chambre.",
            "L'IA n'a <b>jamais vu VOTRE chambre</b> : votre caméra, votre éclairage, votre alcool et votre "
            "température sont différents. Sa précision réelle peut être plus basse.",
            "Pour connaître sa vraie précision, corrigez à la main quelques dizaines de traces de vos vidéos (onglet "
            "« 🧾 Traces », « Corriger la classe »), puis cliquez sur « Mesurer la précision sur mes traces "
            "corrigées » ci-dessous.",
            "Quand deux particules laissent physiquement la même trace, l'application répond <b>« incertaine »</b> au "
            "lieu de deviner. Exemple : vus de dessus, la plupart des muons traversent la mince couche sensible "
            "(≈ 1 cm) presque à la verticale — seulement environ 1 % laissent une trace de plus de 3 cm (flux en "
            "cos²θ, PDG) — et ressemblent alors à un électron rapide. Des traces « incertaines » sont donc "
            "normales : c'est de l'honnêteté, pas une panne.",
        ]
        n = test.get("n_confident") or test.get("n")
        k = test.get("n_correct")
        if n and k is not None:
            ci = wilson_interval(k, n)
            if ci and n < 2000:
                items.append(f"L'examen ne compte que <b>{fmt_num(test.get('n') or n, 0)} traces</b> : le pourcentage "
                             f"d'exactitude est connu à environ ± {fmt_num(100 * (ci[1] - ci[0]) / 2, 0)} points près "
                             "(intervalle de confiance à 95 %). Les classes rares (protons) sont encore plus "
                             "incertaines.")
        if s["threshold"]:
            items.append(f"Seuil de confiance : {_pct(s['threshold'])}. En dessous, la trace est classée "
                         "« incertaine » plutôt que d'afficher une réponse peu sûre.")
        if s.get("legacy"):
            items.append("Le modèle installé est un <b>modèle pilote</b>"
                         + (f" ({fmt_num(s['n_tracks_total'], 0)} traces)" if s["n_tracks_total"] else "")
                         + " : ses chiffres d'examen ont été mesurés avant l'ajout du seuil de confiance. Ils "
                           "seront remplacés par ceux du prochain entraînement.")
        self.honest_text.setText("".join(f"<p style='margin:0 0 6px 0'>•&nbsp; {t}</p>" for t in items))

    # ------------------------------------------------------------------ validation --
    def show_validation(self, result: dict | None):
        """Affiche la précision mesurée sur les traces corrigées à la main.

        result = {"n": int, "accuracy": float, "per_class": {cls: {"precision","recall","n"}},
                  "confusion": {"labels_true","labels_pred","matrix"}, "note": str}  (None = état initial).
        """
        self.validation = result if isinstance(result, dict) else None
        self._clear(self._valid_lay)
        r = self.validation
        if r is None:
            lab = _rich("Aucune mesure pour l'instant. Corrigez quelques traces dans l'onglet « 🧾 Traces » (« Corriger "
                        "la classe »), puis cliquez sur « Mesurer la précision sur mes traces corrigées ». Plus vous "
                        "corrigez de traces, plus la mesure est fiable (visez au moins 30 traces par classe).",
                        "lcMuted")
            self._valid_lay.addWidget(lab)
            return
        n = _int(r.get("n")) or 0
        if n <= 0:
            lab = _rich("⚠  Aucune trace corrigée à la main : impossible de mesurer la précision sur votre chambre. "
                        "Corrigez d'abord quelques traces dans l'onglet « 🧾 Traces ».", "lcText")
            self._valid_lay.addWidget(lab)
            note = str(r.get("note") or "")
            if note:
                self._valid_lay.addWidget(_rich(html.escape(note), "lcNote"))
            return
        acc = _float(r.get("accuracy"))
        row = QHBoxLayout()
        row.setSpacing(10)
        acc_txt = _pct(acc)
        n_conf = _int(r.get("n_confident"))
        n_conf = n_conf if n_conf is not None and 0 <= n_conf <= n else n   # traces classées (hors « incertaine »)
        ci = r.get("accuracy_ci95")
        try:
            ci = (float(ci[0]), float(ci[1])) if ci is not None else None
        except (TypeError, ValueError, IndexError):
            ci = None
        if ci is None and acc is not None and n_conf:
            ci = wilson_interval(round(acc * n_conf), n_conf)
        if ci:
            acc_txt += (f"<span style='font-size:9pt; color:{MUTED}'> ±{NBSP}"
                        f"{fmt_num(100 * (ci[1] - ci[0]) / 2, 0)}</span>")
        sim = (self.summary["splits"].get("test") or {}).get("accuracy_confident")
        cov = _float(r.get("coverage"))
        row.addWidget(_Tile("Traces vérifiées", fmt_num(n, 0),
                            f"dont {fmt_num(n_conf, 0)} classées ({_pct(cov)})" if cov is not None
                            else "corrigées ou confirmées à la main"), 1)
        row.addWidget(_Tile("Exactitude réelle", acc_txt, "parmi les traces classées, sur VOTRE chambre"), 1)
        row.addWidget(_Tile("Sur simulations", _pct(sim) if sim is not None else "—", "pour comparaison"), 1)
        self._valid_lay.addLayout(row)
        pair = QWidget()
        flow = FlowLayout(pair, 36, 14)          # tableau et matrice côte à côte (l'un sous l'autre si étroit)
        sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        pair.setSizePolicy(sp)
        per = r.get("per_class")
        if isinstance(per, dict) and per:
            m = normalize_metrics({"per_class": per, "n": n})
            if m:
                table = _metrics_table(m)
                table.setMinimumWidth(max(420, table.sizeHint().width()))
                table.setFixedHeight(table.sizeHint().height())
                flow.addWidget(table)
        conf = normalize_confusion(r.get("confusion"))
        if conf:
            flow.addWidget(ConfusionMatrix(conf))
        if flow.count():
            self._valid_lay.addWidget(pair)
        else:
            _discard(pair)
        note = str(r.get("note") or "")
        if n_conf < 30 and "petit" not in note and "incertain" not in note:
            note = (note + "  " if note else "") + (f"Seulement {n_conf} traces classées : la mesure est encore "
                                                    "très incertaine (visez au moins 30 traces par classe).")
        if note:
            self._valid_lay.addWidget(_rich(html.escape(note), "lcNote"))

    # ------------------------------------------------------------------ disposition --
    def _reflow(self):
        m = self._v.contentsMargins()
        w = self._center.width() - m.left() - m.right()
        cols = 4 if w >= 1000 else (2 if w >= 540 else 1)
        if cols != self._step_cols:
            g = self._steps_grid
            for c in range(4):
                g.setColumnStretch(c, 0)
            for i, card in enumerate(self.steps):
                g.addWidget(card, i // cols, i % cols)
            for c in range(cols):
                g.setColumnStretch(c, 1)
            self._step_cols = cols
        tcols = 6 if w >= 1120 else (3 if w >= 520 else 2)
        if tcols != self._tile_cols:
            g = self._tiles_grid
            for c in range(6):
                g.setColumnStretch(c, 0)
            for i, t in enumerate(self.tiles):
                g.addWidget(t, i // tcols, i % tcols)
            for c in range(tcols):
                g.setColumnStretch(c, 1)
            self._tile_cols = tcols
        rcols = 2 if w >= 880 else 1
        if rcols != self._res_cols:
            g = self._results_grid
            for c in range(2):
                g.setColumnStretch(c, 0)
            for i, card in enumerate(self.result_cards):
                g.addWidget(card, i // rcols, i % rcols)
            for c in range(rcols):
                g.setColumnStretch(c, 1)
            self._res_cols = rcols

    def eventFilter(self, obj, ev):  # noqa: N802
        if obj is self._center and ev.type() == QEvent.Type.Resize:
            self._reflow()
        return super().eventFilter(obj, ev)

    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        self._reflow()
