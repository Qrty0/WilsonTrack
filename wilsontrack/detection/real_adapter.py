"""Adaptateur « vraies vidéos » : petit modèle appris sur de VRAIES traces de chambres à brouillard.

L'IA de référence a appris sur 66 000 traces simulées ; sur de vraies vidéos (éclairage, compression,
brouillard, texte incrusté, reflets…) elle hésite beaucoup. L'adaptateur a été entraîné sur 856 traces
réelles de 6 vidéos Wikimedia (licences CC, voir exemples/vidéos réelles (Wikimedia)/SOURCES.md),
étiquetées à l'œil, en réutilisant l'avis de l'IA de simulation et des règles physiques.
Mesuré en laissant chaque fois une vidéo de côté : ≈ 80 % de bonnes réponses quand il se prononce
(artefacts : précision ≈ 81 %, traces fines : ≈ 75 %) ; il ne sait PAS encore reconnaître les alphas
(trop peu d'exemples réels) — ces décisions restent aux règles et à l'IA de simulation.

Il ne fait que deux choses, et seulement s'il est sûr (≥ 85 %) :
  - rejeter une détection comme artefact ;
  - confirmer qu'une trace hésitante est une « trace fine (β ou μ) ».
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ADAPTER_PATH = Path(__file__).resolve().parents[1] / "assets" / "adaptateur_reel.joblib"
THRESHOLD = 0.85
_CACHE: dict = {}


class RealAdapter:
    def __init__(self, payload: dict):
        self.model = payload["model"]
        self.keys = list(payload["feature_keys"])
        self.classes = [str(c) for c in payload["classes"]]
        self.n = int(payload.get("n", 0))
        self.videos = list(payload.get("videos", []))

    @classmethod
    def load_default(cls) -> "RealAdapter | None":
        if "m" in _CACHE:
            return _CACHE["m"]
        m = None
        try:
            import joblib
            if ADAPTER_PATH.exists():
                m = cls(joblib.load(ADAPTER_PATH))
        except Exception:
            m = None
        _CACHE["m"] = m
        return m

    def predict_proba(self, f: dict, duration_s, touches_border, rules_scores: dict, sim_model) -> dict:
        row = []
        for k in self.keys:
            v = f.get(k)
            try:
                x = float(v) if v is not None and not isinstance(v, str) else 0.0
            except (TypeError, ValueError):
                x = 0.0
            row.append(x if math.isfinite(x) else 0.0)
        row += [float(duration_s or 0.0), float(bool(touches_border))]
        from .reference_model import ALL_CLASSES, CLASSES
        p = {}
        if sim_model is not None:
            try:
                p = sim_model.predict_proba(dict(f), duration_s, touches_border, False)
            except Exception:
                p = {}
        row += [float(p.get(c, 0.0)) for c in ALL_CLASSES]
        rs = rules_scores or {}
        row += [float(rs.get(c, 0.0)) for c in CLASSES]
        X = np.nan_to_num(np.array([row], float), nan=0.0, posinf=1e6, neginf=-1e6)
        P = self.model.predict_proba(X)[0]
        return {c: float(P[i]) for i, c in enumerate(self.classes)}


def apply(cls: str, conf: float, reason: str, adapter_p: dict | None) -> tuple[str, float, str, bool]:
    """Ajuste la décision. Renvoie (classe, confiance, raison, modifiée ?)."""
    if not adapter_p:
        return cls, conf, reason, False
    top = max(adapter_p, key=adapter_p.get)
    p = adapter_p[top]
    # Jamais contre une alpha ou un proton : l'adaptateur a vu trop peu d'alphas réelles (une alpha
    # brillante et courte peut lui sembler une tache) et les alphas comptent pour le radon.
    if p >= THRESHOLD and top == "artefact" and cls in ("incertain", "beta", "muon", "fine"):
        return "artefact", p, f"Artefact probable d'après l'adaptateur vidéos réelles ({p:.0%}) — " + reason, True
    if p >= THRESHOLD and top == "fine" and cls == "incertain":
        return "fine", p, f"Trace fine (β ou μ) d'après l'adaptateur vidéos réelles ({p:.0%}) — " + reason, True
    return cls, conf, reason, False
