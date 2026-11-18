"""Représentation d'une trace finalisée (un « événement » physique compté une seule fois)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Colonnes exportées (ordre des fichiers CSV / Excel)
EXPORT_COLUMNS = [
    "id", "classe", "confiance", "t_debut_s", "t_analyse_s", "t_meilleure_s", "duree_visible_s", "images_vues",
    "x_mm", "y_mm", "longueur_mm", "corde_mm", "epaisseur_mm", "rectitude", "tortuosite_rad_mm",
    "densite_rel", "epaisseur_rel", "densite", "snr_crete", "asymetrie", "embranchements",
    "orientation_deg", "rayon_courbure_mm", "impulsion_MeV_c", "energie_MeV", "touche_bord",
    "cycle_expansion", "classe_manuelle", "famille", "classe_ia", "proba_ia", "methode", "pourquoi", "raison",
]


def track_family(f: dict) -> str:
    """« fine » (particule au minimum d'ionisation : électron ou muon) ou « lourde » (alpha, proton).

    Utilise la « lourdeur » calculée par les règles (épaisseur et densité relatives) ; à défaut,
    la densité d'ionisation relative.
    """
    h = f.get("ev_heavy")
    if h is not None:
        try:
            return "lourde" if float(h) >= 0.5 else "fine"
        except (TypeError, ValueError):
            pass
    d = f.get("density_rel")
    w = f.get("width_rel")
    try:
        return "lourde" if (float(d or 0) >= 1.9 or float(w or 0) >= 1.9) else "fine"
    except (TypeError, ValueError):
        return "fine"


@dataclass
class TrackEvent:
    id: int
    first_frame: int
    last_frame: int
    best_frame: int
    t_first: float
    t_best: float
    t_last: float
    n_seen: int
    bbox_proc: tuple                  # (x0, y0, x1, y1) à l'échelle de traitement
    centroid_proc: tuple              # (x, y)
    features: dict
    path_full: np.ndarray             # polyligne (x, y) en px de l'image native
    thumb: np.ndarray | None = None   # vignette couleur (BGR)
    thumb_mask: np.ndarray | None = None
    touches_border: bool = False
    cls: str = "incertain"
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)
    reason: str = ""
    manual_cls: str = ""              # étiquette donnée par l'utilisateur (entraînement IA)
    valid: bool = True
    reject_reason: str = ""
    expansion_cycle: int = -1
    px_per_mm_full: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def final_cls(self) -> str:
        return self.manual_cls or self.cls

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_last - self.t_first)

    def why_short(self) -> str:
        """Résumé d'une ligne de l'explication « pourquoi cette classe ? » (mis en cache)."""
        key = (self.final_cls, round(float(self.confidence), 4), self.manual_cls)
        cached = self.extra.get("_why_short")
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from .explain import explain_event
            txt = explain_event(self, self.extra.get("cls_params"), self.extra.get("phys_params")).short()
        except Exception:
            txt = self.reason
        self.extra["_why_short"] = (key, txt)
        return txt

    def to_row(self) -> dict:
        f = self.features
        cx, cy = self.centroid_proc
        sc = self.extra.get("scale", 1.0)
        mm = 1.0 / max(self.px_per_mm_full, 1e-9)
        r = f.get("radius_mm", float("inf"))
        return {
            "id": self.id,
            "classe": self.final_cls,
            "confiance": round(self.confidence, 3),
            "t_debut_s": round(self.t_first, 3),
            "t_analyse_s": round(float(self.extra.get("t_ana", self.t_first)), 3),
            "t_meilleure_s": round(self.t_best, 3),
            "duree_visible_s": round(self.duration_s, 3),
            "images_vues": self.n_seen,
            "x_mm": round(cx / sc * mm, 2),
            "y_mm": round(cy / sc * mm, 2),
            "longueur_mm": round(f.get("length_mm", 0.0), 2),
            "corde_mm": round(f.get("chord_mm", 0.0), 2),
            "epaisseur_mm": round(f.get("width_mm", 0.0), 3),
            "rectitude": round(f.get("straightness", 0.0), 4),
            "tortuosite_rad_mm": round(f.get("tortuosity", 0.0), 4),
            "densite_rel": round(f.get("density_rel", 1.0), 3),
            "epaisseur_rel": round(f.get("width_rel", 1.0), 3),
            "densite": round(f.get("density", 0.0), 3),
            "snr_crete": round(f.get("peak_snr", 0.0), 2),
            "asymetrie": round(f.get("asymmetry", 1.0), 3),
            "embranchements": f.get("branches", 0),
            "orientation_deg": round(f.get("orientation_deg", 0.0), 1),
            "rayon_courbure_mm": round(r, 1) if math.isfinite(r) else "",
            "impulsion_MeV_c": round(f["momentum_mev"], 4) if f.get("momentum_mev") else "",
            "energie_MeV": round(f["energy_mev"], 3) if f.get("energy_mev") else "",   # alphas : énergie MINIMALE
            "touche_bord": int(self.touches_border),
            "cycle_expansion": self.expansion_cycle if self.expansion_cycle >= 0 else "",
            "classe_manuelle": self.manual_cls,
            "famille": self.extra.get("family") or track_family(f),
            "classe_ia": self.extra.get("ml_cls", ""),
            "proba_ia": round(float(self.extra["ml_conf"]), 3) if self.extra.get("ml_conf") is not None else "",
            "methode": "Correction manuelle" if self.manual_cls else (self.extra.get("method") or "Règles physiques"),
            "pourquoi": self.why_short(),
            "raison": self.reason,
        }
