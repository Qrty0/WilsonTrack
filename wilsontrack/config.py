"""Paramètres de l'application (détection, classification, physique).

Tous les paramètres sont regroupés dans des dataclasses sérialisables en JSON,
ce qui permet de sauvegarder / recharger des « préréglages » pour une chambre donnée.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectionParams:
    # --- Pré-traitement -------------------------------------------------------
    process_width: int = 960        # largeur de traitement en px (0 = résolution native)
    blur_sigma: float = 1.0         # lissage spatial (px, à l'échelle de traitement)
    temporal_avg: int = 2           # moyenne glissante sur N images (réduit le bruit du capteur)
    # --- Modèle de fond --------------------------------------------------------
    bg_time_s: float = 3.0          # constante de temps du fond adaptatif (s)
    bg_warmup_frames: int = 12      # images d'apprentissage initial du fond
    # --- Seuillage (en multiples de l'écart-type du bruit) --------------------
    k_high: float = 4.5             # seuil « graine » (hystérésis haute)
    k_low: float = 2.2              # seuil d'extension (hystérésis basse)
    use_ridge: bool = True          # filtre de crêtes de Hessian (traces fines / pointillées)
    ridge_scales: tuple = (1.5, 3.0)
    close_px: int = 3               # fermeture morphologique (relie les gouttelettes)
    min_area_px: int = 12           # aire minimale d'une composante (px de traitement)
    max_blob_frac: float = 0.25     # si > cette fraction de la zone est « allumée » → perturbation globale
    max_motion_px: float = 0.0      # si > 0 : déplacement global (px) au-delà duquel la caméra est jugée bougée
    max_global_jump: float = 6.0    # saut de luminosité globale (niveaux de gris) → lumière allumée / éteinte
    # --- Fusion de fragments colinéaires (traces pointillées) ----------------
    merge_gap_mm: float = 8.0
    merge_angle_deg: float = 14.0
    # --- Suivi temporel --------------------------------------------------------
    min_visible_s: float = 0.15     # durée minimale de visibilité d'une vraie trace (s)
    min_frames: int = 3             # et au moins ce nombre d'images
    max_missed: int = 5             # images manquées avant de clore une trace
    max_track_s: float = 4.0        # durée de vie maximale d'une trace (au-delà : close)
    match_dilate_px: int = 5        # tolérance spatiale d'association
    # --- Filtres de validation d'une trace ------------------------------------
    min_length_mm: float = 4.0
    min_elongation: float = 2.2     # longueur / largeur minimale
    min_peak_snr: float = 6.0       # contraste minimal (en sigma de bruit) le long de la trace
    # --- Région d'intérêt (polygone normalisé 0..1, vide = image entière) ---
    roi: list = field(default_factory=list)
    roi_margin_frac: float = 0.0
    # --- Stabilisation d'image (compense les petites vibrations) ------------
    stabilize: bool = False


@dataclass
class ClassificationParams:
    # "regles" : règles physiques seules ; "hybride" : règles + IA de référence (entraînée sur des
    # traces simulées, livrée avec l'application ; sans le fichier du modèle → règles seules) ;
    # "ia" : modèle entraîné avec VOS corrections. Défaut « hybride » : meilleure exactitude
    # équilibrée que les règles seules sur les jeux de test ET « conditions extrêmes »
    # (voir wilsontrack/assets/modele_reference.json).
    mode: str = "hybride"
    model_path: str = ""
    # Poids de l'IA de référence dans le mode hybride (0 = règles seules, 1 = IA seule) ;
    # −1 = poids optimal mesuré lors de l'entraînement (enregistré dans le modèle).
    ref_model_weight: float = -1.0
    # Seuil de certitude (probabilité calibrée) en mode hybride : en dessous, la trace est déclarée
    # « incertaine » (avec ses deux classes possibles) au lieu de recevoir une classe peu sûre.
    ml_threshold: float = 0.85
    # Largeur (mm) à partir de laquelle une trace est « épaisse » (ionisation forte)
    alpha_width_mm: float = 1.3
    # Longueur maximale d'une alpha naturelle : ≈ 1,05 × portée ASTAR du Po-212 (8,785 MeV, la plus
    # énergétique des alphas naturelles) dans l'air froid : 78 mm à −25 °C, 86 mm à 0 °C
    # (voir classifier.alpha_max_length_for(T)). 80 mm couvre une couche à ≈ −20 °C.
    alpha_max_length_mm: float = 80.0
    # Longueur minimale d'un muon (trace longue et rectiligne)
    muon_min_length_mm: float = 45.0
    # Écart RMS maximal (mm) à la droite pour une trace fine « droite comme un muon » (0 = critère
    # désactivé). Highland : sur 2–5 cm d'air froid, un électron du K-40 s'écarte de 0,5 à 7 mm,
    # un muon de ≈ 0,001 mm. 0,2 mm : meilleur compromis mesuré sur les traces simulées réalistes
    # (précision des « muons » des règles 0,47 → 0,53) ; muon_min_length_mm reste 45 mm (30 mm n'a
    # pas amélioré les traces fines longues sur ces données).
    muon_max_rms_mm: float = 0.2
    # Rectitude (corde / longueur du chemin) : 1 = parfaitement droit
    straight_min: float = 0.95
    curly_max: float = 0.86
    # Densité d'ionisation relative (par rapport aux traces fines de référence)
    heavy_density_rel: float = 2.2
    heavy_width_rel: float = 1.7
    # Référence automatique = percentile des traces fines du même enregistrement
    ref_percentile: float = 35.0
    min_confidence: float = 0.30    # sous ce score → « incertaine »


@dataclass
class PhysicsParams:
    px_per_mm: float = 0.0          # calibration (px de l'image native par mm); 0 = non calibré
    default_fov_mm: float = 250.0   # champ de vision supposé si non calibré
    chamber_temp_c: float = -25.0   # température de la zone sensible (air froid → alphas plus courtes)
    pressure_kpa: float = 101.3
    b_field_mt: float = 0.0         # champ magnétique (mT) — 0 = pas d'aimant
    sensitive_depth_mm: float = 8.0 # épaisseur de la couche sensible (ordre de grandeur)


@dataclass
class ExpansionParams:
    enabled: bool = False           # mode chambre à expansion
    trigger: str = "auto"           # "auto" (saut de luminosité), "serie" (Arduino), "manuel"
    delay_ms: float = 40.0          # délai entre l'expansion et le début de la fenêtre sensible
    window_ms: float = 600.0        # durée de la fenêtre sensible
    jump_sigma: float = 6.0         # sensibilité de la détection automatique de l'expansion
    serial_port: str = ""
    serial_baud: int = 115200


@dataclass
class RunInfo:
    condition: str = "Fond (rayons cosmiques)"
    sample_type: str = "Aucun"
    sample_mass_g: float = 0.0
    trial: int = 1
    operator: str = ""
    notes: str = ""


@dataclass
class AppConfig:
    detection: DetectionParams = field(default_factory=DetectionParams)
    classification: ClassificationParams = field(default_factory=ClassificationParams)
    physics: PhysicsParams = field(default_factory=PhysicsParams)
    expansion: ExpansionParams = field(default_factory=ExpansionParams)
    run: RunInfo = field(default_factory=RunInfo)
    output_dir: str = ""

    # ------------------------------------------------------------------ I/O --
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "AppConfig":
        cfg = cls()
        _update_dataclass(cfg, d)
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self) -> "AppConfig":
        return AppConfig.from_dict(self.to_dict())


def _update_dataclass(obj: Any, d: dict) -> None:
    """Met à jour récursivement une dataclass à partir d'un dict (clés inconnues ignorées)."""
    for f in fields(obj):
        if f.name not in d:
            continue
        cur = getattr(obj, f.name)
        val = d[f.name]
        if is_dataclass(cur) and isinstance(val, dict):
            _update_dataclass(cur, val)
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(obj, f.name, tuple(val))
        else:
            setattr(obj, f.name, val)


def default_output_dir() -> Path:
    docs = Path.home() / "Documents"
    base = docs if docs.exists() else Path.home()
    return base / "WilsonTrack Résultats"
