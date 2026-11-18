"""Configuration commune des tests WilsonTrack.

- ajoute la racine du projet à sys.path (pour « import wilsontrack ») ;
- fournit les enregistrements d'exemple sauvegardés (fond / KCl) comme fixtures.

Les enregistrements d'exemple peuvent être déplacés : définir la variable
d'environnement WILSONTRACK_SAMPLE_RUNS vers le dossier qui contient
« 2026-09-29 Fond (rayons cosmiques) essai-01 » et « 2026-09-29 KCl essai-01 ».
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Interface : pas d'animation d'accueil pendant les tests (préférence « mouvement réduit »)
os.environ.setdefault("WILSONTRACK_NO_ANIM", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEFAULT_RUNS = Path(
    r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-Desktop-Chambre-de-wilson-programme"
    r"\d6e18416-39fc-45d6-957c-9b0e27e49c12\scratchpad\runs"
)
SAMPLE_RUNS_DIR = Path(os.environ.get("WILSONTRACK_SAMPLE_RUNS", str(_DEFAULT_RUNS)))
BG_RUN_NAME = "2026-09-29 Fond (rayons cosmiques) essai-01"
KCL_RUN_NAME = "2026-09-29 KCl essai-01"


def _load_run(name: str):
    from wilsontrack.analysis.run import RunResult
    folder = SAMPLE_RUNS_DIR / name
    if not (folder / "run.json").exists():
        pytest.skip(f"Enregistrement d'exemple introuvable : {folder}")
    return RunResult.load(folder)


@pytest.fixture(scope="session")
def bg_run():
    """Enregistrement « Fond (rayons cosmiques) » sauvegardé (vidéo synthétique, 90 s)."""
    return _load_run(BG_RUN_NAME)


@pytest.fixture(scope="session")
def kcl_run():
    """Enregistrement « KCl » (100 g) sauvegardé (vidéo synthétique avec source β, 90 s)."""
    return _load_run(KCL_RUN_NAME)


@pytest.fixture(scope="session")
def sample_run_dirs():
    return SAMPLE_RUNS_DIR / BG_RUN_NAME, SAMPLE_RUNS_DIR / KCL_RUN_NAME
