"""Préférences de l'utilisateur, conservées entre les sessions.

Fichier : %APPDATA%/WilsonTrack/settings.json (Windows) ou ~/.wilsontrack/settings.json.
Contient la dernière configuration de détection, le dossier de sortie, la caméra
choisie et, si l'utilisateur l'a demandé, la clé API Claude (stockée localement).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import AppConfig


def settings_dir() -> Path:
    base = os.environ.get("APPDATA")
    d = Path(base) / "WilsonTrack" if base else Path.home() / ".wilsontrack"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = path or settings_dir() / "settings.json"
        self.data: dict = {}
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # --- configuration de détection -----------------------------------------
    def load_config(self) -> AppConfig:
        d = self.data.get("config")
        if isinstance(d, dict):
            try:
                return AppConfig.from_dict(d)
            except Exception:
                pass
        return AppConfig()

    def store_config(self, cfg: AppConfig) -> None:
        self.data["config"] = cfg.to_dict()

    # --- clé API --------------------------------------------------------------
    def api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY") or self.data.get("anthropic_api_key", "")
