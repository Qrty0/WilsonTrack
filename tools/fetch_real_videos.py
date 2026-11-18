"""Télécharge de VRAIES vidéos de chambres à brouillard sous licence libre (Wikimedia Commons)
dans exemples/vidéos réelles (Wikimedia)/ et écrit SOURCES.md (auteur, licence, lien).

    python tools/fetch_real_videos.py
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exemples" / "vidéos réelles (Wikimedia)"
UA = {"User-Agent": "WilsonTrack/2.0 (projet etudiant cegep Marie-Victorin)"}
FILES = [
    ("File:Cloud_Chamber.ogv", "1 Cloud Chamber (Stephen Curry).ogv"),
    ("File:Cloud_chamber.ogv", "2 Cloud chamber (aza).ogv"),
    ("File:Home_Made_Cloud_Chamber.webm", "3 Home Made Cloud Chamber (Mickyfitz13).webm"),
    ("File:Nebelkammer-LMU-1.ogv", "4 Nebelkammer LMU (Schoenitzer).ogv"),
    ("File:Wilson_chamber.webm", "5 Wilson chamber (Bilovitskiy).webm"),
    ("File:Самодельная_камера_Вильсона.webm", "6 Camera de Wilson maison (Tilnishok).webm"),
]


def info(title: str) -> dict:
    q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "imageinfo",
                                "iiprop": "url|size|extmetadata", "titles": title})
    with urllib.request.urlopen(urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{q}",
                                                       headers=UA), timeout=60) as r:
        pages = json.load(r)["query"]["pages"]
    return next(iter(pages.values()))["imageinfo"][0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for title, name in FILES:
        i = info(title)
        m = i.get("extmetadata", {})
        artist = re.sub(r"<[^>]+>", "", m.get("Artist", {}).get("value", "")).strip()
        lic = m.get("LicenseShortName", {}).get("value", "")
        dest = OUT / name
        if not dest.exists() or dest.stat().st_size != i["size"]:
            import time
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(urllib.request.Request(i["url"], headers=UA), timeout=300) as r:
                        data = r.read()
                    dest.write_bytes(data)
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429 or attempt == 7:
                        raise
                    wait = int(e.headers.get("Retry-After", "0") or 0) or 30 * (attempt + 1)
                    print(f"  serveur occupé (429), nouvel essai dans {wait} s…", flush=True)
                    time.sleep(wait)
            time.sleep(10)
        rows.append(f"| {name} | {artist} | {lic} | {round(i.get('duration', 0))} s | {i['width']}×{i['height']} | "
                    f"{i['descriptionurl']} |")
        print(f"{name} : {dest.stat().st_size / 1e6:.1f} Mo — {lic} — {artist}", flush=True)
    md = ("# Vraies vidéos de chambres à brouillard (Wikimedia Commons)\n\n"
          f"Téléchargées le {date.today().isoformat()} pour tester WilsonTrack. Licences libres Creative Commons : "
          "citer l'auteur et la licence en cas de réutilisation (rapport, présentation).\n\n"
          "| Fichier | Auteur | Licence | Durée | Résolution | Source |\n|---|---|---|---|---|---|\n"
          + "\n".join(rows) + "\n")
    (OUT / "SOURCES.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
