#!/bin/bash
# Construit WilsonTrack.app et WilsonTrack-2.0-macOS.dmg — à lancer SUR UN MAC, depuis le dossier du projet :
#     bash build_mac.sh
# Prérequis : Python 3.11 à 3.14 (python.org ou Homebrew : brew install python@3.13).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Python 3 introuvable : installez-le depuis https://www.python.org/downloads/macos/"; exit 1; }
echo "Python utilisé : $($PY --version)"

if [ ! -x .venv-mac/bin/python ]; then
  "$PY" -m venv .venv-mac
fi
.venv-mac/bin/python -m pip install --upgrade pip
.venv-mac/bin/python -m pip install -r requirements.txt

echo "Icône macOS…"
mkdir -p build_mac/icon.iconset
for s in 16 32 64 128 256 512; do
  sips -z $s $s wilsontrack/assets/icon.png --out "build_mac/icon.iconset/icon_${s}x${s}.png" >/dev/null
  d=$((s * 2)); sips -z $d $d wilsontrack/assets/icon.png --out "build_mac/icon.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns build_mac/icon.iconset -o build_mac/WilsonTrack.icns

echo "Vérification rapide (tests)…"
.venv-mac/bin/python -m pytest -q -x tests/test_explain.py tests/test_reference_model.py || echo "(avertissement : des tests ont échoué, la construction continue)"

echo "Construction de l'application…"
.venv-mac/bin/python -m PyInstaller --noconfirm --clean --distpath dist_mac --workpath build_mac/work WilsonTrack_mac.spec

APP="dist_mac/WilsonTrack.app"
# Signature « ad hoc » : évite l'erreur « application endommagée » sur les Mac Apple Silicon
codesign --force --deep --sign - "$APP" || true

echo "Image disque…"
STAGE=build_mac/dmg; rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp README.md "$STAGE/LISEZMOI.md" 2>/dev/null || true
[ -d docs ] && cp -R docs "$STAGE/docs"
DMG="dist_mac/WilsonTrack-2.0-macOS.dmg"; rm -f "$DMG"
hdiutil create -volname "WilsonTrack 2.0" -srcfolder "$STAGE" -ov -format UDZO "$DMG"

echo "Auto-test de l'application compilée…"
"$APP/Contents/MacOS/WilsonTrack" --autotest || echo "(auto-test : voir le message ci-dessus)"

echo
echo "Terminé :"
echo "  $APP"
echo "  $DMG"
echo "Premier lancement : clic droit sur WilsonTrack › Ouvrir (application non signée par Apple),"
echo "puis autorisez l'accès à la caméra quand macOS le demande."
