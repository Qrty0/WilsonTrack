# -*- mode: python ; coding: utf-8 -*-
# Spécification PyInstaller pour macOS : application WilsonTrack.app (à lancer SUR un Mac).
# Utilisée par build_mac.sh ; l'icône .icns est générée par ce script à partir de assets/icon.png.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = (
    collect_submodules("skimage.morphology")
    + collect_submodules("sklearn.ensemble")
    + collect_submodules("sklearn.mixture")
    + collect_submodules("sklearn.utils")
    + collect_submodules("serial.tools")
    + collect_submodules("wilsontrack")
    + ["matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg", "pyqtgraph", "anthropic",
       "openpyxl", "joblib", "scipy.sparse.csgraph", "scipy.ndimage", "scipy.stats"]
)

datas = [("wilsontrack/assets", "wilsontrack/assets")]
datas += collect_data_files("skimage", includes=["**/*.pyi", "**/*.txt"])
datas += collect_data_files("anthropic")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "jupyter", "notebook", "pytest", "sphinx",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtMultimedia"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="WilsonTrack",
    debug=False, strip=False, upx=False,
    console=False,
    argv_emulation=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="WilsonTrack")

app = BUNDLE(
    coll,
    name="WilsonTrack.app",
    icon="build_mac/WilsonTrack.icns",
    bundle_identifier="ca.cegepmv.projet.wilsontrack",
    version="2.0.0",
    info_plist={
        "CFBundleDisplayName": "WilsonTrack",
        "CFBundleShortVersionString": "2.0.0",
        "CFBundleVersion": "2.0.0",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        # Obligatoire sur macOS pour utiliser la webcam (sinon la caméra reste noire)
        "NSCameraUsageDescription": "WilsonTrack filme la chambre de Wilson pour détecter les traces de particules.",
        "NSHumanReadableCopyright": "Projet d'intégration — Cégep Marie-Victorin",
    },
)
