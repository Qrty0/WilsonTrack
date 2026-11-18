# -*- mode: python ; coding: utf-8 -*-
# Spécification PyInstaller : deux exécutables (graphique + console) partageant les mêmes bibliothèques.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = (
    collect_submodules("skimage.morphology")
    + collect_submodules("sklearn.ensemble")
    + collect_submodules("sklearn.mixture")
    + collect_submodules("sklearn.utils")
    + collect_submodules("serial.tools")
    + collect_submodules("wilsontrack")
    + [
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_agg",
        "pyqtgraph",
        "anthropic",
        "openpyxl",
        "joblib",
        "scipy.sparse.csgraph",
        "scipy.ndimage",
        "scipy.stats",
    ]
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

exe_gui = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="WilsonTrack",
    debug=False, strip=False, upx=False,
    console=False,
    icon="wilsontrack/assets/icon.ico",
    version="version_info.txt",
)
exe_cli = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="WilsonTrack-console",
    debug=False, strip=False, upx=False,
    console=True,
    icon="wilsontrack/assets/icon.ico",
    version="version_info.txt",
)
coll = COLLECT(
    exe_gui, exe_cli,
    a.binaries, a.datas,
    strip=False, upx=False,
    name="WilsonTrack",
)
