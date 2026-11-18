# Construit WilsonTrack.exe (dossier dist\WilsonTrack) et une archive ZIP prête à partager.
# Usage (PowerShell, depuis le dossier du projet) :  .\build_exe.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}
$py = ".\.venv\Scripts\python.exe"
& $py tools\make_icon.py
& $py -m PyInstaller --noconfirm --clean WilsonTrack.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller a échoué" }
$dist = "dist\WilsonTrack"
Copy-Item README.md $dist -Force
if (Test-Path docs) { Copy-Item docs "$dist\docs" -Recurse -Force }
if (Test-Path arduino) { Copy-Item arduino "$dist\arduino" -Recurse -Force }
if (Test-Path exemples) { Copy-Item exemples "$dist\exemples" -Recurse -Force }
$zip = "dist\WilsonTrack-2.0-Windows.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $dist -DestinationPath $zip
Write-Host "Terminé : $dist\WilsonTrack.exe  et  $zip"
