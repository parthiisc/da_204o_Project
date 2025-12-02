"""Helper script to download models listed in models_manifest.json using the project's downloader.

Run from the repository root:
    python scripts/download_models.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models_manifest.json"

if not MANIFEST.exists():
    print("models_manifest.json not found at", MANIFEST)
    raise SystemExit(1)

manifest = json.load(MANIFEST.open("r", encoding="utf-8"))

try:
    from streamlit_app import model_downloader as md
except Exception as e:
    print("Could not import model_downloader:", e)
    raise

print("Attempting to download joblib models into model_outputs/")
downloaded_joblibs = md.try_download_all(str(ROOT / "model_outputs"))
print("Downloaded joblibs:", downloaded_joblibs)

print("Attempting to download .pth models into CNN/")
pth_names = [k for k in manifest.keys() if k.endswith('.pth')]
pth_downloaded = []
for name in pth_names:
    dest = ROOT / 'CNN' / name
    ok = md.download_file(name, str(dest))
    print(f"{name} -> {ok}")
    if ok:
        pth_downloaded.append(name)

print("Downloaded pths:", pth_downloaded)

print('\nListing model_outputs/:')
mo_dir = ROOT / 'model_outputs'
if mo_dir.exists():
    for p in sorted(mo_dir.iterdir()):
        print(' -', p.name)
else:
    print(' model_outputs/ missing')

print('\nListing CNN/:')
cnn_dir = ROOT / 'CNN'
if cnn_dir.exists():
    for p in sorted(cnn_dir.iterdir()):
        print(' -', p.name)
else:
    print(' CNN/ missing')
