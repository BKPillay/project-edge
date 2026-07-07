from pathlib import Path

APP_PATH = Path("app.py")
text = APP_PATH.read_text(encoding="utf-8")

old = 'def read_csv(name):\n    path = OUTPUT_DIR / name\n    return pd.read_csv(path) if path.exists() else pd.DataFrame()\n'
new = 'def read_csv(name):\n    path = OUTPUT_DIR / name\n    if not path.exists() or path.stat().st_size == 0:\n        return pd.DataFrame()\n    try:\n        return pd.read_csv(path)\n    except pd.errors.EmptyDataError:\n        return pd.DataFrame()\n'

if old not in text:
    raise RuntimeError("Could not find the read_csv function to patch.")

text = text.replace(old, new)
APP_PATH.write_text(text, encoding="utf-8")
print("Patched app.py to handle empty CSV files safely.")
