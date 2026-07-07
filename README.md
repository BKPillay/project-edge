# EDGE AI v2 Full-History Platform

This version rebuilds EDGE AI around the full Daily Lotto history file.

## Key changes

- Uses the master CSV with `draw_number`
- Preserves full history
- Splits the model into independent engines
- Generates outputs offline
- Streamlit reads output files only
- Adds richer number scores:
  - frequency score
  - momentum score
  - gap score
  - final score
- Adds latest draw ball/pair/triplet review
- Adds fast back-test summary

## Run model update

From the project root:

```powershell
python scripts\run_model_update.py
```

Then commit:

```powershell
git add data outputs models scripts app.py .github README.md
git commit -m "EDGE AI v2 full history platform"
git push
```

## Note

Full walk-forward back-testing is deliberately not part of the standard update because it can be expensive. It should be added later as an optional audit script.
