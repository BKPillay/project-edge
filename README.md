
# EDGE AI v1.0 Stable

EDGE AI is now split into a fast Streamlit dashboard and a separate model engine.

## Run the model update

From project root:

```powershell
python scripts\run_model_update.py
```

This version adds the project root to `sys.path`, so it also works from the scripts folder.

## Then commit and push

```powershell
git add data outputs models scripts app.py .github README.md
git commit -m "EDGE AI v1 stable"
git push
```
