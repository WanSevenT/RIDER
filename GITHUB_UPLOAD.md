# Uploading RIDER to GitHub

This package is ready to initialize as a Git repository. A software LICENSE is intentionally not included in this draft; add one only after confirming the appropriate licensing choice with the project/institution owners.

## 1. Inspect the release

```bash
python tools/inspect_release.py
python -m pytest tests/test_release_assets.py
```

## 2. Initialize the repository

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial release of RIDER"
```

Check `git status` before committing. Do not commit private datasets, server-local paths, API keys, unrelated experimental checkpoints, or generated evaluation files under `outputs/`. The committed `results/` directory should contain only the three summary CSV files.

## 3. Create an empty GitHub repository

Create a repository named `RIDER` on GitHub without adding a README, `.gitignore`, or license from the web interface, since this release already supplies the first two files.

Then connect and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/RIDER.git
git push -u origin main
```

## 4. Weights

The compact checkpoints included here are each below GitHub's single-file hard limit, but model weights are binary assets and can make Git history large. For a long-term public release, consider moving them to a model hub or GitHub Release and keeping only `weights/README.md` plus checksums in the main branch.

## 5. Before the public v1.0 tag

- decide and add a software license;
- add final citation metadata after publication/de-anonymization;
- run `scripts/05_eval_rider.sh` in a clean environment;
- verify that the public dataset preparation reproduces the released manifest.
