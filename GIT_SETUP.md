# Git Setup Instructions

## Step 1: Configure Git (Required - Run Once)

Before pushing to GitHub, you need to configure your Git identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Or for this repository only:

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Step 2: Commit Your Changes

```bash
git commit -m "Initial commit: Leukemia Detection System with EfficientNet-B0"
```

## Step 3: Push to GitHub

```bash
git push -u origin main
```

If you encounter authentication issues, you may need to:
- Use a Personal Access Token instead of password
- Set up SSH keys
- Use GitHub CLI

## What's Included

✅ **Included in Repository:**
- All source code (`src/`, `proof/`)
- Model file (`models/best_model.pth`)
- Requirements file
- README.md
- .gitignore

❌ **Excluded from Repository:**
- Data files (`data/`)
- Virtual environment (`venv/`)
- Output files (`outputs/`)
- Temporary files

## Verify Before Pushing

Check what will be pushed:

```bash
git status
git log --oneline
```

