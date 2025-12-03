# 🚀 Push Code to nvidia-brev-launchables Repo

## Current Situation
- Local git remote: https://github.com/akusingh/finnish-tts-brev.git
- New empty repo: https://github.com/akusingh/nvidia-brev-launchables

## Option 1: Change Remote (Recommended)
```bash
cd /Users/arunkumar.singh/nvidia-brev

# Update remote to new repo
git remote set-url origin https://github.com/akusingh/nvidia-brev-launchables.git

# Verify
git remote -v

# Push to new repo
git push -u origin feature/finetuning-on-brev
```

## Option 2: Keep Both Repos
```bash
cd /Users/arunkumar.singh/nvidia-brev

# Add second remote
git remote add launchables https://github.com/akusingh/nvidia-brev-launchables.git

# Push to both
git push origin feature/finetuning-on-brev
git push launchables feature/finetuning-on-brev
```

## What to Push

### Essential Files:
- ✅ setup.sh
- ✅ README.md
- ✅ requirements.txt (if you have one)
- ✅ scripts/
- ✅ .gitignore
- ✅ LICENSE

### Optional but Recommended:
- ✅ launchable.yaml (configuration reference)
- ✅ finnish-tts-brev.ipynb (Jupyter notebook for users)
- ✅ docs/ (all your great documentation)

### DO NOT Push:
- ❌ .env (secrets)
- ❌ datasets/ (too large, users provide their own)
- ❌ checkpoints/ (too large, generated during training)
- ❌ results/ (user-specific)
- ❌ Temporary files

## Commands to Run

```bash
cd /Users/arunkumar.singh/nvidia-brev

# 1. Update remote
git remote set-url origin https://github.com/akusingh/nvidia-brev-launchables.git

# 2. Check what will be pushed
git status

# 3. Add essential files only
git add README.md setup.sh .gitignore LICENSE
git add scripts/
git add launchable.yaml
git add finnish-tts-brev.ipynb  # If you want Jupyter notebook

# 4. Commit
git commit -m "Initial commit: Finnish TTS Launchable"

# 5. Push
git push -u origin feature/finetuning-on-brev

# 6. Create main branch from feature branch
git checkout -b main
git push -u origin main
```
