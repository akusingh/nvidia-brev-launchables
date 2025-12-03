# 🧹 Clean Up Repository for Launchables

## 🎯 Goal: Minimal, Clean Repo with Only Essentials

### ❌ Files to Remove from Repo:
```
MIGRATION_COMPLETE.md (internal doc)
QUICKSTART.md (redundant with README)
data/.gitkeep
data/README.md
data/example_dataset.json
datasets/.gitkeep
datasets/README.md
docs/IMPROVEMENTS_SUMMARY.md (internal)
docs/README_FINNISH_TTS.md (redundant)
finnish-tts-model-training.ipynb (old notebook)
```

### ✅ Files to Keep:
```
.gitattributes
.gitignore
LICENSE
README.md
setup.sh
scripts/convert_finnish_dataset.py
scripts/monitor_training.py
scripts/quick_test.py
```

### ➕ Files to Add:
```
finnish-tts-training.ipynb (new cleaned notebook)
launchable.yaml (config reference)
```

---

## 🚀 Clean Up Commands

### Option 1: Remove Files One by One
```bash
cd /Users/arunkumar.singh/nvidia-brev

# Remove unnecessary files from git
git rm MIGRATION_COMPLETE.md
git rm QUICKSTART.md
git rm data/.gitkeep data/README.md data/example_dataset.json
git rm datasets/.gitkeep datasets/README.md
git rm docs/IMPROVEMENTS_SUMMARY.md docs/README_FINNISH_TTS.md
git rm finnish-tts-model-training.ipynb

# Remove empty dirs
git rm -r data/ datasets/ docs/

# Check status
git status
```

### Option 2: Start Fresh with Clean Main (Recommended)
```bash
cd /Users/arunkumar.singh/nvidia-brev

# Create a new clean branch
git checkout --orphan clean-main

# Add only essential files
git add .gitattributes .gitignore LICENSE
git add README.md setup.sh
git add scripts/
git add finnish-tts-training.ipynb
git add launchable.yaml

# Commit
git commit -m "Clean Launchable: Essential files only"

# Replace main with clean branch
git branch -D main
git branch -m main

# Force push to remote
git push -f origin main
```

---

## 📋 What Each File Does

### Keep These (Essential):
```
.gitattributes         → Git settings
.gitignore            → Protect secrets/large files
LICENSE               → MIT license
README.md             → Main documentation
setup.sh              → Automated setup script
scripts/*.py          → Helper utilities
finnish-tts-training.ipynb → Main training notebook
launchable.yaml       → Config reference
```

### Remove These (Unnecessary):
```
MIGRATION_COMPLETE.md      → Your internal note
QUICKSTART.md             → Redundant with README
data/*                    → Example data (users provide own)
datasets/*                → Empty placeholders
docs/IMPROVEMENTS_SUMMARY.md → Internal doc
docs/README_FINNISH_TTS.md   → Redundant
finnish-tts-model-training.ipynb → Old notebook
```

---

## 🎯 My Recommendation: Option 2 (Start Fresh)

**Why?**
- ✅ Cleanest approach
- ✅ No git history clutter
- ✅ Only what users need
- ✅ Professional appearance

**Steps:**
1. Create orphan branch (no history)
2. Add only essential files
3. Replace main with clean branch
4. Force push

**This gives you a repo with ONLY:**
- README, setup.sh, LICENSE
- Scripts folder
- New cleaned notebook
- Config file

**No clutter, no confusion!**

---

## 🤔 Your Decision

**Option A: Remove files incrementally** (safer, keeps history)
**Option B: Start fresh** (cleaner, no history)

**Which do you prefer?** I'll help with either! 🚀
