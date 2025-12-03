# 📋 Repository Cleanup Analysis

## ✅ Currently in Repo (Good - Keep These)
```
.gitattributes
.gitignore
LICENSE
README.md
setup.sh
MIGRATION_COMPLETE.md
QUICKSTART.md

data/.gitkeep
data/README.md
data/example_dataset.json

datasets/.gitkeep
datasets/README.md

docs/IMPROVEMENTS_SUMMARY.md
docs/README_FINNISH_TTS.md

scripts/convert_finnish_dataset.py
scripts/monitor_training.py
scripts/quick_test.py

finnish-tts-model-training.ipynb (old notebook - consider replacing)
```

## 📝 Staged to Add (Review Before Pushing)
```
✅ finnish-tts-training.ipynb (3.8MB) - CLEANED notebook
✅ launchable.yaml - Configuration reference
```

## 🗑️ Untracked Files (DON'T Push These)
```
❌ .env (secrets)
❌ .env.example (if has secrets)
❌ checkpoints/ (large, generated files)
❌ results/ (user-specific)
❌ tools/ (unnecessary)

📄 Documentation files (decide if needed):
- ACTION_CHECKLIST.md
- DELETE_INSTANCE_NOW.md
- FULL_WORKFLOW_ESTIMATION.md
- IMMEDIATE_ACTIONS_BEFORE_DELETE.md
- INCREMENTAL_TRAINING.md
- LORA_MATH_EXPLAINED.md
- PROJECT_SUMMARY_2025-11-29.md
- PRODUCTION_ROADMAP.md
- And 10+ more planning docs...

🗂️ Other notebooks (probably don't need):
- finnish-tts-brev-optimized.ipynb
- finnish-tts-brev.ipynb

🔧 Other files:
- download_everything.sh (probably don't need)
- test_commands.md (probably don't need)
- test_inference.txt (probably don't need)
```

---

## 🎯 Recommendations

### Option 1: Minimal Launchable (Recommended)
**Push only what users need:**
```bash
# Already staged:
✅ finnish-tts-training.ipynb
✅ launchable.yaml

# Already in repo (keep):
✅ README.md
✅ setup.sh
✅ .gitignore
✅ LICENSE
✅ scripts/
✅ docs/

# Consider adding (useful docs):
git add LORA_MATH_EXPLAINED.md (explains parameters)
git add STEPS_EPOCHS_CHECKPOINTS_MATH.md (explains training)
git add FULL_WORKFLOW_ESTIMATION.md (time/cost)

# DON'T push:
❌ All the planning/action docs
❌ Old notebooks
❌ .env files
❌ checkpoints/
❌ results/
```

### Option 2: Just Push What's Staged (Simplest)
```bash
# Only push:
- finnish-tts-training.ipynb (new cleaned notebook)
- launchable.yaml (config)

# That's it! Everything else is already there or shouldn't be pushed
```

---

## 🔍 Things to Review

### 1. Do we need the old notebook?
```
Currently in repo:
- finnish-tts-model-training.ipynb (old)

Adding:
- finnish-tts-training.ipynb (new, cleaned)

Options:
A) Keep both (confusing for users)
B) Remove old, keep new (cleaner)
C) Rename old to "finnish-tts-training-reference.ipynb"
```

### 2. Should we add useful docs?
```
Could be helpful for users:
✅ LORA_MATH_EXPLAINED.md (explains rank/alpha)
✅ FULL_WORKFLOW_ESTIMATION.md (time/cost estimates)

Probably not needed:
❌ PROJECT_SUMMARY_2025-11-29.md (your personal notes)
❌ ACTION_CHECKLIST.md (your todo list)
❌ DELETE_INSTANCE_NOW.md (your reminder)
```

---

## ✅ My Recommendation: Clean & Simple

### Step 1: Check what we're pushing
```bash
git status
```

**Currently staged:**
- finnish-tts-training.ipynb ✅
- launchable.yaml ✅

### Step 2: Optionally add useful docs
```bash
# Add helpful documentation for users (optional)
git add docs/
git add LORA_MATH_EXPLAINED.md
git add FULL_WORKFLOW_ESTIMATION.md
```

### Step 3: Commit and push
```bash
git commit -m "Add cleaned training notebook for Launchables

- finnish-tts-training.ipynb: Production-ready training notebook
  - Cleaned from successful A100 training
  - Environment variables instead of hardcoded paths
  - Welcome cell with user instructions
- launchable.yaml: Configuration reference for Brev"

git push origin main
```

---

## 🚨 Files That Should NEVER Be Pushed

**Already in .gitignore (good!):**
```
checkpoints/
datasets/
results/
*.env
```

**Make sure these stay untracked:**
- .env (secrets)
- checkpoints/ (large files)
- results/ (user data)
- datasets/ (user data)

---

## 🎯 What Should I Do?

**Tell me:**
1. **Option A:** Push only what's staged (finnish-tts-training.ipynb + launchable.yaml)
2. **Option B:** Add helpful docs too (LORA_MATH_EXPLAINED.md, etc.)
3. **Option C:** Let me review setup.sh first (you just edited it)

**I recommend Option C** - let's check setup.sh changes first! 🔍
