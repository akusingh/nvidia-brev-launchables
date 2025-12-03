# 📓 Using A100 Notebook for Launchables

## ✅ **YES, Use the A100 Notebook!**

**Why it's perfect:**
- ✅ **Proven to work** - Successfully trained on A100
- ✅ **Complete workflow** - Setup → Training → Export
- ✅ **Well documented** - 12 markdown cells with explanations
- ✅ **Real commands** - Actual working code, not theoretical

**File:** `~/Downloads/finnish-tts-training-COMPLETED.ipynb`
- 43 cells total
- 31 code cells (actual commands)
- 12 markdown cells (explanations)

---

## 🔧 **What Needs to Be Cleaned Up**

### 1. **Remove Hardcoded Paths**

**Find and replace:**
```python
# BEFORE (A100-specific):
/home/shadeform/finnish-tts-brev/
/home/shadeform/data/

# AFTER (generic):
${HOME}/nvidia-brev-launchables/
${HOME}/data/
```

### 2. **Remove Instance-Specific Cells**

**Remove these:**
- Duplicate cells (e.g., two `nvidia-smi` cells)
- Cells specific to Brev instance (pip install in venv)
- Debug cells you added during troubleshooting
- Cells with hardcoded speaker IDs

### 3. **Add User Instructions**

**Add at the top:**
```markdown
## 🚀 Welcome to Finnish TTS Training!

This notebook will guide you through training a Finnish TTS model.

**Before starting:**
1. ✅ Setup script has run (automatic)
2. 📁 Upload your Finnish audio data to `datasets/finnish-tts-raw/`
3. 💰 Training takes ~4 hours (~$5 on A100)

**What you need:**
- Finnish audio files (.wav, 44.1kHz)
- Text transcripts (.lab files)
- Minimum 500 samples (1 hour audio)
```

### 4. **Update Training Command**

**Current (in notebook):**
```python
!python fish_speech/train.py \
  --config-name text2semantic_finetune \
  pretrained_ckpt_path=/home/shadeform/finnish-tts-brev/checkpoints/openaudio-s1-mini \
  train_dataset.proto_files=[/home/shadeform/finnish-tts-brev/data/protos] \
  ...
```

**Fixed:**
```python
!python fish_speech/train.py \
  --config-name text2semantic_finetune \
  pretrained_ckpt_path=${HOME}/nvidia-brev-launchables/checkpoints/openaudio-s1-mini \
  train_dataset.proto_files=[${HOME}/data/protos] \
  ...
```

### 5. **Add Cost Estimates**

**Add markdown cell:**
```markdown
## 💰 Cost Estimate

This cell will run for ~4 hours.

| GPU | Cost/hr | Total Cost |
|-----|---------|------------|
| A100-80GB | $1.20 | ~$4.80 |
| L40S | $0.60 | ~$2.40 |

**Progress:** You can monitor training in real-time below.
Press Ctrl+C to stop (checkpoints are saved every 100 steps).
```

---

## 🎯 **Quick Clean-Up Plan**

### Option A: Manual Clean (15 min)

1. Copy notebook to repo:
   ```bash
   cp ~/Downloads/finnish-tts-training-COMPLETED.ipynb \
      /Users/arunkumar.singh/nvidia-brev/finnish-tts-training.ipynb
   ```

2. Open in VS Code

3. Find/Replace:
   - `/home/shadeform/finnish-tts-brev` → `${HOME}/nvidia-brev-launchables`
   - `/home/shadeform/data` → `${HOME}/data`

4. Remove duplicate cells

5. Add user instructions at top

6. Save and commit

### Option B: I Clean It For You (5 min)

I can:
1. ✅ Create a cleaned version
2. ✅ Remove hardcoded paths
3. ✅ Add user instructions
4. ✅ Add cost estimates
5. ✅ Remove debug cells
6. ✅ Make it Launchable-ready

---

## 📋 **Recommended Structure for Launchables**

```
Cell 1: Welcome + Prerequisites
Cell 2: Environment Check (GPU, Python, Fish Speech)
Cell 3: Data Upload Instructions
Cell 4: Convert Dataset (WAV → VQ tokens)
Cell 5: Pack Dataset (create protos)
Cell 6: Training Configuration (show parameters)
Cell 7: Start Training (the big one, ~4 hours)
Cell 8: Monitor Progress (loss curves)
Cell 9: Export Model (merge LoRA)
Cell 10: Test Inference (generate sample)
Cell 11: Download Results
```

---

## 🚀 **What Should We Do?**

**Option 1: I clean the notebook for you** (Recommended, fast)
- Takes 5 minutes
- I remove all hardcoded paths
- Add user instructions
- Ready to push to GitHub

**Option 2: You clean it manually**
- Takes 15 minutes
- You have full control
- I can guide you

**Option 3: Create new simplified notebook**
- Takes 30 minutes
- Start fresh, cleaner
- But loses your proven workflow

---

## 💡 **My Strong Recommendation**

**Use Option 1: Let me clean the A100 notebook**

**Why?**
- ✅ It **already works** (proven on A100)
- ✅ Has the **exact commands** that succeeded
- ✅ Includes all the **fixes** you discovered (protobuf, workers, etc.)
- ✅ Just needs **path updates** (5 min work)
- ✅ You can verify it still makes sense

**Then:**
1. I clean it → 5 min
2. You review → 2 min
3. Push to GitHub → 1 min
4. Create Launchable → 5 min
5. **Total: 13 minutes to submission!** 🚀

---

## 🤔 **What Do You Want?**

**A)** Let me clean the A100 notebook for you (fast)  
**B)** You'll clean it manually (you control)  
**C)** Create new simplified notebook (fresh start)

**I recommend A!** Want me to do it? 🚀
