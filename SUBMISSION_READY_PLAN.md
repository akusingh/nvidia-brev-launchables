# 🚀 NVIDIA Launchables - Ready to Submit Plan
**Date:** December 3, 2025  
**Current State:** 90% Ready - Need Final Steps

---

## ✅ What You Already Have

### 1. Core Files Ready ✓

```
nvidia-brev/
├── launchable.yaml          ✅ Complete (213 lines)
├── README.md                ✅ Complete (356 lines)
├── setup.sh                 ✅ Complete (233 lines)
├── .gitignore               ✅ Configured
├── LICENSE                  ✅ MIT license
│
├── docs/                    ✅ Comprehensive docs
│   ├── PROJECT_SUMMARY_2025-11-29.md
│   ├── LORA_MATH_EXPLAINED.md
│   ├── STEPS_EPOCHS_CHECKPOINTS_MATH.md
│   └── FULL_WORKFLOW_ESTIMATION.md
│
└── scripts/                 ✅ Helper scripts
    ├── monitor_training.py
    ├── quick_test.py
    └── convert_finnish_dataset.py
```

### 2. Trained Model ✓
```
✅ Training completed: Nov 29, 2024
✅ Steps: 2800/3000 (early stopping)
✅ Cost: $5.28
✅ Quality: Working Finnish TTS
✅ Files downloaded: 7.5GB (checkpoints, logs, dataset)
```

### 3. Documentation ✓
```
✅ Complete training documentation
✅ Cost breakdowns
✅ Technical explanations (LoRA math, metrics)
✅ Workflow estimations
✅ Production roadmap
```

---

## ❌ What's Missing (Critical)

### 1. Dataset NOT on HuggingFace ❌

**Current launchable.yaml says:**
```yaml
source: "https://huggingface.co/datasets/yourusername/finnish-tts-dataset/resolve/main/finnish-speaker-2000-complete.tar.gz"
```

**Reality:**
- Dataset is in `~/Downloads/finnish-dataset.tar.gz` (locally)
- NOT uploaded to HuggingFace yet
- URL is placeholder "yourusername"

**Fix Required:**
```bash
# 1. Upload to HuggingFace
huggingface-cli login
huggingface-cli upload yourusername/finnish-tts-dataset \
  ~/Downloads/finnish-dataset.tar.gz \
  finnish-speaker-2000-complete.tar.gz

# 2. Update launchable.yaml with real URL
```

### 2. Repository NOT on GitHub ❌

**Current launchable.yaml says:**
```yaml
repository: "https://github.com/yourusername/finnish-tts-brev"
```

**Reality:**
- Code is local only (`/Users/arunkumar.singh/nvidia-brev`)
- NOT pushed to GitHub yet
- URL is placeholder

**Fix Required:**
```bash
# 1. Create GitHub repo
gh repo create finnish-tts-brev --public --source=. --remote=origin

# 2. Push code
git add .
git commit -m "Initial commit: Finnish TTS Launchable"
git push -u origin main

# 3. Update launchable.yaml with real URL
```

### 3. Test Scripts Missing ❌

**Current scripts/ has:**
- monitor_training.py ✓
- quick_test.py ✓
- convert_finnish_dataset.py ✓

**Missing scripts needed:**
- `scripts/download_dataset.sh` - Auto-download from HuggingFace
- `scripts/test_inference.py` - Generate sample audio (for demos)
- `scripts/validate_setup.sh` - Check environment is correct

### 4. No Smoke Test Done ❌

**Need to verify:**
- Clean setup.sh works on fresh instance
- Dataset downloads correctly
- Training starts without errors
- Checkpoints save properly

**Estimated cost:** $0.50 (30 minutes on cheap GPU)

---

## 🎯 Action Plan: Get Submission-Ready

### Phase 1: Prepare Assets (Local, Free, 1 hour)

**Step 1: Create GitHub Repository**
```bash
cd /Users/arunkumar.singh/nvidia-brev

# Initialize if needed
git init
git add .
git commit -m "Finnish TTS Launchable - Initial commit"

# Create repo (requires GitHub CLI or web)
gh repo create finnish-tts-brev --public --source=. --push
# OR manually: Create on github.com, then:
git remote add origin https://github.com/YOURUSERNAME/finnish-tts-brev.git
git push -u origin main
```

**Step 2: Upload Dataset to HuggingFace**
```bash
# Login to HuggingFace
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens

# Create dataset repo (on HuggingFace web)
# Go to: https://huggingface.co/new-dataset
# Name: finnish-tts-dataset
# Public, MIT license

# Upload dataset
huggingface-cli upload USERNAME/finnish-tts-dataset \
  ~/Downloads/finnish-dataset.tar.gz \
  finnish-speaker-2000-complete.tar.gz

# Verify upload
# Check: https://huggingface.co/datasets/USERNAME/finnish-tts-dataset
```

**Step 3: Update launchable.yaml with Real URLs**
```yaml
# Replace placeholders:
repository: "https://github.com/YOURUSERNAME/finnish-tts-brev"
datasets[0].source: "https://huggingface.co/datasets/YOURUSERNAME/finnish-tts-dataset/resolve/main/finnish-speaker-2000-complete.tar.gz"

# Also update README.md author info
```

**Step 4: Create Missing Scripts**

**A. scripts/download_dataset.sh:**
```bash
#!/bin/bash
# Auto-download Finnish dataset from HuggingFace

DATASET_URL="https://huggingface.co/datasets/YOURUSERNAME/finnish-tts-dataset/resolve/main/finnish-speaker-2000-complete.tar.gz"
DATA_DIR="${HOME}/data"

echo "📥 Downloading Finnish dataset..."
mkdir -p ${DATA_DIR}
wget -q --show-progress ${DATASET_URL} -O ${DATA_DIR}/finnish-dataset.tar.gz

echo "📦 Extracting dataset..."
tar -xzf ${DATA_DIR}/finnish-dataset.tar.gz -C ${DATA_DIR}/
rm ${DATA_DIR}/finnish-dataset.tar.gz

echo "✅ Dataset ready at: ${DATA_DIR}/FinnishSpeaker"
```

**B. scripts/test_inference.py:**
```python
#!/usr/bin/env python3
"""Generate test audio samples from trained model"""

import sys
from pathlib import Path

# Test sentences
TESTS = [
    "Hyvää huomenta, kuinka voit tänään?",
    "Tämä on testi suomenkieliselle puhesynteesille.",
    "Kaunis sää tänään Helsingissä."
]

def generate_samples(model_path, output_dir):
    """Generate test samples"""
    print(f"🎤 Generating {len(TESTS)} test samples...")
    # TODO: Implement inference call
    # For now, just validate paths exist
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory: {output_dir}")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/merged"
    output = sys.argv[2] if len(sys.argv) > 2 else "test_outputs"
    generate_samples(model, output)
```

**C. scripts/validate_setup.sh:**
```bash
#!/bin/bash
# Validate environment is ready for training

echo "🔍 Validating setup..."

# Check Python
python3 --version || { echo "❌ Python not found"; exit 1; }

# Check PyTorch
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || { echo "❌ PyTorch not installed"; exit 1; }

# Check GPU
python3 -c "import torch; assert torch.cuda.is_available(); print(f'✅ GPU: {torch.cuda.get_device_name(0)}')" || { echo "❌ No GPU found"; exit 1; }

# Check Fish Speech
test -d "$HOME/fish-speech" || { echo "❌ Fish Speech not installed"; exit 1; }
echo "✅ Fish Speech installed"

# Check dataset
test -d "$HOME/data/FinnishSpeaker" || { echo "⚠️  Dataset not found (will download)"; }

echo ""
echo "✅ Environment validation complete!"
```

**Step 5: Commit and Push**
```bash
git add launchable.yaml README.md scripts/
git commit -m "Update URLs and add helper scripts"
git push
```

---

### Phase 2: Smoke Test (Optional but Recommended, $0.50, 30 min)

**Goal:** Verify setup works on clean instance

```bash
# 1. Launch cheap GPU instance
brev open --gpu L40S --instance-type g6.xlarge  # $0.60/hr

# 2. Clone repo
git clone https://github.com/YOURUSERNAME/finnish-tts-brev.git
cd finnish-tts-brev

# 3. Run setup
bash setup.sh

# 4. Download dataset
bash scripts/download_dataset.sh

# 5. Validate environment
bash scripts/validate_setup.sh

# 6. Start training (just 50 steps to verify)
cd $HOME/fish-speech
python fish_speech/train.py \
  --config-name text2semantic_finetune \
  trainer.max_steps=50 \
  ... # (full command from launchable.yaml)

# 7. Check if training starts without errors
# If yes: ✅ Ready to submit!
# If no: Debug and fix

# 8. Delete instance
brev delete finnish-tts-brev
```

**Cost:** ~$0.30 (30 min × $0.60/hr)

---

### Phase 3: Submit to NVIDIA (Free, 15 min)

**Step 1: Prepare Submission Email**

```
To: brev-support@nvidia.com
Subject: New Launchable Submission: Finnish TTS Training

Hi NVIDIA Launchables Team,

I'd like to submit a new Launchable for the showcase:

Project: Finnish Text-to-Speech Training with VQ Caching
Repository: https://github.com/YOURUSERNAME/finnish-tts-brev
Demo Video: [Optional: Link to demo]

Key Features:
- One-click Finnish TTS model training
- Pre-extracted VQ tokens (15 min faster setup)
- LoRA fine-tuning (memory efficient)
- 4.5 hours training, ~$5.40 cost
- Production-ready pipeline

The launchable.yaml is in the repo root with complete setup instructions.

Technical Details:
- GPU: A100-80GB (min 40GB)
- Framework: PyTorch + Lightning
- Model: Fish Speech (868M params)
- Dataset: 2000 Finnish samples (included)

Let me know if you need any additional information!

Best regards,
Arun Kumar Singh
```

**Step 2: Wait for Review**
- Typical response time: 1-2 weeks
- May request changes or demos
- Once approved: Listed on brev.nvidia.com/launchables

---

## 📋 Quick Checklist

### Before Submission:
- [ ] Create GitHub repo
- [ ] Upload dataset to HuggingFace
- [ ] Update launchable.yaml URLs (no "yourusername")
- [ ] Update README.md author info
- [ ] Add missing scripts (download, test, validate)
- [ ] Test locally if possible
- [ ] (Optional) Smoke test on Brev instance ($0.50)
- [ ] Commit and push all changes
- [ ] Email brev-support@nvidia.com

### After Submission:
- [ ] Respond to any feedback quickly
- [ ] Make requested changes
- [ ] Provide demos if asked
- [ ] Celebrate when accepted! 🎉

---

## 💡 Key Decisions to Make NOW

### 1. GitHub Username?
**Current:** `yourusername` (placeholder)  
**Action:** Replace with your actual GitHub username everywhere

### 2. HuggingFace Username?
**Current:** `yourusername` (placeholder)  
**Action:** Replace with your actual HF username everywhere

### 3. Do Smoke Test?
**Option A:** Skip smoke test, submit now (risky but fast)  
**Option B:** Spend $0.50 to verify (recommended)  
**My Recommendation:** Do smoke test - $0.50 is worth the confidence

### 4. Make Repo Public?
**Current:** Need to create repo  
**Required:** Must be public for NVIDIA submission  
**Privacy:** Your dataset will be on HuggingFace (public), code on GitHub (public)

---

## 🚀 Fastest Path to Submission (Today!)

**If you want to submit TODAY:**

```bash
# 1. Create accounts (if needed)
# - GitHub account
# - HuggingFace account (get API token)

# 2. Upload dataset (20 min)
huggingface-cli login
huggingface-cli upload USERNAME/finnish-tts-dataset ~/Downloads/finnish-dataset.tar.gz

# 3. Create GitHub repo (10 min)
cd /Users/arunkumar.singh/nvidia-brev
gh repo create finnish-tts-brev --public --source=. --push

# 4. Update URLs (5 min)
# Edit launchable.yaml - replace "yourusername" with real username

# 5. Add scripts (15 min)
# Copy templates above into scripts/

# 6. Commit and push (2 min)
git add .
git commit -m "Ready for NVIDIA Launchables submission"
git push

# 7. Send email (3 min)
# Use template above

# TOTAL TIME: ~1 hour
# COST: $0 (no smoke test)
```

---

## ⚠️ Current Blockers

1. **GitHub repo doesn't exist yet** - Need to create
2. **Dataset not on HuggingFace yet** - Need to upload
3. **URLs are placeholders** - Need to update
4. **Missing helper scripts** - Need to add
5. **No smoke test done** - Optional but recommended

**Estimated Time to Fix All:** 1-2 hours  
**Estimated Cost:** $0 (free) or $0.50 (with smoke test)

---

## 🎯 My Recommendation

### Today (1 hour, $0):
1. ✅ Create GitHub repo
2. ✅ Upload dataset to HuggingFace  
3. ✅ Update all placeholder URLs
4. ✅ Add missing scripts
5. ✅ Push to GitHub

### Tomorrow ($0.50, 30 min):
6. ✅ Smoke test on cheap GPU
7. ✅ Fix any issues found
8. ✅ Push fixes

### Day After (free, 5 min):
9. ✅ Email NVIDIA submission
10. ✅ Wait for approval

**Total Investment:** 2 hours work + $0.50 testing = Ready to submit! 🚀

---

## 📞 Need Help With?

**Common questions:**

**Q: Do I need to re-train the model?**  
A: No! You already have trained model. Just package it.

**Q: Will dataset be public?**  
A: Yes, on HuggingFace. If privacy concern, use demo dataset instead.

**Q: Can I use private repos?**  
A: No, NVIDIA requires public repos for Launchables showcase.

**Q: What if smoke test fails?**  
A: Debug and fix. That's why we do it! Better to find issues now.

**Q: How long until approved?**  
A: Typically 1-2 weeks, sometimes faster for good submissions.

---

## 🎬 Next Steps

**Tell me:**
1. What's your GitHub username? (I'll update URLs)
2. What's your HuggingFace username? (I'll update URLs)
3. Do you want to do smoke test? ($0.50, recommended)
4. Any privacy concerns about dataset being public?

**Then I can:**
- Generate exact commands with your usernames
- Create the missing scripts
- Guide you through upload process
- Help with smoke test if you choose to do it

**Ready to submit this week?** Let's do it! 🚀
