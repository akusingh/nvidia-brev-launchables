# ⚠️ IMMEDIATE ACTIONS - Before Deleting Brev Instance

**CRITICAL: Your instance can only be DELETED, not stopped!**

Once you delete it, **everything is gone forever**. Follow this checklist carefully.

---

## 🚨 Priority 1: Download Trained Model (BEFORE DELETION!)

### What You MUST Download:

```bash
# Check if training is complete
ssh shadeform@64.247.196.21 'tail -20 ~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log'

# Look for: "Training finished" or step 3000/3000
```

### If Training Complete → Download IMMEDIATELY:

```bash
# 1. Download final checkpoint (~47MB)
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt \
  ~/Downloads/finnish-tts-final-checkpoint.ckpt

# 2. Download ALL checkpoints (optional, ~235MB for 5 checkpoints)
scp -r shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/ \
  ~/Downloads/finnish-tts-checkpoints/

# 3. Download training logs
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log \
  ~/Downloads/finnish-tts-train.log

# 4. Download tensorboard logs (for graphs)
scp -r shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/tensorboard/ \
  ~/Downloads/finnish-tts-tensorboard/
```

**☐ Final checkpoint downloaded**
**☐ Training logs downloaded**

---

## 🚨 Priority 2: Package & Upload Dataset (BEFORE DELETION!)

### Create the Dataset Archive:

```bash
# SSH into instance
ssh shadeform@64.247.196.21

# Package dataset with VQ tokens (takes ~10 minutes)
cd ~/finnish-tts-brev/data
tar -czf finnish-speaker-2000-complete.tar.gz FinnishSpeaker/

# Check size
ls -lh finnish-speaker-2000-complete.tar.gz
# Expected: ~8-10GB

# Move to home directory for easy access
mv finnish-speaker-2000-complete.tar.gz ~/

# Exit SSH
exit
```

### Option A: Download to Local Machine (Slow but Safe)

```bash
# Download the archive (may take 30-60 minutes for 8GB)
scp shadeform@64.247.196.21:~/finnish-speaker-2000-complete.tar.gz \
  ~/Downloads/
```

**☐ Dataset downloaded to local machine**

### Option B: Upload to HuggingFace (Faster, Recommended) ⭐

```bash
# SSH back in
ssh shadeform@64.247.196.21

# Install HF CLI if not already
pip install huggingface_hub

# Login (use your HF token from .env)
huggingface-cli login
# Paste your token when prompted

# Create dataset repository
huggingface-cli repo create finnish-tts-dataset --type dataset

# Upload (takes ~20 minutes for 8GB)
huggingface-cli upload yourusername/finnish-tts-dataset \
  ~/finnish-speaker-2000-complete.tar.gz \
  --repo-type dataset

# Verify upload succeeded
echo "Check: https://huggingface.co/datasets/yourusername/finnish-tts-dataset"

# Exit
exit
```

**☐ Dataset uploaded to HuggingFace**
**☐ Download URL verified: `https://huggingface.co/datasets/yourusername/finnish-tts-dataset/resolve/main/finnish-speaker-2000-complete.tar.gz`**

---

## 🚨 Priority 3: Generate Demo Audio (BEFORE DELETION!)

```bash
# SSH in
ssh shadeform@64.247.196.21

# Generate 4 demo samples
cd ~/fish-speech

cat > generate_demos.sh << 'EOF'
#!/bin/bash
CKPT="~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt"

python tools/llama/generate.py --checkpoint $CKPT \
  --text "Hyvää huomenta! Tervetuloa Suomeen." \
  --output demo_1_greeting.wav

python tools/llama/generate.py --checkpoint $CKPT \
  --text "Tämä on korkealaatuinen suomenkielinen puhesynteesi." \
  --output demo_2_technical.wav

python tools/llama/generate.py --checkpoint $CKPT \
  --text "Opiskele tekoälyä NVIDIA:n GPU:illa." \
  --output demo_3_nvidia.wav

python tools/llama/generate.py --checkpoint $CKPT \
  --text "Kiitos käytöstä. Näkemiin!" \
  --output demo_4_goodbye.wav

echo "✅ All demos generated!"
EOF

chmod +x generate_demos.sh
./generate_demos.sh

# Exit
exit

# Download all demos
scp "shadeform@64.247.196.21:~/fish-speech/demo_*.wav" ~/Downloads/
```

**☐ Demo audio files downloaded**

---

## 🚨 Priority 4: Save Configuration & Logs

```bash
# Download important config files
scp shadeform@64.247.196.21:~/finnish-tts-brev/setup.sh ~/nvidia-brev/
scp shadeform@64.247.196.21:~/finnish-tts-brev/.env ~/nvidia-brev/.env.backup

# Download Fish Speech config
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/.hydra/config.yaml \
  ~/Downloads/training_config.yaml

# Download any error logs
scp shadeform@64.247.196.21:~/finnish-tts-brev/setup_log.txt ~/Downloads/
```

**☐ Configuration files backed up**

---

## ✅ CHECKLIST: Safe to Delete Instance

**Before you run `brev delete`:**

### Critical (Must Have):
- [ ] Final checkpoint downloaded (step_000003000.ckpt)
- [ ] Dataset uploaded to HuggingFace OR downloaded locally
- [ ] HuggingFace dataset URL verified working

### Important (Should Have):
- [ ] All checkpoints downloaded (step_300, 400, 500, etc.)
- [ ] Training logs downloaded
- [ ] Demo audio samples generated & downloaded
- [ ] Training config downloaded

### Nice to Have:
- [ ] TensorBoard logs downloaded
- [ ] Setup logs downloaded
- [ ] Any custom scripts saved

---

## ⏱️ Time Estimate for All Downloads

| Task | Time | Notes |
|------|------|-------|
| Final checkpoint | 2 min | 47MB |
| All checkpoints | 5 min | 235MB |
| Generate demos | 5 min | GPU time |
| Download demos | 1 min | Small WAV files |
| Package dataset | 10 min | Tar compression |
| Upload to HF | 20 min | 8GB upload |
| Training logs | 1 min | Small text files |
| **Total** | **45 min** | **Do this all before deletion!** |

---

## 💰 Cost to Complete Downloads

**45 minutes @ $1.20/hour = $0.90**

**Total cost to safely extract everything: $0.90**

---

## 🎯 IMMEDIATE ACTION PLAN

### Right Now (when training finishes):

```bash
# 1. Check training status
ssh shadeform@64.247.196.21 'tail -50 ~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log'

# 2. If complete, start this script (run on LOCAL machine):
cat > ~/nvidia-brev/download_everything.sh << 'EOF'
#!/bin/bash
set -e

echo "🔍 Checking training status..."
ssh shadeform@64.247.196.21 'tail -5 ~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log'

read -p "Is training complete? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏳ Wait for training to complete first!"
    exit 1
fi

echo "📥 Downloading checkpoints..."
mkdir -p ~/Downloads/finnish-tts-backup/checkpoints
scp -r shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/ \
  ~/Downloads/finnish-tts-backup/

echo "📥 Downloading logs..."
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log \
  ~/Downloads/finnish-tts-backup/

echo "🎵 Generating demo audio..."
ssh shadeform@64.247.196.21 << 'REMOTE'
cd ~/fish-speech
CKPT="~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt"

python tools/llama/generate.py --checkpoint $CKPT \
  --text "Hyvää huomenta! Tervetuloa Suomeen." --output demo_1.wav

python tools/llama/generate.py --checkpoint $CKPT \
  --text "Opiskele tekoälyä NVIDIA:n GPU:illa." --output demo_2.wav

echo "✅ Demos generated"
REMOTE

echo "📥 Downloading demos..."
scp "shadeform@64.247.196.21:~/fish-speech/demo_*.wav" ~/Downloads/finnish-tts-backup/

echo "📦 Packaging dataset..."
ssh shadeform@64.247.196.21 << 'REMOTE'
cd ~/finnish-tts-brev/data
tar -czf ~/finnish-speaker-2000-complete.tar.gz FinnishSpeaker/
ls -lh ~/finnish-speaker-2000-complete.tar.gz
REMOTE

echo "⬆️  Uploading to HuggingFace..."
ssh shadeform@64.247.196.21 << 'REMOTE'
pip install -q huggingface_hub
huggingface-cli login --token $(grep HF_TOKEN ~/.env | cut -d= -f2)
huggingface-cli repo create finnish-tts-dataset --type dataset || true
huggingface-cli upload yourusername/finnish-tts-dataset \
  ~/finnish-speaker-2000-complete.tar.gz \
  --repo-type dataset
echo "✅ Upload complete!"
REMOTE

echo ""
echo "✅ ALL DONE! Safe to delete instance now."
echo ""
echo "📋 What you have:"
echo "   - Checkpoints: ~/Downloads/finnish-tts-backup/checkpoints/"
echo "   - Logs: ~/Downloads/finnish-tts-backup/train.log"
echo "   - Demos: ~/Downloads/finnish-tts-backup/demo_*.wav"
echo "   - Dataset: https://huggingface.co/datasets/yourusername/finnish-tts-dataset"
echo ""
echo "⚠️  Now run: brev delete shadeform@64.247.196.21"
EOF

chmod +x ~/nvidia-brev/download_everything.sh

# 3. Run the download script
~/nvidia-brev/download_everything.sh
```

**This script downloads EVERYTHING in one go!**

---

## 🗑️ After Downloads Complete

### Verify Everything:

```bash
# Check downloads
ls -lh ~/Downloads/finnish-tts-backup/

# Should see:
# checkpoints/step_000003000.ckpt (~47MB)
# checkpoints/step_000002900.ckpt
# ... (other checkpoints)
# train.log (~12KB)
# demo_1.wav
# demo_2.wav

# Verify HuggingFace upload
open https://huggingface.co/datasets/yourusername/finnish-tts-dataset

# Test audio samples
open ~/Downloads/finnish-tts-backup/demo_1.wav
```

### Delete Instance:

```bash
# Only after verifying everything is safe!
brev delete shadeform@64.247.196.21

# Or whatever the delete command is
brev instances delete shadeform@64.247.196.21
```

**☐ Instance deleted**
**☐ No more charges**

---

## 📊 What You'll Have After Deletion

### On Your Local Machine:
```
~/Downloads/finnish-tts-backup/
├── checkpoints/
│   ├── step_000003000.ckpt (final model)
│   ├── step_000002900.ckpt
│   ├── step_000002800.ckpt
│   └── ... (5-10 checkpoints)
├── train.log (training history)
├── demo_1.wav (audio sample 1)
└── demo_2.wav (audio sample 2)
```

### On HuggingFace:
```
https://huggingface.co/datasets/yourusername/finnish-tts-dataset
└── finnish-speaker-2000-complete.tar.gz (8GB)
    Contains: 2000 WAV + LAB + NPY files
```

### Total Storage Needed:
- Local: ~300MB (checkpoints + logs + demos)
- HuggingFace: 8GB (dataset with VQ tokens)

---

## 🚨 EMERGENCY: Training Not Complete?

If training hasn't finished step 3000:

### Option 1: Wait (Recommended if close)
- Check step count: `tail -1 train.log`
- If at step 2800+, wait for completion (~30 min)
- Cost: $0.60 more

### Option 2: Use Latest Checkpoint (If urgent)
- Download step_000002500.ckpt or whatever is latest
- Model will be slightly lower quality
- But saves waiting time

### Option 3: Abort & Save Budget
- Download current checkpoint
- Don't generate demos
- Upload dataset only
- Delete instance
- Cost: $0.30

---

## ⏰ Timeline

### Assuming Training Just Finished:

**00:00 - Start downloads**
- Run download_everything.sh

**00:05 - Checkpoints downloaded**
**00:06 - Logs downloaded**
**00:11 - Demos generated**
**00:12 - Demos downloaded**
**00:22 - Dataset packaged**
**00:42 - Dataset uploaded to HF**
**00:45 - DONE!**

### Then:
- Verify everything
- Delete instance
- Save $4-5 remaining budget

---

## ✅ Final Checklist

Before deleting instance:

**Critical:**
- [ ] `step_000003000.ckpt` exists on local machine
- [ ] HuggingFace dataset URL works: `wget [URL]` succeeds
- [ ] At least 2 demo audio files downloaded

**Important:**
- [ ] Training log downloaded
- [ ] Multiple checkpoints downloaded (for safety)
- [ ] Demo audio sounds good quality

**Optional:**
- [ ] TensorBoard logs downloaded
- [ ] Config files saved

**Action:**
- [ ] Instance DELETED
- [ ] Billing STOPPED

---

## 💾 Backup Strategy

After downloading everything:

```bash
# Create permanent backup
cd ~/Downloads
zip -r finnish-tts-complete-backup.zip finnish-tts-backup/

# Upload to cloud storage (optional)
# Google Drive, Dropbox, iCloud, etc.

# Or create another HuggingFace repo for model
huggingface-cli repo create finnish-tts-model --type model
huggingface-cli upload yourusername/finnish-tts-model \
  finnish-tts-backup/checkpoints/step_000003000.ckpt
```

---

## 📞 Quick Reference Commands

```bash
# Check if training complete
ssh shadeform@64.247.196.21 'tail -1 ~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log'

# Download final checkpoint
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt ~/Downloads/

# Delete instance (only after downloads!)
brev delete shadeform@64.247.196.21
```

---

**🚨 REMEMBER: Once deleted, it's GONE FOREVER. Download everything first!**

**Estimated time: 45 minutes**
**Estimated cost: $0.90**
**Then you can safely delete and save remaining $4+ budget!**
