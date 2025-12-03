# Testing Your Launchable - The Right Way

## 🎯 The Problem

**You need to verify:**
1. One-click setup actually works
2. Dataset downloads correctly
3. Training starts without manual intervention
4. Costs are accurate
5. Docs match reality

**But you only have $12 left!**

---

## 💰 Budget-Conscious Testing Strategy

### Option A: Quick Validation on Current Instance ($0)
**Pro:** Free, fast
**Con:** Doesn't test clean-room setup

```bash
# On your current Brev instance
cd ~/finnish-tts-brev

# Simulate fresh start
./verify_launchable.sh

# Check all dependencies
python -c "import torch; print(torch.cuda.is_available())"
python -c "import fish_speech; print('Fish Speech OK')"

# Verify dataset is accessible
ls -la ~/data/FinnishSpeaker/*.npy | wc -l  # Should be 2000
```

**Cost: $0**
**Risk: High** (might work for you but fail for new users)

---

### Option B: Partial Fresh Instance Test ($0.50) ⭐ **RECOMMENDED**
**Pro:** Tests critical parts, super affordable
**Con:** Doesn't test full 3000-step training

```bash
# Step 1: Create new Launchable instance
brev create finnish-tts-launchable-test --instance-type A100-40GB

# Step 2: SSH in
brev ssh finnish-tts-launchable-test

# Step 3: Test setup (5 min, $0.10)
git clone https://github.com/yourusername/finnish-tts-brev.git ~/finnish-tts-brev
cd ~/finnish-tts-brev
bash setup.sh  # Test this works

# Step 4: Test dataset download (5 min, $0.10)
cd ~
mkdir -p data
wget https://huggingface.co/datasets/.../finnish-speaker-2000-complete.tar.gz
tar -xzf finnish-speaker-2000-complete.tar.gz -C data/
ls data/FinnishSpeaker/*.npy | wc -l  # Should be 2000

# Step 5: Test dataset packing (5 min, $0.10)
cd ~/fish-speech
python fish_speech/data/packer.py ~/data/FinnishSpeaker ~/data/protos
ls ~/data/protos/*.proto*  # Verify files created

# Step 6: Test training START (10 min, $0.20)
python fish_speech/train.py \
  --config-name text2semantic_finetune \
  pretrained_ckpt_path=~/finnish-tts-brev/checkpoints/openaudio-s1-mini/model.pth \
  project=FinnishSpeaker_test \
  train_dataset.proto_files=~/data/protos \
  trainer.max_steps=50

# Watch loss for 3-4 iterations, then kill
tail -f ~/fish-speech/results/FinnishSpeaker_test/train.log
# If loss decreasing → SUCCESS!
# Ctrl+C, then: pkill -f train.py

# Step 7: STOP INSTANCE IMMEDIATELY
exit
brev stop finnish-tts-launchable-test
```

**Cost: ~$0.50 (30 minutes)**
**Risk: Low** (you verified setup, download, packing, training start)
**Confidence: 85%**

---

### Option C: Full End-to-End Test ($11.00)
**Pro:** 100% confidence it works
**Con:** Uses almost entire remaining budget

```bash
# Same as Option B, but run full 3000 steps
trainer.max_steps=3000

# Wait 4.5 hours
# Cost: $5.40 for training + $0.60 for setup = $6.00 total
```

**Cost: ~$6.00**
**Risk: None**
**Confidence: 100%**

---

## ✅ RECOMMENDED: Hybrid Approach ($0.50)

Test everything EXCEPT full training:

```
✅ Setup works (5 min)
✅ Dataset downloads (5 min)  
✅ Dataset extracts correctly (3 min)
✅ Dataset packing works (5 min)
✅ Training STARTS and loss decreases (10 min)
❌ Full 3000-step training (skip to save $5)

Total: 30 minutes, $0.50
Confidence: 85% (good enough to submit!)
```

---

## 📋 Complete Testing Checklist

### Before Testing
- [ ] Current training finished
- [ ] Dataset packaged (tar.gz)
- [ ] Dataset uploaded to HuggingFace
- [ ] launchable.yaml has correct URLs
- [ ] Current Brev instance STOPPED

### During Testing (30 min, $0.50)
- [ ] Fresh instance created
- [ ] Repository clones successfully
- [ ] setup.sh completes without errors
- [ ] Dataset downloads from HuggingFace
- [ ] Dataset extracts (2000 WAV + LAB + NPY present)
- [ ] Dataset packing creates proto files
- [ ] Training command starts successfully
- [ ] Loss begins decreasing
- [ ] Logs are being written
- [ ] Checkpoints directory created

### After Testing
- [ ] Document any issues found
- [ ] Update launchable.yaml if needed
- [ ] Test instance STOPPED
- [ ] Budget tracking updated
- [ ] Ready to submit!

---

## 🎯 Testing Timeline

### Today (After current training completes):
**9:00 AM**: Training finishes
**9:00-9:30 AM**: Test final model, generate sample audio ($0.60)
**9:30-9:45 AM**: Package dataset ($0.30)
**9:45-10:00 AM**: Upload to HuggingFace (free)
**10:00 AM**: STOP current instance

**Budget used: $6.90 total**
**Remaining: $5.10**

### Tomorrow:
**9:00 AM**: Create fresh test instance
**9:00-9:30 AM**: Test setup, download, pack, training start ($0.50)
**9:30 AM**: STOP test instance
**9:30-11:00 AM**: Update documentation (local, free)
**11:00 AM**: GitHub Pages setup (free)

**Budget used: $7.40 total**
**Remaining: $4.60**

### Day 3:
**Submit to NVIDIA** (free)

**Final budget: $4.60 remaining for future improvements!**

---

## 💡 Why This Approach Works

### You're Testing the CRITICAL PATH:
1. ✅ **Setup**: Does it install all dependencies?
2. ✅ **Data**: Can users get the dataset?
3. ✅ **Preprocessing**: Does packing work?
4. ✅ **Training**: Does it start and run?

### You're NOT Testing:
5. ❌ **Full Training**: You already know this works (your current training!)

**Logic:** If steps 1-4 work on a fresh instance, and you already have proof that full training completes (your current run), then the Launchable works!

---

## 🚨 What Could Go Wrong?

### Scenario 1: Setup Fails
**Cost so far:** $0.10 (5 minutes)
**Action:** Fix setup.sh, retry
**Max cost to debug:** $0.50

### Scenario 2: Dataset Download Fails
**Cost so far:** $0.20 (10 minutes)
**Action:** Fix HuggingFace permissions, retry
**Max cost to debug:** $0.30

### Scenario 3: Training Won't Start
**Cost so far:** $0.40 (20 minutes)
**Action:** Check paths, dependencies, retry
**Max cost to debug:** $1.00

**Worst case: You spend $2 debugging instead of $0.50**
**Still have $3+ left over!**

---

## 📊 Risk vs Cost Analysis

| Approach | Cost | Time | Risk of Failure | Recommended? |
|----------|------|------|-----------------|--------------|
| No testing | $0 | 0 | **HIGH** (50%+) | ❌ Don't do this |
| Current instance only | $0 | 30 min | Medium (30%) | ⚠️ Risky |
| **Setup test (Hybrid)** | **$0.50** | **30 min** | **Low (15%)** | **✅ BEST** |
| Full end-to-end | $6.00 | 5 hrs | Very Low (5%) | 💰 If budget allows |

---

## ✅ Final Answer: YES, Test on Launchables!

**But do it smartly:**

1. ✅ Test on fresh instance ($0.50)
2. ✅ Verify setup → download → pack → training starts
3. ❌ Skip full 3000-step training (you already know it works)
4. ✅ Save $4-5 for future improvements
5. ✅ Submit with 85% confidence

**This is the industry-standard approach for CI/CD:**
- Smoke test (does it start?) ← This is what you're doing
- Integration test (do parts work together?) ← This too
- Full regression test (3000 steps) ← Skip, you have proof from current run

---

## 🎬 Exact Commands for Tomorrow

```bash
# Create instance
brev create finnish-tts-test --instance-type A100-40GB

# SSH in
brev ssh finnish-tts-test

# Run test sequence (copy-paste this entire block)
git clone https://github.com/yourusername/finnish-tts-brev.git ~/finnish-tts-brev && \
cd ~/finnish-tts-brev && \
bash setup.sh && \
cd ~ && mkdir -p data && \
wget https://huggingface.co/datasets/yourusername/finnish-tts-dataset/resolve/main/finnish-speaker-2000-complete.tar.gz && \
tar -xzf finnish-speaker-2000-complete.tar.gz -C data/ && \
cd ~/fish-speech && \
python fish_speech/data/packer.py ~/data/FinnishSpeaker ~/data/protos && \
python fish_speech/train.py \
  --config-name text2semantic_finetune \
  pretrained_ckpt_path=~/finnish-tts-brev/checkpoints/openaudio-s1-mini/model.pth \
  project=FinnishSpeaker_test \
  train_dataset.proto_files=~/data/protos \
  trainer.max_steps=50 &

# Watch training for 5 minutes
sleep 300
tail ~/fish-speech/results/FinnishSpeaker_test/train.log

# If you see decreasing loss → SUCCESS!
# Kill training
pkill -f train.py

# Exit
exit

# STOP instance
brev stop finnish-tts-test
```

**Time: 30 minutes**
**Cost: $0.50**
**Result: 85% confidence your Launchable works!**

---

**TL;DR: YES, test on Launchables! But do a smart "smoke test" for $0.50 instead of full $6 end-to-end. Save your budget for future improvements!** 🎯
