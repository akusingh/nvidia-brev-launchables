# Testing Strategy for NVIDIA Launchables Submission
**Date:** December 3, 2025  
**Question:** Should we test before submitting, or just send the repo?

---

## 🤔 Two Approaches

### Option A: Submit Now, Test Later (Risky)
```
Timeline:
1. Push code to GitHub (5 min)
2. Email NVIDIA (2 min)
3. Wait for their review (1-2 weeks)
4. If they find issues → Fix and resubmit

Pros:
✅ Fastest (7 minutes)
✅ Free (no GPU testing cost)
✅ NVIDIA might test it for you

Cons:
❌ Risk: Setup might not work
❌ Risk: Broken dependencies
❌ Risk: Path issues
❌ Looks unprofessional if it fails
❌ Delays acceptance (need to fix + resubmit)
```

### Option B: Smoke Test First (Recommended)
```
Timeline:
1. Push code to GitHub (5 min)
2. Launch cheap GPU instance (L40S, $0.60/hr)
3. Test the Launchable workflow (30 min)
   - Clone repo
   - Run setup.sh
   - Download sample dataset
   - Start training (run 50 steps, just to verify)
4. Fix any issues found
5. Push fixes to GitHub
6. Email NVIDIA with confidence

Pros:
✅ Know it actually works
✅ Find issues before NVIDIA does
✅ Professional impression
✅ Faster acceptance (no back-and-forth)
✅ Can include "tested on X GPU" in email

Cons:
❌ Costs $0.30 (30 min × $0.60/hr)
❌ Takes 30 extra minutes
```

---

## 🎯 My Recommendation: DO SMOKE TEST

### Why?

**1. Your setup.sh was tested in Jupyter, not clean shell**
```bash
# In Jupyter notebook:
- Variables might persist between cells
- Paths might be relative to notebook location
- Some commands might work differently
- conda/venv behavior different

# In clean Brev instance:
- Fresh shell environment
- No pre-existing paths
- Must work from bash script only
- This is what users will experience
```

**2. You'll find issues you didn't expect**

Common problems I've seen:
- `$HOME` vs `~` path differences
- Missing `cd` commands
- Race conditions (file not ready yet)
- Wrong Python being called
- Permission issues
- Protobuf version conflicts (you found this before!)

**3. NVIDIA might not test thoroughly before listing**

From their email:
> "Want to be featured? Email us at brev-support@nvidia.com"

They likely:
- ✅ Check repo exists
- ✅ Review launchable.yaml format
- ✅ Check documentation quality
- ❓ Might not run full test (too expensive)

If it breaks for first user → bad reviews, delisted

**4. $0.30 is cheap insurance**

```
Cost of smoke test: $0.30
Cost of bad review: Priceless (reputation)
Cost of rejection: Time to fix + resubmit (weeks)

ROI: Spend $0.30, avoid weeks of delays
```

---

## 🧪 Smoke Test Plan (30 min, $0.30)

### What to Test:

```bash
# 1. Launch instance (cheap GPU is fine)
brev open --gpu L40S --name test-finnish-launchable

# 2. Clone your repo
git clone https://github.com/YOURUSERNAME/finnish-tts-brev.git
cd finnish-tts-brev

# 3. Run setup (this is what users do)
bash setup.sh

# Expected: 
✅ All dependencies install
✅ Fish Speech cloned
✅ No errors in output

# 4. Verify environment
bash scripts/validate_setup.sh  # (if you add this script)

# Expected:
✅ Python 3.10+
✅ PyTorch with CUDA
✅ GPU detected
✅ Fish Speech installed

# 5. Test with tiny dataset (just to verify pipeline)
# Download 50 samples from Common Voice Finnish
mkdir -p datasets/finnish-tts-raw/audio
# ... download 50 samples ...

# 6. Run conversion
python scripts/convert_finnish_dataset.py

# Expected:
✅ WAV files processed
✅ LAB files created
✅ VQ tokens extracted
✅ Proto files packed

# 7. Start training (just 10 steps to verify)
cd $HOME/fish-speech
python fish_speech/train.py \
  --config-name text2semantic_finetune \
  trainer.max_steps=10 \
  ...other params...

# Expected:
✅ Training starts
✅ No errors
✅ Loss decreases
✅ Checkpoint saves

# 8. If all works:
✅ Delete instance
✅ Push any fixes to GitHub
✅ Email NVIDIA with confidence

# 9. If issues found:
❌ Fix them
❌ Test again
❌ Then submit
```

---

## 💰 Cost Breakdown

### Testing Costs:

```
L40S GPU: $0.60/hr

Smoke test (verify pipeline works):
- 30 minutes × $0.60/hr = $0.30
- Tests: setup, conversion, 10 training steps
- Result: High confidence it works ✓

Medium test (50 steps):
- 45 minutes × $0.60/hr = $0.45
- Tests: setup, conversion, 50 training steps  
- Result: Very high confidence ✓

Full test (complete training):
- 4 hours × $1.20/hr (A100) = $4.80
- Tests: Everything end-to-end
- Result: 100% confidence but expensive ✗
```

**Recommendation: Do smoke test ($0.30)**

---

## 📋 Testing Checklist

### Before Submission:

**Minimal Testing (Free, 5 min):**
- [ ] Code pushed to GitHub
- [ ] launchable.yaml valid YAML syntax
- [ ] README.md has clear instructions
- [ ] No hardcoded paths (use $HOME)
- [ ] setup.sh has proper shebang (#!/bin/bash)

**Smoke Test ($0.30, 30 min):**
- [ ] Launch L40S instance
- [ ] Clone repo works
- [ ] setup.sh completes without errors
- [ ] Dependencies all install
- [ ] GPU detected
- [ ] Training starts (10 steps)
- [ ] No obvious bugs

**Medium Test ($0.45, 45 min):**
- [ ] All of smoke test
- [ ] Train 50 steps
- [ ] Checkpoint saves
- [ ] Loss decreases properly
- [ ] Can resume from checkpoint

**Full Test ($4.80, 4 hours):**
- [ ] Complete training run
- [ ] Model merges successfully
- [ ] Inference works
- [ ] Audio quality good

---

## 🎯 My Specific Recommendation for You

### Do This:

**1. Today (Free, 15 min):**
- Push code to GitHub
- Fix any obvious issues in launchable.yaml
- Update README with data instructions

**2. Tomorrow ($0.30, 30 min):**
- Launch L40S instance
- Run smoke test with 50 samples
- Fix any issues found
- Push fixes

**3. Day After (Free, 5 min):**
- Email NVIDIA with tested repo
- Mention "Tested on L40S" in email
- Include any limitations found

### Why This Timeline?

- ✅ Gives you time to review code with fresh eyes
- ✅ Catches obvious bugs before spending $0.30
- ✅ Professional approach
- ✅ Still fast (3 days total)

---

## 🚨 Red Flags to Check Before Testing

**Review these in your code NOW (free):**

```bash
# 1. Check for hardcoded paths
grep -r "/home/shadeform" .
# Should be: $HOME or ~/

# 2. Check for notebook-specific code
grep -r "get_ipython" .
# Should be: none (use plain Python)

# 3. Check for missing error handling
grep -r "set -e" setup.sh
# Should be: present (exit on error)

# 4. Check for absolute vs relative paths
grep -r "cd " setup.sh
# Make sure all paths work from any directory

# 5. Check Python package versions
cat requirements.txt  # (if you have one)
# Make sure versions are pinned (torch==2.0.1, not torch>=2.0)
```

---

## 📊 Risk Assessment

### If You Submit Without Testing:

```
Best Case (50% probability):
- Everything works
- NVIDIA accepts
- Users happy
- Outcome: Success! 🎉

Medium Case (30% probability):
- Small issues (typos, wrong paths)
- NVIDIA asks you to fix
- You fix and resubmit
- Outcome: 2 week delay

Worst Case (20% probability):
- Major issues (setup.sh fails)
- NVIDIA rejects
- Bad first impression
- Need complete overhaul
- Outcome: Might not get featured
```

### If You Test First:

```
Best Case (80% probability):
- Find small issues
- Fix them before submitting
- NVIDIA accepts quickly
- Outcome: Success! 🎉
- Cost: $0.30

Medium Case (15% probability):
- Find major issues
- Takes 2 hours to fix ($1.20)
- But saves rejection
- Outcome: Still success
- Cost: $1.50 total

Worst Case (5% probability):
- Discover fundamental flaw
- Need to redesign
- But discovered early (before public)
- Outcome: Dodged bullet
- Cost: $0.30 (saved reputation)
```

**Expected Value:**
- Test first: 95% success, $0.30-1.50 cost
- Submit blind: 50% immediate success, 50% delays/rejection

---

## ✅ Final Answer

### YES, DO SMOKE TEST!

**Recommended flow:**

```
Today (15 min, free):
1. Push code to GitHub
2. Review for obvious issues

Tomorrow (30 min, $0.30):
3. Smoke test on L40S
4. Fix any issues
5. Push fixes

Day After (5 min, free):
6. Email NVIDIA with confidence
7. Mention "Tested on L40S" 
```

**Total investment:**
- Time: 50 minutes
- Cost: $0.30
- Confidence: 95%+
- Professional: ✓

**vs.**

**Skip testing:**
- Time: 20 minutes
- Cost: $0
- Confidence: 50%
- Risk: Rejection/delays

---

## 🎯 Action Plan

**Want me to:**

1. ✅ Give you GitHub username to update files
2. ✅ Generate smoke test commands
3. ✅ Create validate_setup.sh script
4. ✅ Help you test tomorrow

**Your choice:**
- **Option A:** Test first ($0.30, recommended) 
- **Option B:** Submit now (free, risky)

**What's your GitHub username?** I'll prepare everything for you to test tomorrow. 🚀
