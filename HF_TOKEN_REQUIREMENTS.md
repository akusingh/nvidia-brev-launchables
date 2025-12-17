# HuggingFace Token Requirements

## TL;DR

| Phase | HF Token | Can Skip? |
|-------|----------|-----------|
| **setup.sh** | Optional | ✅ Yes |
| **Training (model download)** | **Required** | ❌ No |

---

## Detailed Breakdown

### Phase 1: Running `setup.sh` (Optional)

```bash
bash setup.sh
```

**What happens:**
- ✅ Installs system packages (no token needed)
- ✅ Installs Python dependencies (no token needed)
- ✅ Clones Fish Speech repository (no token needed)
- ⚠️ Tries to download base model (needs token)

**If you DON'T have HF token:**
- setup.sh will warn you but **won't fail**
- It will skip the base model download
- It will tell you the model will auto-download during training

**Result:** Setup completes successfully, but model isn't pre-downloaded.

---

### Phase 2: Training (Required if using gated model)

When you start training in the Jupyter notebook:

```python
# Training notebook
trainer.fit(model, train_loader, val_loader)
```

The training code needs to download the base model: `fishaudio/openaudio-s1-mini`

**This is a GATED model on HuggingFace**, meaning:
- ❌ Cannot download without authentication
- ❌ Training will fail if you don't have HF token set up

**Error you'll see:**
```
gated_repo_error: You don't have access to this model
```

---

## Recommended Flow

### For Community/Brev Launchables Users

**Step 1: Before deploying (1 minute)**
1. Get HF token: https://huggingface.co/settings/tokens
2. Accept model license: https://huggingface.co/fishaudio/openaudio-s1-mini

**Step 2: Deploy on Brev**
1. Launch instance
2. SSH in

**Step 3: Authenticate (5 seconds)**
```bash
huggingface-cli login
# Paste token, press Enter
```

**Step 4: Run setup.sh**
```bash
bash setup.sh
# Now model WILL download (you're authenticated)
```

**Step 5: Start training**
```bash
jupyter notebook finnish-tts-model-training.ipynb
# Model is ready, training runs immediately
```

**Total time:** 10-15 minutes (first time), includes model download

---

## What if you skip the token?

### Scenario: Run setup WITHOUT token

```bash
# No HF_TOKEN set
bash setup.sh
```

**Output:**
```
▶  Checking HuggingFace authentication...
⚠️  HuggingFace authentication not configured
⚠️  Some models (like fishaudio/openaudio-s1-mini) may require authentication.
   To authenticate:
   1. Get a token: https://huggingface.co/settings/tokens
   2. Run: huggingface-cli login
   OR
   2. Run: export HF_TOKEN='your_token_here'

▶  Attempting to download base model...
⚠️  Skipping model download (authentication not configured)
✅ Model will be downloaded automatically during training
```

**Then start training:**
```bash
jupyter notebook finnish-tts-model-training.ipynb
```

**What happens:**
1. Training code starts
2. Tries to import the base model
3. Fails with: `gated_repo_error: You don't have access to this model`
4. Training crashes ❌

**You'd have to:**
1. Stop training
2. Run `huggingface-cli login`
3. Restart training
4. Wait for model to download (15+ minutes)

---

## Best Practices

### ✅ DO THIS (Recommended)

```bash
# Before or immediately after SSH into Brev
huggingface-cli login
bash setup.sh
# Model downloads during setup (5-10 min)
# Training starts immediately after

# Later, just run:
jupyter notebook finnish-tts-model-training.ipynb
# No delays, model is ready
```

### ❌ DON'T DO THIS

```bash
# Don't skip auth and hope it works
bash setup.sh  # Skips model
jupyter notebook  # Fails during training
# Now stuck waiting for model download mid-training
```

---

## Token Workflow Summary

### Method 1: Interactive Login (RECOMMENDED for Launchables)

```bash
huggingface-cli login
# Prompts for token interactively
# Stores in ~/.cache/huggingface/token
# Works for all subsequent runs
```

**Best for:** One-time setup on Brev

### Method 2: Environment Variable (for automation)

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
bash setup.sh
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
jupyter notebook finnish-tts-model-training.ipynb
```

**Best for:** CI/CD or scripted deployments

### Method 3: .env File (for local development)

```bash
cp .env.example .env
# Edit .env and add: HF_TOKEN=hf_xxxxxxx
bash setup.sh
```

**Best for:** Local Mac/dev machine

---

## Troubleshooting

### "gated_repo_error: You don't have access to this model"

```
This happens during training if:
1. HF token not set, OR
2. Token is valid but you haven't accepted the license
```

**Solution:**
```bash
# 1. Accept license: https://huggingface.co/fishaudio/openaudio-s1-mini
# 2. Authenticate:
huggingface-cli login

# 3. Restart training (model will download)
jupyter notebook finnish-tts-model-training.ipynb
```

### "Unauthorized: Invalid token"

```
Token is invalid or expired
```

**Solution:**
```bash
# Get a new token: https://huggingface.co/settings/tokens
huggingface-cli logout
huggingface-cli login
# Paste new token
```

### Can I pre-download the model to save time?

Yes! After authenticating:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

Then training uses the cached model immediately.

---

## Cost Impact

**With early token setup:**
- Setup: 5-10 min (model downloads)
- Training: Starts immediately after setup

**Without token setup:**
- Setup: 2-3 min (model skipped)
- Training: Fails immediately
- Restart + model download: 15+ min delay
- Training: Finally starts

**Verdict:** Getting token upfront saves 15+ minutes and frustration.

---

## For Community Users

If sharing your Launchable with others:

1. **Document clearly** that they need:
   - HuggingFace account
   - Token from https://huggingface.co/settings/tokens
   - License acceptance: https://huggingface.co/fishaudio/openaudio-s1-mini

2. **In DEPLOYMENT.md**, emphasize:
   ```
   ⚠️  IMPORTANT: Authenticate BEFORE running setup.sh
   ```

3. **Provide clear commands:**
   ```bash
   huggingface-cli login   # Do this first!
   bash setup.sh           # Then this
   ```

---

**Summary:** HF token is **not required for setup.sh**, but **essential for training**. Get it upfront to avoid mid-training failures!
