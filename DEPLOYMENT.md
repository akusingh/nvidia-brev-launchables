# Deployment Guide - Finnish TTS Training on Brev Launchables

This guide walks you through deploying a Finnish TTS training instance on Brev using the new production-ready setup.

## Pre-Deployment Checklist

- [ ] HuggingFace token ready (get from https://huggingface.co/settings/tokens)
- [ ] Accepted the model license: https://huggingface.co/fishaudio/openaudio-s1-mini
- [ ] Brev account created and verified
- [ ] Budget/credits available for compute

## Step 1: Launch on Brev

1. Go to: **https://brev.nvidia.com/launchable/deploy/now?launchableID=env-36LaSgYL2gg2G3UC7N9ZmGugpSc**
   
2. Select GPU (recommended):
   - **L40S (48GB)** - Good for 1000-2000 samples (~$1.74/hr)
   - **A100 (40GB)** - Good for 2000+ samples (~$2.00/hr)
   - **H100 (80GB)** - Faster training (~$4.00/hr)

3. Configure storage:
   - Recommended: **256GB** (leaves room for datasets + models + outputs)

4. Click **Deploy**

## Step 2: Connect to Instance

Once deployed, SSH into your Brev instance:

```bash
# Brev will give you SSH connection string
ssh user@your-instance-ip
```

## Step 3: Authenticate with HuggingFace

### Option A: Interactive Login (RECOMMENDED)

```bash
huggingface-cli login
```

Follow the prompts:
1. Paste your HF token
2. Press Enter
3. Done! You're authenticated.

### Option B: Environment Variable

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxx"
```

### Option C: .env File

```bash
cd ~/nvidia-brev-launchables
cp .env.example .env
# Edit .env with your text editor
nano .env  # Add HF_TOKEN=hf_xxx
```

## Step 4: Run Setup

```bash
cd ~/nvidia-brev-launchables
bash setup.sh
```

Expected output:
```
🇫🇮 Finnish TTS Training Setup for Brev
========================================

▶  Checking HuggingFace authentication...
✅ HuggingFace credentials found (already authenticated)

▶  Creating project directories...
✅ Project directories created

▶  Attempting to download base model...
✅ Base model downloaded successfully

...

========================================
✅ Setup Complete!
========================================
```

**Duration:** 5-10 minutes (first time)

**Tip:** If setup fails, just run it again. It will skip completed steps.

## Step 5: Prepare Your Dataset

Option A: Upload via SFTP/SCP

```bash
# Local machine:
scp -r your_dataset/ user@instance:/root/nvidia-brev-launchables/datasets/finnish-tts-raw/
```

Option B: Download from cloud storage

```bash
# On Brev instance:
cd ~/nvidia-brev-launchables/datasets/finnish-tts-raw
# Download from S3, GCS, etc.
```

Expected structure:
```
datasets/finnish-tts-raw/
├── metadata.csv
└── audio/
    ├── file001.wav
    ├── file002.wav
    └── ...
```

## Step 6: Convert Dataset

```bash
cd ~/nvidia-brev-launchables
bash convert.sh
```

**Duration:** Depends on dataset size (usually 10-30 min)

## Step 7: Start Training

```bash
cd ~/nvidia-brev-launchables
jupyter notebook finnish-tts-model-training.ipynb
```

Jupyter will output a URL with a token. Copy it and open in your browser.

In the notebook:
1. Follow the step-by-step guide
2. Run cells to start training
3. Monitor progress in real-time

## Step 8: Monitor (Optional, Different Terminal)

```bash
cd ~/nvidia-brev-launchables
bash monitor.sh --watch
```

Shows:
- Training loss
- GPU utilization
- Recent logs

## Step 9: Download Results

When training completes:

```bash
# Local machine:
scp -r user@instance:/root/nvidia-brev-launchables/results ~/local-results/
```

Or download from Brev's dashboard.

## Troubleshooting

### Setup script fails on first run

```bash
# Re-run, it will skip completed steps
bash setup.sh

# Or check logs
tail -f ~/.french-tts-setup-state  # This tracks completed steps
```

### "HuggingFace authentication not configured"

```bash
# Interactive login
huggingface-cli login

# OR set env var
export HF_TOKEN="hf_xxx"

# Then re-run setup
bash setup.sh
```

### Model download fails

1. Check you accepted the license: https://huggingface.co/fishaudio/openaudio-s1-mini
2. Wait 5-10 minutes for approval
3. Try again:
   ```bash
   huggingface-cli logout
   huggingface-cli login
   bash setup.sh
   ```

### Out of memory during training

In the Jupyter notebook, reduce batch size:
```python
batch_size = 4  # instead of 8
num_workers = 4  # instead of 8
```

### GPU not detected

```bash
nvidia-smi  # Should show GPU info

# If not, GPU may not be properly allocated
# Stop instance and relaunch with correct GPU selected
```

## Cost Management

**Cost breakdown (L40S):**
- Setup: $0.10 (5 min)
- Dataset conversion: $0.04 (2 min)
- Training (2000 steps): $5.40 (4.5 hr)
- **Total: ~$5.54**

**Ways to save:**
- Use spot instances if available
- Stop instance when not training
- Optimize batch size for your GPU
- Download results before stopping (they're lost if you don't!)

## Next Steps

- Check [AUTHENTICATION.md](AUTHENTICATION.md) for security best practices
- See [README.md](README.md) for full technical docs
- Join the community: GitHub Discussions or Issues

## Support

- **Setup issues**: See [AUTHENTICATION.md](AUTHENTICATION.md) troubleshooting
- **Training issues**: Check [README.md](README.md) FAQ
- **Brev support**: brev-support@nvidia.com
- **Fish Speech support**: https://github.com/fishaudio/fish-speech

---

**Happy Training! 🚀🇫🇮**
