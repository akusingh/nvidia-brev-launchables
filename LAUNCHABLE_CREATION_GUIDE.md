# 🚀 Create Finnish TTS Launchable - Step-by-Step Guide
**Date:** December 3, 2025  
**Based on:** Official NVIDIA Brev Launchables Documentation

---

## 📋 Prerequisites

Before starting, make sure:
- [ ] GitHub repo is public and pushed
- [ ] setup.sh is working
- [ ] README.md has clear instructions
- [ ] All placeholder "yourusername" URLs are updated

---

## 🎯 Launchables Creation Flow

### Step 1: Files and Runtime

**Go to:** [brev.nvidia.com](https://brev.nvidia.com) → Launchables tab → Create Launchable

**Choose:** Git Repository

**Configuration:**
```
Repository URL: https://github.com/YOURUSERNAME/finnish-tts-brev
(Or specific notebook if you have one)

Runtime: VM Mode (recommended)
✅ GPU VM with Ubuntu 22.04
✅ Docker, Python, CUDA pre-installed
✅ Can install additional dependencies in terminal

Click Next →
```

**Why VM Mode?**
- ✅ Simpler than Custom Container
- ✅ Users can customize in terminal
- ✅ Works with setup.sh script
- ✅ No Docker Compose needed

---

### Step 2: Configure the Runtime

**Upload setup script:**

```bash
# Option 1: Upload file
# Click "Upload" → Select setup.sh from your repo

# Option 2: Paste script
# Click "Paste" → Copy contents of setup.sh
```

**Your setup.sh will:**
1. Install system dependencies (sox, ffmpeg)
2. Install PyTorch with CUDA
3. Clone Fish Speech repo
4. Install Python packages
5. Download base model

**Click Next →**

---

### Step 3: Jupyter and Networking

**Jupyter Notebook Experience:**
```
Select: Yes
✅ Installs Jupyter on host
✅ Provides one-click button to access code
✅ Users can run notebooks directly
```

**Pre-expose tunnels:**
```
Optional: No tunnels needed for TTS training
(Unless you want to expose web UI)

Click Next →
```

---

### Step 4: Compute Configuration

**Select GPU:**

**Recommended Configuration:**
```
GPU: A100-80GB
- VRAM: 80GB (plenty for training)
- Cost: ~$1.20/hr
- Training time: ~4 hours
- Total cost: ~$5

Alternative (cheaper):
GPU: L40S
- VRAM: 48GB (sufficient)
- Cost: ~$0.60/hr
- Training time: ~4 hours
- Total cost: ~$2.40
```

**Disk Storage:**
```
Default: 50GB
Recommended: 50GB (enough for dataset + model + checkpoints)
```

**Click Next →**

---

### Step 5: Final Review

**Launchable Name:**
```
Finnish Text-to-Speech Training with LoRA
```

**Launchable Description:**
```
Train high-quality Finnish TTS models using Fish Speech and LoRA fine-tuning.

Features:
• LoRA fine-tuning (memory efficient)
• Automated setup with one command
• ~4 hours training time
• ~$5 cost on A100 (or $2.40 on L40S)
• Production-ready pipeline

Users bring their own Finnish audio data (500+ samples) or use public datasets.

Requirements:
• Finnish audio files (WAV format)
• Text transcripts (.lab files)
• Minimum 500 samples (1 hour audio)
• Recommended 2000+ samples (4+ hours)
```

**Preview Deploy Page:**
- Check that everything looks correct
- Test the shareable link

**Click Create Launchable →**

---

## ✅ After Creation

**You'll get:**
1. ✅ Shareable link (e.g., brev.nvidia.com/launchables/your-launchable-id)
2. ✅ Markdown badge to embed in README
3. ✅ Deploy page for users to launch

**Next steps:**
1. Test the Launchable yourself (deploy it)
2. Verify setup.sh runs correctly
3. Fix any issues
4. Share with community!

---

## 📝 What to Put in Each Field

### Detailed Configuration:

**Step 1 - Repository:**
```
Repository URL: https://github.com/YOURUSERNAME/finnish-tts-brev
Branch: main (default)
Path to code: / (root)
Runtime: VM Mode
```

**Step 2 - Setup Script:**
```bash
# Upload your setup.sh or paste this content:

#!/bin/bash
set -e

echo "🇫🇮 Finnish TTS Training Setup"
echo "================================"

# Update system
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs sox libsox-dev ffmpeg

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone Fish Speech
cd $HOME
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
pip install -e .

# Install dependencies
pip install protobuf==3.20.3 hydra-core omegaconf loguru

# Download base model
python tools/download_models.py

echo "✅ Setup complete! Ready to train."
```

**Step 3 - Jupyter:**
```
Jupyter Notebook: Yes
Expose tunnels: No (unless you want web UI)
```

**Step 4 - Compute:**
```
GPU: A100-80GB (recommended) or L40S (budget)
Disk: 50GB
Region: Any available
```

**Step 5 - Final:**
```
Name: Finnish TTS Training with LoRA

Tags: (add these for discoverability)
- tts
- speech-synthesis
- finnish
- lora
- fish-speech
- audio
- pytorch

Description: (see above)
```

---

## 🧪 Test Your Launchable

**After creation:**

1. **Deploy your own Launchable**
   ```
   Go to: Launchables tab
   Find: Your Finnish TTS Launchable
   Click: Deploy Launchable
   ```

2. **Wait for instance to start** (~2-3 minutes)

3. **Access Jupyter**
   ```
   Click: "Open Jupyter" button
   Check: setup.sh ran successfully
   ```

4. **Test training**
   ```bash
   # In Jupyter terminal:
   cd ~/finnish-tts-brev
   
   # Verify setup
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Should print: True
   ```

5. **If issues found:**
   - Fix in your GitHub repo
   - Launchable will use latest code automatically

---

## 💡 Pro Tips

### Optimize for Users:

**1. Add Quick Start in README:**
```markdown
## 🚀 Quick Start with Brev Launchable

[![Launch on Brev](https://brev.nvidia.com/badge.svg)](YOUR_LAUNCHABLE_LINK)

1. Click the badge above
2. Select your GPU (A100-80GB recommended)
3. Wait 5 minutes for setup
4. Upload your Finnish audio data
5. Start training!
```

**2. Add Usage Instructions:**
```markdown
## After Deployment

Your Brev instance will have:
- ✅ Fish Speech installed
- ✅ PyTorch with CUDA
- ✅ Base model downloaded
- ✅ All dependencies ready

Next steps:
1. Upload your Finnish dataset to `datasets/finnish-tts-raw/`
2. Open the training notebook
3. Run all cells
4. Download your trained model after ~4 hours
```

**3. Add Cost Estimate:**
```markdown
## 💰 Cost Estimate

| GPU | Time | Cost | Quality |
|-----|------|------|---------|
| L40S | 4 hrs | $2.40 | Good ✓ |
| A100-40GB | 4 hrs | $4.00 | Better ✓✓ |
| A100-80GB | 4 hrs | $4.80 | Best ✓✓✓ |

Recommendation: Start with L40S for testing ($2.40)
```

---

## 🎯 Checklist Before Creating Launchable

**GitHub Repo:**
- [ ] Code is public
- [ ] setup.sh is in root
- [ ] README has clear instructions
- [ ] No hardcoded paths (use $HOME)
- [ ] All placeholders replaced

**Setup Script:**
- [ ] Has shebang (#!/bin/bash)
- [ ] Has error handling (set -e)
- [ ] Installs all dependencies
- [ ] Downloads base model
- [ ] Ends with success message

**Documentation:**
- [ ] README explains data requirements
- [ ] Links to public datasets
- [ ] Shows expected folder structure
- [ ] Includes cost estimates

**Testing:**
- [ ] (Optional) Tested on fresh instance
- [ ] setup.sh runs without errors
- [ ] All dependencies install
- [ ] Training starts successfully

---

## 🚀 Ready to Create?

**Steps:**

1. **Update GitHub** (if not done):
   ```bash
   cd /Users/arunkumar.singh/nvidia-brev
   git add setup.sh README.md launchable.yaml
   git commit -m "Prepare for Launchables"
   git push
   ```

2. **Go to Brev:**
   - Visit: https://brev.nvidia.com
   - Click: Launchables tab
   - Click: Create Launchable

3. **Follow the 5 steps above**

4. **Test your Launchable**

5. **Share with community!**

---

## 📞 Need Help?

**Common Issues:**

**Q: Setup script fails**
- Check logs in Jupyter terminal
- Verify all dependencies install
- Check for typos in commands

**Q: GPU not detected**
- Verify CUDA installed: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Q: Can't access Jupyter**
- Wait 2-3 minutes for setup to complete
- Check instance status in Brev dashboard
- Try refreshing browser

**Q: Training crashes**
- Check GPU memory: `nvidia-smi`
- Reduce batch size if OOM
- Check logs for error details

---

## 🎉 After Launch

**Promote your Launchable:**

1. **Add badge to GitHub README:**
   ```markdown
   [![Launch on Brev](https://brev.nvidia.com/badge.svg)](YOUR_LINK)
   ```

2. **Share on social media:**
   - Twitter/X
   - Reddit (r/MachineLearning)
   - LinkedIn

3. **Email NVIDIA for featuring:**
   ```
   To: brev-support@nvidia.com
   Subject: Feature Request: Finnish TTS Launchable
   
   Hi team,
   
   I created a Launchable for Finnish TTS training:
   [YOUR_LAUNCHABLE_LINK]
   
   Would love to be featured on brev.nvidia.com/launchables
   
   Thanks!
   ```

4. **Monitor metrics:**
   - Views
   - Deployments
   - User feedback

Good luck! 🚀
