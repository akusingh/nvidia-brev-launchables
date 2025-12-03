# 🇫🇮 Finnish TTS Training on Brev

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Fish Speech](https://img.shields.io/badge/Fish%20Speech-TTS-green.svg)](https://github.com/fishaudio/fish-speech)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-orange.svg)](https://huggingface.co/fishaudio)

**Production-ready Finnish Text-to-Speech model training using Fish Speech + LoRA**

Train high-quality Finnish TTS models on Brev's GPU infrastructure with automated configuration, monitoring, and deployment tools.

---

## 🎯 Quick Start

### 1. Launch Brev Instance

**Recommended GPU:**
- Small dataset (<1000 samples): L40S (12GB)
- Medium dataset (1000-2000 samples): L40S or A100 (24-40GB)
- Large dataset (2000+ samples): A100 or H100 (40-80GB)

### 2. Clone and Setup

```bash
# Clone this repository (or download it to your Brev instance)
cd ~/nvidia-brev

# Run setup script
bash setup.sh
```

### 3. Prepare Your Dataset

**⚠️ IMPORTANT: Your audio data is private and will NOT be committed to Git!**

Place your Finnish audio data in `datasets/finnish-tts-raw/`:

```
datasets/finnish-tts-raw/
├── metadata.csv
└── audio/
    ├── file001.wav
    ├── file002.wav
    └── ...
```

> 🔒 **Privacy Note**: All files in `datasets/` are excluded in `.gitignore`. Your audio data, transcripts, and metadata will remain on your local machine and Brev instances only.

**metadata.csv format:**
```csv
audio_file,text,speaker_name,source_dataset
audio/file001.wav,Hei, miten menee?,Speaker_0,dataset1
audio/file002.wav,Kiitos hyvää.,Speaker_0,dataset1
```

### 4. Convert Dataset

```bash
cd scripts
python convert_finnish_dataset.py
```

### 5. Start Training

Open the training notebook:

```bash
jupyter notebook finnish-tts-model-training.ipynb
```

Or use VS Code's notebook interface!

---

## 📁 Project Structure

```
nvidia-brev/
├── README.md                           # This file
├── setup.sh                            # Setup script for Brev
├── finnish-tts-model-training.ipynb   # Main training notebook
│
├── scripts/                            # Utility scripts
│   ├── convert_finnish_dataset.py     # Dataset converter
│   ├── monitor_training.py            # Training monitor
│   └── quick_test.py                  # Inference tester
│
├── docs/                               # Documentation
│   ├── README_FINNISH_TTS.md          # Full technical guide
│   └── IMPROVEMENTS_SUMMARY.md        # What's new
│
├── datasets/                           # Your datasets
│   └── finnish-tts-raw/               # Raw input data
│
├── data/                               # Processed data
│   └── FinnishSpeaker/                # Converted dataset
│
├── checkpoints/                        # Model checkpoints
│   ├── openaudio-s1-mini/             # Base model
│   └── FinnishSpeaker_trained/        # Your trained model
│
└── results/                            # Training outputs
    └── FinnishSpeaker_2000_finetune/
        ├── checkpoints/               # Training checkpoints
        └── train.log                  # Training logs
```

---

## 🚀 Features

### ✨ What Makes This Special

- **🎯 GPU Auto-Configuration**: Automatically optimizes for your GPU (12GB-80GB)
- **📊 Real-Time Monitoring**: Track training progress with live dashboard
- **🔄 Smart Checkpointing**: Auto-resume from crashes or interruptions
- **✅ Dataset Validation**: Comprehensive quality checks before training
- **📈 Progress Visualization**: See metrics, loss, and GPU stats
- **🛠️ Complete Toolkit**: Converter, monitor, and testing tools included
- **📚 Comprehensive Docs**: Full guides and troubleshooting

### 🔧 Tools Included

1. **Training Notebook** - Step-by-step guided training
2. **Dataset Converter** - Convert your audio to Fish Speech format
3. **Training Monitor** - Real-time CLI dashboard
4. **Quick Tester** - Test your model instantly
5. **Setup Script** - One-command environment setup

---

## 📖 Documentation

- **Quick Start**: This README (you are here!)
- **Full Technical Guide**: `docs/README_FINNISH_TTS.md`
- **What's New**: `docs/IMPROVEMENTS_SUMMARY.md`
- **Training Notebook**: In-depth inline documentation

---

## 🛠️ Usage Guide

### Convert Your Dataset

```bash
cd scripts
python convert_finnish_dataset.py
```

Output: `data/FinnishSpeaker/` with WAV and LAB files

### Monitor Training (Real-time)

```bash
cd scripts
python monitor_training.py --watch
```

Shows:
- Training progress and ETA
- Loss metrics and trends
- GPU utilization
- Recent log entries

### Test Your Model

```bash
cd scripts
python quick_test.py --model ../checkpoints/FinnishSpeaker_trained --text "Hei maailma"
```

---

## 🎓 Training Process

1. **Setup** (5 min)
   - Run `setup.sh`
   - Download base model

2. **Data Preparation** (10-30 min)
   - Convert dataset
   - Extract VQ tokens
   - Pack for training

3. **Training** (1.5-2 hours)
   - Auto-configured for your GPU
   - Monitors progress automatically
   - Saves checkpoints regularly

4. **Export** (5 min)
   - Merge LoRA weights
   - Create deployment package
   - Test inference

**Total Time: ~2-3 hours**

---

## 📊 Expected Results

### Training Metrics

| Steps | Time | Quality |
|-------|------|---------|
| 500   | 30min | Basic |
| 1000  | 1hr | Good |
| 2000  | 2hr | Excellent |

### GPU Utilization

| GPU | Batch Size | Training Time (2000 steps) |
|-----|------------|----------------------------|
| L40S (12GB) | 2 | ~2.5 hours |
| A100 (40GB) | 4 | ~1.5 hours |
| H100 (80GB) | 4 | ~1 hour |

---

## 🐛 Troubleshooting

### Out of Memory?
```bash
# In the notebook, it auto-adjusts, but you can force:
# Reduce batch_size or increase grad_accumulation
```

### Training Stalls?
```bash
# Check logs
tail -f results/FinnishSpeaker_2000_finetune/train.log

# Check GPU
nvidia-smi

# Monitor progress
cd scripts && python monitor_training.py
```

### Poor Quality?
- Train for more steps (2000+)
- Validate dataset quality
- Check audio sample rate (must be 24kHz)

---

## 🎯 Brev-Specific Tips

### Instance Selection

- **Development/Testing**: L40S (12GB) - $0.60-0.80/hr
- **Production Training**: A100 (40GB) - $1.50-2.00/hr
- **Large-Scale**: H100 (80GB) - $3.00-4.00/hr

### Persistent Storage

Your training checkpoints are saved in:
```bash
results/FinnishSpeaker_2000_finetune/checkpoints/
```

Make sure to download before shutting down!

### Create a Launchable

Share your setup with the community:

1. Complete your first successful training
2. Document any custom changes
3. Submit to brev.nvidia.com/launchables
4. Email: brev-support@nvidia.com

### Cost Optimization

- Use spot instances when possible
- Monitor GPU utilization (aim for 85%+)
- Stop instances when not training
- Download checkpoints regularly

---

## 🤝 Contributing

Improvements and suggestions welcome!

### Ideas for Enhancement

- Multi-speaker support
- Real-time audio preview
- Automatic quality metrics
- Web-based training dashboard
- Integration with other TTS engines

---

## 📝 License

This training pipeline: MIT License  
Fish Speech: See fishaudio/fish-speech repository  

---

## 🙏 Credits

- **Fish Speech** - Excellent TTS framework
- **Brev** - GPU cloud infrastructure
- **Finnish TTS Community** - Dataset contributions

---

## � About Fish Speech

This project uses **Fish Speech** - a powerful multilingual TTS framework:

- **GitHub Repository**: https://github.com/fishaudio/fish-speech (training code & tools)
- **HuggingFace Models**: https://huggingface.co/fishaudio (pre-trained base models)
- **How we use it**:
  1. Clone the Fish Speech codebase from GitHub (training scripts, tools)
  2. Download pre-trained models from HuggingFace (`fishaudio/openaudio-s1-mini`)
  3. Fine-tune with LoRA on your Finnish dataset
  4. Export and deploy your custom Finnish TTS model

**Architecture**: Transformer-based with VQ-VAE audio codec + LoRA fine-tuning

---

## �📧 Support

- **Technical Docs**: `docs/README_FINNISH_TTS.md`
- **Fish Speech GitHub**: https://github.com/fishaudio/fish-speech
- **Fish Speech HF**: https://huggingface.co/fishaudio
- **Brev Support**: brev-support@nvidia.com

---

## 🚀 Ready to Start?

```bash
# 1. Run setup
bash setup.sh

# 2. Prepare your dataset
cd scripts
python convert_finnish_dataset.py

# 3. Open the notebook
jupyter notebook finnish-tts-model-training.ipynb

# 4. Follow the step-by-step guide!
```

**Happy Training! 🎉🇫🇮**

Built with ❤️ for the Brev community
