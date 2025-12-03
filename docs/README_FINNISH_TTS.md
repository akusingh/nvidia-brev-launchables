# 🇫🇮 Finnish TTS Training - Enhanced Edition

**Optimized for Brev GPU Cloud Infrastructure**

Train high-quality Finnish Text-to-Speech models using Fish Speech with LoRA fine-tuning on Brev's powerful GPU infrastructure.

---

## 🎯 Project Overview

This enhanced training pipeline transforms the Fish Speech base model into a Finnish-speaking TTS system through parameter-efficient LoRA fine-tuning.

### Key Features

✅ **GPU-Optimized**: Automatic batch size and memory configuration  
✅ **Smart Checkpointing**: Auto-resume from crashes or interruptions  
✅ **Real-time Monitoring**: Track training progress, loss, and GPU utilization  
✅ **Dataset Validation**: Comprehensive quality checks before training  
✅ **Progress Visualization**: See training metrics in real-time  
✅ **Memory Efficient**: Works on GPUs from 12GB to 80GB+

### What's New in This Version

- **Enhanced Notebook**: Better organization, monitoring, and error handling
- **Training Monitor**: Real-time CLI tool for tracking progress
- **Dataset Converter**: Progress bars and improved error handling
- **Auto-Configuration**: GPU-aware batch size and accumulation settings
- **Quality Checks**: Validate dataset before expensive training runs

---

## 📋 Requirements

### Hardware
- **GPU**: 12GB+ VRAM (recommended: L40S, A100, H100)
- **Storage**: 10GB+ for dataset and checkpoints
- **Time**: ~1.5-2 hours for full training (750 → 2000 steps)

### Software
- Python 3.10+
- CUDA 11.8+ or 12.1+
- Fish Speech (cloned from GitHub)
- Dependencies: see `requirements.txt`

---

## 🚀 Quick Start Guide

### 1. Clone Fish Speech

```bash
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
```

### 2. Install Dependencies

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y sox libsox-dev ffmpeg portaudio19-dev

# Python dependencies
pip install -e .
pip install hydra-core omegaconf pyrootutils
pip install lightning tensorboard transformers
pip install loralib descript-audio-codec
pip install soundfile numpy tqdm loguru
```

### 3. Prepare Your Dataset

Place your Finnish audio dataset in `finnish-tts-raw/`:

```
finnish-tts-raw/
├── metadata.csv
└── audio/
    ├── file001.wav
    ├── file002.wav
    └── ...
```

**metadata.csv format:**
```csv
audio_file,text,speaker_name,source_dataset
audio/file001.wav,Hei, miten menee?,Speaker_0,dataset1
audio/file002.wav,Kiitos hyvää.,Speaker_0,dataset1
```

### 4. Convert Dataset

```bash
python convert_finnish_dataset.py
```

This creates:
- `data/FinnishSpeaker/clip_*.wav` (24kHz mono)
- `data/FinnishSpeaker/clip_*.lab` (transcriptions)

### 5. Open the Training Notebook

```bash
jupyter notebook finnish-tts-model-training.ipynb
```

Or use VS Code's notebook interface!

### 6. Follow the Notebook Steps

The enhanced notebook guides you through:
1. ✅ GPU verification and configuration
2. 📦 Dependency installation
3. 🔐 HuggingFace login (for base model)
4. 📥 Download base model (openaudio-s1-mini)
5. 📊 Load and validate dataset
6. 🔧 Extract VQ tokens
7. 📦 Pack dataset for training
8. 🚀 Start training (with auto-configuration)
9. 📈 Monitor progress in real-time
10. 🔗 Merge LoRA weights
11. 💾 Export and download model

---

## 🛠️ Tools & Utilities

### Training Monitor (CLI)

Watch your training in real-time:

```bash
# Single snapshot
python monitor_training.py

# Continuous monitoring (updates every 10 seconds)
python monitor_training.py --watch

# Custom interval
python monitor_training.py --watch --interval 5
```

Shows:
- Current training step and progress bar
- Checkpoint status
- Loss metrics and trends
- GPU utilization and memory
- Recent log entries
- Estimated time remaining

### Dataset Converter

Enhanced with progress bars:

```bash
python convert_finnish_dataset.py
```

Features:
- ✅ Progress bar (with tqdm)
- ✅ Validation checks
- ✅ Error handling
- ✅ Statistics summary

### Validate Dataset

Check dataset quality before training:

```bash
python validate_dataset.py data/FinnishSpeaker
```

Checks:
- File counts (WAV, LAB, NPY)
- Audio format and quality
- Transcription lengths
- Missing files
- Corrupted data

---

## 📊 Training Configuration

The notebook automatically configures training based on your GPU:

| GPU Memory | Batch Size | Grad Accum | Effective Batch |
|------------|------------|------------|-----------------|
| 40GB+      | 4          | 1          | 4               |
| 24GB+      | 2          | 2          | 4               |
| 12GB+      | 2          | 2          | 4               |
| <12GB      | 1          | 4          | 4               |

### Manual Configuration

Edit training parameters in the notebook:

```python
# LoRA Configuration
LORA_R = 8           # Rank (higher = more capacity)
LORA_ALPHA = 16      # Alpha scaling

# Training
MAX_STEPS = 2000     # Total training steps
VAL_INTERVAL = 50    # Validation frequency
BATCH_SIZE = 2       # Per-device batch size
GRAD_ACCUM = 2       # Gradient accumulation
```

---

## 📈 Monitoring Training

### In the Notebook

Run the monitoring cell periodically:

```python
monitor_training()  # Shows checkpoints, logs, GPU stats
```

### Via CLI

Use the dedicated monitor tool:

```bash
python monitor_training.py --watch
```

### TensorBoard

```bash
tensorboard --logdir results/FinnishSpeaker_2000_finetune
```

### Check GPU Utilization

```bash
watch -n 1 nvidia-smi
```

---

## 🎨 Advanced Features

### Resume Training

The notebook automatically detects and resumes from the latest checkpoint!

```python
# Training config helper finds latest checkpoint
config = TrainingConfig()
command, current_step, ckpt = config.generate_command()

# Shows: "✅ Resuming from checkpoint: step_000000750.ckpt"
```

### Custom Dataset

Use your own data:

```python
# In the notebook
dataset_dir = "data/YourCustomDataset"
```

### Multi-GPU Training

For multiple GPUs, modify:

```bash
# In training command
trainer.devices=2  # Use 2 GPUs
```

### Export Formats

After training:

1. **LoRA Adapters** (small, ~100MB)
2. **Merged Model** (full model with LoRA merged)
3. **Compressed Archive** (for deployment)

---

## 🚢 Deployment

### 1. Download Trained Model

```bash
# Archive is created automatically
# FinnishSpeaker_2000_trained.tar.gz
```

### 2. Extract on Deployment Server

```bash
tar -xzf FinnishSpeaker_2000_trained.tar.gz
```

### 3. Use with Fish Speech WebUI

```bash
python tools/webui.py --model checkpoints/FinnishSpeaker_2000_finetuned
```

### 4. Or CLI Inference

```bash
python tools/llama/generate.py \
  --text "Hei, tämä on testi." \
  --checkpoint checkpoints/FinnishSpeaker_2000_finetuned \
  --output output.wav
```

---

## 🐛 Troubleshooting

### Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce batch size in notebook
2. Increase gradient accumulation
3. Enable gradient checkpointing
4. Use smaller LoRA rank

### Training Stalls

**Symptoms:** No checkpoint updates, GPU idle

**Solutions:**
1. Check logs: `tail -f results/*/train.log`
2. Verify dataset was packed correctly
3. Check for disk space issues
4. Restart training from last checkpoint

### Poor Audio Quality

**Symptoms:** Robotic/garbled output

**Solutions:**
1. Train for more steps (2000+)
2. Use more training data
3. Validate dataset quality
4. Check audio sample rate (must be 24kHz)

### Cannot Resume Training

**Symptoms:** Training starts from scratch

**Solutions:**
1. Check checkpoint path in config
2. Ensure checkpoint file exists
3. Use absolute paths for external checkpoints

---

## 📚 File Structure

```
fish-speech/repo/
├── finnish-tts-model-training.ipynb    # Main training notebook (ENHANCED)
├── convert_finnish_dataset.py          # Dataset converter (with progress)
├── monitor_training.py                 # Training monitor tool (NEW)
├── validate_dataset.py                 # Dataset validator
├── README_FINNISH_TTS.md              # This file
│
├── finnish-tts-raw/                    # Your raw dataset
│   ├── metadata.csv
│   └── audio/
│
├── data/FinnishSpeaker/                # Converted dataset
│   ├── clip_0001.wav
│   ├── clip_0001.lab
│   ├── clip_0001.npy
│   └── ...
│
├── checkpoints/                        # Models
│   ├── openaudio-s1-mini/             # Base model
│   └── FinnishSpeaker_2000_finetuned/ # Your model
│
└── results/                            # Training outputs
    └── FinnishSpeaker_2000_finetune/
        ├── checkpoints/
        ├── train.log
        └── ...
```

---

## 🎓 Tips & Best Practices

### Dataset Quality

- **Use clean audio**: 24kHz, mono, minimal background noise
- **Diverse speakers**: Multiple speakers improve robustness
- **Balanced lengths**: Mix of short and long utterances
- **Clean transcripts**: Accurate, properly punctuated text

### Training

- **Monitor regularly**: Check loss and audio samples
- **Save checkpoints**: Don't lose progress to crashes
- **Start small**: Test with 500 steps first
- **Iterate**: Fine-tune hyperparameters based on results

### GPU Utilization

- **Batch size**: Maximize without OOM
- **Num workers**: Match CPU cores
- **Mixed precision**: Always use bf16/fp16
- **Monitor temperature**: Keep GPU cool

### Brev Specific

- **Use Launchables**: Create reusable environments
- **Save checkpoints**: Use Brev's persistent storage
- **GPU Selection**: Pick based on dataset size:
  - Small (< 1000 samples): L40S (12GB)
  - Medium (1000-5000): A100 (40GB)
  - Large (5000+): H100 (80GB)

---

## 🤝 Contributing

Improvements welcome! Ideas:

- Better hyperparameter tuning
- Multi-speaker support
- Real-time audio preview
- Automatic quality metrics
- Integration with other TTS engines

---

## 📝 License

This enhanced training pipeline follows Fish Speech's license.
Helper scripts are MIT licensed.

---

## 🙏 Acknowledgments

- **Fish Speech** team for the excellent TTS framework
- **Brev** for GPU cloud infrastructure
- **Finnish TTS community** for datasets

---

## 📧 Support

Having issues? 

1. Check the troubleshooting section above
2. Review Fish Speech docs: https://github.com/fishaudio/fish-speech
3. Contact Brev support: brev-support@nvidia.com
4. Open an issue on GitHub

---

**Happy Training! 🎉🇫🇮**

Built with ❤️ for the Finnish TTS community on Brev
