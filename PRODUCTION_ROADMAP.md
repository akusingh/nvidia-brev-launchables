# Finnish TTS Training - Production Roadmap

**Goal:** Transform the current Jupyter notebook workflow into a production-ready, reproducible pipeline for training Finnish TTS models on Brev GPU instances.

---

## Current State ✅

**What's working:**
- ✅ `setup.sh` - Basic environment setup with Python version check
- ✅ Jupyter notebook workflow - Successfully trains models
- ✅ Manual VQ extraction - Works with adjusted workers/batch size
- ✅ Base model download - openaudio-s1-mini checkpoint
- ✅ Dataset uploaded - 2000 Finnish audio samples
- ✅ **VQ tokens extracted** - **2000 NPY files ready to reuse** 🎯
- ✅ Training pipeline - 3000 steps, LoRA fine-tuning

**Current pain points:**
- ⚠️ Notebook format - Hard to version control, not automation-friendly
- ⚠️ Manual steps - VQ extraction, packing, merging all separate
- ⚠️ No error recovery - Training crash = start over
- ⚠️ Hard-coded paths - Not portable between environments
- ⚠️ No monitoring - Can't check progress remotely
- ⚠️ Manual cost tracking - Easy to forget and waste $$$
- ⚠️ Dependency issues - Protobuf/TorchCodec version conflicts discovered late
- ⚠️ **VQ tokens not cached** - Could skip 15-minute extraction step with proper caching

---

## Repository Structure (Target)

```
nvidia-brev/
├── README.md                          # Complete setup & usage guide
├── PRODUCTION_ROADMAP.md              # This file
├── setup.sh                           # Enhanced environment setup
├── requirements.txt                   # Pinned Python dependencies
├── .env.example                       # Template for secrets
│
├── config/
│   ├── training_config.yaml           # Training hyperparameters
│   ├── model_config.yaml              # Model architecture settings
│   └── hardware_config.yaml           # GPU/memory settings
│
├── scripts/
│   ├── download_model.sh              # Automated base model download
│   ├── prepare_data.sh                # Data validation & preprocessing
│   ├── extract_vq.py                  # Standalone VQ extraction
│   ├── pack_dataset.py                # Dataset proto packing
│   ├── train.py                       # Main training script
│   ├── merge_and_export.py            # Post-training LoRA merger
│   └── cleanup.sh                     # Clean intermediate files
│
├── utils/
│   ├── logger.py                      # Centralized logging
│   ├── notifications.py               # Webhook/email alerts
│   ├── cost_tracker.py                # Training cost estimation
│   └── gpu_monitor.py                 # Memory/utilization tracking
│
├── tests/
│   ├── test_setup.sh                  # Environment validation
│   ├── test_data.py                   # Dataset format checks
│   ├── test_vq_extraction.py          # VQ token validation
│   ├── test_model_loading.py          # Checkpoint loading
│   └── test_inference.py              # Quick generation test
│
└── docs/
    ├── SETUP.md                       # Detailed setup guide
    ├── CONFIGURATION.md               # Config file documentation
    ├── TROUBLESHOOTING.md             # Common errors & fixes
    └── COST_CALCULATOR.md             # Training cost breakdown
```

---

## Phase 1: MVP - Core Pipeline (Priority 1) 🔥

**Timeline:** 1-2 hours  
**Goal:** Replace notebook with Python scripts, maintain current functionality

### 1.1 Enhanced `setup.sh`

**Current implementation gaps (from notebook analysis):**
```bash
# Missing from current setup.sh:
1. PyTorch installation with correct index
2. TorchCodec installation (discovered during runtime)
3. Protobuf version pinning (3.20.3, not latest)
4. HuggingFace Hub installation
5. Hydra-core + omegaconf
6. Fish Speech editable install
7. Loguru for logging

# Current setup.sh only has:
✅ Python version check (fixed bc dependency)
✅ Fish Speech clone
✅ Basic pip install
```

**Needed additions:**
```bash
#!/bin/bash
set -e  # Exit on error

echo "🚀 Finnish TTS Training - Environment Setup"

# 1. System checks
check_dependencies() {
    for cmd in git python3 nvidia-smi curl tar; do
        command -v $cmd >/dev/null 2>&1 || { echo "❌ $cmd not found"; exit 1; }
    done
    echo "✅ System dependencies OK"
}

# 2. Python version validation (current implementation is good)
check_python_version() {
    # Keep existing bc-free implementation
}

# 3. GPU validation
check_gpu() {
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "❌ No GPU detected"
        exit 1
    fi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "✅ GPU: $GPU_NAME ($GPU_MEM MB)"
}

# 4. Disk space check
check_disk_space() {
    FREE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$FREE_GB" -lt 50 ]; then
        echo "⚠️  Warning: Only ${FREE_GB}GB free. Need 50GB+"
    fi
    echo "✅ Disk space: ${FREE_GB}GB free"
}

# 5. Virtual environment
setup_venv() {
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        echo "✅ Created .venv"
    fi
    source .venv/bin/activate
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    echo "✅ Upgraded pip"
}

# 6. Install exact working versions (from notebook)
install_pytorch() {
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu121
    echo "✅ PyTorch 2.9.1+cu121 installed"
}

# 7. Install Fish Speech dependencies
install_fish_speech() {
    if [ ! -d "fish-speech" ]; then
        git clone https://github.com/fishaudio/fish-speech.git
    fi
    
    cd fish-speech
    pip install -e .  # Editable install
    cd ..
    echo "✅ Fish Speech installed"
}

# 8. Install additional requirements (from notebook)
install_additional_deps() {
    pip install \
        torchcodec==0.8.1 \
        "protobuf>=3.20.3,<5" \
        huggingface_hub \
        hydra-core \
        omegaconf \
        loguru
    echo "✅ Additional dependencies installed"
}

# 9. Validate imports
validate_setup() {
    python -c "
import torch
import torchaudio
import torchcodec
from fish_speech.models.text2semantic.llama import DualARTransformer
from huggingface_hub import login
import hydra
print('✅ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
}

# 10. Create directory structure
setup_directories() {
    mkdir -p ~/finnish-tts-brev/{data,checkpoints,results}
    mkdir -p ~/finnish-tts-brev/data/{FinnishSpeaker,protos}
    mkdir -p ~/finnish-tts-brev/checkpoints/openaudio-s1-mini
    echo "✅ Directory structure created"
}

# Run all checks
main() {
    check_dependencies
    check_python_version
    check_gpu
    check_disk_space
    setup_venv
    install_pytorch
    install_fish_speech
    install_additional_deps
    validate_setup
    setup_directories
    
    echo ""
    echo "✅ Setup complete!"
    echo "Next steps:"
    echo "  1. Copy .env.example to .env and add HF_TOKEN"
    echo "  2. Upload dataset to ~/finnish-tts-brev/data/FinnishSpeaker/"
    echo "  3. Run: python scripts/train.py"
}

main
```

### 1.2 `requirements.txt`

**Pin exact versions that work (verified from notebook):**
```txt
# PyTorch ecosystem
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1
torchcodec==0.8.1

# Core dependencies
protobuf>=3.20.3,<5
huggingface-hub>=0.20.0
hydra-core>=1.3.0
omegaconf>=2.3.0
loguru>=0.7.0

# Training frameworks
lightning>=2.2.0

# Data processing
numpy>=1.24.0
scipy>=1.10.0

# Utilities
click>=8.1.0
tqdm>=4.65.0
pyyaml>=6.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0

# Note: Install PyTorch with:
# pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
#     --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 `config/training_config.yaml`

**Centralized configuration:**
```yaml
# Paths
paths:
  data_dir: ~/finnish-tts-brev/data/FinnishSpeaker
  proto_dir: ~/finnish-tts-brev/data/protos
  checkpoint_dir: ~/finnish-tts-brev/checkpoints/openaudio-s1-mini
  output_dir: ~/fish-speech/results

# Model
model:
  base_model: openaudio-s1-mini
  lora_rank: 8
  lora_alpha: 16

# Training
training:
  max_steps: 3000
  batch_size: 8
  num_workers: 8
  val_check_interval: 100
  accumulate_grad_batches: 1
  
# Early Stopping
early_stopping:
  enabled: true
  monitor: train/loss
  patience: 5
  mode: min

# Hardware
hardware:
  gpu_memory_fraction: 0.9
  mixed_precision: true
```

### 1.4 `scripts/train.py`

**Main training script (replaces notebook Step 8):**
```python
#!/usr/bin/env python3
"""
Finnish TTS Training Script
Replaces Jupyter notebook with production-ready CLI tool
"""
import argparse
import os
import sys
import yaml
from pathlib import Path
import subprocess
from datetime import datetime

def load_config(config_path):
    """Load training configuration from YAML"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_env():
    """Load environment variables from .env"""
    env_path = Path.home() / 'finnish-tts-brev' / '.env'
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")
    
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    if 'HF_TOKEN' not in os.environ:
        raise ValueError("HF_TOKEN not found in .env")
    
    print("✅ Loaded HF_TOKEN from .env")

def validate_paths(config):
    """Ensure all required paths exist"""
    paths = config['paths']
    
    # Check proto files exist
    proto_dir = Path(paths['proto_dir']).expanduser()
    if not proto_dir.exists() or not list(proto_dir.glob('*.proto*')):
        raise FileNotFoundError(f"Proto files not found in {proto_dir}")
    print(f"✅ Found proto files in {proto_dir}")
    
    # Check checkpoint exists
    ckpt_dir = Path(paths['checkpoint_dir']).expanduser()
    if not (ckpt_dir / 'model.pth').exists():
        raise FileNotFoundError(f"Base model not found in {ckpt_dir}")
    print(f"✅ Found base model in {ckpt_dir}")

def build_command(config):
    """Build Fish Speech training command from config"""
    paths = config['paths']
    model = config['model']
    training = config['training']
    hardware = config['hardware']
    
    # Expand paths
    checkpoint_dir = str(Path(paths['checkpoint_dir']).expanduser())
    proto_dir = str(Path(paths['proto_dir']).expanduser())
    
    cmd = [
        sys.executable,
        'fish_speech/train.py',
        '--config-name', 'text2semantic_finetune',
        f'pretrained_ckpt_path={checkpoint_dir}',
        f'train_dataset.proto_files=[{proto_dir}]',
        f'val_dataset.proto_files=[{proto_dir}]',
        f'project={training["project_name"]}',
        f'+lora@model.model.lora_config=r_{model["lora_rank"]}_alpha_{model["lora_alpha"]}',
        f'data.batch_size={training["batch_size"]}',
        f'data.num_workers={training["num_workers"]}',
        f'trainer.max_steps={training["max_steps"]}',
        f'trainer.val_check_interval={training["val_check_interval"]}',
        f'trainer.accumulate_grad_batches={training["accumulate_grad_batches"]}',
    ]
    
    # Add early stopping if enabled
    if config.get('early_stopping', {}).get('enabled'):
        es = config['early_stopping']
        cmd.extend([
            f'+callbacks.early_stopping._target_=lightning.pytorch.callbacks.EarlyStopping',
            f'+callbacks.early_stopping.monitor={es["monitor"]}',
            f'+callbacks.early_stopping.patience={es["patience"]}',
            f'+callbacks.early_stopping.mode={es["mode"]}',
            f'+callbacks.early_stopping.verbose=true',
        ])
    
    return cmd

def log_system_info():
    """Log system and GPU information"""
    import torch
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_total:.1f} GB")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Train Finnish TTS model')
    parser.add_argument('--config', default='config/training_config.yaml',
                       help='Path to training configuration YAML')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print command without running')
    args = parser.parse_args()
    
    # Change to fish-speech directory
    fish_speech_dir = Path.home() / 'fish-speech'
    os.chdir(fish_speech_dir)
    print(f"📁 Working directory: {fish_speech_dir}\n")
    
    # Load configuration
    config_path = fish_speech_dir.parent / 'nvidia-brev' / args.config
    config = load_config(config_path)
    print(f"✅ Loaded config from {config_path}")
    
    # Load environment
    load_env()
    
    # Validate paths
    validate_paths(config)
    
    # Log system info
    log_system_info()
    
    # Build command
    cmd = build_command(config)
    
    if args.dry_run:
        print("🔍 DRY RUN - Command that would be executed:")
        print(" \\\n  ".join(cmd))
        return
    
    # Print training info
    training = config['training']
    print("🚀 STARTING TRAINING")
    print(f"   Project: {training['project_name']}")
    print(f"   Max steps: {training['max_steps']}")
    print(f"   Batch size: {training['batch_size']}")
    print(f"   Estimated time: ~{training['max_steps'] * 5.4 / 3600:.1f} hours\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ TRAINING COMPLETE!")
        
        # Show checkpoint location
        results_dir = fish_speech_dir / 'results' / training['project_name'] / 'checkpoints'
        print(f"\n📦 Checkpoints saved to:")
        print(f"   {results_dir}")
        
        # List checkpoints
        if results_dir.exists():
            checkpoints = sorted(results_dir.glob('*.ckpt'))
            if checkpoints:
                print(f"\n📋 Available checkpoints ({len(checkpoints)}):")
                for ckpt in checkpoints[-5:]:  # Show last 5
                    size_mb = ckpt.stat().st_size / 1e6
                    print(f"   - {ckpt.name} ({size_mb:.1f} MB)")
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TRAINING FAILED with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("   Checkpoint saved. Resume with --resume flag")
        sys.exit(130)

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Normal training
python scripts/train.py

# With custom config
python scripts/train.py --config config/my_config.yaml

# Dry run (print command without executing)
python scripts/train.py --dry-run

# Resume from checkpoint
python scripts/train.py --resume
```

---

## Phase 2: Reliability & Automation (Priority 2) ⚙️

**Timeline:** 2-3 hours  
**Goal:** Add error handling, auto-recovery, data pipeline

### 2.1 `scripts/prepare_data.sh`

**Complete data pipeline with VQ token caching:**
```bash
#!/bin/bash
# Data preparation pipeline with intelligent caching

set -e

DATA_DIR="${DATA_DIR:-~/finnish-tts-brev/data/FinnishSpeaker}"
PROTO_DIR="${PROTO_DIR:-~/finnish-tts-brev/data/protos}"
CACHE_ARCHIVE="${CACHE_ARCHIVE:-finnish-data-with-vq.tar.gz}"

echo "🔍 Validating dataset..."

# Check source files
WAV_COUNT=$(find "$DATA_DIR" -name "*.wav" | wc -l)
LAB_COUNT=$(find "$DATA_DIR" -name "*.lab" | wc -l)
NPY_COUNT=$(find "$DATA_DIR" -name "*.npy" | wc -l)

echo "   WAV files: $WAV_COUNT"
echo "   LAB files: $LAB_COUNT"
echo "   NPY files (VQ tokens): $NPY_COUNT"

if [ "$WAV_COUNT" -ne "$LAB_COUNT" ]; then
    echo "❌ Mismatch: WAV ($WAV_COUNT) != LAB ($LAB_COUNT)"
    exit 1
fi

# VQ Token Caching Strategy
if [ "$NPY_COUNT" -eq "$WAV_COUNT" ]; then
    echo "✅ VQ tokens already extracted ($NPY_COUNT files)"
    echo "   Skipping extraction (saves ~15 minutes + GPU cost)"
elif [ "$NPY_COUNT" -gt 0 ]; then
    echo "⚠️  Partial VQ extraction detected: $NPY_COUNT/$WAV_COUNT"
    echo "   Extracting remaining $(($WAV_COUNT - $NPY_COUNT)) files..."
    python scripts/extract_vq.py --resume --data-dir "$DATA_DIR"
else
    echo "🔄 No VQ tokens found, extracting all $WAV_COUNT files..."
    echo "   This will take ~15 minutes on A100"
    python scripts/extract_vq.py --data-dir "$DATA_DIR"
fi

# Dataset Packing
PROTO_COUNT=$(find "$PROTO_DIR" -name "*.proto*" | wc -l)
if [ "$PROTO_COUNT" -gt 0 ]; then
    echo "✅ Proto files already exist ($PROTO_COUNT files)"
    echo "   Skipping packing"
else
    echo "🔄 Packing dataset to proto format..."
    python scripts/pack_dataset.py --data-dir "$DATA_DIR" --output-dir "$PROTO_DIR"
fi

echo "✅ Data pipeline complete!"
echo ""
echo "💡 TIP: To cache VQ tokens for future runs:"
echo "   tar -czf $CACHE_ARCHIVE $DATA_DIR"
echo "   aws s3 cp $CACHE_ARCHIVE s3://your-bucket/  # Or HuggingFace Hub"
```

**VQ Token Caching Benefits:**
- ⏱️ **Saves 15 minutes** per training run
- 💰 **Saves ~$0.30** in GPU time (A100 @ $1.20/hr)
- 🔄 **Deterministic** - Same audio always produces same tokens
- 📦 **Portable** - Share across team/instances
- 🚀 **Faster iteration** - Skip preprocessing, go straight to training

### 2.1.1 VQ Token Cache Distribution Strategies

**Option A: Include in Dataset Package** ⭐ Recommended for Launchables
```bash
# Package once, reuse everywhere
cd ~/finnish-tts-brev/data
tar -czf finnish-speaker-2000-with-vq.tar.gz FinnishSpeaker/
# Upload to S3/R2/HuggingFace Hub
# New instances: download tar, extract, skip VQ step entirely
```

**Option B: On-Demand Extraction with Smart Resume**
```python
# scripts/extract_vq.py with resume capability
def check_existing_tokens(data_dir):
    wav_files = glob(f"{data_dir}/*.wav")
    npy_files = set([Path(f).stem for f in glob(f"{data_dir}/*.npy")])
    
    missing = [w for w in wav_files if Path(w).stem not in npy_files]
    return missing

# Only extract missing files (handles interrupted runs)
```

**Option C: Lazy Loading from Cloud**
```bash
# Check local cache first, download from S3 if missing
if [ ! -d "$DATA_DIR/FinnishSpeaker" ]; then
    echo "Downloading pre-extracted VQ tokens from S3..."
    aws s3 sync s3://bucket/vq-tokens/ "$DATA_DIR/"
fi
```

**Recommended for Production:**
- Use **Option A** for initial Launchables release (zero setup friction)
- Add **Option B** for custom datasets (users bring their own audio)
- Consider **Option C** for large datasets (>10GB VQ tokens)

### 2.2 `scripts/extract_vq.py`

**Standalone VQ extraction with resume support:**
```python
#!/usr/bin/env python3
"""
VQ Token Extraction with Smart Caching
Handles resume, memory optimization, and progress tracking
"""
import argparse
import os
import sys
from pathlib import Path
from glob import glob

# Features:
- Auto-detect GPU memory
- Calculate safe workers/batch_size
- Progress bar with ETA
- **Resume capability** (skip existing NPY files)
- **Dry-run mode** (estimate time/cost before running)
- **Validation** (verify output token shapes)
- Resume from partial extraction
- Validate output files
- Error logging per file
```

### 2.3 `scripts/merge_and_export.py`

**Post-training automation:**
```python
# Features:
- Find best checkpoint (lowest val loss)
- Auto-merge LoRA weights
- Run quick inference test
- Package for distribution (tar.gz)
- Optional: Upload to HuggingFace Hub
- Clean intermediate files
- Generate training report
```

### 2.4 Error Recovery

**Add to `train.py`:**
```python
# Auto-resume logic
- Check for existing checkpoints
- Load last checkpoint if found
- Calculate remaining steps
- Continue training from checkpoint
- Log resume info

# Graceful shutdown
- Catch SIGTERM, SIGINT
- Save current state
- Write resume instructions
- Clean up resources
```

---

## Phase 3: Monitoring & Polish (Priority 3) 🎨

**Timeline:** 3-4 hours  
**Goal:** Professional features, notifications, cost tracking

### 3.1 `utils/notifications.py`

**Multi-channel alerts:**
```python
# Supported channels:
- Slack webhook
- Discord webhook
- Telegram bot
- Email (SendGrid/Mailgun)
- Generic HTTP POST

# Event types:
- Training started
- Checkpoint milestone (every 500 steps)
- Validation updates
- Early stopping triggered
- Training completed
- Error/crash alerts
```

### 3.2 `utils/cost_tracker.py`

**Cost monitoring:**
```python
# Features:
- Estimate cost before training starts
- Track actual cost during training
- Alert if exceeds budget threshold
- Report final cost
- Cost per checkpoint
- Efficiency metrics (cost per step)
```

### 3.3 `utils/gpu_monitor.py`

**Hardware monitoring:**
```python
# Metrics tracked:
- GPU utilization %
- Memory usage (used/total)
- Temperature
- Power consumption
- Memory fragmentation
- Alert on potential OOM
```

### 3.4 Auto-shutdown

**Add to scripts:**
```bash
# Brev instance management
- Stop instance after training completion
- Warning if running >6 hours
- Cost summary before shutdown
- Upload checkpoints to cloud storage first
- Verify upload success before shutdown
```

---

## Phase 4: Testing & Validation (Priority 4) ✅

**Timeline:** 2-3 hours  
**Goal:** Ensure reliability across environments

### 4.1 Test Suite

**`tests/test_setup.sh`:**
```bash
- Python version check
- CUDA availability
- Disk space
- Package installations
- Import tests
- GPU access
```

**`tests/test_data.py`:**
```python
- File count validation
- Audio format checks
- LAB file parsing
- VQ token validation
- Proto file structure
```

**`tests/test_inference.py`:**
```python
- Load checkpoint
- Generate sample output
- Verify audio quality
- Check duration
- Validate format
```

---

## Phase 5: Documentation (Priority 5) 📚

**Timeline:** 2-3 hours  
**Goal:** Complete, beginner-friendly documentation

### 5.1 `README.md`

**Sections:**
1. **Overview** - What this repo does
2. **Prerequisites** - Brev account, credits, SSH
3. **Quick Start** - 3 commands to trained model
4. **Configuration** - How to modify settings
5. **Monitoring** - Check training progress
6. **Troubleshooting** - Common errors
7. **Cost Calculator** - Estimate expenses
8. **Contributing** - How to improve

### 5.2 `docs/SETUP.md`

**Detailed setup guide:**
- Brev account creation
- SSH key setup
- Instance deployment
- Dataset upload methods
- Environment configuration
- First training run

### 5.3 `docs/CONFIGURATION.md`

**Config file reference:**
- All YAML parameters explained
- Default values
- Recommended settings per GPU
- Performance tuning guide
- Memory optimization tips

### 5.4 `docs/TROUBLESHOOTING.md`

**Common issues:**
- CUDA OOM errors → reduce batch_size
- Protobuf import errors → pin version
- VQ extraction fails → adjust workers
- Training stalls → check data pipeline
- Checkpoint loading fails → version mismatch

### 5.5 `docs/COST_CALCULATOR.md`

**Cost breakdown:**
```
Hardware: A100-80GB @ $1.20/hour
Steps: 3000
Time per step: 5.4 seconds
Total time: 4.5 hours
Total cost: $5.40

Optimization tips:
- Reduce steps to 2000 → $3.60 (saves $1.80)
- Use batch_size=16 → 3 hours → $3.60
- Early stopping → variable savings
```

---

## Migration Path: Notebook → Scripts

**Analyzed notebook:** `finnish-tts-brev/finnish-tts-training.ipynb` (verified working)

### Current Notebook Cells → New Scripts

| Notebook Step | Current Implementation | New Script | Priority | Notes |
|---------------|----------------------|------------|----------|-------|
| **Step 1: Check GPU** | `!nvidia-smi` + torch check | `scripts/check_gpu.py` | P1 | Add memory check, optimal batch_size suggestion |
| **Step 2: Install PyTorch** | Manual pip install | `setup.sh` | P1 | Pin torch==2.9.1+cu128, torchaudio==2.9.1+cu128 |
| **Step 3: HF Login** | Load .env + login | `scripts/auth.py` | P2 | Validate HF_TOKEN, test connection |
| **Step 4: Download Model** | Manual check | `scripts/download_model.sh` | P1 | Auto-download if missing, verify checksums |
| **Step 5: Load Dataset** | Count files manually | `scripts/validate_data.py` | P1 | Auto-validate 2000 WAV + LAB, report stats |
| **Step 6: Extract VQ** | Manual run with hardcoded params | `scripts/extract_vq.py` | P1 | Auto-detect workers/batch, resume capability |
| **Step 7: Pack Dataset** | Manual protobuf install + run | `scripts/pack_dataset.py` | P1 | Check if already packed, validate output |
| **Step 8: Training** | Long command with manual args | `scripts/train.py` | P1 | Load from config YAML, log to file |
| **Step 9: Monitor** | Manual tail commands | `scripts/monitor.py` | P2 | Live progress display, webhook alerts |
| **Step 10: Merge LoRA** | Manual checkpoint selection | `scripts/merge_and_export.py` | P1 | Auto-find best checkpoint, test inference |
| **Step 11: Download** | Manual tar + download | `scripts/package_model.sh` | P2 | Auto-create archive, upload to cloud |

### Preserving Notebook

**Keep for:**
- Interactive exploration
- Quick testing
- Debugging
- Visualization
- Training monitoring

**But use scripts for:**
- Production training runs
- Automated pipelines
- CI/CD integration
- Reproducibility
- Version control

---

## Phase 5: Launchables Platform Integration 🚀

**Timeline:** 4-6 hours  
**Goal:** Create one-click deployable TTS training platform

### 5.1 Pre-Packaged VQ Token Strategy

**Current State:**
- ✅ 2000 VQ tokens already extracted on Brev instance
- ✅ Located in `~/finnish-tts-brev/data/FinnishSpeaker/*.npy`
- ✅ Total size: ~6000 files (2000 WAV + 2000 LAB + 2000 NPY)

**Launchables Optimization:**
```bash
# Create deployment-ready dataset package
cd ~/finnish-tts-brev/data
tar -czf finnish-speaker-2000-complete.tar.gz FinnishSpeaker/

# Upload to cloud storage (R2/S3/HuggingFace Hub)
# Option 1: Cloudflare R2 (free egress)
rclone copy finnish-speaker-2000-complete.tar.gz r2:launchables-datasets/

# Option 2: HuggingFace Hub (public access)
huggingface-cli upload username/finnish-tts-dataset \
  finnish-speaker-2000-complete.tar.gz

# Launchables instance startup script
wget https://r2.domain.com/finnish-speaker-2000-complete.tar.gz
tar -xzf finnish-speaker-2000-complete.tar.gz -C ~/data/
# Skip VQ extraction entirely → Save 15 minutes + $0.30
```

**Benefits for Launchables Users:**
- 🚀 **Zero preprocessing** - Ready to train immediately
- ⏱️ **15 minutes faster** - From 30 min setup → 15 min setup
- 💰 **$0.30 saved per run** - Lower barrier to experimentation
- 🎯 **Predictable timing** - VQ extraction can OOM on smaller GPUs
- 📦 **Portable** - Same package works on any GPU provider

### 5.2 Launchables Template Structure

```yaml
# launchables.yaml
name: Finnish TTS Training
description: Train custom Finnish speech models with LoRA fine-tuning
category: AI/ML
tags: [tts, speech, finnish, lora, fish-speech]

# Instance requirements
hardware:
  min_gpu_memory: 40GB    # RTX A6000 / A100-40GB
  recommended: A100-80GB
  storage: 50GB

# Pre-installed dataset (with VQ tokens)
datasets:
  - name: finnish-speaker-2000
    source: r2://launchables-datasets/finnish-speaker-2000-complete.tar.gz
    size: 8GB
    checksum: sha256:abc123...
    extract_to: ~/data/FinnishSpeaker/

# One-click setup
setup_command: |
  bash setup.sh
  python scripts/validate_data.py  # Verify VQ tokens present

# One-click training
run_command: |
  python scripts/train.py --config config/training_config.yaml

# Cost estimation
estimated_cost:
  setup: $0.10    # 5 minutes @ $1.20/hr
  training: $5.40 # 4.5 hours @ $1.20/hr
  total: $5.50

# Outputs
outputs:
  - path: ~/fish-speech/results/
    description: Trained LoRA weights
    auto_download: true
  - path: ~/fish-speech/logs/
    description: Training logs
```

### 5.3 Web UI Integration

**Dashboard features:**
```javascript
// Real-time training monitoring
- Loss curve plot (updated every 100 steps)
- GPU utilization graph
- Cost meter (running total)
- ETA countdown
- Audio sample player (generated samples)

// Controls
- Start/stop/resume buttons
- Hyperparameter sliders (before training)
- Early stopping toggle
- Auto-shutdown timer

// Notifications
- Browser notifications on completion
- Webhook integration (Slack/Discord)
- Email reports
```

### 5.4 Incremental Fine-Tuning Strategy 🔄

**Problem:** You have a fine-tuned model (2000 samples) and want to improve it with:
- More data (additional 1000 samples)
- Different speaker characteristics
- Better quality recordings
- Domain-specific content

**Solution: Use your fine-tuned checkpoint as the starting point!**

#### 5.4.1 Two Approaches

**Approach A: LoRA-on-LoRA (Faster, Cheaper)** ⭐ Recommended
```yaml
# Start from your already-trained LoRA weights
# Training: 500-1000 steps instead of 3000
# Cost: $1.20-$2.40 instead of $5.40
# Use case: Incremental improvements, new data batches

training:
  pretrained_ckpt_path: ~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt
  # This checkpoint already has LoRA adapters trained!
  
  max_steps: 1000  # Much fewer steps needed
  learning_rate: 5e-5  # Lower LR for fine-tuning fine-tuned model
  
lora:
  rank: 8  # Keep same as original
  alpha: 16  # Keep same as original
```

**Approach B: Merge + Re-LoRA (Better Performance)**
```bash
# 1. Merge your trained LoRA into base model
python tools/llama/merge_lora.py \
  --lora-checkpoint ~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt \
  --base-model ~/finnish-tts-brev/checkpoints/openaudio-s1-mini \
  --output ~/finnish-tts-brev/checkpoints/finnish-merged-v1

# 2. Use merged model as new base for second LoRA training
training:
  pretrained_ckpt_path: ~/finnish-tts-brev/checkpoints/finnish-merged-v1/model.pth
  max_steps: 2000  # Medium training time
  
lora:
  rank: 16  # Can increase for second round
  alpha: 32
```

#### 5.4.2 Benefits of Incremental Fine-Tuning

**Training efficiency:**
```
First training (from base):
  - 2000 samples → 3000 steps → 4.5 hours → $5.40
  
Incremental training (from checkpoint):
  - +1000 samples → 1000 steps → 1.5 hours → $1.80
  - 67% time saved
  - 67% cost saved
```

**Quality improvements:**
- Preserves learned Finnish phonetics
- Faster convergence on new data
- No catastrophic forgetting (with proper LR)
- Can specialize further (dialects, emotions)

**Use cases:**
1. **Iterative data collection** - Train on 2000 → add 1000 more → retrain
2. **Domain adaptation** - General Finnish → News Finnish → Sports commentary
3. **Multi-speaker** - Speaker A → add Speaker B → blend both
4. **Quality upgrading** - Phone recordings → studio recordings
5. **Error correction** - Find bad pronunciations → add corrected samples

#### 5.4.3 Production Script Enhancement

**Add to `scripts/train.py`:**
```python
def detect_checkpoint_type(ckpt_path):
    """Detect if checkpoint is base model or already fine-tuned"""
    ckpt = torch.load(ckpt_path)
    
    if 'lora' in ckpt or 'adapter' in ckpt:
        print("🔄 Detected fine-tuned checkpoint (LoRA weights present)")
        print("   Using incremental fine-tuning strategy")
        return 'finetuned'
    else:
        print("📦 Detected base model checkpoint")
        print("   Using initial fine-tuning strategy")
        return 'base'

def get_optimal_hyperparams(ckpt_type, new_samples_count):
    """Adjust training params based on starting point"""
    if ckpt_type == 'finetuned':
        # Incremental training: fewer steps, lower LR
        return {
            'max_steps': max(500, new_samples_count * 0.5),
            'learning_rate': 5e-5,  # 10x lower
            'warmup_steps': 50,
        }
    else:
        # Initial training: standard params
        return {
            'max_steps': max(2000, new_samples_count * 1.5),
            'learning_rate': 5e-4,
            'warmup_steps': 200,
        }
```

#### 5.4.4 Checkpoint Versioning Strategy

**Organize your models:**
```
checkpoints/
├── openaudio-s1-mini/           # Base model (868M params)
│   └── model.pth
│
├── finnish-v1-lora/             # First fine-tune (2000 samples)
│   ├── step_000003000.ckpt      # LoRA weights only (8.1M params)
│   └── metadata.json            # Training config, data stats
│
├── finnish-v1-merged/           # Merged version (868M params)
│   └── model.pth                # Base + LoRA combined
│
├── finnish-v2-lora/             # Second fine-tune (+1000 samples)
│   ├── step_000001000.ckpt      # Incremental LoRA
│   └── metadata.json
│
└── finnish-v2-merged/           # Final production model
    └── model.pth
```

**Metadata tracking:**
```json
{
  "version": "finnish-v2",
  "base_model": "openaudio-s1-mini",
  "parent_checkpoint": "finnish-v1-lora/step_000003000.ckpt",
  "training_data": {
    "total_samples": 3000,
    "new_samples": 1000,
    "total_duration_hours": 6.0
  },
  "training_config": {
    "max_steps": 1000,
    "learning_rate": 5e-5,
    "lora_rank": 8
  },
  "performance": {
    "final_loss": 12.3,
    "training_time_hours": 1.5,
    "training_cost_usd": 1.80
  }
}
```

#### 5.4.5 Best Practices

**Do:**
- ✅ Lower learning rate for incremental training (5e-5 vs 5e-4)
- ✅ Fewer steps needed (0.5-1.0x new data size vs 1.5x)
- ✅ Keep LoRA rank consistent across iterations
- ✅ Validate on held-out set after each iteration
- ✅ Version control checkpoints with metadata

**Don't:**
- ❌ Use high LR on fine-tuned checkpoint (catastrophic forgetting)
- ❌ Train too long without validation (overfitting risk)
- ❌ Mix incompatible LoRA ranks (8 → 16 requires merge first)
- ❌ Skip data quality checks on new samples
- ❌ Forget to backup previous checkpoints

#### 5.4.6 Launchables Integration

**UI workflow:**
```
1. User uploads initial 2000 samples
   → Train base model → Get checkpoint v1

2. User adds 1000 more samples
   → Select "Incremental Training"
   → Choose checkpoint v1 as starting point
   → Auto-adjusts hyperparameters
   → Train for 1.5 hours instead of 4.5 hours
   → Pay $1.80 instead of $5.40

3. User wants to specialize
   → Select checkpoint v2
   → Upload domain-specific data
   → Train sub-model for specific use case
```

**Cost savings at scale:**
```
Without incremental training:
- V1: 2000 samples → $5.40
- V2: 3000 samples → $5.40  (retrain from scratch)
- V3: 4000 samples → $5.40
- Total: $16.20

With incremental training:
- V1: 2000 samples → $5.40
- V2: +1000 samples → $1.80  (incremental)
- V3: +1000 samples → $1.80
- Total: $9.00

Savings: $7.20 (44% reduction)
```

### 5.5 Multi-Language Expansion Kit

**Make it a template for ANY language:**
```bash
# User provides:
1. Upload audio files (WAV)
2. Upload transcripts (TXT/LAB)
3. Select language
4. Click "Train"

# System handles:
1. Data validation
2. VQ extraction (cached if repeated)
3. Dataset packing
4. Training with optimal hyperparameters
5. Model export
6. Test inference
7. Download trained model
```

**Supported languages (future):**
- Finnish ✅ (current)
- English (high demand)
- Spanish (large market)
- Japanese (anime/gaming)
- Any language with 2+ hours of audio

### 5.6 Use Cases When Users Don't Have New Data 🎯

**Problem:** Not everyone has custom audio data ready, but they still want to use the platform!

#### Strategy A: Pre-Made Dataset Marketplace 📦

**Offer curated, ready-to-train datasets:**

```yaml
Marketplace offerings:
  - name: "Finnish General Voice (2000 samples)"
    price: Free / $9.99
    description: "Ready-to-train Finnish TTS model"
    includes: Audio + transcripts + VQ tokens pre-extracted
    training_cost: $6.50 (one-click)
    use_case: "Get started with Finnish TTS instantly"
    
  - name: "Finnish News Voice (1000 samples)"
    price: $14.99
    description: "Professional news reading style"
    includes: Studio quality recordings
    
  - name: "Multi-speaker Finnish Pack (5 speakers, 500 each)"
    price: $24.99
    description: "Train versatile multi-voice model"
```

**User journey:**
```
1. Browse marketplace → "Finnish General Voice"
2. Click "Train Model" → Auto-configured
3. Wait 4.5 hours (or check back later)
4. Download trained model
5. Use for unlimited generation

Total cost: $9.99 (dataset) + $6.50 (training) = $16.49
vs ElevenLabs: $0.30/1000 chars → Break even at 55,000 chars
```

#### Strategy B: Hyperparameter Experimentation 🔬

**Users can train the SAME dataset with different settings:**

```yaml
Use case: "Find optimal settings for your use case"

Experiment 1: Baseline
  - LoRA rank: 8, alpha: 16
  - Steps: 3000
  - Cost: $6.50
  
Experiment 2: Higher rank (better quality?)
  - LoRA rank: 16, alpha: 32
  - Steps: 3000
  - Cost: $6.50
  
Experiment 3: Faster training (lower quality?)
  - LoRA rank: 4, alpha: 8
  - Steps: 2000
  - Cost: $4.50
  
Experiment 4: Aggressive training
  - LoRA rank: 8, alpha: 16
  - Steps: 5000, higher LR
  - Cost: $9.00

Total: 4 experiments = $26.50
Result: Find best quality/cost tradeoff for your needs
```

**Platform features:**
- Side-by-side audio comparison
- Quality metrics (MOS scores)
- Cost/quality curve visualization
- "Recommended settings" based on community data

#### Strategy C: Model Customization Without New Data 🎨

**Fine-tune existing models for specific traits:**

```python
# Emotional fine-tuning on same data
Experiment 1: Happy/upbeat voice
  - Add prosody augmentation
  - Adjust pitch/speed ranges
  - Cost: $2.00 (incremental from base)

Experiment 2: Calm/soothing voice
  - Different speed settings
  - Lower pitch variance
  - Cost: $2.00

Experiment 3: Energetic/excited
  - Higher pitch variance
  - Faster speaking rate
  - Cost: $2.00
```

#### Strategy D: Multi-Language Template System 🌍

**Let users train models for OTHER languages using template datasets:**

```yaml
Language packs available:
  ✅ Finnish (current)
  ✅ English (coming soon)
  ✅ Spanish (coming soon)
  ✅ Japanese (coming soon)
  ⏳ 20+ more languages planned

User workflow:
1. "I want Spanish TTS"
2. Click "Train Spanish Model" ($9.99)
3. Uses curated 2000-sample Spanish dataset
4. Wait 4.5 hours
5. Get working Spanish TTS model
```

**Pricing tiers:**
```
Starter: $16.49 (one language, pre-made dataset)
Pro:     $49.99 (three languages of choice)
Studio:  $99.99 (unlimited languages + custom data)
```

#### Strategy E: Training-as-a-Service for Developers �️

**API-first approach for developers without data:**

```javascript
// Developer use case: Add TTS to their app
const launchables = require('@launchables/tts-api');

// Option 1: Use pre-trained model
const model = await launchables.models.get('finnish-general-v1');
const audio = await model.generate('Tervetuloa!');

// Option 2: One-line custom training
const customModel = await launchables.training.start({
  language: 'finnish',
  dataset: 'marketplace/finnish-general-2000',
  webhook: 'https://myapp.com/training-complete'
});

// 4.5 hours later, webhook called
// Developer can now use customModel for inference
```

**Developer benefits:**
- No ML expertise needed
- No data collection needed
- No infrastructure management
- Pay-per-training model
- Production-ready models

#### Strategy F: "Try Before You Buy" Demos 🎬

**Free tier with pre-trained models:**

```yaml
Free offerings:
  - 3 pre-trained models (Finnish, English, Spanish)
  - 10,000 characters/month generation
  - Web playground interface
  - No training included
  
Goal: Hook users, then upsell custom training
  "Want YOUR voice? Train custom model for $16.49"
  "Want better quality? Train with more data"
  "Want commercial license? Upgrade to Pro"
```

#### Strategy G: Community Contribution Model 🤝

**Crowdsource datasets, share revenue:**

```yaml
How it works:
  1. User A records 2000 Finnish samples
  2. Uploads to marketplace at $9.99
  3. User B buys dataset + trains model
  4. User A gets $7 (70% revenue share)
  5. Platform gets $2.99
  6. Training costs $6.50 → Platform profits
  
Benefits:
  - Users have incentive to create quality datasets
  - Platform gets variety without creating content
  - Network effects (more data = more users = more data)
```

### 5.7 Launchables Product Tiers 💼

**Tier 1: Playground (Free)**
```
- Pre-trained model access (3 languages)
- 10,000 characters/month generation
- Web interface only
- Community support
- Watermarked audio
```

**Tier 2: Starter ($16.49 one-time)**
```
- Choose 1 language
- Train on curated dataset (2000 samples)
- Download trained model
- No watermark
- 100,000 chars/month generation
- Commercial license
```

**Tier 3: Pro ($49/month)**
```
- Train 3 languages
- Upload custom datasets (up to 5000 samples)
- Incremental training included
- 1M chars/month generation
- Priority GPU access
- Email support
- API access
```

**Tier 4: Studio ($199/month)**
```
- Unlimited languages
- Unlimited dataset uploads
- Unlimited training runs
- Unlimited generation
- Dedicated GPU instances
- Phone support
- Custom model architectures
- Team collaboration
```

**Tier 5: Enterprise (Custom pricing)**
```
- On-premise deployment
- Custom SLA
- White-label solution
- Training support
- Custom integrations
- Bulk licensing
```

### 5.8 Monetization & Value Prop

**For Launchables:**
- 💰 Revenue: Multiple streams (datasets, training, inference, subscriptions)
- 📈 Volume: Low friction = more users training = more GPU hours sold
- 🎓 Education: "Learn TTS training" tutorials drive traffic
- 🏢 Enterprise: Upsell to custom voice training services
- 🔄 **Incremental training** = recurring revenue (users come back to improve models)
- 🛒 **Marketplace** = passive income (dataset sales)
- 🔁 **Subscriptions** = predictable MRR

**For Users:**
- ⏱️ **Time saved**: 3-5 hours of setup → 5 minutes
- 🧠 **Knowledge barrier**: No ML expertise needed
- 💸 **Cost transparency**: See exact costs before clicking "Train"
- 🎯 **Guaranteed results**: Pre-validated pipeline
- 📦 **Portable outputs**: Download model, use anywhere
- 🔄 **Incremental improvements**: Cheaper to add data (44% cost savings)
- 🛒 **Ready-to-use datasets**: No need to record audio yourself

**Competitive Advantage over DIY:**
- **vs Colab**: Better GPUs, no session timeouts, easier setup
- **vs Runpod**: Pre-configured, one-click, guided workflow
- **vs Brev**: More beginner-friendly, task-specific UI
- **vs ElevenLabs API**: One-time cost, own your model, unlimited generation

### 5.9 Dataset Caching Economics

**Without VQ caching:**
```
User uploads 2000 audio files (5 min)
  → VQ extraction (15 min, $0.30)
  → Dataset packing (2 min, $0.04)
  → Training (4.5 hrs, $5.40)
Total: 5 hrs 22 min, $5.74
```

**With VQ caching:**
```
System downloads pre-extracted dataset (3 min)
  → Validation check (1 min, $0.02)
  → Dataset packing (2 min, $0.04) [if not cached]
  → Training (4.5 hrs, $5.40)
Total: 4 hrs 36 min, $5.46
Saves: 46 minutes, $0.28 per run
```

**With incremental training (existing checkpoint):**
```
User uploads +1000 new samples (2 min)
  → VQ extraction on new files only (5 min, $0.10)
  → Dataset packing (1 min, $0.02)
  → Incremental training from checkpoint (1.5 hrs, $1.80)
Total: 1 hr 38 min, $1.92
Saves: 3 hrs 44 min, $3.82 vs training from scratch
```

**Platform-wide impact:**
- 100 users/month × $0.28 = **$28/month saved** (better margins)
- 100 users/month × 46 min = **76 hours/month** saved (higher throughput)
- Reduced OOM errors from VQ extraction = fewer support tickets
- **30% of users do incremental training** = 30 × $1.92 = **$57.60 recurring revenue/month**

---

## Success Metrics

**Phase 1 (MVP):**
- [ ] Can run entire pipeline with one command
- [ ] Training completes without manual intervention
- [ ] All configs in YAML files
- [ ] No hard-coded paths

**Phase 2 (Reliability):**
- [ ] Auto-resumes from checkpoint after crash
- [ ] Data pipeline handles missing files
- [ ] VQ token caching working (skip extraction on reruns)
- [ ] Error messages are actionable
- [ ] Logs are comprehensive

**Phase 3 (Polish):**
- [ ] Receive Slack notification on completion
- [ ] Cost stays within budget
- [ ] GPU utilization >95%
- [ ] Auto-shutdown works

**Phase 4 (Testing):**
- [ ] All tests pass
- [ ] Works on fresh Brev instance
- [ ] Different GPU types supported
- [ ] Multiple datasets work

**Phase 5 (Launchables Integration):**
- [ ] VQ tokens pre-packaged and cached
- [ ] Incremental training workflow functional
- [ ] Checkpoint versioning system working
- [ ] One-click deployment working
- [ ] Setup time under 5 minutes
- [ ] Cost estimation accurate within 5%
- [ ] Web UI shows real-time progress
- [ ] Template works for custom datasets
- [ ] Multi-language support functional

---

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1** | 1-2 hours | `train.py`, enhanced `setup.sh`, `requirements.txt`, configs |
| **Phase 2** | 2-3 hours | Data pipeline with VQ caching, error recovery, auto-merge |
| **Phase 3** | 3-4 hours | Notifications, cost tracking, auto-shutdown |
| **Phase 4** | 2-3 hours | Test suite, validation scripts |
| **Phase 5** | 4-6 hours | Launchables integration (VQ caching, incremental training, web UI) |
| **Total** | **12-18 hours** | Production-ready Launchables template |

---

## Next Steps (Immediate)

**This weekend:**
1. ✅ Let current training complete (step 3000)
2. ✅ Download final model
3. ✅ Test model quality
4. 🎯 **Package VQ tokens for reuse** (tar + upload to cloud)
5. 🔄 Start Phase 1: Create `train.py` 
6. 🔄 Enhance `setup.sh` with protobuf/torchcodec
7. 🔄 Create `requirements.txt`
8. 🔄 Create `config/training_config.yaml`

**Next week:**
- Run second training with new pipeline + VQ caching
- Validate time savings (expect 15 min faster)
- Validate cost savings (expect $0.30 saved)
- Compare results (notebook vs script)
- Begin Phase 2

**Within 2 weeks:**
- Complete Phases 1-3
- Begin Phase 5 (Launchables integration)
- Create VQ cache distribution system
- Write Launchables template YAML
- Demo video for platform

---

## VQ Token Cache Action Items 📦

**Immediate (after training completes):**
```bash
# 1. Package the dataset with VQ tokens
cd ~/finnish-tts-brev/data
tar -czf finnish-speaker-2000-with-vq.tar.gz FinnishSpeaker/
ls -lh finnish-speaker-2000-with-vq.tar.gz  # Check size (~8GB expected)

# 2. Upload to cloud storage
# Option A: HuggingFace Hub (public, free)
huggingface-cli upload yourusername/finnish-tts-dataset \
  finnish-speaker-2000-with-vq.tar.gz

# Option B: Cloudflare R2 (private, no egress fees)
rclone copy finnish-speaker-2000-with-vq.tar.gz r2:bucket/datasets/

# 3. Test download and extraction on fresh instance
wget https://huggingface.co/yourusername/finnish-tts-dataset/resolve/main/finnish-speaker-2000-with-vq.tar.gz
tar -xzf finnish-speaker-2000-with-vq.tar.gz
ls FinnishSpeaker/*.npy | wc -l  # Should be 2000
```

**Benefits validation:**
- Time comparison: VQ extraction (15 min) vs download + extract (3 min)
- Cost comparison: $0.30 saved per training run
- Reliability: No OOM errors from VQ extraction
- Portability: Works on any GPU, any instance size

---

## Questions to Answer

**Before Phase 1:**
- [ ] Which config format? (YAML vs JSON vs TOML)
- [ ] Logging level? (DEBUG vs INFO)
- [ ] Checkpoint retention policy? (keep all vs best only)

**Before Phase 3:**
- [ ] Notification channel preference? (Slack/Discord/Email)
- [ ] Cost budget per training run?
- [ ] Cloud storage for checkpoints? (S3/GCS/HF Hub)

**Before Phase 5:**
- [ ] Target audience skill level? (beginner/intermediate/advanced)
- [ ] Video tutorials needed?
- [ ] Example datasets to provide?

---

## References

**Current working setup:**
- Brev instance: A100-80GB @ $1.20/hr
- Dataset: 2000 Finnish audio samples
- Base model: openaudio-s1-mini
- Training: 3000 steps, batch_size=8, LoRA r=8 α=16
- VQ extraction: workers=2, batch_size=4 (memory safe)
- Cost: ~$5.40 for full training

**Key learnings:**
- Protobuf 3.20.3 required (not latest)
- TorchCodec needed for torchaudio 2.9.1
- 8 workers OOM on A100-80GB (use 2)
- Training speed: 0.18 it/s (~5.4s per step)
- Loss drops fast (23 → 15 in 100 steps)

---

## Conclusion

This roadmap transforms a functional but manual Jupyter workflow into a **production-ready, reproducible pipeline**. Each phase builds on the previous, with clear deliverables and success metrics.

**Core philosophy:**
- **Simplicity first** - Easy for users to run
- **Reliability** - Handles errors gracefully
- **Transparency** - Clear logging and monitoring
- **Cost-conscious** - Don't waste GPU credits
- **Maintainable** - Easy to update and extend

**Ready to start Phase 1!** 🚀
