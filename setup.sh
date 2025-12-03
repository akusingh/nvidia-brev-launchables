#!/bin/bash
# Finnish TTS Training Setup Script for Brev
# Optimized for Fish Speech + LoRA fine-tuning

set -e  # Exit on error

echo "🇫🇮 Finnish TTS Training Setup for Brev"
echo "========================================"
echo ""

# Check if running on Linux (typical for Brev)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  Detected macOS. This script is optimized for Brev (Linux)."
    echo "Some commands may need adjustment for local development."
    echo ""
fi

# Update system packages (if on Linux)
if command -v apt-get &> /dev/null; then
    echo "📦 Updating system packages..."
    sudo apt-get update -qq
    
    # Install system dependencies
    echo "🔧 Installing system dependencies..."
    sudo apt-get install -y -qq \
        git \
        git-lfs \
        sox \
        libsox-dev \
        ffmpeg \
        portaudio19-dev \
        build-essential \
        python3-dev
    
    git lfs install
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
echo ""
echo "🐍 Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 8 ]; then
    echo "❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel -q

# Check if Fish Speech is available
FISH_SPEECH_DIR="../fish-speech"
if [ -d "$FISH_SPEECH_DIR" ]; then
    echo ""
    echo "✅ Fish Speech found at: $FISH_SPEECH_DIR"
else
    echo ""
    echo "📥 Fish Speech not found. Cloning repository..."
    git clone https://github.com/fishaudio/fish-speech.git $FISH_SPEECH_DIR
    echo "✅ Fish Speech cloned successfully"
fi

# Install Fish Speech dependencies
echo ""
echo "📚 Installing Fish Speech dependencies..."
cd $FISH_SPEECH_DIR

# Core dependencies
pip install -q hydra-core omegaconf pyrootutils
pip install -q lightning tensorboard
pip install -q transformers tokenizers
pip install -q loralib

# Audio processing
pip install -q descript-audio-codec
pip install -q git+https://github.com/descriptinc/audiotools

# Text processing
pip install -q nemo_text_processing WeTextProcessing

# Install Fish Speech (skip pyaudio - not needed for training)
pip install -q -e . --no-deps
pip install -q "einx[torch]==0.2.2" "kui>=1.6.0" "modelscope==1.17.1" \
    "opencc-python-reimplemented==0.1.7" ormsgpack "resampy>=0.4.3" silero-vad

# Additional utilities for our scripts
echo ""
echo "🛠️  Installing utility packages..."
pip install -q soundfile numpy pandas tqdm loguru protobuf

# Jupyter and notebook support
pip install -q jupyter ipywidgets notebook

# Return to nvidia-brev directory
cd - > /dev/null

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p datasets/finnish-tts-raw/audio
mkdir -p data/FinnishSpeaker
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

# Download base model (if not exists)
BASE_MODEL_DIR="checkpoints/openaudio-s1-mini"
if [ -d "$BASE_MODEL_DIR" ]; then
    echo ""
    echo "✅ Base model already downloaded"
else
    echo ""
    echo "📥 Downloading base model..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download fishaudio/openaudio-s1-mini --local-dir $BASE_MODEL_DIR
    else
        echo "⚠️  huggingface-cli not found. Install it with:"
        echo "    pip install huggingface-hub"
        echo "    Then run: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini"
    fi
fi

# Set up environment variables
echo ""
echo "🔧 Configuring environment..."
export HF_HOME=~/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# Add to bashrc if not already there
if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Finnish TTS Training Environment" >> ~/.bashrc
    echo "export HF_HOME=~/.cache/huggingface" >> ~/.bashrc
    echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc
fi

# Verify GPU availability
echo ""
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read gpu; do
        echo "  ✅ $gpu"
    done
else
    echo "  ⚠️  nvidia-smi not found. GPU may not be available."
fi

# Verify Python packages
echo ""
echo "✅ Verifying installations..."
python3 << 'EOF'
import sys

packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'lightning': 'PyTorch Lightning',
    'soundfile': 'SoundFile',
    'numpy': 'NumPy',
}

all_good = True
for package, name in packages.items():
    try:
        __import__(package)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} - not installed")
        all_good = False

if not all_good:
    print("\n⚠️  Some packages missing. Please install them manually.")
    sys.exit(1)
EOF

# Create helper script shortcuts
echo ""
echo "🔗 Creating helper scripts..."
cat > monitor.sh << 'EOF'
#!/bin/bash
cd scripts && python monitor_training.py "$@"
EOF
chmod +x monitor.sh

cat > convert.sh << 'EOF'
#!/bin/bash
cd scripts && python convert_finnish_dataset.py "$@"
EOF
chmod +x convert.sh

cat > test.sh << 'EOF'
#!/bin/bash
cd scripts && python quick_test.py "$@"
EOF
chmod +x test.sh

# Print completion message
echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Prepare your dataset:"
echo "   - Place audio files in: datasets/finnish-tts-raw/audio/"
echo "   - Create metadata.csv in: datasets/finnish-tts-raw/"
echo ""
echo "2. Convert dataset:"
echo "   bash convert.sh"
echo ""
echo "3. Start training:"
echo "   jupyter notebook finnish-tts-model-training.ipynb"
echo ""
echo "4. Monitor training (in another terminal):"
echo "   bash monitor.sh --watch"
echo ""
echo "📚 Documentation:"
echo "   - Quick Start: README.md"
echo "   - Full Guide: docs/README_FINNISH_TTS.md"
echo "   - What's New: docs/IMPROVEMENTS_SUMMARY.md"
echo ""
echo "🎯 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
fi
echo ""
echo "Happy Training! 🚀🇫🇮"
