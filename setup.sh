#!/bin/bash
# Finnish TTS Training Setup Script for Brev
# Optimized for Fish Speech + LoRA fine-tuning
# Production-ready for NVIDIA Launchables deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}✅${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠️${NC}  $1"; }
log_error() { echo -e "${RED}❌${NC} $1"; }
log_step() { echo -e "${BLUE}▶${NC}  $1"; }

echo "🇫🇮 Finnish TTS Training Setup for Brev"
echo "========================================"
echo ""

# Track setup state for idempotency
SETUP_STATE_FILE="${HOME}/.finnish-tts-setup-state"
mark_completed() {
    echo "$1" >> "$SETUP_STATE_FILE"
}
is_completed() {
    [ -f "$SETUP_STATE_FILE" ] && grep -q "^$1$" "$SETUP_STATE_FILE" 2>/dev/null
}

# Check if running on Linux (typical for Brev)
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_warn "Detected macOS. This script is optimized for Brev (Linux)."
    log_warn "Some commands may need adjustment for local development."
    echo ""
fi

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================

if ! is_completed "system_packages"; then
    if command -v apt-get &> /dev/null; then
        log_step "Updating system packages..."
        sudo apt-get update -qq
        
        log_step "Installing system dependencies..."
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
        mark_completed "system_packages"
        log_info "System dependencies installed"
    else
        log_warn "apt-get not available, skipping system packages"
    fi
else
    log_info "System packages already installed (skipping)"
fi

echo ""

# ============================================================================
# PYTHON VERSION CHECK
# ============================================================================

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

log_step "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ]; then
    log_error "Python 3.8+ required (found Python $PYTHON_MAJOR.$PYTHON_MINOR)"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; then
    log_error "Python 3.8+ required (found Python $PYTHON_MAJOR.$PYTHON_MINOR)"
    exit 1
fi

log_info "Python version check passed"

# ============================================================================
# FIX PATH FOR USER-INSTALLED SCRIPTS
# ============================================================================

if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    log_step "Adding ~/.local/bin to PATH..."
    export PATH="$HOME/.local/bin:$PATH"
    log_info "PATH updated (jupyter, huggingface-cli, etc. will work)"
fi

echo ""

# ============================================================================
# UPGRADE PIP
# ============================================================================

if ! is_completed "pip_upgrade"; then
    log_step "Upgrading pip..."
    python3 -m pip install --upgrade pip setuptools wheel -q --no-warn-script-location
    mark_completed "pip_upgrade"
    log_info "pip upgraded"
else
    log_info "pip already upgraded (skipping)"
fi

echo ""

# ============================================================================
# FISH SPEECH REPOSITORY
# ============================================================================

# Try multiple possible locations for Fish Speech
FISH_SPEECH_DIR="../fish-speech"

# If relative path fails, try absolute path in home directory
if [ ! -d "$FISH_SPEECH_DIR" ] && [ ! -w "$(dirname $FISH_SPEECH_DIR 2>/dev/null)" ]; then
    log_warn "Cannot write to parent directory, trying home directory..."
    FISH_SPEECH_DIR="${HOME}/fish-speech"
fi

if [ -d "$FISH_SPEECH_DIR" ] && [ -d "$FISH_SPEECH_DIR/.git" ]; then
    log_info "Fish Speech found at: $FISH_SPEECH_DIR"
else
    log_step "Cloning Fish Speech repository..."
    
    # Ensure parent directory exists and is writable
    PARENT_DIR="$(dirname "$FISH_SPEECH_DIR")"
    if [ ! -d "$PARENT_DIR" ]; then
        mkdir -p "$PARENT_DIR" || {
            log_error "Cannot create directory: $PARENT_DIR"
            exit 1
        }
    fi
    
    if git clone https://github.com/fishaudio/fish-speech.git "$FISH_SPEECH_DIR" 2>&1; then
        log_info "Fish Speech cloned successfully to: $FISH_SPEECH_DIR"
    else
        log_error "Failed to clone Fish Speech repository to: $FISH_SPEECH_DIR"
        log_error "Ensure the parent directory is writable: $PARENT_DIR"
        exit 1
    fi
fi

# Verify the directory is accessible
if ! cd "$FISH_SPEECH_DIR" 2>/dev/null; then
    log_error "Cannot access Fish Speech directory: $FISH_SPEECH_DIR"
    log_error "Check permissions and ensure the directory exists"
    exit 1
fi

echo ""

# ============================================================================
# FISH SPEECH DEPENDENCIES
# ============================================================================

if ! is_completed "fish_speech_deps"; then
    log_step "Installing Fish Speech dependencies..."
    
    # Fix versions before installing fish-speech to avoid conflicts
    log_step "Pinning dependency versions (numpy, pydantic)..."
    pip install -q 'numpy<1.26.5' 'pydantic==2.9.2' --no-warn-script-location
    
    # Core dependencies
    pip install -q hydra-core omegaconf pyrootutils --no-warn-script-location
    pip install -q lightning tensorboard --no-warn-script-location
    pip install -q transformers tokenizers --no-warn-script-location
    pip install -q loralib --no-warn-script-location
    
    # Audio processing
    pip install -q descript-audio-codec --no-warn-script-location
    pip install -q git+https://github.com/descriptinc/audiotools --no-warn-script-location
    
    # Text processing
    pip install -q nemo_text_processing WeTextProcessing --no-warn-script-location
    
    # Install missing fish-speech dependencies
    log_step "Installing additional dependencies..."
    pip install -q cachetools 'datasets==2.18.0' 'gradio>5.0.0' loguru natsort \
        pydub tiktoken uvicorn wandb zstandard --no-warn-script-location
    
    # Install Fish Speech (skip pyaudio - not needed for training)
    pip install -q -e . --no-deps --no-warn-script-location
    pip install -q "einx[torch]==0.2.2" "kui>=1.6.0" "modelscope==1.17.1" \
        "opencc-python-reimplemented==0.1.7" ormsgpack "resampy>=0.4.3" silero-vad --no-warn-script-location
    
    mark_completed "fish_speech_deps"
    log_info "Fish Speech dependencies installed"
else
    log_info "Fish Speech dependencies already installed (skipping)"
fi

echo ""

# ============================================================================
# UTILITY PACKAGES
# ============================================================================

if ! is_completed "utility_packages"; then
    log_step "Installing utility packages..."
    pip install -q soundfile pandas tqdm protobuf --no-warn-script-location
    pip install -q jupyter ipywidgets notebook --no-warn-script-location
    mark_completed "utility_packages"
    log_info "Utility packages installed"
else
    log_info "Utility packages already installed (skipping)"
fi

# Return to original directory
cd - > /dev/null 2>&1 || cd /root/nvidia-brev-launchables || true

echo ""

# ============================================================================
# PROJECT DIRECTORIES
# ============================================================================

log_step "Creating project directories..."

# Detect and handle read-only directories
if [ ! -w "$PWD" ]; then
    log_warn "Current directory ($PWD) is read-only"
    log_warn "Using home directory for project files..."
    PROJECT_DIR="${HOME}/finnish-tts-training"
    mkdir -p "$PROJECT_DIR" || {
        log_error "Cannot create project directory: $PROJECT_DIR"
        exit 1
    }
    cd "$PROJECT_DIR" || {
        log_error "Cannot change to project directory: $PROJECT_DIR"
        exit 1
    }
    log_info "Project directory: $PROJECT_DIR"
fi

mkdir -p datasets/finnish-tts-raw/audio
mkdir -p data/FinnishSpeaker
mkdir -p checkpoints
mkdir -p results
mkdir -p logs
log_info "Project directories created at: $(pwd)"

echo ""

# ============================================================================
# HUGGINGFACE AUTHENTICATION (SECURE)
# ============================================================================

log_step "Checking HuggingFace authentication..."

HF_AUTHENTICATED=false

# Method 1: Check if already logged in via huggingface-cli
if [ -f "$HOME/.cache/huggingface/token" ] || [ -f "$HOME/.huggingface/token" ]; then
    log_info "HuggingFace credentials found (already authenticated)"
    HF_AUTHENTICATED=true
fi

# Method 2: Check for HF_TOKEN environment variable
if [ -n "$HF_TOKEN" ] && [ "$HF_AUTHENTICATED" = false ]; then
    log_info "HF_TOKEN environment variable detected"
    
    # Install huggingface_hub if not present
    pip install -q huggingface_hub
    
    # Login securely via stdin (NOT command line args)
    echo "$HF_TOKEN" | python3 -c "
import sys
from huggingface_hub import login
token = sys.stdin.read().strip()
try:
    login(token=token, add_to_git_credential=False)
    print('✅ HuggingFace authentication successful', file=sys.stderr)
except Exception as e:
    print(f'⚠️  HuggingFace login failed: {e}', file=sys.stderr)
" 2>&1
    
    HF_AUTHENTICATED=true
fi

# Method 3: Check for .env file
if [ -f ".env" ] && [ "$HF_AUTHENTICATED" = false ]; then
    log_info "Loading credentials from .env file"
    source .env
    
    if [ -n "$HF_TOKEN" ]; then
        pip install -q huggingface_hub
        echo "$HF_TOKEN" | python3 -c "
import sys
from huggingface_hub import login
token = sys.stdin.read().strip()
try:
    login(token=token, add_to_git_credential=False)
except: pass
" 2>&1
        HF_AUTHENTICATED=true
    fi
fi

if [ "$HF_AUTHENTICATED" = false ]; then
    echo ""
    log_warn "HuggingFace authentication not configured"
    log_warn "Some models (like fishaudio/openaudio-s1-mini) may require authentication."
    echo ""
    echo "   ${BLUE}To authenticate:${NC}"
    echo "   1. Get a token: https://huggingface.co/settings/tokens"
    echo "   2. Run: ${GREEN}huggingface-cli login${NC}"
    echo "   OR"
    echo "   2. Run: ${GREEN}export HF_TOKEN='your_token_here'${NC}"
    echo ""
fi

# Set environment variables
export HF_HOME="${HOME}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo ""

# ============================================================================
# BASE MODEL DOWNLOAD
# ============================================================================

BASE_MODEL_DIR="checkpoints/openaudio-s1-mini"

if [ -d "$BASE_MODEL_DIR" ] && [ "$(ls -A $BASE_MODEL_DIR 2>/dev/null)" ]; then
    log_info "Base model already downloaded"
else
    log_step "Attempting to download base model..."
    
    if command -v huggingface-cli &> /dev/null && [ "$HF_AUTHENTICATED" = true ]; then
        if huggingface-cli download fishaudio/openaudio-s1-mini --local-dir "$BASE_MODEL_DIR" 2>&1; then
            log_info "Base model downloaded successfully"
        else
            log_warn "Model download failed (may require approval)"
            echo "   1. Visit: https://huggingface.co/fishaudio/openaudio-s1-mini"
            echo "   2. Click 'Accept' to request access"
            echo "   3. Once approved, run: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini"
            echo ""
            log_info "Model will be downloaded automatically during training if needed"
        fi
    else
        log_warn "Skipping model download (authentication not configured)"
        log_info "Model will be downloaded automatically during training"
    fi
fi

echo ""

# ============================================================================
# GPU VERIFICATION
# ============================================================================

log_step "Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    if [ -n "$GPU_INFO" ]; then
        while IFS= read -r gpu; do
            log_info "GPU: $gpu"
        done <<< "$GPU_INFO"
    else
        log_warn "nvidia-smi found but no GPUs detected"
    fi
else
    log_warn "nvidia-smi not found. GPU may not be available."
fi

echo ""

# ============================================================================
# VERIFY PYTHON PACKAGES
# ============================================================================

log_step "Verifying Python package installations..."

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
    print("\n⚠️  Some packages missing. Training may fail.")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    log_info "All required packages verified"
else
    log_error "Package verification failed"
    exit 1
fi

echo ""

# ============================================================================
# CREATE HELPER SCRIPTS
# ============================================================================

if ! is_completed "helper_scripts"; then
    log_step "Creating helper scripts..."
    
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
    
    mark_completed "helper_scripts"
    log_info "Helper scripts created"
else
    log_info "Helper scripts already exist (skipping)"
fi

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
echo "   ${GREEN}bash convert.sh${NC}"
echo ""
echo "3. Start training:"
echo "   ${GREEN}jupyter notebook finnish-tts-model-training.ipynb${NC}"
echo ""
echo "4. Monitor training (in another terminal):"
echo "   ${GREEN}bash monitor.sh --watch${NC}"
echo ""
echo "📚 Documentation:"
echo "   - README: https://github.com/akusingh/nvidia-brev-launchables"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "🎯 Current GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "   GPU monitoring unavailable"
fi

echo ""
echo "Happy Training! 🚀🇫🇮"
echo ""
echo "💡 Tip: To re-run this setup script safely, just run it again."
echo "   It will skip already completed steps."
