#!/bin/bash
# Download all trained assets from Brev instance
# Estimated time: 30 minutes, Cost: $0.60

set -e  # Exit on error

echo "=== STEP 1: Download final checkpoint (2 min) ==="
echo "Downloading step_000002800.ckpt (47MB)..."
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000002800.ckpt ~/Downloads/finnish-final-checkpoint.ckpt
echo "✓ Checkpoint downloaded"

echo ""
echo "=== STEP 2: Download merged model (5 min) ==="
echo "Downloading merged model (3.3GB)..."
scp -r shadeform@64.247.196.21:~/finnish-merged-model ~/Downloads/
echo "✓ Merged model downloaded"

echo ""
echo "=== STEP 3: Download all checkpoints (5 min) ==="
echo "Downloading all training checkpoints..."
scp -r shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints ~/Downloads/finnish-all-checkpoints/
echo "✓ All checkpoints downloaded"

echo ""
echo "=== STEP 4: Package dataset on instance (10 min) ==="
echo "Creating tarball on instance..."
ssh shadeform@64.247.196.21 "tar -czf ~/finnish-dataset.tar.gz ~/finnish-tts-brev/data/FinnishSpeaker/"
echo "✓ Dataset packaged"

echo ""
echo "=== STEP 5: Download dataset (15 min) ==="
echo "Downloading dataset tarball (~8GB)..."
scp shadeform@64.247.196.21:~/finnish-dataset.tar.gz ~/Downloads/
echo "✓ Dataset downloaded"

echo ""
echo "=== STEP 6: Download training logs (1 min) ==="
echo "Downloading logs..."
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log ~/Downloads/finnish-train.log
ssh shadeform@64.247.196.21 "tar -czf ~/tensorboard-logs.tar.gz -C ~/fish-speech/results/FinnishSpeaker_2000_finetune tensorboard/"
scp shadeform@64.247.196.21:~/tensorboard-logs.tar.gz ~/Downloads/
echo "✓ Logs downloaded"

echo ""
echo "=== STEP 7: Download base model for reference (3 min) ==="
echo "Downloading base model..."
scp -r shadeform@64.247.196.21:~/finnish-tts-brev/checkpoints/openaudio-s1-mini ~/Downloads/base-model-openaudio-s1-mini/
echo "✓ Base model downloaded"

echo ""
echo "==================================="
echo "✅ ALL DOWNLOADS COMPLETE!"
echo "==================================="
echo ""
echo "Downloaded files in ~/Downloads/:"
echo "  1. finnish-final-checkpoint.ckpt (47MB) - Final LoRA weights"
echo "  2. finnish-merged-model/ (3.3GB) - Ready-to-use merged model"
echo "  3. finnish-all-checkpoints/ (235MB) - All training checkpoints"
echo "  4. finnish-dataset.tar.gz (~8GB) - Complete dataset with VQ tokens"
echo "  5. finnish-train.log - Training logs"
echo "  6. tensorboard-logs.tar.gz - TensorBoard logs"
echo "  7. base-model-openaudio-s1-mini/ - Base model"
echo ""
echo "Total size: ~11.5GB"
echo ""
echo "Next steps:"
echo "  1. Verify all files exist: ls -lh ~/Downloads/finnish*"
echo "  2. DELETE the Brev instance to stop billing!"
echo "  3. Test model locally with Fish Speech"
echo ""
