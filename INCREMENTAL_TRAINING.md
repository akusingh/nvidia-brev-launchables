# Incremental Fine-Tuning Guide 🔄

## Overview

**Problem:** You've trained a Finnish TTS model on 2000 samples. Now you want to:
- Add more data (1000 new samples)
- Improve quality with better recordings
- Specialize for a specific domain
- Add new speakers

**Solution:** Start from your existing checkpoint instead of training from scratch!

---

## Quick Start

### Option A: LoRA-on-LoRA (Recommended)

**Fastest and cheapest approach - reuse your existing LoRA weights**

```bash
# 1. Check your existing checkpoint
ls -lh ~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/
# You should see: step_000003000.ckpt (47MB)

# 2. Add new audio samples to data directory
cp new_samples/*.wav ~/finnish-tts-brev/data/FinnishSpeaker_v2/
cp new_samples/*.lab ~/finnish-tts-brev/data/FinnishSpeaker_v2/

# 3. Extract VQ tokens for new samples only
python scripts/extract_vq.py \
  --data-dir ~/finnish-tts-brev/data/FinnishSpeaker_v2 \
  --resume  # Skip existing files

# 4. Pack combined dataset
python scripts/pack_dataset.py \
  --data-dir ~/finnish-tts-brev/data/FinnishSpeaker_v2 \
  --output-dir ~/finnish-tts-brev/data/protos_v2

# 5. Train incrementally
python ~/fish-speech/fish_speech/train.py \
  --config-name text2semantic_finetune \
  pretrained_ckpt_path=~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt \
  project=FinnishSpeaker_v2 \
  train_dataset.proto_files=~/finnish-tts-brev/data/protos_v2 \
  trainer.max_steps=1000 \
  +optimizer.lr=5e-5
```

**Results:**
- Training time: **1.5 hours** (vs 4.5 hours from scratch)
- Cost: **$1.80** (vs $5.40 from scratch)
- Savings: **67% time, 67% cost**

---

## Two Approaches Compared

| Aspect | LoRA-on-LoRA | Merge + Re-LoRA |
|--------|--------------|-----------------|
| **Speed** | ⚡️ Fastest (1.5 hrs) | 🐢 Medium (2-3 hrs) |
| **Cost** | 💰 Cheapest ($1.80) | 💰💰 Medium ($2.40-3.60) |
| **Quality** | ⭐️⭐️⭐️⭐️ Very Good | ⭐️⭐️⭐️⭐️⭐️ Best |
| **Setup** | ✅ Simple | 🔧 Requires merge step |
| **Use case** | Incremental improvements | Major dataset additions |
| **Recommended for** | <1000 new samples | >1000 new samples |

---

## Cost Breakdown

### Scenario 1: Training from Scratch Each Time ❌

```
V1: 2000 samples → 3000 steps → 4.5 hrs → $5.40
V2: 3000 samples → 3000 steps → 4.5 hrs → $5.40  (starts over!)
V3: 4000 samples → 3000 steps → 4.5 hrs → $5.40

Total: $16.20
Total time: 13.5 hours
```

### Scenario 2: Incremental Training ✅

```
V1: 2000 samples → 3000 steps → 4.5 hrs → $5.40
V2: +1000 samples → 1000 steps → 1.5 hrs → $1.80  (from V1 checkpoint)
V3: +1000 samples → 1000 steps → 1.5 hrs → $1.80  (from V2 checkpoint)

Total: $9.00
Total time: 7.5 hours

Savings: $7.20 (44%)
Time saved: 6 hours (44%)
```

---

## Best Practices

### ✅ Do This

1. **Lower the learning rate** for incremental training
   ```yaml
   # Initial training
   optimizer.lr: 5e-4
   
   # Incremental training
   optimizer.lr: 5e-5  # 10x lower
   ```

2. **Adjust training steps** based on new data
   ```python
   # Rule of thumb:
   max_steps = new_samples_count * 0.5  # For incremental
   max_steps = total_samples_count * 1.5  # For initial
   ```

3. **Keep LoRA parameters consistent**
   ```yaml
   lora:
     rank: 8      # Same as V1
     alpha: 16    # Same as V1
   ```

4. **Version your checkpoints**
   ```
   checkpoints/
   ├── finnish-v1/ → Initial 2000 samples
   ├── finnish-v2/ → +1000 samples
   └── finnish-v3/ → +1000 samples
   ```

5. **Track metadata**
   ```json
   {
     "version": "v2",
     "parent": "v1/step_000003000.ckpt",
     "new_samples": 1000,
     "total_samples": 3000,
     "training_cost": "$1.80"
   }
   ```

### ❌ Don't Do This

1. **Don't use high learning rate** - Causes catastrophic forgetting
2. **Don't train too long** - Risk overfitting on new data
3. **Don't mix LoRA ranks** - Must merge first if changing rank
4. **Don't skip validation** - Always test on held-out samples
5. **Don't delete parent checkpoints** - You'll need them for rollback

---

## Use Cases

### 1. Iterative Data Collection
**Scenario:** You start with 500 samples, then collect more over time

```
Week 1: 500 samples → train v1 → $1.80
Week 2: +500 samples → train v2 from v1 → $0.90
Week 3: +500 samples → train v3 from v2 → $0.90
Week 4: +500 samples → train v4 from v3 → $0.90

Total: 2000 samples, $4.50
vs from scratch: $5.40 for final model only
Advantage: Have working model each week!
```

### 2. Domain Adaptation
**Scenario:** General Finnish → News Finnish

```
Base: General Finnish (2000 samples) → v1
Add: News recordings (500 samples) → v2 from v1
Result: Specialized news model in 45 min, $0.90
```

### 3. Multi-Speaker
**Scenario:** Single speaker → Multiple speakers

```
Base: Speaker A (2000 samples) → v1
Add: Speaker B (1000 samples) → v2 from v1
Add: Speaker C (1000 samples) → v3 from v2
Result: 3-speaker model with gradual learning
```

### 4. Quality Upgrade
**Scenario:** Phone recordings → Studio recordings

```
Base: Phone quality (2000 samples) → v1
Add: Studio quality (500 samples) → v2 from v1
Result: Model learns better quality while keeping coverage
```

### 5. Error Correction
**Scenario:** Fix pronunciation issues

```
Base: Initial dataset with some errors → v1
Identify: 200 samples with wrong pronunciations
Fix: Re-record those 200 samples correctly
Train: v2 from v1 with corrected samples
Result: Targeted fixes without losing other knowledge
```

---

## Production Config

### config/incremental_training.yaml

```yaml
# Incremental Training Configuration
# Use this for training from an existing checkpoint

paths:
  parent_checkpoint: ~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt
  data_dir: ~/finnish-tts-brev/data/FinnishSpeaker_v2
  proto_dir: ~/finnish-tts-brev/data/protos_v2
  output_dir: ~/fish-speech/results

model:
  base_model: openaudio-s1-mini
  lora_rank: 8      # Must match parent checkpoint
  lora_alpha: 16    # Must match parent checkpoint

training:
  max_steps: 1000   # Reduced from 3000
  batch_size: 8
  num_workers: 8
  val_check_interval: 100
  accumulate_grad_batches: 1

optimizer:
  lr: 5e-5          # 10x lower than initial training
  warmup_steps: 50  # Reduced from 200

early_stopping:
  enabled: true
  monitor: train/loss
  patience: 5
  mode: min

metadata:
  version: v2
  parent_version: v1
  new_samples_count: 1000
  total_samples_count: 3000
```

---

## Checkpoint Types Detection

### Auto-detect checkpoint type in production script:

```python
import torch

def detect_checkpoint_type(ckpt_path):
    """Detect if checkpoint is base model or fine-tuned"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Check for LoRA weights
    has_lora = any('lora' in key or 'adapter' in key 
                   for key in ckpt.get('state_dict', {}).keys())
    
    if has_lora:
        print("🔄 Fine-tuned checkpoint detected")
        print("   → Using incremental training strategy")
        return 'finetuned'
    else:
        print("📦 Base model checkpoint detected")
        print("   → Using initial training strategy")
        return 'base'

def get_optimal_config(ckpt_type, new_samples):
    """Get recommended hyperparameters"""
    if ckpt_type == 'finetuned':
        return {
            'max_steps': max(500, new_samples * 0.5),
            'lr': 5e-5,
            'warmup_steps': 50,
        }
    else:
        return {
            'max_steps': max(2000, new_samples * 1.5),
            'lr': 5e-4,
            'warmup_steps': 200,
        }
```

---

## Validation Strategy

### After incremental training, test these:

```python
test_cases = [
    # Old data - should still work well
    {"text": "Hyvää huomenta", "expected": "good_morning.wav"},
    
    # New data - should work well
    {"text": "Uusi ääninäyte", "expected": "new_sample.wav"},
    
    # Edge cases - check for catastrophic forgetting
    {"text": "Erikoismerkit !@#", "expected": "special_chars.wav"},
]

for test in test_cases:
    generate_audio(test['text'])
    compare_quality(generated, test['expected'])
```

---

## Rollback Strategy

If incremental training makes things worse:

```bash
# 1. Keep all checkpoints
checkpoints/
├── v1/step_000003000.ckpt  ← Don't delete this!
└── v2/step_000001000.ckpt  ← New training

# 2. If v2 is worse, just use v1
python inference.py --checkpoint v1/step_000003000.ckpt

# 3. Or try different hyperparameters for v2
# Lower LR even more: 1e-5
# Fewer steps: 500
# Different data mix
```

---

## Launchables UI Mockup

```
┌─────────────────────────────────────────────┐
│  Finnish TTS Training                       │
├─────────────────────────────────────────────┤
│                                             │
│  Previous Models:                           │
│  ○ V1 - 2000 samples (Nov 29, $5.40)       │
│  ○ V2 - 3000 samples (Dec 1, $1.80)        │
│  ○ Start from scratch                       │
│                                             │
│  New Data:                                  │
│  [Upload Audio Files] 1000 files selected   │
│                                             │
│  Estimated Cost:                            │
│  ┌───────────────────────────────────────┐ │
│  │ From V2 checkpoint:     $1.80         │ │
│  │ Training time:          1.5 hours     │ │
│  │                                       │ │
│  │ From scratch:           $5.40         │ │
│  │ Training time:          4.5 hours     │ │
│  │                                       │ │
│  │ ✅ You save: $3.60 (67%)              │ │
│  └───────────────────────────────────────┘ │
│                                             │
│  [Start Training]  [Advanced Settings]      │
│                                             │
└─────────────────────────────────────────────┘
```

---

## FAQ

**Q: Can I change LoRA rank in incremental training?**  
A: Not directly. You must first merge the existing LoRA into the base model, then start a new LoRA training with the new rank.

**Q: How many times can I do incremental training?**  
A: Theoretically unlimited, but quality may degrade after 4-5 iterations. Consider doing a full retrain every few versions.

**Q: What if I want to remove bad samples from V1?**  
A: You'll need to retrain from scratch with the corrected dataset. Incremental training only adds knowledge, doesn't remove it.

**Q: Can I mix different languages in incremental training?**  
A: Not recommended. Better to train separate models or use a larger base model trained on multiple languages.

**Q: Should I always use incremental training?**  
A: Use incremental when:
- Adding <50% more data
- Fine-tuning for specific domain
- Cost/time is critical

Retrain from scratch when:
- Doubling dataset size or more
- Significant quality issues in V1
- Changing fundamental approach
- Every 3-4 incremental versions (refresh)

---

## Next Steps

1. ✅ Complete your current V1 training (3000 steps)
2. 📦 Save the final checkpoint
3. 🧪 Test V1 model quality
4. 📊 Collect new data or identify improvements needed
5. 🚀 Try incremental training with this guide
6. 📈 Compare V1 vs V2 quality and cost

**Resources:**
- Main roadmap: `PRODUCTION_ROADMAP.md`
- Training logs: `~/fish-speech/results/*/train.log`
- Checkpoints: `~/fish-speech/results/*/checkpoints/`
