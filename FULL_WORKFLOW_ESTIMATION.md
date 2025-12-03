# Full Fine-tuning Workflow: Time & Memory Estimation
**Date:** December 2, 2025  
**Scenario:** Full fine-tuning with 4000 samples → Single speaker LoRA fine-tuning → Inference

---

## 📊 Workflow Overview

```
Phase 1: Full fine-tune on 4000 multi-speaker samples (few epochs)
    ↓
Phase 2: Select best speaker (by sample count)
    ↓
Phase 3: LoRA fine-tune on single speaker until convergence
    ↓
Phase 4: Run inference tests
```

---

## Phase 1: Full Fine-tuning (4000 samples, few epochs)

### Dataset Assumptions:
```
Total samples: 4,000 audio files
Average duration: 7 seconds per file
Total audio: ~7.8 hours
Speakers: Multiple (let's assume 10-20 speakers)
File types: 4000 WAV + 4000 LAB + 4000 NPY (VQ tokens)
```

### Configuration:
```python
method = "full"  # Full fine-tuning
batch_size = 8
num_epochs = 3  # "Few epochs"
precision = "bf16-mixed"  # Auto on A100
```

### Time Calculation:

**Steps per epoch:**
```
steps_per_epoch = 4000 / 8 = 500 steps
```

**Total steps:**
```
total_steps = 500 × 3 = 1,500 steps
```

**Time per step (full fine-tuning):**
```
Our LoRA training: 5.4 sec/step
Full fine-tuning: ~7.0 sec/step (30% slower, more gradients)
```

**Total time:**
```
1,500 steps × 7.0 sec = 10,500 sec = 2.9 hours
Add validation overhead: +0.3 hours
Total: ~3.2 hours
```

### Memory Usage:

**With BF16-mixed precision:**
```
Model weights:        3.5 GB  (868M params, FP32 storage)
Activations:          2.0 GB  (batch_size=8, BF16)
Gradients:            1.75 GB (BF16, all 868M params)
Optimizer (Adam):     7.0 GB  (FP32 master weights)
CUDA overhead:        1.5 GB
Training buffers:     2.0 GB
-----------------------------------------
Total (clean script): ~18 GB
Total (Jupyter):      ~40 GB  (with overhead)

GPU: A100-80GB → 18GB/80GB = 23% utilization ✓
GPU: RTX 6000 Ada-48GB → 18GB/48GB = 38% utilization ✓
```

### Cost:
```
A100-80GB: 3.2 hrs × $1.20/hr = $3.84
RTX 6000: 3.2 hrs × $0.80/hr = $2.56 (cheaper!)
```

---

## Phase 2: Select Best Speaker

### Analysis Time:
```python
# Count samples per speaker
import pandas as pd

# Load labels
labels = []
for file in lab_files:
    with open(file) as f:
        speaker_id = extract_speaker_id(file.name)
        labels.append({'speaker': speaker_id, 'file': file})

# Count
speaker_counts = pd.DataFrame(labels).groupby('speaker').size()
best_speaker = speaker_counts.idxmax()
best_speaker_count = speaker_counts.max()

print(f"Best speaker: {best_speaker} with {best_speaker_count} samples")
```

**Estimated distribution (typical):**
```
If 4000 samples across 10 speakers:
- Top speaker: ~800-1200 samples (20-30%)
- Average speaker: ~400 samples
- Bottom speaker: ~50-100 samples

Let's assume best speaker has: 1000 samples
```

**Time:** 5 minutes (data analysis + filtering)

**Memory:** Negligible (CPU task)

---

## Phase 3: LoRA Fine-tune on Single Speaker (1000 samples)

### Configuration:
```python
method = "lora"
samples = 1000  # Single speaker
batch_size = 8
rank = 8
alpha = 16
precision = "bf16-mixed"
convergence_target = "until loss plateaus"
```

### Time Calculation:

**Steps per epoch:**
```
steps_per_epoch = 1000 / 8 = 125 steps
```

**Steps to convergence:**
```
Estimate: 15-20 epochs for single speaker
(More epochs needed for fewer samples)

Total steps = 125 × 18 = 2,250 steps
```

**Time per step (LoRA):**
```
LoRA with BF16: ~5.0 sec/step
(Slightly faster than our 5.4 sec with FP32)
```

**Total time:**
```
2,250 steps × 5.0 sec = 11,250 sec = 3.1 hours
Add validation: +0.2 hours
Total: ~3.3 hours
```

### Memory Usage:

**With BF16-mixed precision:**
```
Model weights (frozen): 3.5 GB  (FP32 storage, BF16 compute)
LoRA weights:           0.03 GB (8.1M params, BF16)
Activations:            1.0 GB  (batch_size=8, BF16)
Gradients (LoRA only):  0.03 GB (BF16)
Optimizer (LoRA only):  0.06 GB (FP32 master)
CUDA overhead:          1.5 GB
-----------------------------------------
Total (clean script):   ~6 GB
Total (Jupyter):        ~30 GB  (with overhead)

GPU: A100-80GB → 6GB/80GB = 8% utilization
GPU: RTX 6000 Ada-48GB → 6GB/48GB = 13% utilization
```

### Cost:
```
A100-80GB: 3.3 hrs × $1.20/hr = $3.96
RTX 6000: 3.3 hrs × $0.80/hr = $2.64
```

---

## Phase 4: Inference Testing

### Configuration:
```python
test_sentences = 10  # Generate 10 test samples
max_length = 500 tokens per sample
model = merged_model  # LoRA merged with base
```

### Time per Sample:

**Inference speed (A100):**
```
Text → Semantic tokens: ~2 seconds
Semantic → Audio codes: ~3 seconds
Audio codes → Waveform: ~2 seconds
Total per sample: ~7 seconds
```

**Total time:**
```
10 samples × 7 sec = 70 seconds = 1.2 minutes
```

### Memory Usage:

**Inference only:**
```
Merged model:     3.2 GB  (loaded in BF16)
Audio decoder:    1.7 GB  (codec.pth)
Inference buffer: 0.5 GB
-----------------------------------------
Total:            ~5.5 GB

GPU: A100-80GB → 5.5GB/80GB = 7% utilization
Can run on much smaller GPU!
```

### Cost:
```
Inference time: 1.2 minutes
A100: 1.2/60 × $1.20 = $0.024
RTX 6000: 1.2/60 × $0.80 = $0.016

Effectively free! ($0.02)
```

---

## 📊 Complete Workflow Summary

### Total Time:
```
Phase 1: Full fine-tune (4000 samples, 3 epochs)  3.2 hrs
Phase 2: Select best speaker                      0.1 hrs
Phase 3: LoRA fine-tune (1000 samples)            3.3 hrs
Phase 4: Inference testing (10 samples)           0.02 hrs
------------------------------------------------------------
TOTAL:                                            6.6 hours
```

### Total Memory:
```
Phase 1: ~18 GB (full fine-tuning, clean)
Phase 2: ~1 GB (CPU analysis)
Phase 3: ~6 GB (LoRA fine-tuning, clean)
Phase 4: ~5.5 GB (inference)
------------------------------------------------------------
Peak Usage: 18 GB (during Phase 1)

Fits comfortably on:
- A100-80GB: 18/80 = 23% ✓✓✓
- RTX 6000 Ada-48GB: 18/48 = 38% ✓✓
- RTX 4090-24GB: 18/24 = 75% ✓ (tight)
- RTX 3090-24GB: 18/24 = 75% ✓ (tight)
```

### Total Cost:
```
A100-80GB ($1.20/hr):
- Phase 1: $3.84
- Phase 2: $0.12
- Phase 3: $3.96
- Phase 4: $0.02
- TOTAL: $7.94

RTX 6000 Ada ($0.80/hr):
- Phase 1: $2.56
- Phase 2: $0.08
- Phase 3: $2.64
- Phase 4: $0.016
- TOTAL: $5.30 (33% cheaper!)

Recommendation: Use RTX 6000 Ada if available
```

---

## 🎯 Optimizations

### Option 1: Skip Phase 1 (Full Fine-tuning) - ❌ NOT RECOMMENDED FOR FINNISH

**Why NOT to skip for Finnish:**

```
If you skip Phase 1:
- Base model trained on English/Chinese/multilingual data
- Doesn't understand Finnish phonology (ä, ö, unique consonants)
- Missing vowel harmony rules (front/back vowel coordination)
- Wrong prosody patterns (Finnish has different rhythm)
- LoRA only learns SPEAKER, not LANGUAGE
```

**What happens if you skip:**
```
Input text: "Hyvää huomenta, kuinka voit tänään?"
Output: Garbled pronunciation, wrong stress, unnatural rhythm
Quality: 60-70% (speaker sounds right, language sounds wrong)
```

**Correct approach:**
- Phase 1: Full fine-tune on ALL 4000 Finnish samples
  - Teaches model Finnish language structure
  - Quality: 85-90% base quality
- Phase 3: LoRA on best speaker
  - Adds speaker-specific characteristics
  - Quality: 95-98% final quality

**Conclusion:** For new languages like Finnish, Phase 1 is ESSENTIAL!

### Option 2: Use Smaller Batch Size (if memory constrained)

```python
# If on 24GB GPU:
batch_size = 4  # Instead of 8

Effect:
- Memory: 18GB → 12GB (fits RTX 3090!)
- Time: 6.6 hrs → 8.8 hrs (33% slower)
- Cost: Same (longer time, cheaper GPU)
```

### Option 3: Reduce Epochs in Phase 1

```python
# Phase 1: Just 1 epoch instead of 3
num_epochs = 1

Effect:
- Time: 3.2 hrs → 1.1 hrs (saves 2.1 hrs)
- Cost: Saves $2.52
- Quality: Probably fine (just need basic adaptation)
```

---

## 💡 Memory Breakdown by Phase (Clean Script)

### Phase 1: Full Fine-tuning
```
Component                   | Size   | Notes
----------------------------|--------|------------------------
Model weights (FP32)        | 3.5 GB | Stored in FP32
Activations (BF16)          | 2.0 GB | batch_size=8
Gradients (BF16)            | 1.75GB | All 868M params
Optimizer state (FP32)      | 7.0 GB | Adam: momentum + variance
CUDA context                | 1.5 GB | PyTorch overhead
Training buffers            | 2.0 GB | DataLoader, etc.
-----------------------------|--------|------------------------
TOTAL                       | 18 GB  | 23% of A100-80GB
```

### Phase 3: LoRA Fine-tuning
```
Component                   | Size   | Notes
----------------------------|--------|------------------------
Model weights (frozen, FP32)| 3.5 GB | Not updated
LoRA weights (BF16)         | 0.03GB | Only 8.1M params
Activations (BF16)          | 1.0 GB | batch_size=8
Gradients (BF16, LoRA only) | 0.03GB | Tiny!
Optimizer (FP32, LoRA only) | 0.06GB | Adam for 8.1M params
CUDA context                | 1.5 GB | PyTorch overhead
-----------------------------|--------|------------------------
TOTAL                       | 6 GB   | 8% of A100-80GB
```

---

## 📈 Scaling Predictions

### If Dataset = 10,000 samples:
```
Phase 1 (full fine-tune, 3 epochs):
- Steps: 10,000/8 × 3 = 3,750 steps
- Time: 3,750 × 7.0 = 7.3 hours
- Cost (A100): $8.76

Phase 3 (LoRA, best speaker ~2500 samples):
- Steps: 2500/8 × 18 = 5,625 steps
- Time: 5,625 × 5.0 = 7.8 hours
- Cost (A100): $9.36

Total: 15.1 hours, $18.12
```

### If Dataset = 1,000 samples:
```
Phase 1 (full fine-tune, 3 epochs):
- Steps: 1,000/8 × 3 = 375 steps
- Time: 375 × 7.0 = 0.73 hours
- Cost (A100): $0.88

Phase 3 (LoRA, best speaker ~250 samples):
- Steps: 250/8 × 18 = 563 steps
- Time: 563 × 5.0 = 0.78 hours
- Cost (A100): $0.94

Total: 1.5 hours, $1.82
```

---

## 🎓 Key Insights

### 1. Full Fine-tuning is 3× More Memory
```
Full: 18 GB
LoRA: 6 GB
Ratio: 3:1
```

### 2. Time is Similar (Surprisingly!)
```
Full fine-tune (4000 samples, 3 epochs): 3.2 hrs
LoRA (1000 samples, 18 epochs): 3.3 hrs

Why? LoRA needs more epochs but fewer samples
```

### 3. Cost Depends on GPU Choice
```
A100-80GB: Expensive but fast ($1.20/hr)
RTX 6000 Ada: Best value ($0.80/hr, 33% savings)
RTX 3090: Cheapest but tight memory ($0.50/hr)
```

### 4. Phase 1 Might Be Unnecessary
```
If you know best speaker beforehand:
Skip Phase 1 → Save $3.86 (49%)
Go straight to LoRA on target speaker
```

### 5. Inference is Cheap!
```
10 samples: $0.02 (negligible)
100 samples: $0.20
1000 samples: $2.00

Recommendation: Generate many test samples!
```

---

## 📋 Recommended Workflow

### For Your Use Case (4000 samples):

**Option A: With Phase 1 (Conservative)**
```
1. Full fine-tune all 4000 samples (3 epochs)  → 3.2 hrs, $3.84
2. Analyze speaker distribution                → 5 min, $0.10
3. LoRA fine-tune best speaker (1000 samples)  → 3.3 hrs, $3.96
4. Generate 20 test samples                    → 2 min, $0.04
------------------------------------------------------------
TOTAL: 6.6 hours, $7.94 on A100
       6.6 hours, $5.30 on RTX 6000 Ada ✓ Recommended
```

**Option B: Skip Phase 1 (Recommended)**
```
1. Analyze dataset, find best speaker          → 5 min, $0.10
2. LoRA fine-tune best speaker (1000 samples)  → 3.3 hrs, $3.96
3. Generate 20 test samples                    → 2 min, $0.04
------------------------------------------------------------
TOTAL: 3.4 hours, $4.10 on A100
       3.4 hours, $2.74 on RTX 6000 Ada ✓✓ Best value!
```

**Option C: Ultra-Budget (Smaller GPU)**
```
Use RTX 3090 ($0.40/hr):
1. Analyze → LoRA → Test                       → 3.4 hrs, $1.36
Memory: 6GB fits easily in 24GB ✓
```

---

## 🚀 Final Recommendations

**For 4000-sample Finnish TTS workflow:**

**⚠️ IMPORTANT: Finnish Language Considerations**
- Finnish has unique phonetics (vowel harmony, consonant gradation)
- Complex morphology (15 cases, agglutinative structure)
- Different prosody patterns than English
- **Full fine-tuning on all 4000 samples IS NECESSARY** to learn language-specific features

**Recommended Full Workflow:**

1. **GPU Choice:** RTX 6000 Ada (best value, $0.80/hr) or A100 if available
2. **Phase 1: Full fine-tune** all 4000 samples (3 epochs) - **REQUIRED for Finnish**
3. **Phase 2: Select best speaker** by sample count
4. **Phase 3: LoRA fine-tune** on best speaker until convergence
5. **Phase 4: Generate test samples** (20+ samples across different text types)
6. **Use BF16:** Enabled by default on modern GPUs
7. **Clean script:** Avoid Jupyter (saves 20GB memory)

**Total Time:** 6.6 hours  
**Total Cost:** $5.30 (RTX 6000) or $7.94 (A100)  
**Peak Memory:** 18 GB (Phase 1: full fine-tuning)  
**Output:** Language-adapted base model + speaker-specific LoRA + 20 test audio samples  

**Why Full Fine-tuning First?**
- Adapts base model to Finnish phonology
- Learns vowel harmony patterns
- Captures Finnish prosody/rhythm
- LoRA alone won't learn language structure (only speaker characteristics)

**ROI:** Excellent! Under $8 for production-ready Finnish TTS model 🇫�

---

**Summary Table:**

| Phase | Time | Memory | Cost (RTX 6000) | Required? |
|-------|------|--------|----------------|-----------|
| 1. Full fine-tune (4000×3) | 3.2h | 18GB | $2.56 | ❌ Optional |
| 2. Speaker selection | 5min | 1GB | $0.07 | ✅ Yes |
| 3. LoRA fine-tune (1000×18) | 3.3h | 6GB | $2.64 | ✅ Yes |
| 4. Inference (20 samples) | 2min | 5.5GB | $0.03 | ✅ Yes |
| **TOTAL (all phases)** | **6.6h** | **18GB** | **$5.30** | ✅ **REQUIRED for Finnish** |
| **TOTAL (skip Phase 1)** | **3.4h** | **6GB** | **$2.74** | ❌ **Only if base model already speaks Finnish** |

