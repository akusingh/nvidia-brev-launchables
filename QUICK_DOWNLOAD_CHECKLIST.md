# ⚡ QUICK CHECKLIST - Before Deleting Instance

**Time: 45 minutes | Cost: $0.90 | CRITICAL!**

---

## 🚨 DO THIS BEFORE DELETING:

### 1. Download Final Checkpoint (2 min)
```bash
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt \
  ~/Downloads/finnish-final.ckpt
```

### 2. Package Dataset (10 min)
```bash
ssh shadeform@64.247.196.21
cd ~/finnish-tts-brev/data
tar -czf ~/finnish-dataset.tar.gz FinnishSpeaker/
exit
```

### 3. Upload to HuggingFace (20 min)
```bash
ssh shadeform@64.247.196.21
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create finnish-tts-dataset --type dataset
huggingface-cli upload yourusername/finnish-tts-dataset ~/finnish-dataset.tar.gz --repo-type dataset
exit
```

### 4. Generate & Download Demos (10 min)
```bash
ssh shadeform@64.247.196.21
cd ~/fish-speech
python tools/llama/generate.py --checkpoint ~/fish-speech/results/FinnishSpeaker_2000_finetune/checkpoints/step_000003000.ckpt \
  --text "Hyvää huomenta! Tervetuloa Suomeen." --output demo.wav
exit

scp shadeform@64.247.196.21:~/fish-speech/demo.wav ~/Downloads/
```

### 5. Download Training Log (1 min)
```bash
scp shadeform@64.247.196.21:~/fish-speech/results/FinnishSpeaker_2000_finetune/train.log ~/Downloads/
```

---

## ✅ Verify Before Deleting:

- [ ] `~/Downloads/finnish-final.ckpt` exists (47MB)
- [ ] HuggingFace URL works: https://huggingface.co/datasets/yourusername/finnish-tts-dataset
- [ ] `~/Downloads/demo.wav` plays audio
- [ ] `~/Downloads/train.log` exists

---

## 🗑️ Then Delete:

```bash
brev delete shadeform@64.247.196.21
```

---

## 📋 What You'll Have:

✅ Trained model checkpoint (47MB)
✅ Complete dataset on HuggingFace (8GB, with VQ tokens!)
✅ Demo audio samples
✅ Training logs

**Total time: 45 minutes**
**Total cost: $0.90**
**Remaining budget: ~$4-5 for testing later!**

---

## 🚀 Next Steps (Tomorrow):

1. Update `launchable.yaml` with HuggingFace URL
2. Test on fresh Brev instance ($0.50, 30 min)
3. Submit to NVIDIA
4. Delete test instance

**Everything is saved, ready for Launchables submission!** 🎉
