# Finnish TTS Training - Quick Reference

## 🚀 Quick Commands

### Setup
```bash
bash setup.sh
```

### Convert Dataset
```bash
cd scripts
python convert_finnish_dataset.py
```

### Start Training
```bash
jupyter notebook finnish-tts-model-training.ipynb
```

### Monitor Training (Real-time)
```bash
bash monitor.sh --watch
```

### Test Model
```bash
bash test.sh --model checkpoints/FinnishSpeaker_trained --text "Hei maailma"
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `finnish-tts-model-training.ipynb` | Main training notebook |
| `scripts/convert_finnish_dataset.py` | Dataset converter |
| `scripts/monitor_training.py` | Training monitor |
| `scripts/quick_test.py` | Model tester |
| `setup.sh` | Environment setup |

---

## 📊 Directory Layout

```
nvidia-brev/
├── finnish-tts-model-training.ipynb    # ⭐ Start here
├── setup.sh                            # Run first
├── scripts/                            # Utilities
├── docs/                               # Full documentation
├── datasets/finnish-tts-raw/          # Your raw data goes here
├── data/FinnishSpeaker/               # Converted data
├── checkpoints/                        # Models
└── results/                            # Training outputs
```

---

## 🎯 Typical Workflow

1. **Setup** → `bash setup.sh`
2. **Data** → Place files in `datasets/finnish-tts-raw/`
3. **Convert** → `bash convert.sh`
4. **Train** → Open notebook and run cells
5. **Monitor** → `bash monitor.sh --watch` (in another terminal)
6. **Export** → Follow notebook steps 10-11
7. **Test** → `bash test.sh --model <path> --text "..."`

---

## 💡 Tips

- Always validate dataset before training (Step 5.1 in notebook)
- Monitor GPU with: `watch -n 1 nvidia-smi`
- Training takes ~1.5-2 hours for 2000 steps
- Checkpoints saved every 50 steps
- Download checkpoints before shutdown!

---

## 📚 Full Documentation

- **README.md** - Complete project guide
- **docs/README_FINNISH_TTS.md** - Technical details
- **docs/IMPROVEMENTS_SUMMARY.md** - All enhancements

---

## 🐛 Common Issues

### Out of Memory
- Notebook auto-adjusts batch size
- Or reduce manually in Step 8.1

### Training Stalls
- Check: `tail -f results/*/train.log`
- Monitor: `bash monitor.sh`

### Poor Quality
- Train longer (2000+ steps)
- Validate dataset quality
- Check audio is 24kHz

---

**Need help? Check docs/README_FINNISH_TTS.md**
