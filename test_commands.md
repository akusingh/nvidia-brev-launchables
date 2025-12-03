# Test Finnish TTS Model - Run in Jupyter Notebook

## Option 1: Quick CLI Test (5 minutes)

```python
# Test with a simple Finnish sentence
!cd ~/fish-speech && python -m tools.api_client \
  --text "Hyvää päivää! Tämä on testi." \
  --reference-audio ~/finnish-tts-brev/data/FinnishSpeaker/wavs/$(ls ~/finnish-tts-brev/data/FinnishSpeaker/wavs/ | head -1) \
  --reference-text "$(head -1 ~/finnish-tts-brev/data/FinnishSpeaker/labels.txt)" \
  --output ~/test-output.wav
```

## Option 2: Start WebUI (fastest, 2 minutes)

```python
# Start the Fish Speech WebUI
!cd ~/fish-speech && python -m tools.webui \
  --llama-checkpoint-path ~/finnish-merged-model \
  --decoder-checkpoint-path ~/finnish-tts-brev/checkpoints/openaudio-s1-mini/firefly-gan-base-generator.ckpt \
  --listen 0.0.0.0:7860
```

Then open in browser: `http://64.247.196.21:7860`

## Option 3: Simple Inference Test

```python
# Generate audio from text using merged model
!cd ~/fish-speech && python -m fish_speech.models.text2semantic.inference \
  --checkpoint-path ~/finnish-merged-model \
  --text "Tervetuloa! Minun nimeni on suomalainen äänimalli." \
  --output ~/test-finnish.wav
```

## Download the test output

After generation, download to check quality:
```python
from IPython.display import Audio, display
display(Audio("~/test-output.wav"))
```

Or download via SCP:
```bash
scp shadeform@64.247.196.21:~/test-output.wav ~/Downloads/
```
