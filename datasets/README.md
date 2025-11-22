# Dataset Directory

Place your Finnish TTS dataset here.

## Expected Structure

```
datasets/finnish-tts-raw/
├── metadata.csv
└── audio/
    ├── file001.wav
    ├── file002.wav
    └── ...
```

## metadata.csv Format

```csv
audio_file,text,speaker_name,source_dataset
audio/file001.wav,Hei, miten menee?,Speaker_0,cv-15
audio/file002.wav,Kiitos hyvää.,Speaker_0,cv-15
```

## Requirements

- **Audio Format**: WAV, 24kHz, mono preferred
- **Text**: UTF-8 encoded Finnish text
- **Speaker**: Consistent speaker_name for same voice

## Processing

After placing your data here, run:

```bash
cd scripts
python convert_finnish_dataset.py
```

This will create `data/FinnishSpeaker/` with processed files.
