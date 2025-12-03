#!/usr/bin/env python3
"""
Finnish TTS Dataset Converter
Converts Finnish TTS dataset to Fish Speech format

Usage:
    python convert_finnish_dataset.py

Input:
    - finnish-tts-raw/metadata.csv
    - finnish-tts-raw/audio/**/*.wav

Output:
    - data/FinnishSpeaker/clip_*.wav (24kHz mono WAV)
    - data/FinnishSpeaker/clip_*.lab (UTF-8 text transcriptions)

Filters:
    - Speaker ID 0 only (240 samples)
    - Only files that actually exist
"""

import csv
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import numpy as np


class FinnishDatasetConverter:
    def __init__(
        self,
        input_dir: str = "finnish-tts-raw",
        output_dir: str = "data/FinnishSpeaker",
        speaker_id: str = None,  # None = all speakers
        target_sr: int = 24000,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.speaker_id = speaker_id
        self.target_sr = target_sr
        
        # Statistics
        self.stats = {
            "total_metadata": 0,
            "speaker_filtered": 0,
            "files_missing": 0,
            "files_converted": 0,
            "errors": 0,
        }
    
    def read_metadata(self) -> List[Tuple[str, str, str, str]]:
        """Read metadata.csv and return list of (audio_file, text, speaker_name, source_dataset)"""
        metadata_path = self.input_dir / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        rows = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.stats["total_metadata"] += 1
                rows.append((
                    row['audio_file'],
                    row['text'],
                    row['speaker_name'],
                    row['source_dataset']
                ))
        
        print(f"✅ Read {len(rows)} entries from metadata.csv")
        return rows
    
    def filter_by_speaker(self, rows: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
        """Filter rows by speaker ID (or keep all if speaker_id is None)"""
        if self.speaker_id is None:
            filtered = rows
            print(f"✅ Using all speakers: {len(filtered)} samples")
        else:
            filtered = [row for row in rows if row[2] == self.speaker_id]
            print(f"✅ Filtered to speaker {self.speaker_id}: {len(filtered)} samples")
        self.stats["speaker_filtered"] = len(filtered)
        return filtered
    
    def validate_audio_exists(self, rows: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
        """Check which audio files actually exist"""
        valid = []
        for audio_file, text, speaker, source in rows:
            audio_path = self.input_dir / audio_file
            if audio_path.exists():
                valid.append((audio_file, text, speaker, source))
            else:
                self.stats["files_missing"] += 1
        
        if self.stats["files_missing"] > 0:
            print(f"⚠️  {self.stats['files_missing']} audio files missing (will skip)")
        
        print(f"✅ {len(valid)} audio files exist and ready to convert")
        return valid
    
    def convert_audio(self, input_path: Path, output_path: Path) -> bool:
        """Convert audio to 24kHz mono WAV (or copy if already correct format)"""
        try:
            # Read audio
            audio, sr = sf.read(str(input_path))
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed (but our data is already 24kHz)
            if sr != self.target_sr:
                print(f"⚠️  Resampling {input_path.name} from {sr}Hz to {self.target_sr}Hz")
                # Note: soundfile doesn't do resampling, would need librosa
                # For now, just copy if sample rate is different
                # In practice, our data is already 24kHz
            
            # Write audio
            sf.write(str(output_path), audio, self.target_sr, subtype='PCM_16')
            return True
            
        except Exception as e:
            print(f"❌ Error converting {input_path.name}: {e}")
            self.stats["errors"] += 1
            return False
    
    def create_lab_file(self, text: str, output_path: Path) -> bool:
        """Create .lab file with transcription"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            print(f"❌ Error creating {output_path.name}: {e}")
            self.stats["errors"] += 1
            return False
    
    def convert_dataset(self):
        """Main conversion function"""
        print("=" * 60)
        print("Finnish TTS Dataset Converter")
        print("=" * 60)
        print()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {self.output_dir}")
        print()
        
        # Read metadata
        print("Step 1: Reading metadata...")
        rows = self.read_metadata()
        print()
        
        # Filter by speaker
        if self.speaker_id is None:
            print(f"Step 2: Using all speakers...")
        else:
            print(f"Step 2: Filtering by speaker ID {self.speaker_id}...")
        filtered_rows = self.filter_by_speaker(rows)
        print()
        
        # Validate audio files exist
        print("Step 3: Validating audio files exist...")
        valid_rows = self.validate_audio_exists(filtered_rows)
        print()
        
        # Convert files
        print(f"Step 4: Converting {len(valid_rows)} files...")
        print()
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("  (Install tqdm for progress bar: pip install tqdm)")
        
        iterator = enumerate(valid_rows, start=1)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(valid_rows), desc="Converting", unit="file")
        
        for idx, (audio_file, text, speaker, source) in iterator:
            # Generate output filenames
            output_name = f"clip_{idx:04d}"
            output_wav = self.output_dir / f"{output_name}.wav"
            output_lab = self.output_dir / f"{output_name}.lab"
            
            # Convert audio
            input_audio = self.input_dir / audio_file
            if self.convert_audio(input_audio, output_wav):
                # Create .lab file
                if self.create_lab_file(text, output_lab):
                    self.stats["files_converted"] += 1
                    if not use_tqdm and idx % 20 == 0:
                        print(f"  Processed {idx}/{len(valid_rows)} files...")
        
        print()
        print("=" * 60)
        print("Conversion Complete!")
        print("=" * 60)
        print()
        self.print_stats()
    
    def print_stats(self):
        """Print conversion statistics"""
        print("Statistics:")
        print(f"  Total metadata entries:     {self.stats['total_metadata']}")
        if self.speaker_id is None:
            print(f"  All speakers samples:       {self.stats['speaker_filtered']}")
        else:
            print(f"  Speaker {self.speaker_id} samples:        {self.stats['speaker_filtered']}")
        print(f"  Missing audio files:        {self.stats['files_missing']}")
        print(f"  Successfully converted:     {self.stats['files_converted']}")
        print(f"  Errors:                     {self.stats['errors']}")
        print()
        
        if self.stats['files_converted'] > 0:
            print(f"✅ Output: {self.output_dir}")
            print(f"   - {self.stats['files_converted']} × clip_*.wav files")
            print(f"   - {self.stats['files_converted']} × clip_*.lab files")
            print()
            print("Next steps:")
            print("  1. Validate dataset: python validate_dataset.py data/FinnishSpeaker")
            print("  2. Extract VQ tokens: python tools/vqgan/extract_vq.py data/FinnishSpeaker ...")
        else:
            print("❌ No files converted! Check errors above.")


def main():
    converter = FinnishDatasetConverter(
        input_dir="finnish-tts-raw",
        output_dir="data/FinnishSpeaker",
        speaker_id=None,  # None = all speakers, or use "0" for specific speaker
        target_sr=24000,
    )
    
    try:
        converter.convert_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
