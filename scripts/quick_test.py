#!/usr/bin/env python3
"""
Quick Inference Tester for Finnish TTS
Test your trained model with sample Finnish text

Usage:
    python quick_test.py --model checkpoints/FinnishSpeaker_2000_finetuned --text "Hei maailma"
"""

import argparse
import sys
from pathlib import Path

def test_model_exists(model_path):
    """Check if model exists"""
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Check for required files
    required_files = []
    # Adjust based on Fish Speech model structure
    
    print(f"✅ Model found: {model_path}")
    return True

def run_inference(model_path, text, output_file="output.wav"):
    """Run inference with the model"""
    print(f"\n{'='*60}")
    print("INFERENCE TEST")
    print(f"{'='*60}\n")
    print(f"Model: {model_path}")
    print(f"Text: {text}")
    print(f"Output: {output_file}\n")
    
    # NOTE: This is a placeholder
    # You'll need to implement actual Fish Speech inference here
    # The exact command depends on Fish Speech's API
    
    print("📝 Sample inference command:")
    print(f"""
python tools/llama/generate.py \\
  --text "{text}" \\
  --checkpoint {model_path} \\
  --output {output_file} \\
  --temperature 0.5 \\
  --max-new-tokens 256
""")
    
    print("\n💡 Adjust the command based on Fish Speech documentation")
    print("💡 For full inference, use the Fish Speech WebUI:\n")
    print(f"   python tools/webui.py --model {model_path}\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Quick inference test for Finnish TTS")
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/FinnishSpeaker_2000_finetuned',
        help='Path to trained model'
    )
    parser.add_argument(
        '--text',
        type=str,
        default='Hei, olen suomalainen puhesyntetisaattori.',
        help='Finnish text to synthesize'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.wav',
        help='Output audio file'
    )
    parser.add_argument(
        '--list-samples',
        action='store_true',
        help='Show sample Finnish texts for testing'
    )
    
    args = parser.parse_args()
    
    if args.list_samples:
        print("\n📝 Sample Finnish Texts for Testing:\n")
        samples = [
            "Hei, miten menee?",
            "Kiitos paljon avusta.",
            "Suomi on kaunis maa.",
            "Minä rakastan suomalaista kulttuuria.",
            "Tämä on testi suomalaiselle puhesyntetisaattorille.",
            "Helsinki on Suomen pääkaupunki.",
            "Onko sinulla kysymyksiä?",
            "Hyvää huomenta! Toivon sinulle hyvää päivää.",
        ]
        
        for i, text in enumerate(samples, 1):
            print(f"{i}. {text}")
        print()
        return
    
    # Check model exists
    if not test_model_exists(args.model):
        sys.exit(1)
    
    # Run inference
    run_inference(args.model, args.text, args.output)

if __name__ == "__main__":
    main()
