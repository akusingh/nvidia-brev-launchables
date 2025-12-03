# Dataset Directory

Place your custom training datasets here.

## Supported Formats

### JSON (.json)
```json
[
  {
    "instruction": "Your instruction here",
    "input": "Optional input context",
    "output": "Expected output"
  }
]
```

### JSONL (.jsonl)
One JSON object per line:
```
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

### CSV (.csv)
```
instruction,input,output
"Instruction 1","Input 1","Output 1"
"Instruction 2","Input 2","Output 2"
```

### ShareGPT Format
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Question here"},
      {"from": "gpt", "value": "Answer here"}
    ]
  }
]
```

## Example Datasets

See `example_dataset.json` for a sample format.

## Tips

1. **Quality over Quantity**: 1000 high-quality examples > 10,000 low-quality ones
2. **Diverse Examples**: Cover different aspects of your use case
3. **Consistent Format**: Maintain uniform formatting across examples
4. **Clear Instructions**: Make instructions specific and unambiguous
5. **Balanced Length**: Mix short and long examples

## Dataset Sources

- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [Dolly](https://github.com/databrickslabs/dolly)
- [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
