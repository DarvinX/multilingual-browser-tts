---
license: apache-2.0
language:
- bn
pipeline_tag: text-to-speech
datasets:
- Hansda/bengali-maya2-jahnavi-20h
tags:
- text-to-speech
- bengali
- onnx
- browser
- int8
---

# Bengali Voice · compact browser TTS

Single-voice 24 kHz Bengali speech synthesis, adapted from [Tiny Voca](https://github.com/n4ze3m/tiny-voca). The compact model uses 112 hidden channels, 3 text blocks, 2 prosody blocks, 5 acoustic blocks, a source-filter vocoder, and a learned waveform residual. It has 930,224 parameters. The INT8 browser package comprises the compressed ONNX graph, packed weights, and 1,167-byte unpacker. A browser demo must also provide the frontend and shared ONNX runtime.

| Artifact | Bytes | Naturalness | CER |
| --- | ---: | ---: | ---: |
| INT8 browser model package | 997,518 | 2.255 predicted MOS | 2.510% |
| INT8 native archive | 989,986 | Same quantized weights | Same quantized weights |
| FP32 checkpoint | 3,785,809 | — | — |

The INT8 naturalness score is the mean of UTMOSv2 predictions for 24 Bengali validation clips under crop seeds 17, 29, and 43. Bengali IndicConformer CER covers 72 validation clips and 3,785 characters. These are automated validation scores. UTMOSv2 is a proxy for perceived naturalness. See `evaluation/` for the source records. The FP32 checkpoint is provided for training and conversion.

## Training data

Trained on [Bengali Maya 2 Native Jahnavi 20h](https://huggingface.co/datasets/Hansda/bengali-maya2-jahnavi-20h), a corpus of 16,315 clips with public-domain [Mozilla Common Voice Bengali prompts](https://github.com/common-voice/common-voice/blob/605c25e48f9fb87cb25ca794cb39bf91d3741f29/server/data/bn/sentence-collector.txt) and [Maya 2 Native](https://huggingface.co/maya-research/maya2-native) using the built-in Jahnavi voice. The dataset card documents its CC BY 4.0 release terms, upstream attribution, and train, validation and test splits.

```bibtex
@misc{hansda2026bengalimaya2jahnavi,
  author = {Hansda, Tamal},
  title = {Bengali synthetic speech: Maya 2 Native Jahnavi, 20 hours},
  year = {2026},
  url = {https://huggingface.co/datasets/Hansda/bengali-maya2-jahnavi-20h}
}
```

## Files and credits

`browser/` contains the model-specific artifacts. The weight and graph archives are meant for the browser loader. `samples/` contains example WAVs from this INT8 model. The acoustic checkpoint and adapted inference code use Apache 2.0. The separate bundled Bengali phonemizer assets credit the Google Bengali pronunciation lexicon, copyright 2015 and 2016 Google Inc., CC BY 4.0. See `frontend/README.md` and `frontend/google-LICENSE`.
