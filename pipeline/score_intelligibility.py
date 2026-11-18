"""Score synthesized speech against held-out prompts with an independent ASR."""

import argparse
import json
from pathlib import Path

from audit import ASR, Recognizer, distance, normal, read_pcm


def measure(output, language, device, indic_models=None):
    refs = json.loads((output / "transcripts.json").read_text())
    ids = sorted(refs)
    name = ASR
    if indic_models:
        from indic_asr import REPO, IndicRecognizer

        recognizer = IndicRecognizer(indic_models, language, device)
        name = REPO
    else:
        recognizer = Recognizer(device)
    pairs = []
    for start in range(0, len(ids), 24):
        batch = ids[start:start + 24]
        waves = [read_pcm(output / "cer-audio" / f"{uid}.wav") for uid in batch]
        for uid, guess in zip(batch, recognizer.transcribe(waves, language), strict=True):
            reference = normal(refs[uid])
            pairs.append({"id": uid, "reference": refs[uid], "hypothesis": guess["hypothesis"],
                          "edits": distance(reference, normal(guess["hypothesis"])), "characters": len(reference)})
    report = {"asr": name, "utterances": len(pairs), "cer": sum(row["edits"] for row in pairs) / sum(row["characters"] for row in pairs),
              "clips": pairs}
    (output / "cer.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{language}: {report['cer']:.2%} CER over {len(pairs)} clips", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("language")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--indic-models", type=Path)
    args = parser.parse_args()
    measure(args.output, args.language, args.device, args.indic_models)
