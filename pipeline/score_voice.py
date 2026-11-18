"""Render held-out prompts and score them with the Bengali UTMOSv2 protocol."""

import argparse
import json
from pathlib import Path
import random
import wave

import numpy as np
import torch

from export_voice import load_voice
from language_frontend import load_frontend
from tiny_voca.infer import speak


def render(checkpoint, frontend, manifest, output, device):
    load_frontend(frontend)
    _, model = load_voice(checkpoint)
    model.to(device)
    utterances = {row["id"]: row for row in map(json.loads, manifest.read_text().splitlines())}
    if len(utterances) < 24:
        raise ValueError(f"Need at least 24 held-out utterances in {manifest}")
    rng = random.Random(8767)
    screen = sorted(rng.sample(sorted(utterances), min(72, len(utterances))))
    ids = sorted(random.Random(8767).sample(screen, 24))
    output.mkdir(parents=True, exist_ok=True)
    audio = output / "mos-audio"
    audio.mkdir(exist_ok=True)
    cer_audio = output / "cer-audio"
    cer_audio.mkdir(exist_ok=True)
    for uid in screen:
        row = utterances[uid]
        with torch.inference_mode():
            pcm, _, _ = speak(model, row["text"], model.config.speakers[0], seed=0)
        samples = np.round(pcm.numpy().clip(-.9999, .9999) * 32767).astype("<i2")
        if not len(samples) or not (abs(samples).max() > 10):
            raise ValueError(f"Invalid model output at {uid}")
        path = cer_audio / f"{uid}.wav"
        with wave.open(str(path), "wb") as writer:
            writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            writer.writeframes(samples.tobytes())
        if uid in ids and not (audio / path.name).exists():
            (audio / path.name).hardlink_to(path)
    (output / "transcripts.json").write_text(json.dumps({uid: utterances[uid]["text"] for uid in screen}, ensure_ascii=False) + "\n")
    return ids, audio


def mos(ids, audio, device):
    import utmosv2

    scorer = utmosv2.create_model(pretrained=True, device=device)
    scores = []
    for seed in (17, 29, 43):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rows = scorer.predict(input_dir=audio, device=device, num_workers=0, batch_size=4,
                              num_repetitions=1, verbose=False)
        predictions = {Path(row["file_path"]).stem: float(row["predicted_mos"]) for row in rows}
        scores.append([predictions[uid] for uid in ids])
    per_clip = np.array(scores).mean(0)
    if not np.isfinite(per_clip).all():
        raise ValueError("UTMOSv2 returned a non-finite score")
    bootstrap = np.random.default_rng(1740).integers(0, len(ids), (10000, len(ids)))
    return {"predicted_mos": float(per_clip.mean()), "bootstrap_95pct": np.quantile(per_clip[bootstrap].mean(1), [.025, .975]).tolist(),
            "seeds": [17, 29, 43], "ids": ids, "scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("frontend", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(3)
    ids, audio = render(args.checkpoint, args.frontend, args.manifest, args.output, args.device)
    result = mos(ids, audio, args.device)
    result["split"] = args.manifest.stem
    (args.output / "mos.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("predicted_mos", "bootstrap_95pct", "split")}), flush=True)
