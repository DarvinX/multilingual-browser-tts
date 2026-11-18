"""Adapt all weights of the final Bengali architecture with resumable GAN training."""

import argparse
from functools import lru_cache
import gzip
import json
from pathlib import Path
import signal

import torch
from torch import nn

from language_frontend import load_frontend
from tiny_voca.config import CHECKPOINT_FORMAT
from tiny_voca.data import AnnotatedDataset, BucketSampler, collate
from tiny_voca.text import frontend_fingerprint, frontend_identity
from tiny_voca.train import Discriminators, TrainingSystem, WEIGHTS, atomic_save, crops, optimize_step
from tiny_voca.weight_storage import pack_weights
from wave_refiner import RefinedVoice


class SpeechFeatures(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        from torchaudio.transforms import Resample
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        name = "facebook/wav2vec2-base"
        config = Wav2Vec2Config.from_pretrained(name)
        config.num_hidden_layers = 3
        self.encoder = Wav2Vec2Model.from_pretrained(name, config=config).eval()
        self.resample = Resample(sample_rate, 16000)
        self.requires_grad_(False)

    def forward(self, wave):
        samples = self.resample(wave.float())
        variance, mean = torch.var_mean(samples, dim=-1, unbiased=False, keepdim=True)
        return self.encoder((samples - mean) * (variance + 1e-7).rsqrt()).last_hidden_state.float()


class PerceptualTraining(TrainingSystem):
    def __init__(self, voice, feature_weight):
        super().__init__(voice.config)
        self.model = voice
        self.features = SpeechFeatures(voice.config.sample_rate) if feature_weight else None
        self.feature_weight = feature_weight

    def objective(self, batch, teacher_probability, weights=WEIGHTS, random_phase=True):
        loss, metrics, auxiliary = super().objective(batch, teacher_probability, weights, random_phase)
        if self.features is None:
            return loss, metrics, auxiliary
        fake, real, _ = crops(auxiliary["wave"], batch["wave"], batch["frame_lengths"] * self.model.config.hop, 12288, 1)
        with torch.no_grad():
            target = self.features(real)
        feature_error = (self.features(fake) - target).abs().mean()
        metrics["speech_feature_distance"] = feature_error
        return loss + self.feature_weight * feature_error, metrics, auxiliary


def initialize(voice, reference, frontend, source_vocab):
    weights = dict(reference["model"])
    old = weights.pop("tokens.weight")
    old_symbols = list(reference["frontend"]["symbols"])
    pronunciations = json.loads(source_vocab.read_text())
    broad = {row["symbol"]: row["ipa"] for row in pronunciations if row["ipa"]}
    old_ids = {broad.get(label, label): i for i, label in enumerate(old_symbols)}
    with torch.no_grad():
        for index, label in enumerate(frontend.symbols):
            if label in old_ids:
                voice.tokens.weight[index].copy_(old[old_ids[label]])
    missing, unexpected = voice.load_state_dict(weights, strict=False)
    if missing != ["tokens.weight"] or unexpected:
        raise ValueError(f"Reference architecture differs: missing {missing}, extra {unexpected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("frontend", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("reference_vocab", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--feature-weight", type=float, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gan-ratio", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=43103)
    parser.add_argument("--refiner-only", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.78)
    frontend = load_frontend(args.frontend)
    voice = RefinedVoice(json.loads((args.data / "config.json").read_text()))
    data = AnnotatedDataset(args.data / "train.jsonl", voice.config, cache=False, voiced_vowels=True)
    cached = lru_cache(maxsize=700)(data.__getitem__)
    sampler = BucketSampler(data.lengths, 2800, 7, args.seed)
    recipe = dict(weights=WEIGHTS, learning_rate=args.lr, lr_half_life=20000, warmup=350, teacher_hold=250,
                  teacher_ramp=1850, teacher_final=.1, amp_bf16=args.device == "cuda", grad_clip=10.0, gan_start=400,
                  gan_ramp=2200, gan_ratio=args.gan_ratio, gan_scale_max=2000., crop_samples=8192, crops=1,
                  feature_matching=30.)
    identity = {"data": data.fingerprint, "frontend": frontend_fingerprint(), "recipe": recipe,
                "feature_weight": args.feature_weight, "refiner_only": args.refiner_only, "seed": args.seed}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "resume.pt"
    if not latest.exists():
        if args.init:
            state = torch.load(args.init, map_location="cpu", weights_only=True)
            if state["frontend"]["fingerprint"] != frontend_fingerprint():
                raise ValueError("Training stage uses a different frontend")
            voice.load_state_dict(state["model"])
        else:
            reference = torch.load(args.reference, map_location="cpu", weights_only=True)
            initialize(voice, reference, frontend, args.reference_vocab)
    system = PerceptualTraining(voice, args.feature_weight).to(args.device).train()
    if system.features is not None:
        system.features.eval()
    if args.refiner_only:
        voice.eval().requires_grad_(False)
        voice.vocoder.refiner.train().requires_grad_(True)
        system.mel_head.requires_grad_(False)
    critics = Discriminators().to(args.device).train()
    optimizer = torch.optim.AdamW((p for p in system.parameters() if p.requires_grad), lr=args.lr, betas=(.8, .99), fused=args.device == "cuda")
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=args.lr, betas=(.8, .99), fused=args.device == "cuda")
    done = 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume state does not match data or training settings")
        system.model.load_state_dict(state["model"])
        system.mel_head.load_state_dict(state["mel_head"])
        critics.load_state_dict(state["critics"])
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        sampler.load_state(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        done = state["step"]
        print(f"Resuming from step {done}", flush=True)

    def inference_state(step):
        return {"format_version": CHECKPOINT_FORMAT, "arch": "wave-refiner", "config": voice.config.to_dict(),
                "step": step, "annotation": data.annotation,
                "frontend": {**frontend_identity(), "fingerprint": frontend_fingerprint()},
                "model": {name: tensor.detach().cpu() for name, tensor in voice.state_dict().items()}}

    def save(step):
        state = inference_state(step)
        atomic_save({"identity": identity, "step": step, "model": state["model"], "mel_head": system.mel_head.state_dict(),
                     "critics": critics.state_dict(), "optimizer": optimizer.state_dict(),
                     "critic_optimizer": critic_optimizer.state_dict(), "sampler": sampler.state(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if args.device == "cuda" else None}, latest)
        if step % 2000 == 0 or step == args.steps:
            atomic_save(state, output / f"step-{step:05d}-fp32.pt")
            if step == args.steps:
                for mode in ("int8", "int4"):
                    native = output / f"step-{step:05d}-{mode}.pt"
                    atomic_save(pack_weights(state, mode), native)
                    archive = native.with_suffix(".pt.gz")
                    temporary = archive.with_suffix(".tmp")
                    temporary.write_bytes(gzip.compress(native.read_bytes(), mtime=0))
                    temporary.replace(archive)
        print(f"Saved checkpoint step {step}", flush=True)

    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(f"Training {sum(p.numel() for p in voice.parameters()):,} parameters from {done} to {args.steps}", flush=True)
    for step in range(done + 1, args.steps + 1):
        batch = collate([cached(index) for index in sampler.next()], args.device)
        stats = optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe)
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, **stats}), flush=True)
        if stop or step % args.save_every == 0 or step % 2000 == 0 or step == args.steps:
            save(step)
        if stop:
            return
    if not (output / f"step-{args.steps:05d}-int8.pt.gz").exists():
        save(args.steps)


if __name__ == "__main__":
    main()
