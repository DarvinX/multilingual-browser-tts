"""Clip calibrated 5–7-bit weights, using the existing lossless browser coding."""
import argparse
from pathlib import Path

import torch
from torch.nn.functional import pad

from tiny_voca.train import atomic_save
from tiny_voca.weight_storage import GROUP_SIZE

SENSITIVE = ("duration.", "vocoder.", "prosody_head.", "prosody_projection.", "tokens.")


def quantize(weight, bits):
    count = weight.numel()
    blocks = pad(weight.flatten(), (0, -count % GROUP_SIZE)).reshape(-1, GROUP_SIZE)
    limit = (1 << (bits-1)) - 1
    peak = blocks.abs().amax(dim=1)
    best_error = torch.full_like(peak, torch.inf)
    best_scale = torch.zeros_like(peak, dtype=torch.float16)
    best_codes = torch.zeros_like(blocks, dtype=torch.int8)
    for fraction in (1.0, .98, .96, .94, .92, .89, .86, .82, .78, .74):
        scale = (peak * fraction / limit).half()
        safe = torch.where(scale == 0, 1., scale).float()
        codes = (blocks / safe[:, None]).round().clamp(-limit, limit)
        error = (blocks - codes * scale[:, None].float()).square().sum(1)
        improved = error < best_error
        best_error = torch.minimum(error, best_error)
        best_scale = torch.where(improved, scale, best_scale)
        best_codes = torch.where(improved[:, None], codes.to(torch.int8), best_codes)
    return best_codes.flatten(), best_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=(5, 6, 7), required=True)
    parser.add_argument("--protect-heads", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    state = torch.load(args.source, map_location="cpu", weights_only=True)
    if "packed_weights" in state:
        raise ValueError("use the unquantized source checkpoint")
    weights = dict(state["model"])
    groups = {}
    for name, weight in weights.items():
        if weight.dtype != torch.float32 or weight.ndim < 2 or weight.numel() < 256:
            continue
        bits = 8 if args.protect_heads and name.startswith(SENSITIVE) else args.bits
        codes, scale = quantize(weight, bits)
        weights[name] = codes
        groups[name] = (tuple(weight.shape), scale)
    # The coefficients fit in signed bytes. Keeping this container lets the
    # current browser loader inflate them and gzip compresses the smaller range.
    state.update(model=weights, packed_weights={"bits": 8, "groups": groups},
                 effective_bits=args.bits, protected_heads=args.protect_heads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(state, args.output)


if __name__ == "__main__":
    main()
