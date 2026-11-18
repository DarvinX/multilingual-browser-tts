"""Packed checkpoint weights for the native FP32 inference model."""

from math import prod

import torch
from torch.nn.functional import pad


GROUP_SIZE = 64


def pack_weights(state, precision):
    if precision not in ("fp16", "int8", "int4"):
        raise ValueError(f"unsupported checkpoint precision: {precision}")
    if "packed_weights" in state:
        raise ValueError("expected an unpacked inference checkpoint")
    weights = state["model"]
    if precision == "fp16":
        return {**state, "model": {
            name: weight.half() if weight.dtype == torch.float32 else weight
            for name, weight in weights.items()
        }}

    bits = int(precision[-1])
    limit = 2 ** (bits - 1) - 1
    packed = dict(weights)
    groups = {}
    for name, weight in weights.items():
        if weight.dtype != torch.float32 or weight.ndim < 2 or weight.numel() < 256:
            continue
        flat = weight.flatten()
        blocks = pad(flat, (0, -flat.numel() % GROUP_SIZE)).reshape(-1, GROUP_SIZE)
        scales = (blocks.abs().amax(dim=1) / limit).to(torch.float16)
        safe_scales = torch.where(scales == 0, 1., scales).float()
        codes = (blocks / safe_scales[:, None]).round().clamp(-limit, limit).to(torch.int8).flatten()
        if bits == 4:
            codes = (codes + 8).to(torch.uint8)
            codes = codes[0::2] | (codes[1::2] << 4)
        packed[name] = codes
        groups[name] = (tuple(weight.shape), scales)
    return {**state, "model": packed, "packed_weights": {"bits": bits, "groups": groups}}


def unpack_weights(state):
    """Restore packed tensors before loading the model's state dict."""
    weights = state["model"]
    metadata = state.get("packed_weights")
    if metadata is None:
        return weights
    bits = metadata["bits"]
    if bits not in (4, 8):
        raise ValueError(f"unsupported packed checkpoint precision: {bits}")
    unpacked = dict(weights)
    for name, (shape, scales) in metadata["groups"].items():
        codes = weights[name]
        if bits == 4:
            codes = torch.stack((codes & 15, codes >> 4), dim=1).flatten().to(torch.int8) - 8
        blocks = codes.reshape(-1, GROUP_SIZE).float() * scales.float()[:, None]
        unpacked[name] = blocks.flatten()[:prod(shape)].reshape(shape)
    return unpacked
