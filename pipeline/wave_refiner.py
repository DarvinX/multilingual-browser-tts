"""Small, identity-initialized neural waveform residual for the compact voice."""
import argparse
import gzip
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiny_voca.config import Config
from tiny_voca.export import INPUTS, OUTPUTS, Exportable, example_inputs
from tiny_voca.model import SourceFilter, TinyVoca
from tiny_voca.text import set_frontend
from tiny_voca.train import atomic_save
from tiny_voca.weight_storage import pack_weights


class WaveResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        width = 28
        self.condition = nn.Linear(channels, 8)
        self.input = nn.Conv1d(9, width, 9, padding=4)
        self.layers = nn.ModuleList(nn.Sequential(
            nn.Conv1d(width, width, 9, padding=4*dilation, dilation=dilation, groups=width),
            nn.Conv1d(width, 2*width, 1),
            nn.GLU(dim=1),
        ) for dilation in (1, 2, 4, 8, 16, 32))
        self.output = nn.Conv1d(width, 1, 7, padding=3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, wave, hidden, sample_lengths):
        samples = torch.arange(wave.shape[1], device=wave.device)
        audible = (samples[None, :] < sample_lengths[:, None])[:, None]
        conditioning = F.interpolate(self.condition(hidden).transpose(1, 2), size=wave.shape[1],
                                     mode="linear", align_corners=False)
        x = self.input(torch.cat((wave[:, None], conditioning), dim=1) * audible) * audible
        for layer in self.layers:
            x = (x + layer(x)) * audible
        correction = 0.08 * self.output(x)[:, 0].float().tanh()
        return (wave + correction) * audible[:, 0]


class RefinedFilter(SourceFilter):
    def __init__(self, config):
        super().__init__(config)
        self.refiner = WaveResidual(config.channels)

    def forward(self, hidden, f0, lengths, noise=None, phase=None):
        wave = super().forward(hidden, f0, lengths, noise, phase)
        return self.refiner(wave, hidden, lengths*self.config.hop)


class RefinedVoice(TinyVoca):
    def __init__(self, config):
        super().__init__(Config.from_dict(config))
        self.vocoder = RefinedFilter(self.config)


class BrowserRefined(Exportable):
    def _vocoder(self, hidden, f0, total, noise):
        wave = super()._vocoder(hidden, f0, total, noise)
        return self.model.vocoder.refiner(wave, hidden, total.reshape(1)*self.model.config.hop)


def export_refined(model, path):
    axes = {"tokens": {1: "tokens"}, "stress": {1: "tokens"}, "audio": {1: "samples"},
            "durations": {1: "tokens"}}
    with torch.no_grad():
        torch.onnx.export(BrowserRefined(model).eval(), example_inputs(vocab_size=model.config.vocab_size), str(path),
                          input_names=list(INPUTS), output_names=list(OUTPUTS), dynamic_axes=axes, opset_version=17,
                          do_constant_folding=False, dynamo=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    set_frontend("bengali")
    torch.manual_seed(2059)
    state = torch.load(args.source, map_location="cpu", weights_only=True)
    model = RefinedVoice(state["config"])
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected or any(not name.startswith("vocoder.refiner.") for name in missing):
        raise ValueError("source checkpoint must match the base narrow architecture")
    state.update(arch="wave-refiner", model=model.state_dict())
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/"initial-int8.pt"
    atomic_save(state, args.output/"initial-fp32.pt")
    atomic_save(pack_weights(state, "int8"), target)
    portable = gzip.compress(target.read_bytes(), mtime=0)
    target.with_suffix(".pt.gz").write_bytes(portable)
    print("Parameters", sum(p.numel() for p in model.parameters()), "portable INT8 bytes", len(portable))
