"""Build the static Bengali speech demo and its on-device inference assets."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import struct
import tarfile

import onnx
import torch
from torch import nn

from tiny_voca import bengali, bengali_letter_rules as letters, romanized_bengali as roman
from tiny_voca.export import export
from tiny_voca.infer import load_model
from tiny_voca.text import PUNCTUATION, SPECIALS, set_frontend
from tiny_voca.weight_storage import GROUP_SIZE


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/bengali-dense-20261117"
RUNTIME = ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")
EXTERNAL_DATA = "voice.weights"


class WordEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, source):
        # Each item is one unpadded word. This is the dense equivalent of the
        # trained phonemizer's packed GRU with the same sequence length.
        memory, state = self.model.encoder(self.model.source_embedding(source))
        hidden = self.model.initial(torch.cat((state[0], state[1]), dim=-1)).tanh().unsqueeze(0)
        return memory, self.model.key(memory).transpose(1, 2), hidden


class PhoneDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token, memory, keys, hidden):
        mask = torch.ones_like(memory[..., 0], dtype=torch.bool)
        scores, state = self.model.decode(token, memory, keys, mask, hidden)
        ignored = torch.full_like(scores[..., :1], -torch.inf)
        scores = torch.cat((ignored, ignored, scores[..., 2:3], ignored, scores[..., 4:]), dim=-1)
        return scores.argmax(dim=-1), state


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def gzip_file(source, target):
    with source.open("rb") as plain, target.open("wb") as output, gzip.GzipFile(
        filename="", fileobj=output, mode="wb", compresslevel=6, mtime=0
    ) as packed:
        shutil.copyfileobj(plain, packed)


def export_g2p(phonemizer, build, assets):
    encoder = WordEncoder(phonemizer.model).eval()
    decoder = PhoneDecoder(phonemizer.model).eval()
    source = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    with torch.inference_mode():
        memory, keys, hidden = encoder(source)
        enc = build / "g2p-encode.onnx"
        dec = build / "g2p-decode.onnx"
        torch.onnx.export(encoder, (source,), str(enc), input_names=["source"],
                          output_names=["memory", "keys", "hidden"], opset_version=17, dynamo=False,
                          dynamic_axes={"source": {1: "letters"}, "memory": {1: "letters"}, "keys": {2: "letters"}})
        torch.onnx.export(decoder, (source[:, :1], memory, keys, hidden), str(dec),
                          input_names=["token", "memory", "keys", "hidden"],
                          output_names=["phone", "state"], opset_version=17, dynamo=False,
                          dynamic_axes={"memory": {1: "letters"}, "keys": {2: "letters"}})
    for graph in (enc, dec):
        onnx.checker.check_model(graph)
        gzip_file(graph, assets / (graph.name + ".gz"))


def export_frontend(build, assets):
    _, phonemizer = bengali.frontend_tools()
    transliterator = bengali.romanized_bengali()
    symbols = bengali.symbol_ids()
    spelling = {
        word: letters.spell_sounds(sound for phone in phones for sound in letters.ENGLISH_SOUNDS[phone])
        for word, phones in letters.english_pronunciations().items()
        if word.isascii() and word.isalpha() and word not in transliterator.exact
    }
    data = {
        "lexicon": {word: [symbols[phone] for phone in phones] for word, phones in phonemizer.lexicon.items()},
        "graphemes": phonemizer.graphemes,
        "phoneIds": [symbols.get(phone, symbols["<unk>"]) for phone in phonemizer.phones],
        "special": dict(zip(("pad", "bos", "eos", "space", "unknown"), range(len(SPECIALS)), strict=True)),
        "marks": {mark: symbols[mark] for mark in PUNCTUATION},
        "roman": {"exact": transliterator.exact, "relaxed": transliterator.relaxed,
                  "months": roman.MONTHS, "locative": roman.LOCATIVE_EXCEPTIONS,
                  "english": spelling, "vowels": letters.VOWELS, "consonants": letters.CONSONANTS,
                  "rules": letters.LETTER_RULES, "variants": letters.SPELLING_VARIANTS},
    }
    plain = build / "frontend.json"
    write_json(plain, data)
    gzip_file(plain, assets / "frontend.json.gz")
    export_g2p(phonemizer, build, assets)


def external_graph(path, build, assets):
    graph = onnx.load(path)
    offset = 0
    layout = []
    for tensor in graph.graph.initializer:
        if not tensor.name.startswith("model.") or not tensor.raw_data:
            raise ValueError(f"voice initializer has no checkpoint tensor: {tensor.name}")
        length = len(tensor.raw_data)
        offset += -offset % 4
        layout.append((tensor.name.removeprefix("model."), offset, length))
        onnx.external_data_helper.set_external_data(tensor, EXTERNAL_DATA, offset=offset, length=length)
        tensor.ClearField("raw_data")
        offset += length
    external = build / "voice.onnx"
    external.write_bytes(graph.SerializeToString())
    target = assets / "voice.onnx.gz"
    gzip_file(external, target)
    return layout, offset, target.stat().st_size


def export_weights(checkpoint, layout, raw_bytes, target):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    packing = state.get("packed_weights", {})
    groups = packing.get("groups", {})
    entries, payload = [], []
    for name, offset, length in layout:
        weight = state["model"][name]
        raw = weight.contiguous().numpy().tobytes()
        mode = "raw"
        if name in groups:
            _, scales = groups[name]
            mode = f"int{packing['bits']}"
            raw += scales.contiguous().numpy().tobytes()
        elif weight.dtype == torch.float16:
            mode = "fp16"
        elif len(raw) != length:
            raise ValueError(f"weight byte count changed for {name}")
        entries.append([mode, offset, length, len(raw)])
        payload.append(raw)
    header = json.dumps({"entries": entries, "rawBytes": raw_bytes, "groupSize": GROUP_SIZE},
                        separators=(",", ":")).encode("utf-8")
    with target.open("wb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as packed:
        packed.write(struct.pack("<I", len(header)))
        packed.write(header)
        packed.writelines(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True, help="onnxruntime-web npm tarball")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    set_frontend("bengali")
    output = args.output.resolve()
    assets = output / "assets"
    build = output.parent / "onnx-build"
    for path in (assets, build):
        path.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    runtime_files = []
    with tarfile.open(args.vendor) as package:
        for name in RUNTIME:
            plain = build / name if name.endswith(".wasm") else runtime / name
            with package.extractfile("package/dist/" + name) as source, plain.open("wb") as target:
                shutil.copyfileobj(source, target)
            packed = runtime / (name + ".gz")
            if name.endswith(".wasm"):
                gzip_file(plain, packed)
            runtime_files.append(packed if name.endswith(".wasm") else plain)

    export_frontend(build, assets)
    baseline = RUN / "full/best-inference.pt"
    model = load_model(baseline)
    graph = build / "fp32.onnx"
    export(model, graph, fold_constants=False)
    onnx.checker.check_model(graph)
    layout, raw_bytes, graph_bytes = external_graph(graph, build, assets)
    models = []
    for precision in ("fp32", "fp16", "int8", "int4"):
        checkpoint = baseline if precision == "fp32" else RUN / f"quantization/bengali-{precision}.pt"
        target = assets / f"{precision}.weights.gz"
        export_weights(checkpoint, layout, raw_bytes, target)
        models.append({"id": precision, "file": target.name, "bytes": target.stat().st_size})
        print(precision, target.stat().st_size, flush=True)
    write_json(output / "models.json", {
        "models": models, "sampleRate": model.config.sample_rate, "maxTokens": model.config.max_tokens,
        "frontendBytes": (assets / "frontend.json.gz").stat().st_size,
        "runtimeBytes": graph_bytes + sum(path.stat().st_size for path in runtime_files),
    })
    for path in (ROOT / "web/bengali-browser").iterdir():
        if path.is_file():
            shutil.copyfile(path, output / path.name)


if __name__ == "__main__":
    main()
