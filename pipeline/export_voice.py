"""Export standalone compact browser artifacts and quantization size reports."""

import argparse
import gzip
import io
import json
from pathlib import Path
import shutil

import torch

from language_frontend import load_frontend, STRESS
from tiny_voca.text import PUNCTUATION, SPECIALS
from tiny_voca.weight_storage import unpack_weights
from wave_refiner import RefinedVoice, export_refined


def load_voice(checkpoint):
    payload = io.BytesIO(gzip.decompress(checkpoint.read_bytes())) if checkpoint.suffix == ".gz" else checkpoint
    state = torch.load(payload, map_location="cpu", weights_only=True)
    model = RefinedVoice(state["config"])
    model.load_state_dict(unpack_weights(state))
    return state, model.eval()


def export_frontend(frontend, build, output):
    from build_bengali_web import export_g2p

    ids = frontend.ids
    data = {"lexicon": {word: [[ids[p.lstrip("ˈˌ")], STRESS.get(p[0], 0)] for p in phones]
                        for word, phones in frontend.lexicon.items()}, "graphemes": frontend.graphemes,
            "phoneIds": [ids.get(phone.lstrip("ˈˌ"), ids["<unk>"]) for phone in frontend.phones],
            "phoneStress": [STRESS.get(phone[0], 0) for phone in frontend.phones],
            "special": dict(zip(("pad", "bos", "eos", "space", "unknown"), range(len(SPECIALS)), strict=True)),
            "marks": {mark: ids[mark] for mark in PUNCTUATION}}
    (output / "frontend.json.gz").write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))
    export_g2p(frontend, build, output)


def export(checkpoint, frontend_path, output):
    import onnx
    from build_bengali_web import export_weights, external_graph
    from compact_graph import compact_graph
    from shuffle_weights import shuffle_weights

    output.mkdir(parents=True, exist_ok=True)
    build = output / ".build"
    build.mkdir(exist_ok=True)
    frontend = load_frontend(frontend_path)
    state, model = load_voice(checkpoint)
    graph = export_refined(model, build / "raw.onnx")
    reduced = build / "compact.onnx"
    compact_graph(graph, reduced, model)
    onnx.checker.check_model(reduced)
    layout, raw_bytes, graph_bytes = external_graph(reduced, build, output)
    source = checkpoint
    if checkpoint.suffix == ".gz":
        source = build / "native.pt"
        source.write_bytes(gzip.decompress(checkpoint.read_bytes()))
    plain = build / "plain.weights.gz"
    export_weights(source, layout, raw_bytes, plain)
    target = output / "model.weights.gz"
    shuffle_weights(plain, target)
    unpacker = Path(__file__).with_name("weight-shuffle.js")
    shutil.copy2(unpacker, output)
    export_frontend(frontend, build, output)
    model_bytes = target.stat().st_size + graph_bytes + unpacker.stat().st_size
    stats = {"model_bytes": model_bytes, "weights_bytes": target.stat().st_size, "graph_bytes": graph_bytes,
             "unpacker_bytes": unpacker.stat().st_size, "parameters": sum(p.numel() for p in model.parameters()),
             "sample_rate": model.config.sample_rate, "step": state["step"]}
    (output / "size.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("frontend", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    export(args.checkpoint, args.frontend, args.output)
