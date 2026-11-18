"""Synthesize deterministic source-filter matrices from small ONNX tables."""
import numpy as np
import onnx
from onnx import helper, numpy_helper

from tiny_voca.export import _analysis_kernels, _cepstrum_matrices, _synthesis_kernel

REUSABLE = {"Constant", "Shape", "Size", "Gather", "Unsqueeze", "Squeeze", "Cast", "Reshape",
            "Transpose", "Range", "Slice", "Concat", "Expand"}


class FourierTables:
    def __init__(self, fft, window, positions):
        self.fft = fft
        self.prefix = f"fft{fft}_"
        bins = fft // 2 + 1
        self.nodes = []
        angles = np.arange(fft, dtype=np.float64) * (2 * np.pi / fft)
        cos = np.cos(angles).astype(np.float32)
        sin = -np.sin(angles).astype(np.float32)
        sin[[0, fft//2]] = 0
        cos[[fft//4, 3*fft//4]] = 0
        self.cos = self.constant("cos", cos)
        self.sin = self.constant("sin", sin)
        self.window = self.constant("window", window.numpy()[None, :])
        self.modulus = self.constant("mod", np.array(fft, dtype=np.int64))
        self.axis = self.constant("axis", np.array([1], dtype=np.int64))
        self.bins = np.arange(bins, dtype=np.int64)
        self.samples = np.arange(fft, dtype=np.int64)
        self.positions = np.array(positions, dtype=np.int64)
        scale = np.full((bins, 1), 2 / fft, dtype=np.float32)
        scale[[0, -1]] /= 2
        self.scale = self.constant("scale", scale)
        self.emitted = False
        real, imaginary = _analysis_kernels(window)
        cep_real, cep_imaginary = _cepstrum_matrices(positions, fft)
        self.expected = dict(zip(("analysis_cos", "analysis_sin", "synthesis", "cep_cos", "cep_sin"),
                                 (real, imaginary, _synthesis_kernel(window), cep_real, cep_imaginary), strict=True))

    def node(self, op, inputs, label, **kwargs):
        target = self.prefix + label
        self.nodes.append(helper.make_node(op, inputs, [target], **kwargs))
        return target

    def constant(self, label, value):
        return self.node("Constant", [], label, value=numpy_helper.from_array(value))

    def grid(self, label, rows, columns):
        left = self.constant(label+"_left", rows[:, None])
        right = self.constant(label+"_right", columns[None, :])
        product = self.node("Mul", [left, right], label+"_product")
        index = self.node("Mod", [product, self.modulus], label+"_index")
        return [self.node("Gather", [basis, index], f"{label}_{part}")
                for basis, part in ((self.cos, "cos"), (self.sin, "sin"))]

    def outputs(self):
        analysis = self.grid("analysis", self.bins, self.samples)
        cep = self.grid("cep", self.positions, self.bins)
        windowed = [self.node("Mul", [table, self.window], f"windowed_{part}")
                    for table, part in zip(analysis, ("cos", "sin"), strict=True)]
        analysis_out = [self.node("Unsqueeze", [table, self.axis], f"kernel_{part}")
                        for table, part in zip(windowed, ("cos", "sin"), strict=True)]
        real = self.node("Mul", [windowed[0], self.scale], "synthesis_cos")
        imaginary = self.node("Mul", [windowed[1], self.scale], "synthesis_sin")
        inverse = self.node("Concat", [real, imaginary], "inverse", axis=0)
        synthesis = self.node("Unsqueeze", [inverse, self.axis], "synthesis")
        return dict(zip(self.expected, (*analysis_out, synthesis, *cep), strict=True))


def reuse_values(nodes, remap, outputs):
    produced, unique = {}, []
    for node in nodes:
        node.input[:] = [remap.get(name, name) for name in node.input]
        if not node.domain and node.op_type in REUSABLE and len(node.output) == 1 and node.output[0] not in outputs:
            key = (node.op_type, tuple(node.input), tuple(attr.SerializeToString() for attr in node.attribute))
            if key in produced:
                remap[node.output[0]] = produced[key]
                continue
            produced[key] = node.output[0]
        unique.append(node)
    return unique


def compact_graph(path, output, model):
    config, vocoder = model.config, model.vocoder
    causal = list(range(config.cep_causal))
    anti = [config.synth_fft - 1 - i for i in range(config.cep_anti)]
    banks = [FourierTables(config.synth_fft, vocoder.synth_window, causal+anti),
             FourierTables(config.noise_fft, vocoder.noise_window, list(range(config.noise_cep)))]
    signatures = {}
    for bank in banks:
        values = bank.outputs()
        for label, tensor in bank.expected.items():
            signatures[tensor.numpy().tobytes()] = bank, values[label]
    graph = onnx.load(path)
    updated, remap, matched = [], {}, set()
    for node in graph.graph.node:
        payload = node.attribute[0].t.raw_data if node.op_type == "Constant" else b""
        if payload in signatures:
            bank, name = signatures[payload]
            if not bank.emitted:
                updated.extend(bank.nodes)
                bank.emitted = True
            remap[node.output[0]] = name
            matched.add(name)
        else:
            updated.append(node)
    if len(matched) != len(signatures):
        raise ValueError(f"expected {len(signatures)} Fourier constants but matched {len(matched)}")

    outputs = {value.name for value in graph.graph.output}
    # Resolve the original Fourier substitutions before eliminating repeated
    # shape computations, so both passes can refer to canonical outputs.
    for node in updated:
        node.input[:] = [remap.get(name, name) for name in node.input]
    updated = reuse_values(updated, {}, outputs)
    renamed = {name: f"v{i:x}" for i, name in enumerate(
        name for node in updated for name in node.output if name and name not in outputs)}
    for node in updated:
        node.name = ""
        node.input[:] = [renamed.get(name, name) for name in node.input]
        node.output[:] = [renamed.get(name, name) for name in node.output]
    graph.graph.ClearField("node")
    graph.graph.node.extend(updated)
    graph.graph.ClearField("value_info")
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
