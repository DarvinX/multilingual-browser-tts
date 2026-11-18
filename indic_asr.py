"""Per-language IndicConformer CTC decoding with greedy word anchors."""

from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np


REPO = "sulabhkatiyar/indicconformer-120m-onnx"
REVISION = "16f1697d0631c9bb88e371fca24577efe0aa8617"


class IndicRecognizer:
    def __init__(self, root, language, device):
        import librosa
        import onnxruntime as ort

        directory = Path(root) / language
        entries = json.loads((directory / "vocab.json").read_text())
        self.vocab = entries if isinstance(entries, list) else [label for label, _ in sorted(entries.items(), key=lambda pair: pair[1])]
        self.filters = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000)
        self.librosa = librosa
        self.ort = ort
        self.model_path = directory / "model.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, ("CUDAExecutionProvider", {"gpu_mem_limit": str(2600 * 2**20),
                             "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC"}))
        self.session = ort.InferenceSession(self.model_path, sess_options=options, providers=providers)
        self.cpu_session = None
        self.run_options = ort.RunOptions()
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("IndicConformer CUDA provider is unavailable")
        if device == "cuda":
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")

    def features(self, wave):
        emphasis = np.concatenate((wave[:1], wave[1:] - .97 * wave[:-1]))
        spectral = self.librosa.stft(emphasis, n_fft=512, hop_length=160, win_length=400)
        mel = np.log(self.filters @ (np.abs(spectral) ** 2) + 2**-24)
        return ((mel - mel.mean(axis=-1, keepdims=True)) / (mel.std(axis=-1, ddof=1, keepdims=True) + 1e-5)).astype(np.float32)

    def decode(self, frames, duration):
        words = []
        value, start, end = "", 0., 0.
        previous = -1
        for frame, token in enumerate(frames):
            old = previous
            previous = int(token)
            if token == old or token >= len(self.vocab):
                continue
            piece = self.vocab[token]
            if piece in ("<unk>", "<s>", "</s>", "<pad>"):
                continue
            moment = min(duration, frame * .04)
            if piece.startswith("▁") and value:
                words.append({"word": value, "start": start, "end": end})
                value = ""
            if not value:
                start = moment
            value += piece.lstrip("▁")
            end = min(duration, moment + .04)
        if value:
            words.append({"word": value, "start": start, "end": end})
        return {"words": words, "hypothesis": " ".join(word["word"] for word in words)}

    def infer(self, features):
        lengths = np.array([row.shape[-1] for row in features], dtype=np.int64)
        batch = np.zeros((len(features), 80, int(lengths.max())), dtype=np.float32)
        for target, feature in zip(batch, features, strict=True):
            target[:, :feature.shape[-1]] = feature
        inputs = {"audio_signal": batch, "length": lengths}
        try:
            output = self.session.run(None, inputs, self.run_options)[0]
        except self.ort.capi.onnxruntime_pybind11_state.RuntimeException as exc:
            if not any(message in str(exc).casefold() for message in ("allocate memory", "available memory", "out of memory")):
                raise
            if len(features) > 1:
                mid = len(features) // 2
                return self.infer(features[:mid]) + self.infer(features[mid:])
            if self.cpu_session is None:
                options = self.ort.SessionOptions()
                options.intra_op_num_threads = 2
                self.cpu_session = self.ort.InferenceSession(self.model_path, sess_options=options, providers=["CPUExecutionProvider"])
            output = self.cpu_session.run(None, inputs)[0]
        if output.shape[-1] != len(self.vocab) + 1:
            raise RuntimeError(f"Unexpected IndicConformer output shape {output.shape}")
        return [np.argmax(scores[:(length + 3) // 4], axis=-1) for scores, length in zip(output, lengths, strict=True)]

    def transcribe(self, audio: Sequence[np.ndarray], language):
        result = [None] * len(audio)
        order = sorted(range(len(audio)), key=lambda i: len(audio[i]))
        for offset in range(0, len(order), 4):
            group = order[offset:offset + 4]
            features = [self.features(audio[index]) for index in group]
            for index, frames in zip(group, self.infer(features), strict=True):
                result[index] = self.decode(frames, len(audio[index]) / 16000)
        return result
