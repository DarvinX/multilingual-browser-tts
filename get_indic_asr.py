"""Stage pinned per-language IndicConformer model and vocabulary for auditing."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from indic_asr import REPO, REVISION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("language", choices=("gu", "hi", "kn", "ml", "ta", "te"))
    args = parser.parse_args()
    for asset in ("vocab.json", "model.onnx"):
        if not (args.directory / args.language / asset).is_file():
            hf_hub_download(REPO, f"{args.language}/{asset}", revision=REVISION, local_dir=args.directory)
