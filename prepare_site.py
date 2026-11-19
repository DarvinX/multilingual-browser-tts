"""Build the multilingual demo with verified releases and selected previews."""

import argparse
import json
from pathlib import Path
import shutil

from prompts import LANGUAGES


ROOT = Path(__file__).resolve().parent
PREVIEW_ASSETS = ("frontend.json.gz", "g2p-decode.onnx.gz", "g2p-encode.onnx.gz",
                  "model.weights.gz", "voice.onnx.gz", "weight-shuffle.js")
SCRIPTS = {"bn": ("বাংলা", "অ"), "gu": ("ગુજરાતી", "અ"), "hi": ("हिन्दी", "अ"), "kn": ("ಕನ್ನಡ", "ಅ"),
           "ml": ("മലയാളം", "അ"), "ta": ("தமிழ்", "அ"), "te": ("తెలుగు", "అ"), "en": ("English", "A")}


def published_card(code, kind):
    marker = ROOT / "artifacts/published" / f"{code}-{kind}.txt"
    if marker.is_file():
        return marker.read_text().strip().partition("/commit/")[0]
    return None


def model_choice(directory, url, mode, label=None):
    weights = (directory / "model.weights.gz").stat().st_size
    full = weights + (directory / "voice.onnx.gz").stat().st_size + (directory / "weight-shuffle.js").stat().st_size
    return {"id": mode, "label": label or mode.upper(), "bytes": weights, "modelBytes": full,
            "weights": f"./{url}/model.weights.gz", "graph": f"./{url}/voice.onnx.gz", "frontend": f"./{url}"}


def preview_choice(code, selection, output):
    phase, mode = selection.split("/")
    candidate = ROOT / "artifacts/training" / code / "candidates" / phase / mode
    split = "test" if (candidate / "test/mos.json").is_file() else "validation"
    score = json.loads((candidate / split / "mos.json").read_text())["predicted_mos"]
    url = f"assets/{code}-preview-{phase}-{mode}"
    target = output / url
    target.mkdir(parents=True, exist_ok=True)
    for filename in PREVIEW_ASSETS:
        shutil.copy2(candidate / "browser" / filename, target)
    choice = model_choice(target, url, f"preview-{mode}", f"{mode.upper()} preview")
    choice["preview"] = {"mos": score, "split": split}
    return choice


def best_preview(code):
    work = ROOT / "artifacts/training" / code
    if not (work / "stages/refiner/step-03600-int8.pt.gz").is_file():
        return None
    candidates = work / "candidates"
    scored = []
    for metric in candidates.glob("*/*/validation/mos.json"):
        candidate = metric.parents[1]
        size = json.loads((candidate / "browser/size.json").read_text())["model_bytes"]
        if size < 1_000_000:
            score = json.loads(metric.read_text())["predicted_mos"]
            scored.append((score, -size, candidate.relative_to(candidates).as_posix()))
    return max(scored)[2] if scored else None


def build(bengali, output):
    shutil.copytree(bengali, output, dirs_exist_ok=True)
    shutil.copytree(ROOT / "web", output, dirs_exist_ok=True)
    manifest = json.loads((bengali / "models.json").read_text())
    previews = json.loads((ROOT / "site-previews.json").read_text())
    compact = output / "assets" / "bn-compact"
    compact.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "artifacts/bn-model/browser").iterdir():
        shutil.copy2(path, compact)
    languages = []
    active = {"bn", *(ROOT / "release-languages.txt").read_text().split()}
    for code in (code for code in SCRIPTS if code in active):
        name, voice, _ = LANGUAGES[code]
        native, mark = SCRIPTS[code]
        fixture = ROOT / "converter/fixtures" / f"{code}.jsonl"
        rows = [json.loads(line) for line in fixture.read_text().splitlines()]
        examples = [rows[i]["roman"] for i in (0, 1)] + [rows[i]["native"] for i in (3, 4)]
        entry = {"code": code, "name": name, "nativeName": native, "mark": mark, "voice": voice,
                 "examples": examples, "models": [], "maxTokens": 512, "sampleRate": 24000,
                 "datasetCard": published_card(code, "dataset"), "modelCard": published_card(code, "model")}
        if code == "bn":
            option = model_choice(compact, "assets/bn-compact", "compact-int8", "Compact INT8")
            option["frontend"] = "./assets"
            option["evaluation"] = {"mos": 2.255, "split": "validation"}
            entry.update(manifest)
            entry["datasetCard"] = "https://huggingface.co/datasets/Hansda/bengali-maya2-jahnavi-20h"
            entry["modelCard"] = "https://huggingface.co/Hansda/bengali-voice-compact"
            entry["models"] = [option, *manifest["models"]]
            entry["examples"] = ["Tumi kokhon bari phirbe?", "Kal shokale ami tomar jonno opekkha korbo.",
                                   "আজ বিকেলে বৃষ্টি হলে আমরা বাড়িতেই চা খাব।", "নদীর ধারে বসে আমরা সূর্য ডোবা দেখলাম।"]
        elif entry["modelCard"]:
            for variant in sorted((ROOT / f"artifacts/hf/models/{code}").iterdir()):
                if not (variant / "browser/size.json").is_file():
                    continue
                url = f"assets/{code}-{variant.name}"
                shutil.copytree(variant / "browser", output / url, dirs_exist_ok=True)
                choice = model_choice(output / url, url, variant.name)
                settings = json.loads((variant / "browser/size.json").read_text())
                if "pace_scale" in settings:
                    choice["paceScale"] = settings["pace_scale"]
                score = json.loads((variant / "test/mos.json").read_text())["predicted_mos"]
                choice["evaluation"] = {"mos": score, "split": "test"}
                entry["models"].append(choice)
            entry["models"].sort(key=lambda variant: (variant["id"] != "int8", variant["modelBytes"]))
        else:
            selection = previews.get(code) or best_preview(code)
            if selection:
                entry["models"].append(preview_choice(code, selection, output))
        if code not in {"bn", "en"} and entry["models"]:
            path = output / "assets/converter"
            path.mkdir(exist_ok=True)
            shutil.copy2(ROOT / f"artifacts/converter/assets/{code}.json.gz", path)
        languages.append(entry)
    (output / "languages.json").write_text(json.dumps(languages, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bengali", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.bengali, args.output)
