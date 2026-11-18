"""Build the multilingual demo using only verified model releases."""

import argparse
import json
from pathlib import Path
import shutil

from prompts import LANGUAGES


ROOT = Path(__file__).resolve().parent
SCRIPTS = {"bn": ("বাংলা", "অ"), "gu": ("ગુજરાતી", "અ"), "hi": ("हिन्दी", "अ"), "kn": ("ಕನ್ನಡ", "ಅ"),
           "ml": ("മലയാളം", "അ"), "ta": ("தமிழ்", "அ"), "te": ("తెలుగు", "అ"), "en": ("English", "A")}
ORDER = ("bn", "gu", "hi", "kn", "ml", "ta", "te", "en")


def model_choice(directory, url, mode, label=None):
    weights = (directory / "model.weights.gz").stat().st_size
    full = weights + (directory / "voice.onnx.gz").stat().st_size + (directory / "weight-shuffle.js").stat().st_size
    return {"id": mode, "label": label or mode.upper(), "bytes": weights, "modelBytes": full,
            "weights": f"./{url}/model.weights.gz", "graph": f"./{url}/voice.onnx.gz", "frontend": f"./{url}"}


def build(bengali, output):
    shutil.copytree(bengali, output, dirs_exist_ok=True)
    shutil.copytree(ROOT / "web", output, dirs_exist_ok=True)
    manifest = json.loads((bengali / "models.json").read_text())
    compact = output / "assets" / "bn-compact"
    compact.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "artifacts/bn-model/browser").iterdir():
        shutil.copy2(path, compact)
    languages = []
    for code in ORDER:
        name, voice, _ = LANGUAGES[code]
        native, mark = SCRIPTS[code]
        fixture = ROOT / "converter/fixtures" / f"{code}.jsonl"
        rows = [json.loads(line) for line in fixture.read_text().splitlines()]
        examples = [rows[i]["roman"] for i in (0, 1)] + [rows[i]["native"] for i in (3, 4)]
        entry = {"code": code, "name": name, "nativeName": native, "mark": mark, "voice": voice,
                 "examples": examples, "models": [], "maxTokens": 512, "sampleRate": 24000}
        if code == "bn":
            option = model_choice(compact, "assets/bn-compact", "compact-int8", "Compact INT8")
            option["frontend"] = "./assets"
            entry.update(manifest)
            entry["models"] = [option, *manifest["models"]]
            entry["examples"] = ["Tumi kokhon bari phirbe?", "Kal shokale ami tomar jonno opekkha korbo.",
                                   "আজ বিকেলে বৃষ্টি হলে আমরা বাড়িতেই চা খাব।", "নদীর ধারে বসে আমরা সূর্য ডোবা দেখলাম।"]
        elif (ROOT / f"artifacts/published/{code}-model.txt").is_file():
            for variant in sorted((ROOT / f"artifacts/hf/models/{code}").iterdir()):
                if not (variant / "browser/size.json").is_file():
                    continue
                url = f"assets/{code}-{variant.name}"
                shutil.copytree(variant / "browser", output / url, dirs_exist_ok=True)
                entry["models"].append(model_choice(output / url, url, variant.name))
            if code != "en" and entry["models"]:
                path = output / "assets/converter"
                path.mkdir(exist_ok=True)
                shutil.copy2(ROOT / f"artifacts/converter/assets/{code}.json.gz", path)
            entry["models"].sort(key=lambda variant: (variant["id"] != "int8", variant["modelBytes"]))
        languages.append(entry)
    (output / "languages.json").write_text(json.dumps(languages, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bengali", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.bengali, args.output)
