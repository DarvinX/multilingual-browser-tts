"""Train the Bengali pronunciation model and score held-out word spellings."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from g2p_model import BOS, EOS, PAD, UNK, GraphemeToPhone, encode_word

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def load_lexicon():
    from frontend import ROOT, SPELLING, normalize_spelling

    DATA = ROOT / "data"
    variants = defaultdict(dict)
    letter_fallbacks = {}
    ipa = {}
    for line in (DATA / "google-phonemes.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            symbol, transcription = line.split("\t")
            ipa[symbol] = transcription
    for line in (DATA / "google-lexicon.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        raw_word, transcription, *labels = line.split("\t")
        word = normalize_spelling(raw_word)
        phones = tuple(p for p in transcription.split() if p != ".")
        if not set(phones) <= ipa.keys():
            raise ValueError(f"Unmapped symbols for {raw_word}: {set(phones) - ipa.keys()}")
        label = labels[0].strip() if labels else ""
        if label.lower() in {"letter", "letter-en"}:
            if SPELLING.fullmatch(word):
                letter_fallbacks.setdefault(word, [phones])
            continue
        variants[word][phones] = variants[word].get(phones, False) or not label
    lexicon = {
        word: sorted(pronunciations, key=lambda phones: (not pronunciations[phones], phones))
        for word, pronunciations in variants.items() if pronunciations
    }
    return lexicon, letter_fallbacks, ipa


def split_word(word):
    bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big") % 100
    return "validation" if bucket < 5 else "test" if bucket < 10 else "train"


def batch_samples(examples, batch_size, rng=None):
    buckets = defaultdict(list)
    for sample in examples:
        buckets[len(sample[0])].append(sample)
    batches = []
    remainder = []
    for length in sorted(buckets):
        bucket = buckets[length]
        if rng:
            rng.shuffle(bucket)
        complete = len(bucket) // batch_size * batch_size
        batches.extend(bucket[i:i + batch_size] for i in range(0, complete, batch_size))
        remainder.extend(bucket[complete:])
    batches.extend(remainder[i:i + batch_size] for i in range(0, len(remainder), batch_size))
    if rng:
        rng.shuffle(batches)
    return batches


def tensor(rows, device):
    return pad_sequence([torch.tensor(row, dtype=torch.long) for row in rows], batch_first=True).to(device)


def edits(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (x != y)))
        row = next_row
    return row[-1]


def evaluate(model, examples, device, batch_size):
    model.eval()
    errors = ref_phones = exact = ended = 0
    for samples in batch_samples(examples, batch_size):
        source = tensor([s[0] for s in samples], device)
        predictions = model.predict(source, max_steps=source.shape[1] * 2 + 4).tolist()
        for output, (_, references) in zip(predictions, samples, strict=True):
            if EOS in output:
                output = output[:output.index(EOS)]
                ended += 1
            else:
                output = []
            distance, length = min((edits(output, r), len(r)) for r in references)
            errors += distance
            ref_phones += length
            exact += distance == 0
    return {
        "words": len(examples),
        "phone_error_rate_pct": round(100 * errors / ref_phones, 4),
        "exact_word_pct": round(100 * exact / len(examples), 4),
        "ended_with_eos": ended,
    }


def save_checkpoint(path, checkpoint):
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def main(argv=None, lexicon_data=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1740)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("use positive epochs and batch size")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True

    if lexicon_data is None:
        from frontend import SPELLING

        lexicon, letter_fallbacks, ipa = load_lexicon()
        trainable = {word: refs for word, refs in lexicon.items() if SPELLING.fullmatch(word)}
    else:
        lexicon, letter_fallbacks, ipa = lexicon_data
        trainable = lexicon
    lexicon = letter_fallbacks | lexicon
    alphabet = sorted(set("".join(w for w in trainable if split_word(w) == "train")))
    graphemes = {symbol: i for i, symbol in enumerate(SPECIAL + alphabet)}
    phones = SPECIAL + sorted(ipa)
    ids = {phone: i for i, phone in enumerate(phones)}
    splits = defaultdict(list)
    for word, variants in sorted(trainable.items()):
        splits[split_word(word)].append(
            (encode_word(word, graphemes), [tuple(ids[p] for p in ref) for ref in variants])
        )

    config = {
        "input_size": len(graphemes), "output_size": len(phones), "width": 160,
    }
    checkpoint = {"config": config, "graphemes": graphemes, "phones": phones, "ipa": ipa}
    model = GraphemeToPhone(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps([trainable, ipa, args.seed, args.batch_size], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    resume_path = out / "resume.pt"
    print(json.dumps({
        "device": str(device), "splits": {k: len(v) for k, v in splits.items()},
        "lookup_words": len(lexicon), "phone_count": len(ipa), "grapheme_count": len(alphabet),
        "parameters": sum(p.numel() for p in model.parameters()), "seed": args.seed,
    }), flush=True)

    history = []
    best_per = float("inf")
    first_epoch = 1
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume["identity"] != identity:
            raise ValueError("G2P training inputs changed since the saved run")
        history = resume["history"]
        first_epoch = len(history) + 1
        best_per = resume["best_per"]
        model.load_state_dict(resume["weights"])
        optimizer.load_state_dict(resume["optimizer"])
        rng.setstate(resume["python_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"])
        print(f"Resuming G2P at epoch {first_epoch}", flush=True)
    started = perf_counter()
    for epoch in range(first_epoch, args.epochs + 1):
        for group in optimizer.param_groups:
            group["lr"] = 0.002 * 0.975 ** (epoch - 1)
        model.train()
        losses = torch.zeros((), device=device)
        batches = batch_samples(splits["train"], args.batch_size, rng)
        for samples in batches:
            source = tensor([s[0] for s in samples], device)
            target = tensor([(BOS, *rng.choice(s[1]), EOS) for s in samples], device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(phones)), target[:, 1:].reshape(-1), ignore_index=PAD)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses += loss.detach()

        val = evaluate(model, splits["validation"], device, args.batch_size)
        per = val["phone_error_rate_pct"]
        result = {
            "epoch": epoch, "train_loss": round(losses.item() / len(batches), 5),
            "validation": val, "elapsed_seconds": round(perf_counter() - started, 2),
        }
        if per < best_per:
            best_per = per
            checkpoint["config"] = config
            checkpoint["weights"] = model.state_dict()
            checkpoint["epoch"] = epoch
            save_checkpoint(out / "phonemizer.pt", checkpoint)
        history.append(result)
        save_checkpoint(resume_path, {"identity": identity, "weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                                      "history": history, "best_per": best_per, "python_rng": rng.getstate(),
                                      "torch_rng": torch.get_rng_state(),
                                      "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None})
        print(json.dumps(result), flush=True)
        (out / "training-metrics.json").write_text(json.dumps({"history": history}, indent=2) + "\n")

    saved = torch.load(out / "phonemizer.pt", map_location=device, weights_only=True)
    best = GraphemeToPhone(**saved["config"]).to(device)
    best.load_state_dict(saved["weights"])
    test = evaluate(best, splits["test"], device, args.batch_size)
    report = {
        "seed": args.seed, "device": str(device), "chosen_epoch": saved["epoch"], "training": "full-rank",
        "validation": history[saved["epoch"] - 1]["validation"], "test_neural_only": test,
        "lookup_words": len(lexicon), "train_words": len(splits["train"]),
        "phonemes": len(ipa), "elapsed_seconds": round(perf_counter() - started, 2), "history": history,
    }
    (out / "training-metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "pronunciations.json").write_text(
        json.dumps({word: refs[0] for word, refs in lexicon.items()}, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (out / "phone-vocab.json").write_text(
        json.dumps([{"id": i, "symbol": p, "ipa": ipa.get(p)} for i, p in enumerate(phones)],
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
