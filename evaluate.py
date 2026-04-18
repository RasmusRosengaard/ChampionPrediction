#!/usr/bin/env python3
"""
Model evaluation and comparison script.

Evaluates all trained models (model_*_matches/ directories) on a consistent
held-out test split and prints a comparison table.

Metrics: cross-entropy loss, top-1 / top-3 / top-5 accuracy.
If history.json files exist (written by the updated train.py), also plots
training curves and saves evaluation_curves.png.

Usage:
    python evaluate.py
    python evaluate.py --match-dir match_files
    python evaluate.py --model-dirs model_6330_matches model_10000_matches
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import (
    ALLY_SLOTS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    ENEMY_SLOTS,
    HIDDEN_UNITS,
    MATCH_DIR,
    MODEL_DIR,
    PAD_IDX,
    ROLE_EMBED_DIM,
    ChampionVocab,
    DraftModel,
    MatchParser,
)

# ─── Config ───────────────────────────────────────────────────────────────────

TEST_SPLIT  = 0.10   # fraction of data used as test set
SEED        = 42     # fixed seed for reproducible splits
EVAL_BATCH  = 512


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_all_samples(match_dir: str) -> list:
    files = list(Path(match_dir).glob("*.json"))
    raw = []
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        raw.extend(MatchParser.parse(data))
    rng = random.Random(SEED)
    rng.shuffle(raw)
    return raw


def encode_samples(samples: list, vocab: ChampionVocab):
    allies, enemies, roles, targets = [], [], [], []

    for s in samples:
        target_idx = vocab.encode(s.target_champ)
        if target_idx == PAD_IDX:
            continue  # champion not in this model's vocab — skip

        ally = [vocab.encode(c) for c in s.ally_champs]
        ally = (ally + [PAD_IDX] * ALLY_SLOTS)[:ALLY_SLOTS]

        enemy = [vocab.encode(c) for c in s.enemy_champs]
        enemy = (enemy + [PAD_IDX] * ENEMY_SLOTS)[:ENEMY_SLOTS]

        allies.append(ally)
        enemies.append(enemy)
        roles.append(s.role_idx)
        targets.append(target_idx)

    return (
        {
            "ally":  np.array(allies,  dtype=np.int32),
            "enemy": np.array(enemies, dtype=np.int32),
            "role":  np.array(roles,   dtype=np.int32),
        },
        np.array(targets, dtype=np.int32),
    )


# ─── Metrics ──────────────────────────────────────────────────────────────────

def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    top_k_idx = np.argpartition(logits, -k, axis=1)[:, -k:]
    return float(np.any(top_k_idx == targets[:, None], axis=1).mean())


# ─── Evaluation ───────────────────────────────────────────────────────────────

def find_model_dirs() -> list[Path]:
    return sorted(
        Path(".").glob(f"{MODEL_DIR}_*_matches"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
    )


def load_model(model_dir: Path, vocab: ChampionVocab) -> tf.keras.Model:
    model_path   = model_dir / "draft_model.keras"
    weights_path = model_dir / "best.weights.h5"

    model = DraftModel(
        vocab_size=vocab.size,
        embedding_dim=EMBEDDING_DIM,
        role_embed_dim=ROLE_EMBED_DIM,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT_RATE,
    )
    dummy = {
        "ally":  tf.zeros((1, ALLY_SLOTS),  dtype=tf.int32),
        "enemy": tf.zeros((1, ENEMY_SLOTS), dtype=tf.int32),
        "role":  tf.zeros((1,),             dtype=tf.int32),
    }
    model(dummy, training=False)

    if model_path.exists():
        return tf.keras.models.load_model(
            str(model_path), custom_objects={"DraftModel": DraftModel}
        )
    if weights_path.exists():
        model.load_weights(str(weights_path))
        return model
    raise FileNotFoundError(f"No model file found in {model_dir}")


def evaluate_model(model_dir: Path, test_samples: list) -> dict:
    vocab_path = model_dir / "vocab.pkl"
    if not vocab_path.exists():
        return {"error": "vocab.pkl not found"}

    try:
        vocab = ChampionVocab.load(str(vocab_path))
        model = load_model(model_dir, vocab)
    except Exception as exc:
        return {"error": str(exc)}

    inputs, targets = encode_samples(test_samples, vocab)
    n = len(targets)
    if n == 0:
        return {"error": "no valid test samples for this vocab"}

    # Single forward pass — collect all logits
    ds = tf.data.Dataset.from_tensor_slices(inputs).batch(EVAL_BATCH)
    logits_list = [model(batch, training=False).numpy() for batch in ds]
    logits = np.concatenate(logits_list, axis=0)   # [N, vocab_size]

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = float(loss_fn(targets, logits).numpy())

    history = None
    history_path = model_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    return {
        "n_test_samples": n,
        "vocab_size": vocab.size - 1,
        "loss": loss,
        "top1_acc": top_k_accuracy(logits, targets, 1),
        "top3_acc": top_k_accuracy(logits, targets, 3),
        "top5_acc": top_k_accuracy(logits, targets, 5),
        "history": history,
    }


# ─── Output ───────────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    W = 80
    print("\n" + "═" * W)
    print(
        f"  {'Model':<28} {'#Test':>7} {'Champions':>10} "
        f"{'Loss':>8} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7}"
    )
    print("─" * W)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<28}  ERROR: {r['error']}")
            continue
        print(
            f"  {name:<28} {r['n_test_samples']:>7} {r['vocab_size']:>10} "
            f"{r['loss']:>8.4f} {r['top1_acc']:>7.4f} "
            f"{r['top3_acc']:>7.4f} {r['top5_acc']:>7.4f}"
        )
    print("═" * W + "\n")


def plot_comparison(results: dict) -> None:
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        print("No valid model results to plot.")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots (pip install matplotlib).")
        return

    labels = [k.replace("model_", "").replace("_matches", "\nmatches") for k in valid]
    loss   = [v["loss"]     for v in valid.values()]
    top1   = [v["top1_acc"] for v in valid.values()]
    top3   = [v["top3_acc"] for v in valid.values()]
    top5   = [v["top5_acc"] for v in valid.values()]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("Model comparison — test set metrics", fontsize=13)

    for ax, values, title, color, fmt in zip(
        axes,
        [loss, top1, top3, top5],
        ["Loss", "Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy"],
        ["#e57373", "#64b5f6", "#81c784", "#ffb74d"],
        [".4f", ".1%", ".1%", ".1%"],
    ):
        bars = ax.bar(x, values, color=color, width=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                format(val, fmt),
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    out_path = "evaluation_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Comparison chart saved to {out_path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-dir", default=MATCH_DIR, help="Directory with match JSON files")
    parser.add_argument(
        "--model-dirs", nargs="*",
        help="Model directories to compare (default: auto-discover all model_*_matches/)"
    )
    args = parser.parse_args()

    model_dirs = [Path(d) for d in args.model_dirs] if args.model_dirs else find_model_dirs()
    if not model_dirs:
        print("No model directories found. Run train.py first.")
        return

    print(f"Models to evaluate: {[d.name for d in model_dirs]}")
    print(f"\nLoading samples from {args.match_dir}/ …")
    all_samples = load_all_samples(args.match_dir)

    n = len(all_samples)
    test_n = max(1, int(n * TEST_SPLIT))
    test_samples = all_samples[-test_n:]   # last slice after fixed-seed shuffle
    print(f"Test set: {test_n:,} samples  ({TEST_SPLIT:.0%} of {n:,} total, seed={SEED})")

    results = {}
    for model_dir in model_dirs:
        print(f"\nEvaluating {model_dir.name} …", end=" ", flush=True)
        results[model_dir.name] = evaluate_model(model_dir, test_samples)
        r = results[model_dir.name]
        if "error" not in r:
            print(f"loss={r['loss']:.4f}  top1={r['top1_acc']:.4f}")
        else:
            print(f"FAILED ({r['error']})")

    print_table(results)

    # Save numeric results (no history blob)
    out = {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in results.items()}
    with open("evaluation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → evaluation_results.json")

    plot_comparison(results)


if __name__ == "__main__":
    main()
