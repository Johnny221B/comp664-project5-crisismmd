#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    models = ("text", "image", "fusion")
    predictions = {}
    rows = []
    for model in models:
        summary_path = Path("logs") / args.protocol / model / f"seed_{args.seed}" / "summary.json"
        summary = json.loads(summary_path.read_text())
        frame = pd.read_csv(summary["prediction_path"], dtype={"sample_id": str, "tweet_id": str})
        if frame.sample_id.duplicated().any():
            raise ValueError(f"duplicate predictions for {model}")
        predictions[model] = frame.sort_values("sample_id").reset_index(drop=True)
        metrics = summary["test"]
        rows.append({
            "protocol": args.protocol, "model": model, "seed": args.seed, "n": len(frame),
            "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"],
            "f1_not_informative": metrics["per_class"]["not_informative"]["f1"],
            "f1_informative": metrics["per_class"]["informative"]["f1"],
            "checkpoint": summary["checkpoint"], "predictions": summary["prediction_path"],
        })
    reference = predictions["text"]
    for model, frame in predictions.items():
        if not frame.sample_id.equals(reference.sample_id):
            raise ValueError(f"sample_id alignment mismatch: text vs {model}")
        if not frame.label.equals(reference.label):
            raise ValueError(f"label alignment mismatch: text vs {model}")
    output = Path("results/tables") / ("id_main.csv" if args.protocol.startswith("id_") else "ood_main.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    figure_root = Path("figures"); figure_root.mkdir(exist_ok=True)
    for model, frame in predictions.items():
        matrix = confusion_matrix(frame.label, frame.pred, labels=[0, 1])
        fig, axis = plt.subplots(figsize=(4.5, 4))
        shown = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2): axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label", title=f"{args.protocol}: {model} (n={len(frame)})")
        fig.colorbar(shown, ax=axis); fig.tight_layout()
        stem = figure_root / f"id_confusion_{model}_seed{args.seed}"
        fig.savefig(stem.with_suffix(".png"), dpi=300); fig.savefig(stem.with_suffix(".pdf")); plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(table)); width = 0.34
    axis.bar(positions - width / 2, table.accuracy, width, label="Accuracy")
    axis.bar(positions + width / 2, table.macro_f1, width, label="Macro-F1")
    axis.set_xticks(positions, table.model); axis.set_ylim(0, 1); axis.legend(); axis.set_ylabel("Score")
    axis.set_title(f"{args.protocol}, seed {args.seed}, n={len(reference)}"); fig.tight_layout()
    fig.savefig(figure_root / "id_main_seed42.png", dpi=300); fig.savefig(figure_root / "id_main_seed42.pdf"); plt.close(fig)
    print(table[["model", "n", "accuracy", "macro_f1"]].to_string(index=False))
    print(f"WROTE {output}; prediction alignment PASS")


if __name__ == "__main__":
    main()

