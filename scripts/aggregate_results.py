#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.metrics import classification_metrics

PROTOCOLS = ("id_official_strict", "ood_heldout_strict")
MODELS = ("text", "image", "fusion")
SEEDS = (42, 123, 2026)


def head_for(model: str) -> nn.Module:
    if model == "text": return nn.Sequential(nn.Dropout(.2), nn.Linear(768, 2))
    if model == "image": return nn.Sequential(nn.Dropout(.2), nn.Linear(512, 2))
    return nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(.3), nn.Linear(256, 2))


def val_predictions(protocol: str, model: str, seed: int) -> pd.DataFrame:
    checkpoint_path = Path("checkpoints") / protocol / model / f"seed_{seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cache_path = next(path for path in checkpoint["feature_caches"] if Path(path).name.startswith("val_"))
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    head = head_for(model); head.load_state_dict(checkpoint["head_state"]); head.eval()
    with torch.inference_mode(): logits = head(data["features"])
    return pd.DataFrame({
        "sample_id": data["sample_id"], "tweet_id": data["tweet_id"], "event": data["event"], "split": "val",
        "label": data["labels"].numpy(), "pred": logits.argmax(1).numpy(), "prob_informative": logits.softmax(1)[:, 1].numpy(),
        "model": model, "seed": seed, "checkpoint_id": str(checkpoint_path),
    })


def grouped_bootstrap(left: pd.DataFrame, right: pd.DataFrame, metric: str, b: int = 5000) -> tuple[float, float, float]:
    merged = left.merge(right, on=["sample_id", "tweet_id", "label"], suffixes=("_left", "_right"), validate="one_to_one")
    groups = merged.tweet_id.unique(); rng = np.random.default_rng(664); differences = []
    def score(labels, preds):
        labels = np.asarray(labels); preds = np.asarray(preds)
        if metric == "accuracy":
            return float(np.mean(labels == preds))
        if metric == "macro_f1":
            class_scores = []
            for label in (0, 1):
                true_positive = np.sum((labels == label) & (preds == label))
                false_positive = np.sum((labels != label) & (preds == label))
                false_negative = np.sum((labels == label) & (preds != label))
                denominator = 2 * true_positive + false_positive + false_negative
                class_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
            return float(np.mean(class_scores))
        raise ValueError(f"unsupported bootstrap metric: {metric}")
    observed = score(merged.label, merged.pred_left) - score(merged.label, merged.pred_right)
    grouped = {group: rows.index.to_numpy() for group, rows in merged.groupby("tweet_id")}
    for _ in range(b):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([grouped[group] for group in sampled])
        sample = merged.loc[indices]
        differences.append(score(sample.label, sample.pred_left) - score(sample.label, sample.pred_right))
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(observed), float(low), float(high)


def main() -> None:
    run_rows, event_rows, boot_rows = [], [], []
    prediction_cache = {}
    for protocol in PROTOCOLS:
        for model in MODELS:
            for seed in SEEDS:
                summary_path = Path("logs") / protocol / model / f"seed_{seed}" / "summary.json"
                summary = json.loads(summary_path.read_text())
                test = pd.read_csv(summary["prediction_path"], dtype={"sample_id": str, "tweet_id": str})
                prediction_cache[(protocol, model, seed)] = test
                metrics = summary["test"]
                run_rows.append({"protocol": protocol, "model": model, "seed": seed, "n": len(test),
                    "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
                    "val_macro_f1": summary["val"]["macro_f1"], "domain_gap": summary["val"]["macro_f1"] - metrics["macro_f1"],
                    "checkpoint": summary["checkpoint"], "predictions": summary["prediction_path"]})
                validation = val_predictions(protocol, model, seed)
                val_path = Path("results/predictions") / f"{protocol}_{model}_seed{seed}_val.csv"
                validation.to_csv(val_path, index=False)
                if protocol.startswith("ood_"):
                    for event, part in validation.groupby("event"):
                        result = classification_metrics(part.label, part.pred)
                        event_rows.append({"protocol": protocol, "split": "seen_val", "event": event, "model": model, "seed": seed,
                            "n": len(part), "positive_n": int(part.label.sum()), "positive_rate": float(part.label.mean()),
                            "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
                    result = classification_metrics(test.label, test.pred)
                    event_rows.append({"protocol": protocol, "split": "heldout_test", "event": "california_wildfires", "model": model, "seed": seed,
                        "n": len(test), "positive_n": int(test.label.sum()), "positive_rate": float(test.label.mean()),
                        "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
        for seed in SEEDS:
            fusion = prediction_cache[(protocol, "fusion", seed)]
            validation_scores = {
                model: next(row["val_macro_f1"] for row in run_rows
                            if row["protocol"] == protocol and row["model"] == model and row["seed"] == seed)
                for model in ("text", "image")
            }
            best_unimodal = max(validation_scores, key=validation_scores.get)
            for contrast, comparison in (("fusion-text", "text"), ("fusion-image", "image")):
                other = prediction_cache[(protocol, comparison, seed)]
                for metric in ("macro_f1", "accuracy"):
                    difference, low, high = grouped_bootstrap(fusion, other, metric)
                    boot_rows.append({"protocol": protocol, "seed": seed, "contrast": contrast, "comparator": comparison,
                                      "selection_basis": "fixed",
                                      "metric": metric, "difference": difference, "ci95_low": low, "ci95_high": high,
                                      "ci_crosses_zero": bool(low <= 0 <= high), "bootstrap_groups": "tweet_id", "B": 5000})
            # The validation-selected comparator is one of the two fixed
            # unimodal models, so reuse its identical paired resamples rather
            # than recomputing another 5,000 draws.
            source_contrast = f"fusion-{best_unimodal}"
            for metric in ("macro_f1", "accuracy"):
                source = next(row for row in boot_rows if row["protocol"] == protocol and row["seed"] == seed
                              and row["contrast"] == source_contrast and row["metric"] == metric)
                selected = dict(source)
                selected.update({"contrast": "fusion-best_unimodal", "comparator": best_unimodal,
                                 "selection_basis": "validation_macro_f1"})
                boot_rows.append(selected)
    runs = pd.DataFrame(run_rows); events = pd.DataFrame(event_rows); boots = pd.DataFrame(boot_rows)
    table_root = Path("results/tables"); table_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(table_root / "main_runs.csv", index=False)
    for protocol, filename in (("id_official_strict", "id_main.csv"), ("ood_heldout_strict", "ood_main.csv")):
        part = runs[runs.protocol == protocol]
        summary = part.groupby("model").agg(n=("n", "first"), accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), val_macro_f1_mean=("val_macro_f1", "mean"),
            domain_gap_mean=("domain_gap", "mean"), domain_gap_std=("domain_gap", "std")).reset_index()
        summary.to_csv(table_root / filename, index=False)
    events.to_csv(table_root / "per_event.csv", index=False); boots.to_csv(table_root / "paired_bootstrap.csv", index=False)
    figure_root = Path("figures"); figure_root.mkdir(exist_ok=True)
    ood = events.groupby(["split", "event", "model"]).macro_f1.agg(["mean", "std"]).reset_index()
    ordered_events = sorted(ood.event.unique()); x = np.arange(len(ordered_events)); width = .25
    fig, axis = plt.subplots(figsize=(12, 5))
    for index, model in enumerate(MODELS):
        part = ood[ood.model == model].set_index("event").reindex(ordered_events)
        axis.bar(x + (index - 1) * width, part["mean"], width, yerr=part["std"], label=model, capsize=2)
    axis.set_xticks(x, [event.replace("_", "\n") for event in ordered_events]); axis.set_ylim(0, 1); axis.set_ylabel("Macro-F1")
    axis.set_title("Seen-event validation and California held-out test (mean ± seed std)"); axis.legend(); fig.tight_layout()
    fig.savefig(figure_root / "ood_event_f1.png", dpi=300); fig.savefig(figure_root / "ood_event_f1.pdf"); plt.close(fig)
    print(runs.groupby(["protocol", "model"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).to_string())
    print(f"WROTE {table_root / 'main_runs.csv'}, id_main.csv, ood_main.csv, per_event.csv, paired_bootstrap.csv")


if __name__ == "__main__":
    main()
