#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps


def analyze(protocol: str, seed: int) -> pd.DataFrame:
    frames = {}
    for model in ("text", "image", "fusion"):
        path = Path("results/predictions") / f"{protocol}_{model}_seed{seed}_test.csv"
        frame = pd.read_csv(path, dtype={"sample_id": str, "tweet_id": str})
        frames[model] = frame[["sample_id", "tweet_id", "event", "label", "pred", "prob_informative"]].rename(
            columns={"pred": f"pred_{model}", "prob_informative": f"prob_{model}"})
    merged = frames["text"].merge(frames["image"], on=["sample_id", "tweet_id", "event", "label"], validate="one_to_one").merge(
        frames["fusion"], on=["sample_id", "tweet_id", "event", "label"], validate="one_to_one")
    correct = {model: merged[f"pred_{model}"].eq(merged.label) for model in ("text", "image", "fusion")}
    merged["category"] = "mixed_other"
    merged.loc[correct["text"] & correct["image"] & correct["fusion"], "category"] = "all_correct"
    merged.loc[~correct["text"] & ~correct["image"] & ~correct["fusion"], "category"] = "all_wrong"
    merged.loc[correct["fusion"] & (~correct["text"] | ~correct["image"]), "category"] = "fusion_rescue"
    merged.loc[correct["fusion"] & ~correct["text"] & ~correct["image"], "category"] = "fusion_strong_rescue"
    merged.loc[~correct["fusion"] & (correct["text"] | correct["image"]), "category"] = "fusion_regression"
    merged.loc[~correct["fusion"] & correct["text"] & correct["image"], "category"] = "fusion_strong_regression"
    merged["protocol"], merged["seed"] = protocol, seed
    manifest = pd.read_csv(Path("data/manifests") / f"{protocol}.csv", dtype={"sample_id": str, "tweet_id": str})
    return merged.merge(manifest[["sample_id", "text", "image_path"]], on="sample_id", validate="one_to_one")


def confidence(frame: pd.DataFrame) -> pd.Series:
    return frame.prob_fusion.where(frame.label.eq(1), 1 - frame.prob_fusion)


def case_figure(selected: pd.DataFrame, path: Path, title: str) -> None:
    if selected.empty: return
    fig, axes = plt.subplots(1, len(selected), figsize=(3.2 * len(selected), 3.8), squeeze=False)
    for axis, row in zip(axes[0], selected.itertuples()):
        with Image.open(row.image_path) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.axis("off"); axis.set_title(f"{row.event}\ny={row.label}; T/I/F={row.pred_text}/{row.pred_image}/{row.pred_fusion}", fontsize=8)
    fig.suptitle(title); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    all_cases = pd.concat([analyze(protocol, args.seed) for protocol in ("id_official_strict", "ood_heldout_strict")], ignore_index=True)
    all_cases.to_csv("results/tables/fusion_cases.csv", index=False)
    selected_parts, report = [], ["# Fusion error analysis", "", f"Seed: {args.seed}. Categories are computed directly from aligned prediction CSVs.", ""]
    for protocol, data in all_cases.groupby("protocol"):
        counts = data.category.value_counts(); report += [f"## {protocol}", "", counts.to_markdown(), ""]
        for category in ("fusion_strong_rescue", "fusion_rescue", "fusion_strong_regression", "fusion_regression"):
            candidates = data[data.category == category].copy()
            candidates["fusion_true_confidence"] = confidence(candidates)
            candidates = candidates.sort_values("fusion_true_confidence", ascending=category.startswith("fusion_rescue")).head(10)
            selected_parts.append(candidates)
            case_figure(candidates.head(5), Path("figures/cases") / f"{protocol}_{category}.png", f"{protocol}: {category}")
            report += [f"### {category}", "", f"Count: {int(counts.get(category, 0))}. Selected examples are in `results/tables/fusion_cases_selected.csv`.", ""]
    pd.concat(selected_parts, ignore_index=True).to_csv("results/tables/fusion_cases_selected.csv", index=False)
    Path("reports/error_analysis.md").write_text("\n".join(report) + "\n")
    counts = all_cases.groupby(["protocol", "category"]).size().unstack(fill_value=0)
    axis = counts.plot(kind="bar", figsize=(10, 5)); axis.set_ylabel("Samples"); axis.set_title("Fusion success and failure categories, seed 42")
    axis.figure.tight_layout(); axis.figure.savefig("figures/fusion_case_counts.png", dpi=300); axis.figure.savefig("figures/fusion_case_counts.pdf"); plt.close(axis.figure)
    print(counts.to_string()); print("WROTE fusion_cases.csv, selected cases, report, and case figures")


if __name__ == "__main__":
    main()
