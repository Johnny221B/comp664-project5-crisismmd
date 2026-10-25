#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
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
    # These four outcomes are mutually exclusive and exhaustive. Strong cases
    # are subtypes, not extra top-level categories.
    merged["category"] = "unclassified"
    merged.loc[correct["text"] & correct["image"] & correct["fusion"], "category"] = "all_correct"
    merged.loc[~correct["text"] & ~correct["image"] & ~correct["fusion"], "category"] = "all_wrong"
    merged.loc[correct["fusion"] & (~correct["text"] | ~correct["image"]), "category"] = "fusion_rescue"
    merged.loc[~correct["fusion"] & (correct["text"] | correct["image"]), "category"] = "fusion_regression"
    if merged.category.eq("unclassified").any():
        raise AssertionError("fusion outcome taxonomy is not exhaustive")
    merged["correctness_pattern"] = (
        correct["text"].map({True: "T", False: "F"})
        + correct["image"].map({True: "T", False: "F"})
        + correct["fusion"].map({True: "T", False: "F"})
    )
    merged["strong_rescue"] = correct["fusion"] & ~correct["text"] & ~correct["image"]
    merged["strong_regression"] = ~correct["fusion"] & correct["text"] & correct["image"]
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


CURATED_NOTES = {
    ("id_official_strict", "910586004784009216_0"): "Strong rescue: Formula-1/no-damage context triggers both unimodal false positives; the joint head suppresses them.",
    ("id_official_strict", "913419277830520833_0"): "Text-led rescue: survivor-impact wording is informative while the image branch is strongly negative.",
    ("id_official_strict", "913515779605266433_0"): "Strong regression: both unimodal scores are positive, but the nonlinear fusion head crosses to the wrong side of 0.5.",
    ("id_official_strict", "915287630723665932_0"): "Image-led regression: text correctly treats the political photo caption as non-informative; image evidence dominates fusion.",
    ("ood_heldout_strict", "923324482143518721_0"): "OOD strong rescue: wildfire terms cause two false positives for downstream wine-supply news; fusion recovers label 0.",
    ("ood_heldout_strict", "921097679345733634_0"): "OOD image-led rescue: text misses the informative item; image evidence moves fusion just above the decision threshold.",
    ("ood_heldout_strict", "920045440594268160_0"): "OOD strong regression: both modalities are individually correct, but fusion is unstable at 0.498 under domain shift.",
    ("ood_heldout_strict", "917861564002181120_0"): "OOD conflict regression: highly positive text is pulled below threshold by a negative image prediction.",
}


def curated_figure(selected: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), squeeze=False)
    for axis, row in zip(axes.flat, selected.itertuples()):
        with Image.open(row.image_path) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.axis("off")
        text = textwrap.shorten(str(row.text), width=95, placeholder="...")
        protocol_label = str(row.protocol).replace("_strict", "")
        axis.set_title(
            f"{protocol_label} | {row.category} | {row.correctness_pattern}\n"
            f"y={row.label}; T/I/F={row.pred_text}/{row.pred_image}/{row.pred_fusion}\n{text}",
            fontsize=7,
        )
    fig.suptitle("Representative fusion rescues and regressions (seed 42)")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    all_cases = pd.concat([analyze(protocol, args.seed) for protocol in ("id_official_strict", "ood_heldout_strict")], ignore_index=True)
    all_cases.to_csv("results/tables/fusion_cases.csv", index=False)
    selected_parts, report = [], ["# Fusion error analysis", "", f"Seed: {args.seed}. Categories are computed directly from aligned prediction CSVs.", ""]
    for protocol, data in all_cases.groupby("protocol"):
        counts = data.category.value_counts(); report += [f"## {protocol}", "", counts.to_markdown(), ""]
        report += [
            "The four rows above are mutually exclusive and exhaustive. `strong_rescue` means both unimodal models were wrong while fusion was correct; `strong_regression` means both unimodal models were correct while fusion was wrong.",
            "",
            f"Strong rescue subset: {int(data.strong_rescue.sum())}; strong regression subset: {int(data.strong_regression.sum())}.",
            "",
        ]
        for category in ("fusion_rescue", "fusion_regression"):
            candidates = data[data.category == category].copy()
            candidates["fusion_true_confidence"] = confidence(candidates)
            candidates = candidates.sort_values("fusion_true_confidence", ascending=category.startswith("fusion_rescue")).head(10)
            selected_parts.append(candidates)
            case_figure(candidates.head(5), Path("figures/cases") / f"{protocol}_{category}.png", f"{protocol}: {category}")
            report += [f"### {category}", "", f"Count: {int(counts.get(category, 0))}. Selected examples are in `results/tables/fusion_cases_selected.csv`.", ""]
    pd.concat(selected_parts, ignore_index=True).to_csv("results/tables/fusion_cases_selected.csv", index=False)
    case_keys = list(zip(all_cases.protocol, all_cases.sample_id))
    curated = all_cases[[key in CURATED_NOTES for key in case_keys]].copy()
    curated_keys = list(zip(curated.protocol, curated.sample_id))
    curated["qualitative_note"] = [CURATED_NOTES[key] for key in curated_keys]
    order = {key: index for index, key in enumerate(CURATED_NOTES)}
    curated["display_order"] = [order[key] for key in curated_keys]
    curated = curated.sort_values("display_order").drop(columns="display_order")
    if len(curated) != len(CURATED_NOTES):
        missing = set(CURATED_NOTES) - set(zip(curated.protocol, curated.sample_id))
        raise AssertionError(f"missing curated samples: {sorted(missing)}")
    curated.to_csv("results/tables/qualitative_cases.csv", index=False)
    curated_figure(curated, Path("figures/cases/qualitative_cases.png"))
    report += ["## Curated presentation cases", "", "Eight representative cases (four rescues and four regressions across ID/OOD) are recorded with researcher interpretations in `results/tables/qualitative_cases.csv` and rendered in `figures/cases/qualitative_cases.png`. These notes are interpretations, not dataset annotations.", ""]
    Path("reports/error_analysis.md").write_text("\n".join(report) + "\n")
    counts = all_cases.groupby(["protocol", "category"]).size().unstack(fill_value=0)
    axis = counts.plot(kind="bar", figsize=(10, 5)); axis.set_ylabel("Samples"); axis.set_title("Fusion success and failure categories, seed 42")
    axis.figure.tight_layout(); axis.figure.savefig("figures/fusion_case_counts.png", dpi=300); axis.figure.savefig("figures/fusion_case_counts.pdf"); plt.close(axis.figure)
    print(counts.to_string()); print("WROTE fusion cases, eight curated cases, report, and case figures")


if __name__ == "__main__":
    main()
