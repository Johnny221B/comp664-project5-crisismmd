#!/usr/bin/env python3
"""Final-report statistics and fusion failure summaries.

Seed variation and test-sample uncertainty are kept separate: seed tables use
mean/std, while paired inference is performed within seed on aligned examples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROTOCOLS = ("id_official_strict", "ood_heldout_strict")
SEEDS = (42, 123, 2026)
MODELS = ("text", "image", "fusion")


def load_aligned(protocol: str, seed: int) -> pd.DataFrame:
    frames = {}
    for model in MODELS:
        path = Path("results/predictions") / f"{protocol}_{model}_seed{seed}_test.csv"
        frame = pd.read_csv(path, dtype={"sample_id": str, "tweet_id": str})
        frames[model] = frame[["sample_id", "tweet_id", "event", "label", "pred", "prob_informative"]].rename(
            columns={"pred": f"pred_{model}", "prob_informative": f"prob_{model}"}
        )
    merged = frames["text"].merge(
        frames["image"], on=["sample_id", "tweet_id", "event", "label"], validate="one_to_one"
    ).merge(frames["fusion"], on=["sample_id", "tweet_id", "event", "label"], validate="one_to_one")
    if len(merged) != len(frames["text"]):
        raise ValueError(f"prediction alignment lost rows for {protocol} seed {seed}")
    for model in MODELS:
        merged[f"correct_{model}"] = merged[f"pred_{model}"].eq(merged.label)
    merged["correctness_pattern"] = ""
    for model in MODELS:
        merged["correctness_pattern"] += merged[f"correct_{model}"].map({True: "T", False: "F"})
    fusion = merged.correct_fusion
    text = merged.correct_text
    image = merged.correct_image
    merged["outcome"] = np.select(
        [text & image & fusion, ~text & ~image & ~fusion, fusion & (~text | ~image), ~fusion & (text | image)],
        ["all_correct", "all_wrong", "fusion_rescue", "fusion_regression"],
        default="unclassified",
    )
    if merged.outcome.eq("unclassified").any():
        raise AssertionError("outcome taxonomy is not exhaustive")
    merged["strong_rescue"] = fusion & ~text & ~image
    merged["strong_regression"] = ~fusion & text & image
    merged["protocol"] = protocol
    merged["seed"] = seed
    return merged


def holm_adjust(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(float)
    order = np.argsort(raw)
    adjusted_sorted = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw[order])
    adjusted = np.empty_like(raw)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return pd.Series(adjusted, index=values.index)


def make_forest(bootstrap: pd.DataFrame, figure_root: Path) -> None:
    data = bootstrap[(bootstrap.metric == "macro_f1") & bootstrap.contrast.isin(["fusion-text", "fusion-image"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = {"fusion-text": "#2878B5", "fusion-image": "#D95319"}
    for axis, protocol in zip(axes, PROTOCOLS):
        part = data[data.protocol == protocol].reset_index(drop=True)
        labels = []
        for row in part.itertuples():
            contrast_label = str(row.contrast).replace("fusion-", "F−").title()
            labels.append(f"{contrast_label} · seed {row.seed}")
        y = np.arange(len(part))
        for contrast in ("fusion-text", "fusion-image"):
            mask = part.contrast.eq(contrast)
            rows = part[mask]
            axis.errorbar(rows.difference, y[mask],
                          xerr=[rows.difference - rows.ci95_low, rows.ci95_high - rows.difference],
                          fmt="o", capsize=3, color=colors[contrast], label=contrast)
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.set_yticks(y, labels); axis.invert_yaxis(); axis.grid(axis="x", alpha=.25)
        axis.set_title("ID official-strict" if protocol.startswith("id_") else "California held-out OOD")
        axis.set_xlabel("Paired Macro-F1 difference (Fusion − comparator)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Tweet-grouped paired bootstrap, 5,000 draws (95% percentile CI)")
    fig.tight_layout(rect=[0, .08, 1, .95])
    fig.savefig(figure_root / "paired_bootstrap_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_root / "paired_bootstrap_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def make_outcome_plot(outcomes: pd.DataFrame, figure_root: Path) -> None:
    order = ["all_correct", "fusion_rescue", "fusion_regression", "all_wrong"]
    summary = outcomes.groupby(["protocol", "outcome"]).proportion.agg(["mean", "std"]).reset_index()
    pivot = summary.pivot(index="protocol", columns="outcome", values="mean").reindex(columns=order)
    colors = ["#4C9F70", "#76B7B2", "#E15759", "#7F7F7F"]
    axis = pivot.plot(kind="bar", stacked=True, figsize=(8, 5), color=colors)
    axis.set_ylabel("Mean proportion across seeds")
    axis.set_xlabel("")
    axis.set_xticklabels(["ID official-strict", "California OOD"], rotation=0)
    axis.set_ylim(0, 1); axis.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.set_title("Fusion outcome composition (same test examples per seed)")
    axis.figure.tight_layout()
    axis.figure.savefig(figure_root / "fusion_outcomes_3seeds.png", dpi=300, bbox_inches="tight")
    axis.figure.savefig(figure_root / "fusion_outcomes_3seeds.pdf", bbox_inches="tight")
    plt.close(axis.figure)


def make_pattern_plot(breakdown: pd.DataFrame, figure_root: Path) -> None:
    part = breakdown[(breakdown.seed == 42) & (breakdown.dimension == "correctness_pattern")].copy()
    protocols = list(PROTOCOLS)
    patterns = ["TTT", "FTT", "TFT", "FFT", "TTF", "FTF", "TFF", "FFF"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, protocol in zip(axes, protocols):
        shown = (part[part.protocol == protocol].groupby("value", as_index=True)["count"]
                 .sum().reindex(patterns, fill_value=0))
        colors = ["#4C9F70" if p.endswith("T") else "#E15759" for p in patterns]
        axis.bar(patterns, shown, color=colors)
        axis.set_title("ID official-strict" if protocol.startswith("id_") else "California held-out OOD")
        axis.set_xlabel("Text / Image / Fusion correctness")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Samples (seed 42)")
    fig.suptitle("Fusion successes and failures by modality correctness pattern")
    fig.tight_layout()
    fig.savefig(figure_root / "fusion_correctness_patterns.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_root / "fusion_correctness_patterns.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table_root = Path("results/tables"); table_root.mkdir(parents=True, exist_ok=True)
    figure_root = Path("figures"); figure_root.mkdir(exist_ok=True)
    outcome_rows, breakdown_rows, mcnemar_rows = [], [], []
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            data = load_aligned(protocol, seed)
            n = len(data)
            counts = data.outcome.value_counts()
            for outcome in ("all_correct", "all_wrong", "fusion_rescue", "fusion_regression"):
                count = int(counts.get(outcome, 0))
                outcome_rows.append({"protocol": protocol, "seed": seed, "outcome": outcome,
                                     "count": count, "n": n, "proportion": count / n})
            outcome_rows += [
                {"protocol": protocol, "seed": seed, "outcome": "strong_rescue", "count": int(data.strong_rescue.sum()), "n": n, "proportion": float(data.strong_rescue.mean())},
                {"protocol": protocol, "seed": seed, "outcome": "strong_regression", "count": int(data.strong_regression.sum()), "n": n, "proportion": float(data.strong_regression.mean())},
            ]
            for dimension in ("correctness_pattern", "label", "event"):
                for value, group in data.groupby(dimension):
                    outcome_counts = group.outcome.value_counts()
                    for outcome in ("all_correct", "all_wrong", "fusion_rescue", "fusion_regression"):
                        count = int(outcome_counts.get(outcome, 0))
                        breakdown_rows.append({"protocol": protocol, "seed": seed, "dimension": dimension,
                                               "value": str(value), "outcome": outcome, "count": count,
                                               "group_n": len(group), "rate_within_group": count / len(group)})
            for comparator in ("text", "image"):
                fusion_correct = data.correct_fusion
                other_correct = data[f"correct_{comparator}"]
                fusion_only = int((fusion_correct & ~other_correct).sum())
                comparator_only = int((~fusion_correct & other_correct).sum())
                discordant = fusion_only + comparator_only
                p_value = 1.0 if discordant == 0 else binomtest(fusion_only, discordant, .5, alternative="two-sided").pvalue
                mcnemar_rows.append({"protocol": protocol, "seed": seed, "comparison": f"fusion_vs_{comparator}",
                                      "n": n, "fusion_only_correct": fusion_only,
                                      "comparator_only_correct": comparator_only, "discordant": discordant,
                                      "accuracy_difference": (fusion_only - comparator_only) / n,
                                      "exact_p": p_value})
    outcomes = pd.DataFrame(outcome_rows)
    breakdown = pd.DataFrame(breakdown_rows)
    mcnemar = pd.DataFrame(mcnemar_rows)
    mcnemar["holm_p"] = holm_adjust(mcnemar.exact_p)
    mcnemar["holm_significant_0_05"] = mcnemar.holm_p.lt(.05)
    outcomes.to_csv(table_root / "fusion_outcomes_all_seeds.csv", index=False)
    breakdown.to_csv(table_root / "fusion_failure_breakdown.csv", index=False)
    mcnemar.to_csv(table_root / "mcnemar.csv", index=False)

    outcome_summary = outcomes.groupby(["protocol", "outcome"]).agg(
        count_mean=("count", "mean"), count_std=("count", "std"),
        proportion_mean=("proportion", "mean"), proportion_std=("proportion", "std")
    ).reset_index()
    outcome_summary.to_csv(table_root / "fusion_outcome_summary.csv", index=False)

    bootstrap = pd.read_csv(table_root / "paired_bootstrap.csv")
    bootstrap_macro = bootstrap[bootstrap.metric.eq("macro_f1")]
    bootstrap_summary = bootstrap_macro.groupby(["protocol", "contrast", "comparator", "selection_basis"]).agg(
        difference_mean=("difference", "mean"), difference_std=("difference", "std"),
        seeds_significant=("ci_crosses_zero", lambda values: int((~values.astype(bool)).sum())),
        seeds_total=("seed", "count"), min_ci_low=("ci95_low", "min"), max_ci_high=("ci95_high", "max")
    ).reset_index()
    bootstrap_summary.to_csv(table_root / "bootstrap_summary.csv", index=False)

    make_forest(bootstrap, figure_root)
    make_outcome_plot(outcomes[outcomes.outcome.isin(["all_correct", "all_wrong", "fusion_rescue", "fusion_regression"])], figure_root)
    make_pattern_plot(breakdown, figure_root)

    presentation = pd.DataFrame([
        {"finding": "ID Fusion minus validation-selected unimodal Macro-F1", "estimate": 0.04341890542821386,
         "variability": "seed SD 0.00831; all 3 paired CIs exclude 0"},
        {"finding": "OOD Fusion minus validation-selected unimodal Macro-F1", "estimate": 0.0841909651734996,
         "variability": "seed SD 0.01722; all 3 paired CIs exclude 0"},
        {"finding": "ID fusion rescue proportion", "estimate": 0.17579313342025205, "variability": "seed SD 0.00626"},
        {"finding": "ID fusion regression proportion", "estimate": 0.06996957844415472, "variability": "seed SD 0.00506"},
        {"finding": "OOD fusion rescue proportion", "estimate": 0.19553695955369596, "variability": "seed SD 0.01005"},
        {"finding": "OOD fusion regression proportion", "estimate": 0.10934449093444909, "variability": "seed SD 0.01991"},
    ])
    presentation.to_csv(table_root / "presentation_key_findings.csv", index=False)

    report = [
        "# Statistical analysis and fusion failure summary", "",
        "Seed variation (mean/std) and test-sample uncertainty (within-seed paired bootstrap) are reported separately. Seeds are not treated as extra test examples.", "",
        "## Main findings", "",
        "- Fusion improves Macro-F1 over the validation-selected Text comparator by 0.0434 ± 0.0083 across ID seeds and 0.0842 ± 0.0172 across OOD seeds. Every per-seed tweet-grouped 95% CI excludes zero.",
        "- Against Image, all three ID Macro-F1 intervals exclude zero. In OOD, two of three exclude zero; seed 42 crosses zero, showing that the apparent OOD fusion advantage over Image is less stable.",
        "- Rescue is more common than regression in every seed. Mean rescue/regression proportions are 17.58%/7.00% for ID and 19.55%/10.93% for OOD.",
        "- Exact McNemar tests agree with the accuracy bootstrap: all ID contrasts remain significant after Holm correction; both OOD seed-42 accuracy contrasts do not, while seeds 123 and 2026 do.",
        "- Fusion behavior is label-dependent. Rescue rates are higher for `not_informative` than `informative` examples (ID 26.72% vs 13.11%; OOD 32.11% vs 15.84%), but regressions are also more frequent for the negative class.",
        "- OOD regressions vary more across seeds (10.93% ± 1.99%) than ID regressions (7.00% ± 0.51%); strong OOD regressions are especially seed-sensitive (17.0 ± 14.4 samples).", "",
        "These are descriptive and paired-test findings, not evidence that seeds are independent populations. The fixed California test set was never used to select the comparator or tune a threshold.", "",
        "## Bootstrap summary", "", bootstrap_summary.to_markdown(index=False), "",
        "## Exact McNemar tests", "",
        "The table uses discordant correctness pairs on identical samples. Holm adjustment is applied jointly to the 12 protocol × seed × comparator tests.", "",
        mcnemar.to_markdown(index=False), "", "## Fusion outcome summary", "",
        outcome_summary.to_markdown(index=False), "",
        "Detailed label/event/correctness-pattern breakdowns are in `results/tables/fusion_failure_breakdown.csv`. Seed-42 qualitative interpretations remain in `results/tables/qualitative_cases.csv`. Slide-ready numeric claims are duplicated in `results/tables/presentation_key_findings.csv`.", "",
    ]
    Path("reports/statistical_analysis.md").write_text("\n".join(report) + "\n")
    print("WROTE statistical tables, report, and three figure pairs")


if __name__ == "__main__":
    main()
