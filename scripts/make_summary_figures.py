#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def bars(table_path: str, output_stem: str, title: str) -> None:
    data = pd.read_csv(table_path).set_index("model").loc[["text", "image", "fusion"]].reset_index()
    x = np.arange(len(data)); width = .34
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(x-width/2, data.accuracy_mean, width, yerr=data.accuracy_std, label="Accuracy", capsize=3)
    axis.bar(x+width/2, data.macro_f1_mean, width, yerr=data.macro_f1_std, label="Macro-F1", capsize=3)
    axis.set_xticks(x, data.model); axis.set_ylim(0, 1); axis.set_ylabel("Mean score ± seed std"); axis.set_title(title); axis.legend(); fig.tight_layout()
    fig.savefig(f"figures/{output_stem}.png", dpi=300); fig.savefig(f"figures/{output_stem}.pdf"); plt.close(fig)


def confusions(protocol: str, prefix: str, seed: int = 42) -> None:
    for model in ("text", "image", "fusion"):
        frame = pd.read_csv(f"results/predictions/{protocol}_{model}_seed{seed}_test.csv")
        matrix = confusion_matrix(frame.label, frame.pred, labels=[0, 1])
        fig, axis = plt.subplots(figsize=(4.5, 4)); shown = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2): axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(xticks=[0,1], yticks=[0,1], xlabel="Predicted label", ylabel="True label", title=f"{protocol}: {model}, seed {seed}, n={len(frame)}")
        fig.colorbar(shown, ax=axis); fig.tight_layout(); stem=f"figures/{prefix}_confusion_{model}_seed{seed}"
        fig.savefig(stem+".png", dpi=300); fig.savefig(stem+".pdf"); plt.close(fig)


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    bars("results/tables/id_main.csv", "id_main_3seeds", "ID official-strict test (3 seeds)")
    bars("results/tables/ood_main.csv", "ood_main_3seeds", "California held-out test (3 seeds)")
    confusions("ood_heldout_strict", "ood")
    print("WROTE three-seed summary figures and OOD confusion matrices")


if __name__ == "__main__": main()
