#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    rows=[]
    for protocol in ("id_official_strict","ood_heldout_strict"):
        for variant in ("presence","dropout"):
            for seed in (42,123,2026):
                summary=json.loads((Path("logs")/protocol/f"fusion_{variant}"/f"seed_{seed}"/"summary.json").read_text())
                for condition,metrics in summary["conditions"].items():
                    rows.append({"protocol":protocol,"variant":variant,"seed":seed,"condition":condition,"accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"],"best_epoch":summary["best_epoch"]})
    runs=pd.DataFrame(rows); runs.to_csv("results/tables/modality_dropout_runs.csv",index=False)
    summary=runs.groupby(["protocol","variant","condition"]).agg(accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std")).reset_index()
    summary.to_csv("results/tables/modality_dropout.csv",index=False)
    for protocol in runs.protocol.unique():
        part=summary[summary.protocol==protocol]; conditions=["both","text_only","image_only"]; x=np.arange(3); width=.35
        fig,axis=plt.subplots(figsize=(7,4.5))
        for index,variant in enumerate(("presence","dropout")):
            values=part[part.variant==variant].set_index("condition").loc[conditions]
            axis.bar(x+(index-.5)*width,values.macro_f1_mean,width,yerr=values.macro_f1_std,label=variant,capsize=3)
        axis.set_xticks(x,conditions); axis.set_ylim(0,1); axis.set_ylabel("Macro-F1 mean ± seed std"); axis.set_title(protocol); axis.legend(); fig.tight_layout()
        stem=f"figures/modality_dropout_{'id' if protocol.startswith('id_') else 'ood'}"; fig.savefig(stem+".png",dpi=300); fig.savefig(stem+".pdf"); plt.close(fig)
    comparison=summary.pivot(index=["protocol","condition"],columns="variant",values="macro_f1_mean").reset_index(); comparison["dropout_minus_presence"]=comparison.dropout-comparison.presence
    comparison.to_csv("results/tables/modality_dropout_comparison.csv",index=False)
    print(summary.to_string(index=False)); print("\nDifferences\n",comparison.to_string(index=False))


if __name__=="__main__": main()
