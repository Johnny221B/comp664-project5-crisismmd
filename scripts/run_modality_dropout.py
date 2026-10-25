#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
import yaml

from src.metrics import classification_metrics
from src.models.modality_dropout import PresenceAwareFusionHead, apply_modality_dropout, condition_features
from src.utils import append_jsonl, set_seed


def fusion_caches(protocol: str) -> dict[str, dict]:
    checkpoint = torch.load(Path("checkpoints") / protocol / "fusion" / "seed_42" / "best.pt", map_location="cpu", weights_only=False)
    paths = {Path(path).name.split("_")[0]: path for path in checkpoint["feature_caches"]}
    test_candidates = sorted((Path("data/processed/features") / protocol / "fusion").glob("test_*.pt"))
    paths["test"] = str(max(test_candidates, key=lambda path: path.stat().st_size))
    return {split: torch.load(path, map_location="cpu", weights_only=False) for split, path in paths.items()}


@torch.inference_mode()
def evaluate(head, data, condition, device, batch_size):
    head.eval(); outputs=[]
    for start in range(0, len(data["labels"]), batch_size):
        features, presence = condition_features(data["features"][start:start+batch_size].to(device), condition)
        outputs.append(head(features, presence).cpu())
    logits=torch.cat(outputs); predictions=logits.argmax(1)
    return classification_metrics(data["labels"].numpy(), predictions.numpy()), logits, predictions


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("presence", "dropout"), required=True); parser.add_argument("--device", default="cuda:0"); args=parser.parse_args()
    config=yaml.safe_load(Path("configs/dropout_fusion.yaml").read_text()); set_seed(args.seed)
    probability = 0.0 if args.variant == "presence" else float(config["modality_dropout_probability"])
    data=fusion_caches(args.protocol); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    head=PresenceAwareFusionHead(config["hidden"], config["dropout"]).to(device)
    optimizer=torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]); criterion=nn.CrossEntropyLoss()
    model_name=f"fusion_{args.variant}"; log_dir=Path("logs")/args.protocol/model_name/f"seed_{args.seed}"; ckpt_dir=Path("checkpoints")/args.protocol/model_name/f"seed_{args.seed}"
    log_dir.mkdir(parents=True,exist_ok=True); ckpt_dir.mkdir(parents=True,exist_ok=True); log_path=log_dir/"metrics.jsonl"; best_path=ckpt_dir/"best.pt"
    best=-1.0; stale=0; order_generator=torch.Generator().manual_seed(args.seed); mask_generator=torch.Generator().manual_seed(args.seed+664)
    for epoch in range(1,config["max_epochs"]+1):
        head.train(); order=torch.randperm(len(data["train"]["labels"]),generator=order_generator); total=0.0
        for start in range(0,len(order),config["batch_size"]):
            index=order[start:start+config["batch_size"]]; features=data["train"]["features"][index]
            masked,presence=apply_modality_dropout(features,probability,mask_generator); labels=data["train"]["labels"][index].to(device)
            optimizer.zero_grad(set_to_none=True); logits=head(masked.to(device),presence.to(device)); loss=criterion(logits,labels); loss.backward(); optimizer.step(); total+=loss.item()*len(index)
        metrics,_,_=evaluate(head,data["val"],"both",device,config["batch_size"]); record={"epoch":epoch,"train_loss":total/len(order),"val_macro_f1":metrics["macro_f1"],"val_accuracy":metrics["accuracy"]}
        append_jsonl(log_path,record); print(json.dumps(record))
        if metrics["macro_f1"]>best:
            best=metrics["macro_f1"]; stale=0; torch.save({"head_state":head.state_dict(),"config":config,"protocol":args.protocol,"variant":args.variant,"seed":args.seed,"epoch":epoch,"val_metrics":metrics},best_path)
        else:
            stale+=1
            if stale>=config["patience"]: break
    checkpoint=torch.load(best_path,map_location=device,weights_only=False); head.load_state_dict(checkpoint["head_state"]); summary={"protocol":args.protocol,"variant":args.variant,"seed":args.seed,"checkpoint":str(best_path),"best_epoch":checkpoint["epoch"],"modality_dropout_probability":probability,"conditions":{}}
    for condition in ("both","text_only","image_only"):
        metrics,logits,predictions=evaluate(head,data["test"],condition,device,config["batch_size"]); summary["conditions"][condition]=metrics
        output=Path("results/predictions")/f"{args.protocol}_{model_name}_seed{args.seed}_{condition}_test.csv"
        pd.DataFrame({"sample_id":data["test"]["sample_id"],"tweet_id":data["test"]["tweet_id"],"event":data["test"]["event"],"split":"test","condition":condition,
            "label":data["test"]["labels"].numpy(),"pred":predictions.numpy(),"prob_informative":logits.softmax(1)[:,1].numpy(),"model":model_name,"seed":args.seed,"checkpoint_id":str(best_path)}).to_csv(output,index=False)
    (log_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()

