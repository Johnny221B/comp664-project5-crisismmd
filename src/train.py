from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import PairedCrisisDataset, read_manifest
from src.data.transforms import image_transform
from src.metrics import classification_metrics
from src.models import ImageClassifier, LateFusionClassifier, TextClassifier
from src.utils import append_jsonl, set_seed, sha256


def build_model(config: dict) -> nn.Module:
    if config["model"] == "text":
        return TextClassifier(config["text_model"], config["dropout"])
    if config["model"] == "image":
        return ImageClassifier(config["image_weights"], config["dropout"])
    if config["model"] == "fusion":
        return LateFusionClassifier(config["text_model"], config["image_weights"], config["fusion_hidden"], config["fusion_dropout"])
    raise ValueError(f"unknown model: {config['model']}")


def trainable_head(model: nn.Module) -> nn.Module:
    return model.head


def make_loader(frame: pd.DataFrame, config: dict, tokenizer, shuffle: bool = False) -> DataLoader:
    dataset = PairedCrisisDataset(frame, tokenizer, image_transform(config["image_weights"]), config["max_length"])
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle,
        num_workers=config["num_workers"], pin_memory=True, persistent_workers=config["num_workers"] > 0,
    )


@torch.inference_mode()
def extract_features(model: nn.Module, loader: DataLoader, kind: str, device: torch.device) -> dict:
    model.eval()
    features, labels = [], []
    metadata = {key: [] for key in ("sample_id", "tweet_id", "event", "split")}
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            if kind == "text":
                output = model.encoder(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            elif kind == "image":
                output = model.encoder(batch["image"].to(device))
            else:
                text = model.text_encoder(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                image = model.image_encoder(batch["image"].to(device))
                output = torch.cat([text, image], dim=1)
        features.append(output.float().cpu())
        labels.append(batch["label"].cpu())
        for key in metadata:
            metadata[key].extend(batch[key])
    return {"features": torch.cat(features), "labels": torch.cat(labels), **metadata}


def feature_cache_path(config: dict, split: str, manifest_hash: str, smoke: bool) -> Path:
    signature = json.dumps({
        "manifest": manifest_hash, "model": config["model"], "text_model": config["text_model"],
        "image_weights": config["image_weights"], "max_length": config["max_length"], "image_size": config["image_size"],
        "split": split, "smoke": smoke,
    }, sort_keys=True)
    key = hashlib.sha256(signature.encode()).hexdigest()[:16]
    return Path("data/processed/features") / config["protocol"] / config["model"] / f"{split}_{key}.pt"


def get_features(model, frame, split, config, tokenizer, device, manifest_hash, smoke):
    path = feature_cache_path(config, split, manifest_hash, smoke)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False), path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = extract_features(model, make_loader(frame, config, tokenizer), config["model"], device)
    torch.save(data, path)
    return data, path


def evaluate_head(head: nn.Module, data: dict, device: torch.device, batch_size: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    head.eval()
    logits_all = []
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), batch_size):
            logits_all.append(head(data["features"][start:start + batch_size].to(device)).float().cpu())
    logits = torch.cat(logits_all)
    predictions = logits.argmax(1)
    return classification_metrics(data["labels"].numpy(), predictions.numpy()), logits, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(args.seed, config.get("deterministic", True))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest_hash = sha256(config["manifest"])
    frames = {split: read_manifest(config["manifest"], split) for split in ("train", "val", "test")}
    if args.smoke_test:
        frames["train"] = frames["train"].groupby("label", group_keys=False).head(16).reset_index(drop=True)
        frames["val"] = frames["val"].groupby("label", group_keys=False).head(8).reset_index(drop=True)
        config["max_epochs"], config["patience"], config["batch_size"], config["num_workers"] = 2, 2, 8, 0
    tokenizer = AutoTokenizer.from_pretrained(config["text_model"])
    model = build_model(config).to(device)
    frozen = [name for name, parameter in model.named_parameters() if "head" not in name and parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"encoder parameters unexpectedly trainable: {frozen[:5]}")
    head = trainable_head(model)
    train_data, train_cache = get_features(model, frames["train"], "train", config, tokenizer, device, manifest_hash, args.smoke_test)
    val_data, val_cache = get_features(model, frames["val"], "val", config, tokenizer, device, manifest_hash, args.smoke_test)
    del model
    torch.cuda.empty_cache()
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    run_dir = Path("logs") / config["protocol"] / config["model"] / f"seed_{args.seed}"
    checkpoint_dir = Path("checkpoints") / config["protocol"] / config["model"] / f"seed_{args.seed}"
    if args.smoke_test:
        run_dir, checkpoint_dir = run_dir / "smoke", checkpoint_dir / "smoke"
    log_path, checkpoint_path = run_dir / "metrics.jsonl", checkpoint_dir / "best.pt"
    run_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best, stale = -1.0, 0
    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    for epoch in range(1, config["max_epochs"] + 1):
        head.train(); order = torch.randperm(len(train_data["labels"]), generator=generator); total_loss = 0.0
        for start in range(0, len(order), config["batch_size"]):
            index = order[start:start + config["batch_size"]]
            features = train_data["features"][index].to(device); labels = train_data["labels"][index].to(device)
            optimizer.zero_grad(set_to_none=True); logits = head(features); loss = criterion(logits, labels)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
            loss.backward(); optimizer.step(); total_loss += loss.item() * len(index)
        metrics, _, _ = evaluate_head(head, val_data, device, config["batch_size"])
        record = {"epoch": epoch, "train_loss": total_loss / len(order), **{f"val_{k}": v for k, v in metrics.items() if k != "per_class"}, "elapsed_seconds": time.time() - started}
        append_jsonl(log_path, record); print(json.dumps(record))
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            torch.save({"head_state": head.state_dict(), "config": config, "seed": args.seed, "epoch": epoch, "val_metrics": metrics, "manifest_sha256": manifest_hash, "feature_caches": [str(train_cache), str(val_cache)]}, checkpoint_path)
        else:
            stale += 1
            if stale >= config["patience"]: break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False); head.load_state_dict(checkpoint["head_state"])
    final_val, _, _ = evaluate_head(head, val_data, device, config["batch_size"])
    if args.smoke_test:
        print(json.dumps({"status": "smoke_pass", "checkpoint": str(checkpoint_path), "val": final_val}, indent=2)); return
    # Test is touched only after model selection and best checkpoint reload.
    model = build_model(config).to(device); model.head.load_state_dict(head.state_dict())
    test_data, test_cache = get_features(model, frames["test"], "test", config, tokenizer, device, manifest_hash, False)
    test_metrics, logits, predictions = evaluate_head(head, test_data, device, config["batch_size"])
    probabilities = logits.softmax(1)[:, 1]
    prediction_path = Path("results/predictions") / f"{config['protocol']}_{config['model']}_seed{args.seed}_test.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": test_data["sample_id"], "tweet_id": test_data["tweet_id"], "event": test_data["event"], "split": test_data["split"],
        "label": test_data["labels"].numpy(), "pred": predictions.numpy(), "prob_informative": probabilities.numpy(),
        "model": config["model"], "seed": args.seed, "checkpoint_id": str(checkpoint_path),
    }).to_csv(prediction_path, index=False)
    summary = {"config": config, "seed": args.seed, "checkpoint": str(checkpoint_path), "prediction_path": str(prediction_path), "val": final_val, "test": test_metrics, "test_cache": str(test_cache)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
