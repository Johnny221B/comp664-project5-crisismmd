# M1 status — seed 42 ID baseline

Date: 2026-10-25

## Protocol

- Manifest: `data/manifests/id_official_strict.csv`
- Train/validation/test: 9,535 / 1,570 / 1,534 pairs.
- Released official assignments are retained except documented content-level
  duplicates removed from lower-priority splits. No test samples were removed.
- All three models use exactly the same paired sample IDs.

## Models

| Model | Frozen feature | Trainable head | Trainable parameters |
|---|---:|---|---:|
| Text-only | DistilBERT CLS, 768 | Dropout(0.2) + Linear(768,2) | 1,538 |
| Image-only | ResNet-18 pooled, 512 | Dropout(0.2) + Linear(512,2) | 1,026 |
| Late Fusion | concatenated 1,280 | Linear(1280,256) + ReLU + Dropout(0.3) + Linear(256,2) | 328,450 |

Common budget: seed 42, AdamW, LR 1e-3, weight decay 1e-4, batch 64,
maximum 25 epochs, patience 5, validation macro-F1 checkpoint selection.

## Results

| Model | Best epoch | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Text-only | 12 | 0.812361 | 0.859192 | 0.838593 |
| Image-only | 12 | 0.792351 | 0.824641 | 0.799942 |
| Late Fusion | 2 | 0.870882 | 0.887223 | 0.873877 |

Seed-42 differences: Fusion − Text macro-F1 = +0.035284; Fusion − Image
= +0.073935. These are single-seed results, not the final three-seed table.

## Verification

- Three GPU smoke tests passed and produced reloadable checkpoints.
- Full runs selected checkpoints only on validation macro-F1; test feature
  extraction occurred after best-checkpoint reload.
- Prediction sample IDs and labels align exactly across all three models.
- Final unit test run: 15 passed.

## Commands

```bash
python -m src.train --config configs/id_text.yaml --seed 42 --device cuda:0
python -m src.train --config configs/id_image.yaml --seed 42 --device cuda:1
python -m src.train --config configs/id_fusion.yaml --seed 42 --device cuda:2
python scripts/compare_models.py --protocol id_official_strict --seed 42
python -m pytest -q
```
