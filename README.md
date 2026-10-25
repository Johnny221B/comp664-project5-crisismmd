# COMP 664 Project 5 — CrisisMMD

Reproducible multimodal classification of CrisisMMD v2.0 posts as informative
or not informative. This repository implements the official text-only,
image-only, and late-fusion baselines; evaluates in-distribution (ID) and
California Wildfires held-out-event (OOD) splits; and adds modality dropout.

## Completed milestones

- **M0:** environment capture, data audit, image validation, leakage checks,
  and strict manifests.
- **M1:** frozen DistilBERT and ResNet-18 feature pipelines, three classifiers,
  deterministic training, and tests.
- **M2:** three-seed ID and OOD baseline experiments.
- **M3:** aggregation, paired bootstrap tests, per-event results, domain-gap and
  fusion error analyses.
- **M4:** presence-indicator control versus modality dropout under both-, text-
  only-, and image-only-input conditions.

All runs retain resolved configuration, logs, checkpoints, and predictions.
Raw media, pretrained weights, feature caches, and the environment are excluded.

## Main results

Mean ± sample standard deviation over seeds 42, 123, and 2026:

| Protocol | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| ID strict | Text | 0.8557 ± 0.0040 | 0.8360 ± 0.0032 |
| ID strict | Image | 0.8253 ± 0.0011 | 0.8013 ± 0.0025 |
| ID strict | Late fusion | **0.8931 ± 0.0051** | **0.8794 ± 0.0052** |
| California OOD strict | Text | 0.7891 ± 0.0065 | 0.6735 ± 0.0027 |
| California OOD strict | Image | 0.7883 ± 0.0087 | 0.7260 ± 0.0078 |
| California OOD strict | Late fusion | **0.8265 ± 0.0182** | **0.7577 ± 0.0167** |

M4 has a mixed trade-off: modality dropout substantially improves text-only
robustness, while slightly reducing normal two-modality performance and not
improving OOD image-only performance. See `reports/m4_status.md`.

## Environment

The recorded server used Python 3.11, PyTorch 2.11.0+cu128, torchvision
0.26.0+cu128, and eight NVIDIA RTX A5000 GPUs. Exact snapshots are in
`logs/m0/`; dependencies are in the requirement files.

```bash
conda create -p .venv python=3.11 -y
conda run -p .venv pip install -r requirements.txt
conda run -p .venv pip install -r requirements-m1.txt
# Install the CUDA-matched build described by requirements-torch-cu128.txt.
conda run -p .venv pip install -e .
```

## Data preparation

CrisisMMD media is not redistributed. Download from the official source and
verify archive SHA256
`351b9be6123a5d5074407d792bfdddc1f470d4909f65ddd429b584d24607d43f`,
then run:

```bash
.venv/bin/python scripts/download_data.py --source official --data-root data/raw
.venv/bin/python scripts/inspect_data.py --data-root data/raw --out reports/data_audit.md
.venv/bin/python scripts/prepare_manifests.py
.venv/bin/python scripts/validate_splits.py
.venv/bin/python scripts/validate_content_leakage.py
.venv/bin/python scripts/make_strict_manifests.py
```

Committed manifests contain 12,639 ID and 12,668 OOD samples after checksum-
based duplicate removal. They support private course reproduction and should
not be republished publicly without checking CrisisMMD terms.

## Baseline reproduction

Each command accepts `--seed {42,123,2026}` and an explicit CUDA device:

```bash
.venv/bin/python -m src.train --config configs/id_text.yaml --seed 42 --device cuda:0
.venv/bin/python -m src.train --config configs/id_image.yaml --seed 42 --device cuda:1
.venv/bin/python -m src.train --config configs/id_fusion.yaml --seed 42 --device cuda:2
.venv/bin/python -m src.train --config configs/ood_text.yaml --seed 42 --device cuda:3
.venv/bin/python -m src.train --config configs/ood_image.yaml --seed 42 --device cuda:4
.venv/bin/python -m src.train --config configs/ood_fusion.yaml --seed 42 --device cuda:5
```

Repeat for all seeds, then aggregate:

```bash
.venv/bin/python scripts/aggregate_results.py
.venv/bin/python scripts/compare_models.py
.venv/bin/python scripts/error_analysis.py
.venv/bin/python scripts/make_summary_figures.py
```

## M4 reproduction

Run every protocol/variant/seed combination (`id_official_strict` or
`ood_heldout_strict`; `dropout` or `presence`; seeds 42, 123, 2026):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_modality_dropout.py \
  --protocol id_official_strict --variant dropout --seed 42 --device cuda:0
.venv/bin/python scripts/aggregate_modality_dropout.py
```

## Verification and outputs

```bash
.venv/bin/python -m pytest -q
```

- `configs/`: experiment configuration
- `data/manifests/`: strict split manifests
- `logs/`: environment and per-run logs/configs
- `checkpoints/`: trained classifier states
- `results/predictions/`: per-example predictions and probabilities
- `results/tables/`: aggregated metrics and comparisons
- `figures/`: plots, excluding source-image case montages
- `reports/`: audit and milestone reports

