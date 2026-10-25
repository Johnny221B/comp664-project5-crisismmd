# Experiment log

## 2026-10-25 — M0 environment and data

- Verified 8 NVIDIA RTX A5000 GPUs (24,564 MiB each), driver 570.86.10.
- Created Python 3.11.17 environment and installed PyTorch 2.11.0+cu128.
- Downloaded official CrisisMMD v2.0 archive and official agreed-label split.
- Verified archive SHA256 values in `data/raw/download_manifest.json`.
- Parsed 12,708 paired agreed-label samples from the released split.
- Decoded all 12,708 selected images successfully.
- Found 50 distinct image contents occurring across released ID splits (125 rows,
  including one checksum with conflicting binary labels). The released manifest
  is preserved unchanged and is expected to fail content-level strict validation.
- Generated explicitly named content-disjoint sensitivity manifests by retaining
  evaluation splits over training for duplicated content: ID removes 69 rows;
  OOD removes 40 rows. No final-test rows were removed.
- M0 unit tests: 9 passed.

The strict sensitivity manifests are not described as the unchanged official
split. Training remains gated on an explicit protocol choice in M1.
