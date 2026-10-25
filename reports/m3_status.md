# M3 completion status

- Completed 18/18 primary jobs: two protocols × three models × three seeds.
- All jobs use frozen pretrained encoders and independent random heads.
- OOD jobs were trained from pretrained encoders plus new heads; no ID checkpoint
  was reused. California is absent from OOD train and validation.
- Test prediction IDs and labels align across models for every protocol and seed.
- `results/tables/id_main.csv` and `ood_main.csv` report mean ± seed std.
- `results/tables/per_event.csv` separates six seen validation events from the
  California held-out test and records n and class prevalence.
- `results/tables/paired_bootstrap.csv` contains B=5000 tweet-grouped percentile
  intervals for Fusion−Text and Fusion−Image differences.
- Seed-42 qualitative categories and selected rows are generated strictly from
  aligned prediction files, not hand-invented examples.

Primary finding: Late Fusion has the highest average macro-F1 in ID (0.8794)
and California OOD (0.7577). OOD uncertainty is larger, and the California
domain gap remains meaningful despite the fusion improvement.
