# Fusion error analysis

Seed: 42. Categories are computed directly from aligned prediction CSVs.

## id_official_strict

| category          |   count |
|:------------------|--------:|
| all_correct       |    1100 |
| fusion_rescue     |     261 |
| fusion_regression |     116 |
| all_wrong         |      57 |

The four rows above are mutually exclusive and exhaustive. `strong_rescue` means both unimodal models were wrong while fusion was correct; `strong_regression` means both unimodal models were correct while fusion was wrong.

Strong rescue subset: 4; strong regression subset: 10.

### fusion_rescue

Count: 261. Selected examples are in `results/tables/fusion_cases_selected.csv`.

### fusion_regression

Count: 116. Selected examples are in `results/tables/fusion_cases_selected.csv`.

## ood_heldout_strict

| category          |   count |
|:------------------|--------:|
| all_correct       |     746 |
| fusion_rescue     |     222 |
| fusion_regression |     152 |
| all_wrong         |      75 |

The four rows above are mutually exclusive and exhaustive. `strong_rescue` means both unimodal models were wrong while fusion was correct; `strong_regression` means both unimodal models were correct while fusion was wrong.

Strong rescue subset: 4; strong regression subset: 33.

### fusion_rescue

Count: 222. Selected examples are in `results/tables/fusion_cases_selected.csv`.

### fusion_regression

Count: 152. Selected examples are in `results/tables/fusion_cases_selected.csv`.

## Curated presentation cases

Eight representative cases (four rescues and four regressions across ID/OOD) are recorded with researcher interpretations in `results/tables/qualitative_cases.csv` and rendered in `figures/cases/qualitative_cases.png`. These notes are interpretations, not dataset annotations.

