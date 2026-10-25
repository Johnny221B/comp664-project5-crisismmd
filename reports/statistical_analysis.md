# Statistical analysis and fusion failure summary

Seed variation (mean/std) and test-sample uncertainty (within-seed paired bootstrap) are reported separately. Seeds are not treated as extra test examples.

## Main findings

- Fusion improves Macro-F1 over the validation-selected Text comparator by 0.0434 ± 0.0083 across ID seeds and 0.0842 ± 0.0172 across OOD seeds. Every per-seed tweet-grouped 95% CI excludes zero.
- Against Image, all three ID Macro-F1 intervals exclude zero. In OOD, two of three exclude zero; seed 42 crosses zero, showing that the apparent OOD fusion advantage over Image is less stable.
- Rescue is more common than regression in every seed. Mean rescue/regression proportions are 17.58%/7.00% for ID and 19.55%/10.93% for OOD.
- Exact McNemar tests agree with the accuracy bootstrap: all ID contrasts remain significant after Holm correction; both OOD seed-42 accuracy contrasts do not, while seeds 123 and 2026 do.
- Fusion behavior is label-dependent. Rescue rates are higher for `not_informative` than `informative` examples (ID 26.72% vs 13.11%; OOD 32.11% vs 15.84%), but regressions are also more frequent for the negative class.
- OOD regressions vary more across seeds (10.93% ± 1.99%) than ID regressions (7.00% ± 0.51%); strong OOD regressions are especially seed-sensitive (17.0 ± 14.4 samples).

These are descriptive and paired-test findings, not evidence that seeds are independent populations. The fixed California test set was never used to select the comparator or tune a threshold.

## Bootstrap summary

| protocol           | contrast             | comparator   | selection_basis     |   difference_mean |   difference_std |   seeds_significant |   seeds_total |   min_ci_low |   max_ci_high |
|:-------------------|:---------------------|:-------------|:--------------------|------------------:|-----------------:|--------------------:|--------------:|-------------:|--------------:|
| id_official_strict | fusion-best_unimodal | text         | validation_macro_f1 |         0.0434189 |       0.00831152 |                   3 |             3 |    0.0161812 |     0.0705129 |
| id_official_strict | fusion-image         | image        | fixed               |         0.0781577 |       0.00564526 |                   3 |             3 |    0.0533552 |     0.106091  |
| id_official_strict | fusion-text          | text         | fixed               |         0.0434189 |       0.00831152 |                   3 |             3 |    0.0161812 |     0.0705129 |
| ood_heldout_strict | fusion-best_unimodal | text         | validation_macro_f1 |         0.084191  |       0.0172248  |                   3 |             3 |    0.0403218 |     0.136454  |
| ood_heldout_strict | fusion-image         | image        | fixed               |         0.031716  |       0.017192   |                   2 |             3 |   -0.016446  |     0.0757601 |
| ood_heldout_strict | fusion-text          | text         | fixed               |         0.084191  |       0.0172248  |                   3 |             3 |    0.0403218 |     0.136454  |

## Exact McNemar tests

The table uses discordant correctness pairs on identical samples. Holm adjustment is applied jointly to the 12 protocol × seed × comparator tests.

| protocol           |   seed | comparison      |    n |   fusion_only_correct |   comparator_only_correct |   discordant |   accuracy_difference |     exact_p |      holm_p | holm_significant_0_05   |
|:-------------------|-------:|:----------------|-----:|----------------------:|--------------------------:|-------------:|----------------------:|------------:|------------:|:------------------------|
| id_official_strict |     42 | fusion_vs_text  | 1534 |                   107 |                        64 |          171 |             0.0280313 | 0.00125216  | 0.00500863  | True                    |
| id_official_strict |     42 | fusion_vs_image | 1534 |                   158 |                        62 |          220 |             0.0625815 | 7.55352e-11 | 7.55352e-10 | True                    |
| id_official_strict |    123 | fusion_vs_text  | 1534 |                   121 |                        52 |          173 |             0.0449804 | 1.62411e-07 | 1.29928e-06 | True                    |
| id_official_strict |    123 | fusion_vs_image | 1534 |                   172 |                        62 |          234 |             0.071708  | 3.78585e-13 | 4.16444e-12 | True                    |
| id_official_strict |   2026 | fusion_vs_text  | 1534 |                   114 |                        54 |          168 |             0.0391134 | 4.25352e-06 | 2.97746e-05 | True                    |
| id_official_strict |   2026 | fusion_vs_image | 1534 |                   160 |                        54 |          214 |             0.0691004 | 2.25253e-13 | 2.70304e-12 | True                    |
| ood_heldout_strict |     42 | fusion_vs_text  | 1195 |                   106 |                        85 |          191 |             0.0175732 | 0.147662    | 0.295324    | False                   |
| ood_heldout_strict |     42 | fusion_vs_image | 1195 |                   120 |                       100 |          220 |             0.0167364 | 0.200092    | 0.295324    | False                   |
| ood_heldout_strict |    123 | fusion_vs_text  | 1195 |                   124 |                        47 |          171 |             0.0644351 | 3.37808e-09 | 3.04027e-08 | True                    |
| ood_heldout_strict |    123 | fusion_vs_image | 1195 |                   126 |                        63 |          189 |             0.0527197 | 5.36783e-06 | 3.2207e-05  | True                    |
| ood_heldout_strict |   2026 | fusion_vs_text  | 1195 |                   106 |                        70 |          176 |             0.0301255 | 0.00815045  | 0.0244513   | True                    |
| ood_heldout_strict |   2026 | fusion_vs_image | 1195 |                   132 |                        78 |          210 |             0.0451883 | 0.000237166 | 0.00118583  | True                    |

## Fusion outcome summary

| protocol           | outcome           |   count_mean |   count_std |   proportion_mean |   proportion_std |
|:-------------------|:------------------|-------------:|------------:|------------------:|-----------------:|
| id_official_strict | all_correct       |   1100.33    |     5.50757 |        0.717297   |      0.00359033  |
| id_official_strict | all_wrong         |     56.6667  |     2.51661 |        0.0369405  |      0.00164056  |
| id_official_strict | fusion_regression |    107.333   |     7.76745 |        0.0699696  |      0.00506353  |
| id_official_strict | fusion_rescue     |    269.667   |     9.60902 |        0.175793   |      0.00626403  |
| id_official_strict | strong_regression |      8.66667 |     1.52753 |        0.00564972 |      0.000995779 |
| id_official_strict | strong_rescue     |      7.66667 |     4.72582 |        0.00499783 |      0.00308071  |
| ood_heldout_strict | all_correct       |    754       |     9.84886 |        0.630962   |      0.00824172  |
| ood_heldout_strict | all_wrong         |     76.6667  |     2.08167 |        0.0641562  |      0.00174198  |
| ood_heldout_strict | fusion_regression |    130.667   |    23.7978  |        0.109344   |      0.0199144   |
| ood_heldout_strict | fusion_rescue     |    233.667   |    12.0139  |        0.195537   |      0.0100535   |
| ood_heldout_strict | strong_regression |     17       |    14.4222  |        0.0142259  |      0.0120688   |
| ood_heldout_strict | strong_rescue     |      4.33333 |     0.57735 |        0.00362622 |      0.000483138 |

Detailed label/event/correctness-pattern breakdowns are in `results/tables/fusion_failure_breakdown.csv`. Seed-42 qualitative interpretations remain in `results/tables/qualitative_cases.csv`. Slide-ready numeric claims are duplicated in `results/tables/presentation_key_findings.csv`.

