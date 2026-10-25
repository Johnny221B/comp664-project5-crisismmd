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

Eight representative cases (four rescues and four regressions across ID/OOD) are recorded below and in `results/tables/qualitative_cases.csv`. These notes are researcher interpretations, not dataset annotations.

| protocol           |            sample_id | category          | correctness_pattern   |   label |   pred_text |   pred_image |   pred_fusion |   prob_text |   prob_image |   prob_fusion | mechanism                           | text                                                                                           | qualitative_note                                                                                                             |
|:-------------------|---------------------:|:------------------|:----------------------|--------:|------------:|-------------:|--------------:|------------:|-------------:|--------------:|:------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| id_official_strict | 910586004784009216_0 | fusion_rescue     | FFT                   |       0 |           1 |            1 |             0 |       0.632 |        0.704 |         0.368 | shared_false_positive_suppressed    | https://t.co/AXMk0rTyLf - Mexico Cityâ€™s F1 track escapes damage during huge earthquake...    | Strong rescue: Formula-1/no-damage context triggers both unimodal false positives; the joint head suppresses them.           |
| id_official_strict | 913419277830520833_0 | fusion_rescue     | TFT                   |       1 |           1 |            0 |             1 |       0.889 |        0.091 |         0.52  | text_led_rescue                     | For many of the survivors of Mexico's earthquake, the difficulties are only just beginning...  | Text-led rescue: survivor-impact wording is informative while the image branch is strongly negative.                         |
| id_official_strict | 913515779605266433_0 | fusion_regression | TTF                   |       1 |           1 |            1 |             0 |       0.732 |        0.683 |         0.485 | nonlinear_boundary_failure          | US Military Finally Heading To Puerto Rico And Virgin Islands After StormÂ Maria...            | Strong regression: both unimodal scores are positive, but the nonlinear fusion head crosses to the wrong side of 0.5.        |
| id_official_strict | 915287630723665932_0 | fusion_regression | TFF                   |       0 |           0 |            1 |             1 |       0.274 |        0.742 |         0.529 | misleading_image_dominates          | RT @GideonResnick: Reuters pic of Trump shooting paper towels into the crowd in Puerto Rico... | Image-led regression: text correctly treats the political photo caption as non-informative; image evidence dominates fusion. |
| ood_heldout_strict | 923324482143518721_0 | fusion_rescue     | FFT                   |       0 |           1 |            1 |             0 |       0.667 |        0.533 |         0.233 | shared_false_positive_suppressed    | Global wine supply plummets after wildfires in California, Italy https://t.co/EBosv66f5W...    | OOD strong rescue: wildfire terms cause two false positives for downstream wine-supply news; fusion recovers label 0.        |
| ood_heldout_strict | 921097679345733634_0 | fusion_rescue     | FTT                   |       1 |           0 |            1 |             1 |       0.243 |        0.86  |         0.502 | image_led_rescue                    | ICE Director: Suspected Wine Country Arsonist Is Illegal Alien Mexican National...             | OOD image-led rescue: text misses the informative item; image evidence moves fusion just above the decision threshold.       |
| ood_heldout_strict | 920045440594268160_0 | fusion_regression | TTF                   |       1 |           1 |            1 |             0 |       0.895 |        0.585 |         0.498 | ood_boundary_instability            | California wildfires blanket region in smoky pollutants, raising health risks...               | OOD strong regression: both modalities are individually correct, but fusion is unstable at 0.498 under domain shift.         |
| ood_heldout_strict | 917861564002181120_0 | fusion_regression | TFF                   |       1 |           1 |            0 |             0 |       0.917 |        0.472 |         0.498 | modality_conflict_negative_transfer | California Wildfires Trigger Employer Emergency Action Plans https://t.co/Ey3gNvpiVX...        | OOD conflict regression: highly positive text is pulled below threshold by a negative image prediction.                      |

### Recurrent mechanisms

- **Text-led rescue:** explicit impact, relief, or survivor language overcomes an uninformative or misleading image prediction.
- **Image-led rescue:** visual evidence supplies disaster relevance that terse, political, or noisy text misses.
- **Shared false-positive suppression:** both unimodal heads react to disaster keywords or imagery, while the learned joint configuration restores the non-informative label.
- **Negative transfer under conflict:** one strong but wrong modality pulls fusion away from the correct unimodal decision.
- **Non-monotonic fusion boundary:** because the fusion MLP is not an average of unimodal probabilities, it can be wrong even when both separate heads are correct.
- **OOD boundary instability:** several California regressions lie close to 0.5, consistent with sensitivity to event shift and random seed.

### Slide assets

- Combined panel: `figures/cases/qualitative_cases.png`
- Rescue panel: `figures/cases/qualitative_rescues.png`
- Regression panel: `figures/cases/qualitative_regressions.png`

