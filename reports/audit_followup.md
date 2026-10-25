# Follow-up audit: splits, fusion outcomes, bootstrap, and cases

Audit date: 2026-10-25. All counts below were recomputed from committed
manifests and aligned seed-42 prediction CSVs.

## 1. Official agreement and split integrity

`load_official_agreed` reads only the three course-designated files named
`task_informative_text_img_agreed_lab_{train,dev,test}.tsv`. It requires the
official agreement flag to be positive and now fails unless
`text_label == image_label == label` after normalization. All 12,708 source
pairs pass; labels map only to `not_informative=0` and `informative=1`.

The trained ID protocol is accurately named `id_official_strict`: it preserves
every official test example (n=1,534) and its official split assignment, but
removes 66 training and 3 validation rows whose image content checksum occurs
in a higher-priority split. Thus it is derived from the official agreed-label
split and evaluates the exact official test set, but it is not an unchanged
official training split. Reports must retain the `strict` qualifier. If an
unchanged-official training result is required as a separate headline number,
that experiment remains to be run and must disclose the detected content
overlap.

For `ood_heldout_strict`, all 1,195 California Wildfires pairs occur only in
test. Train (9,808) and validation (1,665) contain only the other six events.
Across both strict protocols, train/validation/test have zero shared tweet IDs,
image IDs, or image-content checksums. Every manifest row has consistent label
mapping. The validator now applies the California rule to both
`ood_heldout` and `ood_heldout_strict`, with a regression test.

## 2. Mutually exclusive fusion outcomes

The corrected taxonomy is exhaustive:

- `all_correct`: text, image, and fusion are correct.
- `all_wrong`: text, image, and fusion are wrong.
- `fusion_rescue`: fusion is correct and at least one unimodal model is wrong.
- `fusion_regression`: fusion is wrong and at least one unimodal model is correct.

`strong_rescue` (both unimodal models wrong) and `strong_regression` (both
unimodal models correct) are subset flags, not additional top-level classes.
The old table displayed them as separate overwritten categories, which explains
the apparent missing 14 ID and 37 OOD samples.

| Protocol | All correct | All wrong | Rescue | Regression | Total |
|---|---:|---:|---:|---:|---:|
| ID strict | 1,100 | 57 | 261 | 116 | 1,534 |
| California OOD strict | 746 | 75 | 222 | 152 | 1,195 |

The ID subsets are 4 strong rescues and 10 strong regressions (14); the OOD
subsets are 4 and 33 (37). Each row also records a T/F correctness pattern for
text, image, and fusion.

## 3. Paired grouped bootstrap

The best unimodal comparator is selected by validation Macro-F1, never by test.
Text is selected for every protocol and seed. Each interval uses 5,000
percentile bootstrap draws over tweet IDs, so all image pairs belonging to a
sampled tweet move together. Split validation independently confirms that a
tweet and all of its images occur in only one split.

Late Fusion minus validation-selected best unimodal, Macro-F1:

| Protocol | Seed | Comparator | Difference | 95% CI | Crosses 0? |
|---|---:|---|---:|---:|---|
| ID strict | 42 | Text | +0.0353 | [0.0162, 0.0542] | No |
| ID strict | 123 | Text | +0.0519 | [0.0331, 0.0705] | No |
| ID strict | 2026 | Text | +0.0431 | [0.0243, 0.0616] | No |
| California OOD strict | 42 | Text | +0.0727 | [0.0403, 0.1055] | No |
| California OOD strict | 123 | Text | +0.1040 | [0.0731, 0.1365] | No |
| California OOD strict | 2026 | Text | +0.0759 | [0.0434, 0.1097] | No |

The complete table also retains fixed Fusion-Text and Fusion-Image contrasts.
Although Image happens to outperform Text on the California test set, selecting
it after seeing test results would be invalid. For context only, the seed-42
Fusion-Image Macro-F1 interval is [-0.0164, 0.0412] and crosses zero; seeds 123
and 2026 do not.

## 4. Qualitative cases and failure mechanisms

`results/tables/qualitative_cases.csv` contains eight real cases: four rescues
and four regressions spanning ID and OOD. Each row includes image path, text,
true label, all three predictions/probabilities, correctness pattern, and a
researcher interpretation. `figures/cases/qualitative_cases.png` renders the
eight images with compact prediction captions for presentation use.

Observed mechanisms include text-led and image-led rescue under modality
conflict, suppression of shared false positives, a misleading modality pulling
fusion across the threshold, and nonlinear fusion-boundary failures where both
unimodal classifiers are correct. Interpretive notes are explicitly presented
as researcher observations, not original CrisisMMD annotations.
