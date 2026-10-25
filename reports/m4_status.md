# M4 — Modality Dropout

## Controlled design

Both the control (`presence`, p=0) and modality-dropout model (p=0.2) use the
same 1,282-dimensional input: 768 text features, 512 image features, and two
explicit modality-presence indicators. Each selected training example masks at
most one modality. Both variants were trained for ID and OOD protocols using
seeds 42, 123, and 2026, with identical optimization budgets.

## Macro-F1, mean ± seed standard deviation

| Protocol | Condition | Presence control | Dropout p=0.2 | Difference |
|---|---|---:|---:|---:|
| ID | both | 0.8843 ± 0.0027 | 0.8813 ± 0.0059 | -0.0030 |
| ID | text only | 0.7407 ± 0.0028 | 0.8270 ± 0.0026 | +0.0863 |
| ID | image only | 0.7569 ± 0.0155 | 0.7894 ± 0.0058 | +0.0324 |
| California OOD | both | 0.7702 ± 0.0086 | 0.7613 ± 0.0165 | -0.0089 |
| California OOD | text only | 0.6002 ± 0.0319 | 0.6708 ± 0.0043 | +0.0706 |
| California OOD | image only | 0.7236 ± 0.0179 | 0.7084 ± 0.0590 | -0.0152 |

## Interpretation

Modality dropout substantially improves text-only missing-modality robustness in
both protocols and improves ID image-only robustness. It slightly reduces normal
two-modality performance. California image-only performance does not improve and
becomes much more variable, so the extension is not uniformly beneficial.

## Execution note

The first six-process control launch allowed each PyTorch process to use many CPU
threads, causing severe oversubscription. Five incomplete processes were stopped;
one completed. The affected jobs were rerun with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. All final tables use only successful reruns, and all 12
required M4 summaries/checkpoints are present.
