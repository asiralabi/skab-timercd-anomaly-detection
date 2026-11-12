# Industrial Sensor Anomaly Detection Using Time-RCD

Final project for *Applied AI with Hugging Face*. Zero-shot anomaly detection on the
[SKAB](https://github.com/waico/SKAB) industrial water-circulation dataset using the pretrained
Hugging Face model [thu-sail-lab/Time-RCD](https://huggingface.co/thu-sail-lab/Time-RCD).

**Research question.** Can a pretrained zero-shot time-series foundation model detect the labeled fault
periods in SKAB without any training on SKAB, and how does it compare with simple classical baselines?

**Short answer.** Yes, partly. Time-RCD clearly beats Isolation Forest and a rolling z-score
(mean per-experiment ROC-AUC 0.71 vs 0.55), but its reliability varies a lot between experiments and it
misses subtle faults. On 24 held-out test experiments it reaches F1 0.60 with recall 0.90 and precision 0.46.

## Repository structure

```
├── README.md
├── requirements.txt
├── notebook/
│   └── project.ipynb        # full pipeline: download → EDA → scoring → evaluation → comparison
├── results/
│   ├── figures/             # fig1–fig5 (produced by the notebook)
│   ├── evaluation.csv       # per-experiment test metrics + pooled test row
│   └── comparison.csv       # baseline / variant comparison (mean per-experiment ROC-AUC)
└── presentation.pdf
```

The raw data (`data/`) and the per-row score files (`results/scores*.csv`) are not committed; the notebook
downloads and regenerates them.

## Dataset: SKAB

SKAB (Skoltech Anomaly Benchmark) contains multivariate sensor recordings from an experimental water
circulation rig with a pump, a valve and a tank. Each CSV file is one experiment in which a fault is
introduced (valve closing, rotor imbalance, leaks, cavitation, ...). Rows are 1-second readings of 8 sensors
plus two label columns, `anomaly` (0 normal, 1 anomaly) and `changepoint`. The label columns are used only
for evaluation and never fed to the model.

| Item | Value |
|---|---|
| Labeled experiments | 34 (valve1: 16, valve2: 4, other: 14) + 1 anomaly-free recording |
| Labeled rows | 37,401 (65.1 % normal, 34.9 % anomaly) |
| Sensors | Accelerometer1RMS, Accelerometer2RMS, Current, Pressure, Temperature, Thermocouple, Voltage, Volume Flow RateRMS |
| Missing values / duplicates within an experiment | 0 / 0 |
| License | GPL-3.0 (data repository) |

**EDA observations (from the notebook, Section 2):**

1. **Class distribution.** 34.9 % of rows are anomalous. That is imbalanced, but far less than typical
   anomaly data, because every experiment contains one deliberately induced fault lasting several minutes.
   Per experiment the anomaly share ranges from 25 % to 49 %.
2. **Sensor relationships.** The two accelerometers are perfectly correlated (r = 1.00), so the model
   effectively sees 7 independent signals. Flow rate correlates with the accelerometers at r = 0.96 under
   normal operation but only r = 0.80 during anomalies. Pressure and Voltage are nearly uncorrelated with everything.
3. **Anomaly duration.** There are exactly 34 anomaly segments, one per experiment, each 188 to 586 s long.
   There are no isolated point anomalies: these are collective anomalies, which suits a context-based model.

Figures: `results/figures/fig1_sensors_over_time.png`, `fig2_class_distribution.png`, `fig3_correlation_heatmap.png`.

## Model: Time-RCD

[Time-RCD](https://huggingface.co/thu-sail-lab/Time-RCD) (Lan et al., 2025, ICML 2026) is a zero-shot
foundation model for time-series anomaly detection, pretrained on synthetic data. Given a series of shape
(T, C) it returns one anomaly score in [0, 1] per time step, based on how much each patch deviates from its
surrounding context ("relative context discrepancy"). No training or fine-tuning on SKAB is performed.
License: Apache-2.0. Checkpoint used: `best_model/pretrain_checkpoint_best_multi.pth` (445 MB, downloaded
automatically on first run).

The alternative candidate `nvidia/Kumo-Anomaly` was not used because it requires its own inference library.

## Method

1. **Split by experiment, fixed before any model run.** 10 validation experiments
   (valve1 0–4, valve2 0, other 1/5/10/13, one of each "other" fault type) are used only to choose the threshold.
   The remaining 24 experiments are the test set. No training set is needed because nothing is trained.
2. **Scoring.** Each experiment is passed to the multivariate model as a (T, 8) array of raw sensor values.
   The model does its own normalisation internally.
3. **Rank normalisation (label-free).** Time-RCD scores are relative to each recording's own context, so the
   same fault can score 0.9 in one file and 0.05 in another. Each experiment's scores are therefore converted
   to within-experiment percentile ranks before thresholding. This raised pooled test ROC-AUC from 0.68 to 0.74.
4. **Threshold.** The rank that maximises F1 on the pooled validation rows (0.310) is frozen and applied to the test set.
5. **Evaluation.** Accuracy, precision, recall, F1, ROC-AUC and a confusion matrix on the test rows, plus
   per-experiment metrics (`results/evaluation.csv`).

## Results

Test set: 24 experiments, 26,516 rows.

| Metric | Value |
|---|---|
| Accuracy | 0.587 |
| Precision | 0.456 |
| Recall | 0.895 |
| F1 | 0.604 |
| ROC-AUC (pooled, rank scores) | 0.737 |
| Per-experiment ROC-AUC | median 0.79, range 0.21 – 1.00 |
| Confusion matrix [[TN, FP], [FN, TP]] | [[7224, 9961], [982, 8349]] |
| Trivial "everything is anomaly" F1 (reference) | 0.521 |

The threshold that maximises F1 favours recall heavily. The model-tutorial method (raw score above the 95 %
quantile of the anomaly-free recording) gives the opposite trade-off: precision 0.57, recall 0.25, F1 0.34.

**Baseline comparison** (mean per-experiment ROC-AUC over all 34 experiments, threshold-free):

| Method | Mean AUC | Median AUC | Experiments with AUC < 0.6 |
|---|---|---|---|
| Time-RCD multivariate (used) | **0.709** | 0.776 | 11 / 34 |
| Time-RCD multivariate, 120 s moving average | 0.710 | 0.798 | 12 / 34 |
| Time-RCD univariate, mean of 8 channels | 0.651 | 0.668 | 14 / 34 |
| Isolation Forest, fit on anomaly-free data | 0.555 | 0.509 | 27 / 34 |
| Rolling z-score, 60 s window | 0.545 | 0.545 | 32 / 34 |

**Five representative test examples** (notebook Section 5):

| Case | Experiment, time | Flow | Temp | Label | Score | Rank | Predicted |
|---|---|---|---|---|---|---|---|
| Normal, correctly normal | valve1/10, 13:34:39 | 32.0 | 67.7 | 0 | 0.004 | 0.146 | normal |
| Anomaly, detected | other/8, 17:24:13 | 127.3 | 88.1 | 1 | 1.000 | 1.000 | anomaly |
| Anomaly, missed | other/4, 17:20:44 | 76.0 | 87.6 | 1 | 0.012 | 0.001 | normal |
| Normal, false alarm | other/14, 19:32:19 | 2.8 | 86.5 | 0 | 1.000 | 1.000 | anomaly |
| Prolonged anomaly, detected | other/11, from 18:20:42 | 128.0 | 84.8 | 1 | 0.159 | 0.877 | anomaly, 451 consecutive seconds |

The false alarm in other/14 is instructive: flow is 2.8 (pump essentially off) at the very start of the
recording, before the labeled fault. The model treats the start-up transient as the most unusual event in the file.

## Error analysis, strengths and weaknesses

**Strengths**

- Works zero-shot: no labels, no training, one function call per experiment, about 6 s per file on CPU.
- Detects the long fault periods in most "other" experiments (rotor imbalance, leaks, water-level changes)
  with ROC-AUC 0.87 to 1.00 and handles all 8 sensors jointly.
- Substantially better than classical label-free baselines on the same data.

**Weaknesses**

- **Reliability varies per experiment.** 11 of 34 experiments have ROC-AUC below 0.6; the worst (valve2/3) is 0.21.
- **Subtle faults are missed.** The valve experiments often change the flow rate by only about 1 unit; the model
  under-reacts to these while reacting strongly to the rig settling during the first minutes of a recording.
- **Score scale is not comparable across recordings**, which forces a per-recording rank normalisation and
  makes a single operational threshold hard to set.
- **Precision is low (0.46)** at the F1-optimal threshold; a deployment would need a stricter threshold and
  would then miss most faults.
- Domain gap: the model is pretrained on synthetic series; SKAB is a small physical rig with step-like,
  low-resolution sensors (Pressure and Flow take only a few discrete values).

**Tried and did not help** (see `results/comparison.csv`): a 120 s moving average on the scores, ignoring the
first 120 s of each recording, and the univariate checkpoint averaged over channels. Flow rate alone scored as
well as the multivariate model, but selecting a channel after seeing results would be test-set leakage, so it
was not adopted.

**Possible improvements**

- Calibrate a per-experiment threshold from a short known-normal warm-up window instead of a global rank.
- Combine Time-RCD with a sensor-specific rule on flow rate for the valve faults.
- Fine-tune Time-RCD or a lighter model on a few SKAB experiments, with proper experiment-level cross-validation.
- Compare against a second pretrained model (e.g. NVIDIA Kumo-Anomaly) under identical splits.

## Technical difficulties and limitations

- Time-RCD is a young model (about 100 downloads at the time of writing). Its Transformers wrapper only
  supports univariate input, so the official `time_rcd` package from GitHub is used for multivariate inference.
- The first run downloads a 445 MB checkpoint. On Windows, `huggingface_hub` warns about missing symlink
  support; this is harmless.
- Only 34 short experiments exist, so all metrics have wide uncertainty. No fine-tuning was attempted.

## Reproduction

```bash
git clone https://github.com/asiralabi/skab-timercd-anomaly-detection.git
cd skab-timercd-anomaly-detection
pip install -r requirements.txt            # installs time-rcd from GitHub; Python 3.10+ (tested on 3.14)
jupyter nbconvert --to notebook --execute --inplace notebook/project.ipynb
```

Or open `notebook/project.ipynb` in Jupyter and run all cells. The notebook downloads the data, downloads the
model checkpoint, scores all experiments (about 5 min CPU for the multivariate model, about 10 min for the
univariate side experiment), and writes `results/evaluation.csv`, `results/comparison.csv` and the figures.
Everything runs on CPU; no GPU is required.

## References

- Lan, T. et al. (2025). *Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging
  Synthetic Data and Relative Context Discrepancy.* arXiv:2509.21190. Model: https://huggingface.co/thu-sail-lab/Time-RCD
- Katser, I. D. & Kozitsin, V. O. (2020). *Skoltech Anomaly Benchmark (SKAB).* https://github.com/waico/SKAB
