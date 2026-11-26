# Finding Machine Faults in Sensor Data with a Hugging Face Model

Final project for *Applied AI with Hugging Face*.

A water pump test rig is watched by 8 sensors. When something goes wrong (a valve closes, a leak, a shaking rotor),
the sensor readings start to look unusual. This project uses the pretrained Hugging Face model
[thu-sail-lab/Time-RCD](https://huggingface.co/thu-sail-lab/Time-RCD) to mark every second that looks like a fault,
on the public [SKAB](https://github.com/waico/SKAB) dataset.

**Research question.** Can a pretrained Hugging Face model find the faults, and does fine-tuning it on our own data
make it better?

**Short answer.** Yes, and yes. Used as it is (zero-shot), Time-RCD already beats a classic Isolation Forest
(ROC-AUC 0.74 vs 0.61). After fine-tuning on only 10 labeled recordings (about 6 minutes on a laptop CPU) it reaches
**F1 0.90 and ROC-AUC 0.97** on 24 unseen test recordings. A sanity check shows that it learned what faults look like,
not just *when* they usually happen.

## What is in this repository

```
├── README.md
├── requirements.txt
├── data/skab/               # the 35 SKAB recordings (CSV), from github.com/waico/SKAB, GPL-3.0
├── notebook/
│   └── project.ipynb        # everything: download → EDA → preprocessing → 3 models → evaluation → examples → conclusion
├── results/
│   ├── figures/             # fig1–fig6, made by the notebook
│   ├── comparison.csv       # main results: the 3 models on the test set
│   ├── evaluation.csv       # ROC-AUC of every model on every test recording
│   ├── clock_check.csv      # the sanity check (step 7 of the notebook)
│   └── examples.csv         # the five example seconds
└── presentation.pdf         # PDF export of notebook/project.ipynb (code, outputs and the final conclusion)
```

## 1. The data: SKAB

SKAB (Skoltech Anomaly Benchmark) has recordings from a real water-circulation rig with a pump, a valve and a tank.
Each recording runs normally for a while, then one fault is made on purpose. Each row is one second.

| | |
|---|---|
| Recordings with labels | 34 (+ 1 recording with no fault at all) |
| Rows | 37,401 seconds |
| Sensors (8) | 2 accelerometers (vibration), current, pressure, 2 temperatures, voltage, flow rate |
| Labels | `anomaly`: 1 = fault, 0 = normal (used only for fine-tuning and grading, never as model input) |
| License | GPL-3.0 |

## 2. Exploring the data (EDA)

1. **The data is clean.** No missing values and no duplicate rows.
2. **About 35 % of all seconds are faults.** Every recording has exactly one fault block, 3 to 10 minutes long.
   So we are looking for long unusual *periods*, not single strange values.
3. **Sensors move together, and the links change during a fault.** The two accelerometers are identical (r = 1.00).
   Flow rate follows vibration closely in normal seconds (r = 0.96) but less during faults (r = 0.80).
   So a good model should read all sensors together.

Figures: `fig1_sensors_over_time.png`, `fig2_class_distribution.png`, `fig3_correlation_heatmap.png`.

## 3. Preparing the data

1. **Sort** each recording by time.
2. **Keep only the 8 sensor columns** as model input. The label columns are hidden from the models.
3. **Scale** every sensor inside each recording to mean 0 and standard deviation 1, because the sensors use very
   different units (temperature about 70, current about 2).
4. **Split by whole recording:** 10 recordings for training, 24 for testing. We do not split random rows, because
   two neighbouring seconds are almost the same and the test would then be far too easy.

## 4. Three models, one fair test

| Model | What it is |
|---|---|
| **Isolation Forest** | Classic machine learning (scikit-learn), the baseline. It learns what "normal" looks like from the fault-free recording. |
| **Time-RCD zero-shot** | The pretrained Hugging Face model, used as it is. It has never seen our data. For every moment it asks: *does this fit with what happens around it?* and returns a score from 0 (normal) to 1 (very unusual). |
| **Time-RCD fine-tuned** (bonus) | The same model, trained a little more on the 10 training recordings and their fault labels: 10 epochs, learning rate 1e-4, about 6 minutes on a CPU. The test recordings are never used for training. |

All three models are graded on the same 24 test recordings, with the same rule for turning scores into alarms:
inside each recording the scores are turned into percentiles, and a second gets an alarm if its percentile is above a
cut-off. Each model's cut-off is picked on the training recordings only.

**Metrics in plain words:** *precision* = of all alarms, how many were real faults; *recall* = of all fault seconds,
how many were caught; *F1* = one number that balances the two; *ROC-AUC* = how well the scores separate faults from
normal seconds without any cut-off (0.5 = guessing, 1.0 = perfect), measured per recording and then averaged.

## 5. Results (24 test recordings, 26,516 seconds)

| Model | Precision | Recall | F1 | Accuracy | ROC-AUC | Test recordings with ROC-AUC > 0.7 |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.40 | 0.79 | 0.53 | 0.50 | 0.61 | 3 of 24 |
| Time-RCD zero-shot | 0.46 | 0.90 | 0.60 | 0.59 | 0.74 | 15 of 24 |
| **Time-RCD fine-tuned** | **0.90** | **0.90** | **0.90** | **0.93** | **0.97** | **24 of 24** |

For reference, calling every second a fault gives F1 = 0.52.

- The pretrained model beats the classic baseline even without any training.
- Fine-tuning made the biggest difference: precision doubled (0.46 → 0.90), so there are far fewer false alarms.

### Sanity check: did the model learn the faults, or just the clock?

In SKAB almost every fault starts about 570 seconds into the recording. A **clock-only guess** (for each second, how
often was it a fault in the training recordings?) never looks at a sensor, yet it gets ROC-AUC 0.95! So we cut the
first 5 minutes off every test recording, which makes every fault come 5 minutes earlier:

| Mean ROC-AUC on the 24 test recordings | Full recordings | First 5 minutes cut |
|---|---|---|
| Clock-only guess | 0.95 | 0.53 (no better than guessing) |
| Time-RCD fine-tuned | 0.97 | 0.93 |

The clock guess breaks, but the fine-tuned model still works. It learned what faults look like, not when they happen.

### Five example seconds (fine-tuned model, notebook step 8)

| Case | Recording, time | Flow | Real | Score | Alarm? |
|---|---|---|---|---|---|
| Normal, no alarm (correct) | other/2, 16:29:17 | 75.0 | normal | 0.03 | no |
| Fault, alarm (caught) | other/8, 17:24:13 | 127.3 | fault | 1.00 | yes |
| Fault, no alarm (missed) | valve1/7, 12:51:32 | 32.0 | fault | 0.01 | no |
| Normal, alarm (false alarm) | other/14, 19:32:19 | 2.8 | normal | 1.00 | yes |
| Start of a long fault, caught | valve1/9, 13:24:40 | 32.0 | fault | 0.22 | yes (400 of 402 fault seconds caught) |

- **The missed fault** is in the last seconds of the fault in valve1/7: the flow is already back at 32, exactly like
  normal running, so the sensors no longer show anything unusual.
- **The false alarm** is the very last second of other/14: the flow is 2.8 because the pump is being switched off.
  The model finds this very unusual, but the label says normal.
- **Example 5:** in valve1/9 the alarm starts 2 seconds after the fault begins and the model catches 400 of the 402
  fault seconds.

## 6. Strengths and weaknesses

**Strengths**
- Works without any training (zero-shot) and still beats the classic baseline.
- A little labeled data goes a long way: 10 recordings and about 6 minutes of CPU time took F1 from 0.60 to 0.90.
- Reads all 8 sensors together, and passed the clock test.

**Weaknesses**
- Zero-shot alone gives many false alarms (precision 0.46) and does badly on some recordings (worst ROC-AUC 0.21).
- Fine-tuning needs labeled faults, which are rare in a real factory.
- SKAB is small (34 recordings) and its faults happen at almost the same time, so it is easier than real life.
  The fine-tuned model drops a little (0.97 → 0.93) when the timing changes.
- Scores are not on the same scale in different recordings, so we needed per-recording percentiles.

**Next steps**
- Test on recordings where faults happen at random times.
- Fine-tune on random cuts of each recording so that timing can never help.
- Use more recordings with k-fold cross-validation, and compare a second pretrained model (for example MOMENT or Chronos).

## 7. How to run it

```bash
git clone https://github.com/asiralabi/skab-timercd-anomaly-detection.git
cd skab-timercd-anomaly-detection
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebook/project.ipynb
```

Or open `notebook/project.ipynb` in Jupyter or VS Code and click **Run All**. The SKAB data is already in `data/skab/`
(the notebook downloads any missing file), and the model (445 MB) downloads by itself on the first run. It takes
about 10 minutes on a laptop CPU; no GPU is needed. Tested with Python 3.14 on Windows 11.

## References

- Lan, T. et al. (2025). *Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic
  Data and Relative Context Discrepancy.* arXiv:2509.21190. Model: https://huggingface.co/thu-sail-lab/Time-RCD
- Katser, I. D. & Kozitsin, V. O. (2020). *Skoltech Anomaly Benchmark (SKAB).* https://github.com/waico/SKAB
