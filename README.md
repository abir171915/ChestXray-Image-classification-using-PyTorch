# Chest X-Ray Pneumonia Classification (PyTorch)

## Problem

Binary classification of chest X-ray images into `NORMAL` vs `PNEUMONIA`, using the Kaggle "Chest X-Ray Images (Pneumonia)" dataset. Goal: build a baseline CNN from scratch, then compare against transfer learning (ResNet) with proper evaluation for a clinically imbalanced dataset.

## Dataset

- Source: Kaggle "Chest X-Ray Images (Pneumonia)"
- Original split: `train` (5216 images), `test` (624 images), `val` (16 images)
- **Issue found:** the provided `val` set only had 16 images (8 NORMAL / 8 PNEUMONIA) — too small to give a meaningful signal during training.
- **Fix:** re-split `train` into a new stratified 85/15 train/val split (per-class, reproducible with `random.seed(42)`), leaving the original `test` folder completely untouched as the final evaluation set.
- **Class imbalance:** train set is ~2.9:1 PNEUMONIA:NORMAL (3875 vs 1341), which turned out to materially affect model behavior (see results below).

## Phase 1: Baseline

**Model:** TinyVGG-style CNN built from scratch (2 conv blocks, each with 2 conv layers + ReLU + maxpool, followed by a flatten + linear classifier). Trained from random initialization — no pretrained weights — to establish a baseline number before introducing transfer learning.

**Training setup:**
- Images resized to 64x64, converted to grayscale (1 channel)
- Loss: `CrossEntropyLoss`
- Optimizer: Adam, lr=0.001
- 5 epochs

**Training curves:** train and validation loss/accuracy tracked closely together across all 5 epochs, with no divergence — indicating stable training and no overfitting at this scale.

**Validation results (final epoch):** ~96% accuracy, ~0.10 loss

**Test results (held-out, evaluated once after training):**

```
                precision   recall   f1-score   support
NORMAL             0.91      0.38      0.53       234
PNEUMONIA          0.72      0.98      0.83       390

accuracy                              0.75       624
```

**Diagnosis:** Test accuracy (75%) is notably lower than validation accuracy (96%), despite healthy training curves. The confusion matrix shows this isn't random error — the model is strongly biased toward predicting PNEUMONIA:

- PNEUMONIA recall: 0.98 (misses only 9/390 real pneumonia cases)
- NORMAL recall: 0.38 (misclassifies 146/234 healthy patients as pneumonia)

This pattern is consistent with the ~2.9:1 class imbalance in the training data. It's also consistent with a documented train/test distribution shift in this specific Kaggle dataset (different image sources/patients between train and test folders), which likely compounds the effect. Since training curves themselves show no overfitting, the gap is attributed to these two factors rather than a bug in the training loop.

**Baseline to beat:** test accuracy 0.75, PNEUMONIA recall 0.98, NORMAL recall 0.38, macro F1 0.68.

## Next steps

- Isolate the imbalance hypothesis: retrain TinyVGG baseline with class-weighted loss, check whether NORMAL recall improves without materially hurting PNEUMONIA recall — before introducing any architecture change, to keep variables isolated.
- Phase 2: transfer learning with pretrained ResNet (frozen vs. partially unfrozen backbone), compared against this baseline.
- Phase 3: experiment tracking table, F1-prioritized evaluation, model checkpointing, standalone `predict.py` inference script.

## Repo structure

```
project/
├── chest_xray/          # dataset (gitignored)
│   ├── train/
│   ├── new_train/       # re-split train (85%)
│   ├── new_val/         # re-split val (15%)
│   └── test/             # untouched original test set
├── notebooks/            # exploration + training notebooks
├── src/                  # modular code (in progress)
├── checkpoints/          # saved model weights (gitignored)
└── README.md
```
## Experiment 2: Weighted loss (isolating the imbalance hypothesis)

Hypothesis: the baseline's strong bias toward predicting PNEUMONIA (NORMAL recall 0.38) is at least partly caused by the ~2.9:1 class imbalance in training data.

Change from baseline: same TinyVGG architecture, same data, same hyperparameters (lr=0.001, 5 epochs) — only the loss function changed. Used CrossEntropyLoss with inverse-frequency class weights computed from the actual re-split train counts (1140 NORMAL, 3294 PNEUMONIA):

weight_normal = total / (2 * normal_count)      # 1.9447
weight_pneumonia = total / (2 * pneumonia_count) # 0.6730
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([weight_normal, weight_pneumonia]))

Training curves were healthy (smooth convergence, train/val tracked closely, no overfitting).

Test results:
precision   recall   f1-score   support
NORMAL             0.94      0.43      0.59       234
PNEUMONIA          0.74      0.98      0.85       390

accuracy                              0.78       624

Comparison to baseline:

| Metric | Baseline | Weighted loss | Change |
|---|---|---|---|
| NORMAL recall | 0.38 | 0.43 | ↑ improved |
| PNEUMONIA recall | 0.98 | 0.98 | unchanged |
| Accuracy | 0.75 | 0.78 | ↑ |
| Macro F1 | 0.68 | 0.72 | ↑ |

Conclusion: class weighting improved NORMAL recall without costing PNEUMONIA recall, confirming imbalance was a genuine contributing factor. However, the improvement is modest — 57% of healthy patients are still misclassified — so imbalance is not the sole explanation for the val/test gap. The most likely remaining factor is the train/test distribution shift documented for this Kaggle dataset (different image sources/patients between folders). This motivates trying a stronger backbone (Phase 2) in addition to, not instead of, the imbalance fix.

Next steps
Phase 2: transfer learning with pretrained ResNet (frozen vs. partially unfrozen backbone), compared against both experiments above. Carry forward weighted loss as the default going into Phase 2.
Optionally: inspect a sample of misclassified test images to check for visible distribution-shift patterns (equipment, contrast, image quality).
Phase 3: experiment tracking table, F1-prioritized evaluation, model checkpointing, standalone predict.py inference script.