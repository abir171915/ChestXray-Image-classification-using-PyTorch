# Chest X-Ray Pneumonia Classification (PyTorch)

Binary image classification of chest X-rays into `NORMAL` vs `PNEUMONIA`, built from scratch through progressively more capable models — a hand-built CNN baseline, a class-imbalance fix, and transfer learning with ResNet18 — with each step evaluated on a held-out test set and the reasoning behind every decision documented below.

## Problem

Given a chest X-ray image, predict whether the patient shows signs of pneumonia. Beyond just training a model, the goal of this project was to practice the full workflow of an image classification problem end to end: diagnosing dataset issues, isolating variables between experiments, handling class imbalance correctly, and knowing when a "more powerful" model is actually the wrong move.

## Dataset

- Source: Kaggle ["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Original split: `train` (5216 images), `test` (624 images), `val` (16 images)

**Issue found:** the provided `val` split only has 16 images (8 `NORMAL` / 8 `PNEUMONIA`) — far too small to give a reliable validation signal during training.

**Fix:** re-split `train` into a new, reproducible 85/15 train/val split, per class (`random.seed(42)`), leaving the original `test` folder completely untouched as the final, only-evaluated-once test set.

| Split | NORMAL | PNEUMONIA |
|---|---|---|
| new_train (85%) | 1140 | 3294 |
| new_val (15%) | 201 | 581 |
| test (untouched) | 234 | 390 |

**Class imbalance:** the training set is skewed ~2.9:1 toward PNEUMONIA. This turned out to materially affect every model's behavior — see results below.

## Methodology

Three models were trained and compared on the identical held-out test set, so results are directly comparable:

1. **TinyVGG (baseline)** — small CNN built from scratch (CNN Explainer architecture), trained from random initialization, plain `CrossEntropyLoss`. Establishes a baseline before any imbalance handling or transfer learning.
2. **TinyVGG (weighted loss)** — same architecture and hyperparameters as the baseline; only change is inverse-frequency class weighting in the loss, to isolate whether class imbalance was driving the baseline's errors.
3. **ResNet18 (transfer learning, frozen backbone)** — ImageNet-pretrained ResNet18 with the backbone frozen and only the final linear layer retrained, using the same weighted loss.

A fourth variant — unfreezing ResNet's last residual block (`layer4`) for fine-tuning — was also tried, and is reported below because *what it revealed* is as important as the metric itself.

## Results

All numbers are on the untouched `test` set (624 images), evaluated once per model.

| Model | Test Acc | NORMAL recall | NORMAL precision | NORMAL F1 | PNEUMONIA recall |
|---|---|---|---|---|---|
| TinyVGG (baseline) | 0.73 | 0.29 | 0.99 | 0.45 | 1.00 |
| TinyVGG (weighted loss) | 0.85 | 0.68 | 0.89 | 0.77 | 0.95 |
| **ResNet18 (frozen backbone)** | **0.87** | **0.70** | 0.93 | **0.80** | 0.97 |
| ResNet18 (layer4 unfrozen, 30 epochs) | 0.82 | 0.53 | 0.98 | 0.69 | 0.99 |
| ResNet18 (layer4 unfrozen + dropout, 8 epochs) | 0.80 | 0.48 | 0.98 | 0.65 | 0.99 |

**Best model: ResNet18 with a frozen backbone.**

### What the baseline revealed

The TinyVGG baseline reached ~95% validation accuracy but only 73% test accuracy, and the confusion matrix showed why: it was defaulting to "PNEUMONIA" almost every time (NORMAL recall of only 0.29, 166/234 healthy patients misclassified). Training curves showed no overfitting, so this wasn't a training bug — it was the ~2.9:1 class imbalance biasing the model toward the majority class.

### What weighted loss fixed

Applying inverse-frequency class weights to `CrossEntropyLoss` — same architecture, same hyperparameters, only the loss changed — raised NORMAL recall from 0.29 to 0.68 and test accuracy from 0.73 to 0.85, confirming the imbalance hypothesis without sacrificing PNEUMONIA recall (0.95, still catching nearly every real case).

### What transfer learning added

Swapping in a frozen, ImageNet-pretrained ResNet18 (only the final linear layer trained) improved every metric further — test accuracy to 0.87, NORMAL recall to 0.70 — while training only ~1,000 parameters. This is the strongest result in the project.

### What unfreezing `layer4` taught (a negative result, kept deliberately)

The intuitive next step — unfreezing ResNet's last residual block to let it adapt to X-ray-specific features — made every metric *worse*, not better, in two separate attempts (with and without dropout, at both 30 and 8 epochs). The training log explains why: train accuracy hit ~99.9–100% within a handful of epochs while validation loss started climbing, the textbook signature of overfitting. `layer4` alone has roughly 8–9 million trainable parameters, well beyond what ~4,400 training images can meaningfully constrain — and dropout on the final feature vector doesn't regularize the convolutional filters actually doing the overfitting.

This is kept in the results table on purpose: **more capacity is not automatically better**, and knowing when to stop adding trainable parameters is as much a modeling decision as knowing when to add them. The frozen-backbone model remains the best model for this dataset size.

## Lessons learned

- **A validation metric that never changes across epochs is a bug, not a stable model.** Early in this project, a shadowed variable name inside the shared `train()` function caused every model's validation loop to silently evaluate against the wrong `DataLoader`. It surfaced as a flat, unmoving validation accuracy during ResNet training — a mismatch between the printed metric and a manually-recomputed confusion matrix was what exposed it.
- **Validation and test can disagree even when a model looks fine.** All the re-split `new_val` data comes from the same source as `new_train`, so it can share the same quirks a model overfits to — while `test` is a truly independent set. The `layer4` experiments looked excellent on validation (~99% accuracy) while quietly degrading on test.
- **Recall means different things for different classes, and the cost of each error type differs in a clinical context.** A missed PNEUMONIA case (false negative) is more costly than a false alarm on a healthy patient (false positive) — this shaped which metric mattered most when comparing models, not just raw accuracy.

## Repo structure

```
Chest_Xray/
├── chest_xray/              # dataset (gitignored)
│   ├── train/                # original Kaggle train split
│   ├── new_train/            # re-split train (85%)
│   ├── new_val/               # re-split val (15%)
│   ├── val/                   # original (too-small) val split, unused
│   └── test/                  # untouched original test set
├── checkpoints/              # saved model weights (gitignored)
│   ├── tinyvgg_baseline.pth
│   ├── tinyvgg_weighted_loss.pth
│   └── resnet18_frozen.pth   # best model (0.87 test acc)
├── classification.ipynb       # full workflow: data prep, all models, evaluation
└── README.md
```

## Next steps

- Add data augmentation (random flips/rotations) as a lower-risk alternative to `layer4` fine-tuning for squeezing out further gains.
- Extract training/evaluation code from the notebook into a `src/` module with a standalone `predict.py` for inference on a single image, loading `checkpoints/resnet18_frozen.pth`.
- Inspect a sample of misclassified test images for visible distribution-shift patterns (equipment, contrast, image quality) between train and test sources.
