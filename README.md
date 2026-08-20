# Chest X-Ray Pneumonia Classification (PyTorch)

Binary image classification of chest X-rays into `NORMAL` vs `PNEUMONIA`, built from scratch through progressively more capable models — a hand-built CNN baseline, a class-imbalance fix, and transfer learning with ResNet18/ResNet34 — with each step evaluated on a held-out test set and the reasoning behind every decision documented below.

## Problem

Given a chest X-ray image, predict whether the patient shows signs of pneumonia. Beyond just training a model, the goal of this project was to practice the full workflow of an image classification problem end to end: diagnosing dataset issues, isolating variables between experiments, handling class imbalance and data leakage correctly, and knowing when a "more powerful" model is actually the wrong move.

## Dataset

- Source: Kaggle ["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Original split: `train` (5216 images), `test` (624 images), `val` (16 images)

**Issue 1 — validation set too small:** the provided `val` split only has 16 images (8 `NORMAL` / 8 `PNEUMONIA`) — far too small to give a reliable validation signal during training.

**Issue 2 — patient-level data leakage:** filenames in this dataset encode a patient/study identifier (e.g. `person1003_bacteria_2934.jpeg` and `person1003_virus_1685.jpeg` are the same patient). A naive image-level train/val split checked directly against the data showed **73% of PNEUMONIA validation "patients" also had other images of themselves in the training set** — meaning models could partly learn to recognize a *patient*, not the disease, and get evaluated on another image of the same person. This inflated validation accuracy in a way that didn't hold up on the untouched test set.

**Fix:** re-split `train` into a new, reproducible 85/15 train/val split — grouped **by patient**, not by image, so every image belonging to one patient lands entirely on one side of the split (`random.seed(42)`). The original `test` folder is left completely untouched as the final, only-evaluated-once test set.

| Split | NORMAL (patients / images) | PNEUMONIA (patients / images) |
|---|---|---|
| new_train (85%) | 1073 / 1137 | 1390 / 3362 |
| new_val (15%) | 189 / 204 | 245 / 513 |
| test (untouched) | — / 234 | — / 390 |

Verified zero patient overlap between `new_train` and `new_val` after the fix.

**Class imbalance:** the training set is skewed ~3:1 toward PNEUMONIA. This turned out to materially affect every model's behavior — see results below.

## Methodology

Four models were trained and compared on the identical held-out test set, so results are directly comparable:

1. **TinyVGG (baseline)** — small CNN built from scratch (CNN Explainer architecture), trained from random initialization, plain `CrossEntropyLoss`. Establishes a baseline before any imbalance handling or transfer learning.
2. **TinyVGG (weighted loss)** — same architecture and hyperparameters as the baseline; only change is inverse-frequency class weighting in the loss, to isolate whether class imbalance was driving the baseline's errors.
3. **ResNet18 (transfer learning, frozen backbone)** — ImageNet-pretrained ResNet18 with the backbone frozen and only the final linear layer retrained, using the same weighted loss.
4. **ResNet34 (transfer learning, frozen backbone)** — same recipe as ResNet18, swapping in a deeper pretrained backbone, to isolate "does more pretrained depth help" from "does more trainable capacity help" (the next experiment below already answers the second question).

All four also use **data augmentation** on the training set only — `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` — with a separate, non-augmented transform for validation/test, so evaluation numbers stay clean and comparable.

A fifth variant — unfreezing ResNet18's last residual block (`layer4`) for fine-tuning — was also tried, and is reported below because *what it revealed* is as important as the metric itself.

## Results

All numbers are on the untouched `test` set (624 images), evaluated once per model, on the patient-level (leak-free) split, with training-time augmentation applied.

| Model | Test Acc | NORMAL recall | NORMAL precision | NORMAL F1 | PNEUMONIA recall |
|---|---|---|---|---|---|
| TinyVGG (baseline) | 0.75 | 0.35 | 0.93 | 0.51 | 0.98 |
| TinyVGG (weighted loss) | 0.78 | 0.49 | 0.86 | 0.63 | 0.95 |
| **ResNet18 (frozen backbone)** | **0.91** | **0.85** | 0.91 | **0.88** | 0.95 |
| ResNet34 (frozen backbone) | 0.85 | 0.71 | 0.85 | 0.78 | 0.93 |

**Best model: ResNet18 with a frozen backbone.**

### What the baseline revealed

The TinyVGG baseline reached high validation accuracy but only ~75% test accuracy, and the confusion matrix showed why: it was defaulting to "PNEUMONIA" far too often (NORMAL recall of only 0.35). This wasn't a training bug — it was the ~3:1 class imbalance biasing the model toward the majority class.

### What weighted loss fixed (partially)

Applying inverse-frequency class weights to `CrossEntropyLoss` — same architecture, same hyperparameters, only the loss changed — raised NORMAL recall from 0.35 to 0.49 without badly hurting PNEUMONIA recall (0.98 → 0.95). A real improvement, but still well behind what transfer learning achieved — the biggest gain in this project came from pretrained features, not loss weighting alone.

### What transfer learning added

Swapping in a frozen, ImageNet-pretrained ResNet18 (only the final linear layer trained) was the single biggest jump in the whole project — test accuracy from 0.78 to 0.91, NORMAL recall from 0.49 to 0.85 — while training only ~1,000 parameters. This is the strongest result in the project.

### What data augmentation added

Adding `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter` to the training transform — no architecture change, no new parameters — improved ResNet18 further: test accuracy from 0.89 to 0.91, NORMAL recall from 0.79 to 0.85, with false positives *and* false negatives both dropping. Unlike every other lever tried in this project, augmentation improves what the model *sees* during training rather than what it's *allowed to learn* — a genuinely different, complementary axis to class weighting (which reshapes the loss) and transfer learning (which reshapes the features available).

### What a deeper backbone (ResNet34) taught

Swapping ResNet18 for the deeper ResNet34, still fully frozen, did **not** improve results — accuracy came in lower (0.91 → 0.85) and NORMAL recall dropped (0.85 → 0.71). With only ~4,700 training images and a frozen backbone, the extra depth in ResNet34's pretrained features wasn't exploitable by the small trainable `fc` head — depth alone isn't a free upgrade. ResNet18 remains the better choice here.

### What unfreezing `layer4` taught (a negative result, kept deliberately)

The intuitive next step — unfreezing ResNet18's last residual block to let it adapt to X-ray-specific features — made every metric *worse*, not better, in two separate attempts (with and without dropout, at 30 and 8 epochs respectively; these runs predate the patient-leakage fix, so are not directly comparable to the table above, but the failure mode itself is the point). Train accuracy hit ~99.9–100% within a handful of epochs while validation loss climbed — the textbook signature of overfitting. `layer4` alone has roughly 8–9 million trainable parameters, well beyond what a few thousand training images can meaningfully constrain — and dropout on the final feature vector doesn't regularize the convolutional filters actually doing the overfitting.

This is kept in the write-up on purpose: **more capacity is not automatically better**, and knowing when to stop adding trainable parameters is as much a modeling decision as knowing when to add them.

## Lessons learned

- **Patient-level leakage is easy to miss and inflates validation numbers silently.** A file-level train/val split looked fine (no duplicate images), but checking patient IDs encoded in the filenames revealed 73% of PNEUMONIA validation patients also appeared in training. Validation accuracy looked great throughout the project until this was fixed — always check whether a dataset can have multiple samples per real-world entity (patient, user, session) before splitting, and split at that entity's level, not the sample level.
- **A validation metric that never changes across epochs is a bug, not a stable model.** Earlier in this project, a shadowed variable name inside the shared `train()` function caused every model's validation loop to silently evaluate against the wrong `DataLoader`. It surfaced as a flat, unmoving validation accuracy during ResNet training — a mismatch between the printed metric and a manually-recomputed confusion matrix was what exposed it.
- **Copy-pasted training cells are a common source of silent bugs.** When reusing a training cell for a new model (e.g. ResNet18 → ResNet34), it's easy to leave one variable pointing at the old model's optimizer. The symptom was a training loss that never moved — always double check every variable in a copy-pasted cell references the *new* model, not the old one.
- **Recall means different things for different classes, and the cost of each error type differs in a clinical context.** A missed PNEUMONIA case (false negative) is more costly than a false alarm on a healthy patient (false positive) — this shaped which metric mattered most when comparing models, not just raw accuracy.

## Repo structure

```
Chest_Xray/
├── chest_xray/              # dataset (gitignored)
│   ├── train/                # original Kaggle train split
│   ├── new_train/            # patient-level re-split train (85%)
│   ├── new_val/               # patient-level re-split val (15%)
│   ├── val/                   # original (too-small) val split, unused
│   └── test/                  # untouched original test set
├── checkpoints/              # saved model weights (gitignored)
│   ├── tinyvgg_baseline.pth
│   ├── tinyvgg_weighted_loss.pth
│   └── resnet18_frozen.pth   # best model (0.91 test acc)
├── src/                       # reusable pipeline, extracted from the notebook
│   ├── data.py                 # patient-level split, DataLoaders
│   ├── model.py                # TinyVGG, ResNet builder
│   ├── train.py                 # train/eval loop + CLI
│   └── predict.py               # single-image inference (ResNet18, hardcoded — the winner)
├── tests/                      # unit tests for src/ (pytest)
│   ├── test_data.py
│   └── test_model.py
├── classification.ipynb       # full experiment log: data prep, all models, evaluation
├── pyproject.toml
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# Train a model (rebuilds the patient-level split automatically the first time)
python -m src.train --model resnet18 --epochs 5

# Predict on a single image, using the best saved checkpoint
python -m src.predict --image path/to/xray.jpeg

# Run the test suite
pytest tests/ -v
```

The notebook (`classification.ipynb`) remains the full experiment log — every model, comparison, and debugging step described in this README, in the order it actually happened. `src/` is the extracted, reusable version of the winning pipeline.

## Next steps

- Re-run the `layer4`-unfrozen experiments on the patient-level, augmented split to get a directly comparable (rather than illustrative) overfitting result.
- Bring `src/data.py`/`src/model.py`/`src/train.py` up to date with the augmentation transforms now used in the notebook.
- Inspect a sample of misclassified test images for visible distribution-shift patterns (equipment, contrast, image quality) between train and test sources.
- Package `src/` as an installable module (`pip install -e .`) once the pipeline settles further.
