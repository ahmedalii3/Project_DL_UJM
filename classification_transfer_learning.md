# Transfer Learning Classification Notebook
## Lavender Water Stress Detection — Multi-Backbone Comparison

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
4. [Backbone Models](#backbone-models)
5. [Model Architecture](#model-architecture)
6. [Training Strategy](#training-strategy)
7. [Data Leakage Prevention](#data-leakage-prevention)
8. [Evaluation](#evaluation)
9. [Results per Fold (EfficientNet-B0 baseline)](#results-per-fold-efficientnet-b0-baseline)
10. [Summary](#summary)

---

## Overview

This notebook implements a **transfer learning** approach for binary classification of lavender plants into two categories:
- `not_stressed` — healthy lavender plants
- `stressed` — water-stressed lavender plants

The approach leverages **five pretrained backbone models** (initialized with ImageNet weights) and fine-tunes each on a small domain-specific dataset using a two-stage training pipeline with 5-Fold Stratified Cross-Validation. All five backbones are trained and their cross-validation metrics compared to identify the best architecture for this task.

---

## Dataset

| Property        | Value                                   |
|-----------------|-----------------------------------------|
| Dataset path    | `lavender_dataset_splitted_stress/`     |
| Total samples   | **118 images**                          |
| Classes         | `not_stressed`, `stressed` (binary)     |
| Image size      | 224 × 224 pixels                        |
| Dataset mean    | [0.2229, 0.2028, 0.1973] (per channel)  |
| Dataset std     | [0.1884, 0.1824, 0.1612] (per channel)  |

> **Note:** Despite the dataset's own mean/std being computed, **ImageNet normalization statistics** (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`) are used for all transforms, since the backbone is pretrained on ImageNet.

---

## Data Preprocessing & Augmentation

### Training Transforms (with augmentation)
| Transform                         | Parameters                                     |
|-----------------------------------|------------------------------------------------|
| `RandomResizedCrop`               | size=224, scale=(0.75, 1.0)                    |
| `RandomHorizontalFlip`            | p=0.5                                          |
| `RandomVerticalFlip`              | p=0.5                                          |
| `RandomRotation`                  | degrees=20                                     |
| `ColorJitter`                     | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05 |
| `RandomAffine`                    | translate=(0.1, 0.1)                           |
| `GaussianBlur`                    | kernel_size=3, sigma=(0.1, 2.0)                |
| `ToTensor` + `Normalize`          | ImageNet mean & std                            |
| `RandomErasing`                   | p=0.2, scale=(0.02, 0.15)                      |

### Validation Transforms (no augmentation)
| Transform                  | Parameters               |
|----------------------------|--------------------------|
| `Resize`                   | (224, 224)               |
| `ToTensor` + `Normalize`   | ImageNet mean & std      |

Heavy augmentation is applied to the training set to compensate for the **very small dataset size (118 images)** and reduce overfitting.

---

## Backbone Models

Five backbones were selected based on their suitability for **fine-grained plant stress detection** on a small dataset. Water stress in lavender manifests as colour changes (yellowing/browning), leaf wilting, texture changes (curling/dryness), and structural changes — all of which are captured well by these architectures.

| Backbone | ImageNet Top-1 | Params | Rationale for this task |
|---|---|---|---|
| **EfficientNet-B0** | 77.7 % | 5.3 M | Compact, compound-scaled CNN; strong baseline for small plant datasets |
| **MobileNet-V3-Large** | 75.3 % | 5.5 M | Lightweight; squeeze-and-excitation blocks aid feature selection; fast inference for agricultural apps |
| **ResNet-50** | 80.9 % | 25.6 M | Classic workhorse; dominant in plant-disease literature (PlantVillage 2016+) |
| **DenseNet-121** | 77.6 % | 8.0 M | Dense feature reuse captures low-level colour/texture cues critical for stress; proven in plant phenotyping |
| **ConvNeXt-Tiny** | 82.5 % | 28.6 M | Modern SOTA CNN (2022); matches ViT quality; excels at fine-grained biological visual tasks |

### Backbone Selector (Cell 0)

```python
BACKBONES_TO_RUN = [
    "efficientnet_b0",
    "mobilenet_v3_large",
    "resnet50",
    "densenet121",
    "convnext_tiny",
]
```

Set the list to a single element to train just one backbone, or leave all five to run a full comparison.

---

## Model Architecture

Each backbone uses **ImageNet pretrained weights** and receives a custom classification head that replaces the original. Dropout is set to `0.4` across all heads to combat overfitting on the small dataset.

### EfficientNet-B0

```
EfficientNet-B0 Backbone (pretrained on ImageNet)
    └── features[0..8]   — MBConv blocks (Inverted Residual + SE)
    └── features[-1]     — Last block (unfrozen in Stage 2)
    └── classifier:
            ├── Dropout(p=0.4)
            └── Linear(1280 → 2)
```

### MobileNet-V3-Large

```
MobileNet-V3-Large Backbone (pretrained on ImageNet)
    └── features[0..16]  — Inverted residuals with SE modules
    └── features[-1]     — Last block (unfrozen in Stage 2)
    └── classifier:
            ├── Linear(960 → 1280)
            ├── Hardswish()
            ├── Dropout(p=0.4)
            └── Linear(1280 → 2)
```

### ResNet-50

```
ResNet-50 Backbone (pretrained on ImageNet)
    └── conv1, bn1, relu, maxpool  — Stem
    └── layer1, layer2, layer3    — Frozen residual stages
    └── layer4                    — Last residual stage (unfrozen in Stage 2)
    └── fc:
            ├── Dropout(p=0.4)
            └── Linear(2048 → 2)
```

### DenseNet-121

```
DenseNet-121 Backbone (pretrained on ImageNet)
    └── features.conv0 … denseblock3, transition3  — Frozen blocks
    └── features.denseblock4 + features.norm5      — Last dense block (unfrozen in Stage 2)
    └── classifier:
            ├── Dropout(p=0.4)
            └── Linear(1024 → 2)
```

### ConvNeXt-Tiny

```
ConvNeXt-Tiny Backbone (pretrained on ImageNet)
    └── features[0..6]   — Downsample stem + ConvNeXt stages 1–3
    └── features[-1]     — Last ConvNeXt stage (unfrozen in Stage 2)
    └── classifier:
            ├── LayerNorm2d(768)
            ├── Flatten
            ├── Dropout(p=0.4)
            └── Linear(768 → 2)
```

---

## Training Strategy

Training is split into **two stages** per fold to avoid destroying pretrained representations in early training:

### Stage 1 — Head-Only Training (Backbone Frozen)
| Parameter          | Value            |
|--------------------|------------------|
| Epochs             | 5                |
| Frozen layers      | All backbone parameters |
| Trainable layers   | Classifier head only |
| Optimizer          | Adam             |
| Head learning rate | `1e-3`           |
| Weight decay       | `1e-4`           |

The backbone is completely frozen. Only the new classification head is trained. This avoids overwriting pretrained features before the head stabilizes.

### Stage 2 — Fine-Tuning (Partial Unfreeze)
| Parameter           | Value                                 |
|---------------------|---------------------------------------|
| Max epochs          | 20                                    |
| Unfrozen layers     | `features[-1]` (last block) + head    |
| Optimizer           | Adam with differential learning rates |
| Backbone LR         | `1e-4` (10× slower than head)         |
| Head LR             | `1e-3`                                |
| LR Scheduler        | `ReduceLROnPlateau` (halves LR on plateau) |
| Early stopping      | Patience = 7 epochs (based on val loss) |

In Stage 2, only the **last convolutional block** of each backbone is unfrozen alongside the head. This allows the model to adapt high-level features to the new domain without disturbing the lower-level pretrained representations. The exact block unfrozen depends on the backbone:

| Backbone | Unfrozen block in Stage 2 |
|---|---|
| EfficientNet-B0 | `features[-1]` |
| MobileNet-V3-Large | `features[-1]` |
| ResNet-50 | `layer4` |
| DenseNet-121 | `features.denseblock4` + `features.norm5` |
| ConvNeXt-Tiny | `features[-1]` |

**Differential learning rates** ensure the backbone fine-tunes cautiously (lr=1e-4) while the head adapts faster (lr=1e-3).

### Loss Function
- **Cross-Entropy Loss** for binary classification

### Model Checkpointing
- The best model per fold is saved based on **lowest validation loss**
- Saved to: `models_weights_tl_{backbone_name}_fold{N}.pth`

---

## Data Leakage Prevention

The K-fold pipeline is designed to guarantee strict separation between training and validation data at every stage:

| Concern | How it is handled |
|---|---|
| Label / metadata extraction | A **no-transform** `ImageFolder` (`ref_dataset`) is created solely to read directory structure and obtain class labels. It is immediately deleted after use — no images are loaded, no augmentation is applied. |
| Training samples seen during validation | `StratifiedKFold` produces **disjoint** `train_idx` / `val_idx` sets. An explicit `assert len(set(train_idx) & set(val_idx)) == 0` fires if overlap is ever detected. |
| Augmentation applied to validation | A **separate** `ImageFolder` is instantiated per fold for training (with full augmentation) and for validation (resize + normalise only). `Subset` restricts each loader to its own indices only. |
| Normalisation statistics | ImageNet statistics (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`) are used — hardcoded constants that carry no information from the dataset splits. |

---

## Evaluation

After training each fold, the **best checkpoint** (lowest val loss) is reloaded and evaluated on the fold's validation set. Metrics reported per backbone per fold:

- **Accuracy**
- **Macro F1-Score**
- **Confusion Matrix**
- **Per-class Precision, Recall, F1** (via `classification_report`)

Loss curves (train vs. validation) are plotted for each fold to visualize training dynamics and potential overfitting.

After all backbones finish, a **cross-backbone comparison table** is printed:

```
Cross-Backbone Comparison Summary
========================================================================
  Backbone               Mean Acc      ±   Mean macroF1     ±
  ------------------------------------------------------------
  efficientnet_b0         0.6783  0.0611        0.6381  0.0933
  mobilenet_v3_large         ...     ...           ...     ...
  resnet50                   ...     ...           ...     ...
  densenet121                ...     ...           ...     ...
  convnext_tiny              ...     ...           ...     ...
========================================================================
```

A **Cell 4** at the end of the notebook renders side-by-side bar charts of mean CV accuracy and mean macro-F1 with ±std error bars for all five backbones.

---

## Results per Fold (EfficientNet-B0 baseline)

| Fold | Val Accuracy | Macro F1 | Not-Stressed F1 | Stressed F1 |
|------|-------------|----------|-----------------|-------------|
| 1    | **75.00%**  | 0.7429   | 0.7857          | 0.7000      |
| 2    | 58.33%      | 0.4958   | 0.7059          | 0.2857      |
| 3    | 66.67%      | 0.5966   | 0.7647          | 0.4286      |
| 4    | 65.22%      | 0.6167   | 0.7333          | 0.5000      |
| 5    | 73.91%      | 0.7386   | 0.7500          | 0.7273      |

### Cross-Validation Summary

| Metric         | Mean     | Std Dev  |
|----------------|----------|----------|
| Accuracy       | **67.83%** | ±6.11% |
| Macro F1       | **63.81%** | ±9.33% |

Early stopping was triggered in most folds (Folds 2, 3, 4, 5), indicating that the validation loss stopped improving well before the 20-epoch limit.

---

## Summary

> **For presentation**

**Architecture:** Five ImageNet-pretrained backbones (EfficientNet-B0, MobileNet-V3-Large, ResNet-50, DenseNet-121, ConvNeXt-Tiny) are compared under identical training conditions. Each uses a two-stage strategy: the backbone is first frozen while only the custom head (Dropout 40% → task-specific Linear) is trained, then the last convolutional block is progressively unfrozen for domain adaptation with differential learning rates (backbone: 1e-4, head: 1e-3). Per-backbone checkpoints are saved at the best validation loss for each fold.

**Data Leakage Prevention:** A no-transform reference dataset is used exclusively to extract labels and fold indices. Each fold creates independent `ImageFolder` objects with the correct transform per split, and a per-fold assertion verifies that training and validation index sets are strictly disjoint.

**Validation:** 5-Fold Stratified Cross-Validation to maintain class balance across folds. Heavy data augmentation (flips, rotation, colour jitter, Gaussian blur, random erasing) is applied to mitigate overfitting on the small 118-image dataset.

**EfficientNet-B0 Baseline Results:** Mean accuracy of **67.83%** (±6.1%) and mean macro F1 of **63.81%** (±9.3%), with performance ranging from 58.3% (Fold 2) to 75.0% (Fold 1). The `stressed` class is consistently harder to classify (lower recall), likely due to class imbalance and limited per-fold training samples (~94 training / ~24 validation).

The variance across folds and the visible train–val accuracy gap (~70–80% train vs. ~58–75% val) suggest **moderate overfitting**, driven primarily by the very small dataset size rather than model capacity. Comparing the remaining four backbones may reveal an architecture that generalises better under these constraints.
