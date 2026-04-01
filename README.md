# Butterfly Species Classifier (Transfer Learning)

This project implements an end-to-end pipeline for butterfly species classification using transfer learning.

## Project Overview

The goal is to classify images of butterflies into 8 species using pretrained convolutional neural networks.

The following models were trained and compared:
- MobileNetV2 (frozen backbone)
- MobileNetV2 (fine-tuned)
- EfficientNetB0 (frozen backbone)

All models were trained on the same dataset and evaluated on a fixed test split.

## Dataset

- 8 butterfly species
- ~150 images per class (after sampling)
- Total dataset size: ~1200 images
- Data split:
  - Train: 70%
  - Validation: 15%
  - Test: 15% (180 images)

### Data collection

The dataset was collected locally using a custom Python script based on the iNaturalist API.

The pipeline includes:
- downloading images from iNaturalist
- filtering and cleaning metadata
- organizing images into class folders

## Models

### 1. MobileNetV2 (Frozen)
- Pretrained on ImageNet
- Backbone fully frozen
- Only classification head trained

### 2. MobileNetV2 (Fine-tuned)
- Stage 1: train classifier head
- Stage 2: unfreeze top layers of backbone and fine-tune

### 3. EfficientNetB0 (Frozen)
- Pretrained on ImageNet
- Backbone frozen
- Only classification head trained

## Results (Test Set)

| Model                     | Accuracy | Macro F1 |
|--------------------------|----------|----------|
| EfficientNetB0 (Frozen)  | 0.8556   | 0.8562   |
| MobileNetV2 (Fine-tuned) | 0.8333   | 0.8320   |
| MobileNetV2 (Frozen)     | 0.8000   | 0.8033   |

Additional metrics:
- ROC-AUC (macro, OvR) — computed
- PR-AUC (macro) — computed
- Top-3 accuracy — computed

## Key Observations

- EfficientNetB0 achieved the best overall performance.
- Fine-tuning MobileNetV2 improved performance compared to the frozen version.
- Most classification errors occur between visually similar species (see confusion matrix).

These results are consistent with typical behavior of transfer learning on small datasets, where accuracy around 80–85% is common depending on class similarity and dataset size. :contentReference[oaicite:0]{index=0}

## Outputs

The project generates the following artifacts:

- `final_model_comparison.csv` — model comparison table
- `best_metrics.json` — best model metrics
- `results_summary.csv` — detailed classification report
- `*_confusion_matrix.png` — confusion matrices
- `misclassified_examples/` — images with incorrect predictions
- `sample_predictions.csv` — predictions on test set

## How to Run

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
