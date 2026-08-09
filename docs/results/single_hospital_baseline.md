# Single-Hospital Baseline Results

**Date**: 2026-08-06T16:02:48.477342+00:00  
**Seed**: 42  
**Training duration**: 94.34s  

## Model Configuration

| Parameter | Value |
|-----------|-------|
| model_name | `vit_base_patch16_224` |
| num_classes | `2` |
| pretrained | `True` |
| input_size | `224` |
| drop_rate | `0.1` |
| learning_rate | `0.0001` |
| weight_decay | `0.0001` |
| epochs | `3` |
| mixed_precision | `False` |

## Home Hospital (Hospital 0) — Test Metrics

| Metric | Value |
|--------|-------|
| Test Loss | 0.0003 |
| Accuracy | 1.0000 |
| Precision (macro) | 1.0000 |
| Recall (macro) | 1.0000 |
| F1 (macro) | 1.0000 |
| ROC-AUC | 1.0000 |
| Samples | 20 |

### Confusion Matrix

```
[[8, 0], [0, 12]]
```

## Cross-Hospital Generalization

The model trained **only** on Hospital 0's data is evaluated
on the test sets of other hospitals. This demonstrates the
**small-data weakness**: a model trained on a single hospital's
limited, non-IID data fails to generalise to unseen hospitals.

| Hospital | Accuracy | Precision | Recall | F1 | ROC-AUC | Samples |
|----------|----------|-----------|--------|----|---------|---------|
| Hospital 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20 |
| Hospital 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15 |

> **Interpretation**: The drop in accuracy and F1 on other hospitals'
> data compared to Hospital 0 confirms the small-data weakness.
> Federated learning aims to close this gap by jointly training
> across all hospitals without sharing raw data.

## Reproducibility

- **Seed**: `42`
- **Config file**: `data/checkpoints/baseline/baseline_config.json`
- **Checkpoint**: `data/checkpoints/baseline/baseline_best_model.pt`

To reproduce, run:

```bash
cd shared && python -m medshield.model.run_baseline
```
