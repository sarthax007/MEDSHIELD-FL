# Selective Encryption Rationale

**Task 33** — Identify critical parameters for selective homomorphic encryption.

## Overview

MedShield-FL uses **selective encryption** rather than encrypting the entire
model weight vector. Only the classification-critical parameters are encrypted
with Homomorphic Encryption (HE) before transmission during federated
aggregation rounds. This document explains which parameters are selected, their
measured sizes, and the security / bandwidth rationale.

## Critical Parameters

A parameter is classified as **critical** if its state_dict key contains any of
the following substrings:

| Pattern      | Role                                                             |
| ------------ | ---------------------------------------------------------------- |
| `cls_token`  | The learnable `[CLS]` embedding that aggregates image-level info |
| `head`       | The final linear classifier (weight matrix and bias vector)      |

For the default **ViT-Base/16 + TumorClassifier** architecture the critical
keys are:

| Key                    | Shape         | Elements |
| ---------------------- | ------------- | -------: |
| `backbone.cls_token`   | (1, 1, 768)   |      768 |
| `head.weight`          | (2, 768)      |    1,536 |
| `head.bias`            | (2,)          |        2 |
| **Total critical**     |               | **2,306** |
| **Total model**        |               | **85,800,194** |

**Critical ratio ≈ 0.003 %** of total parameters.

## Security Rationale

The `cls_token` and `head` parameters are the most informative about the
classification decision boundary. An adversary who intercepts these weights can:

1. **Reconstruct the decision surface** — the head weights directly encode the
   linear separation between "tumor" and "no-tumor" classes in the 768-dim
   feature space.
2. **Infer data distribution** — shifts in the `cls_token` across rounds leak
   information about the local training data distribution of a hospital.

By encrypting these parameters with HE, an honest-but-curious aggregation
server can still perform weighted averaging on the ciphertexts (HE supports
addition and scalar multiplication) but cannot inspect the actual values.

## Bandwidth / Compute Trade-off

| Approach               | Encrypted Elements | HE Overhead Factor | Effective Overhead |
| ---------------------- | -----------------: | -----------------: | -----------------: |
| Full-model encryption  |       85,800,194   |             ~100×  |         ~8.58 B×   |
| Selective encryption   |            2,306   |             ~100×  |         ~230.6 K×  |

Encrypting only 2,306 floats instead of 85.8 M reduces the HE compute and
bandwidth overhead by a factor of **~37,000×**, making real-time federated
rounds practical even on modest hardware.

## Deterministic Ordering

All serialization functions sort state_dict keys **alphabetically** before
flattening. This guarantees that every federated client produces an identical
vector layout, so the server can aggregate corresponding elements without
coordination or metadata exchange.

## API Reference

| Function                          | Purpose                                         |
| --------------------------------- | ----------------------------------------------- |
| `is_critical_parameter(name)`     | Check if a key is critical                      |
| `state_dict_to_critical_vector()` | Flatten only critical params to 1-D tensor      |
| `critical_vector_to_state_dict()` | Merge critical vector back into full state_dict  |
| `model_to_critical_vector()`      | Convenience: model → critical 1-D tensor        |
| `critical_vector_to_model()`      | Convenience: critical 1-D tensor → model        |
| `get_critical_ratio()`            | Log and return critical / total ratio            |
| `get_serialization_layout()`      | Full offset metadata for the encryption layer    |
