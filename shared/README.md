# MedShield-FL — Shared Library
# README for the `shared/medshield` Python package.

The `medshield` Python package contains all code shared between:
- `fl/` — the Flower federated-learning client and server
- `server/` — the FastAPI backend

## Sub-modules

| Module | Purpose |
|---|---|
| `medshield.data` | Dataset loading, preprocessing, and NIfTI / BraTS utilities |
| `medshield.models` | ViT-Base/16 model definition and checkpoint helpers |
| `medshield.crypto` | TenSEAL / CKKS key generation, encryption, and decryption |
| `medshield.explain` | Grad-CAM adapted for Vision Transformers |
| `medshield.active` | Active learning — uncertainty and diversity sampling |
| `medshield.config` | Pydantic Settings — loads from `.env` and environment variables |
| `medshield.utils` | Logging, seeding, metric helpers |

## Installation (editable)

```bash
pip install -e shared/
```

Run this once after cloning. Both `server/` and `fl/` import `medshield` directly.
