# MedShield-FL

**A Unified Privacy-Preserving Federated Learning Framework with Homomorphic Encryption for Secure Healthcare AI**

> Train accurate medical AI across hospitals — without patient data ever leaving the hospital.

Hospitals cannot pool patient scans: HIPAA and GDPR forbid it. But no single hospital
holds enough varied data to train a reliable diagnostic model. MedShield-FL resolves
this by moving the **model to the data** instead of the data to the model. Each hospital
trains locally, encrypts its model update with CKKS homomorphic encryption, and sends
only ciphertext. The central server averages those ciphertexts **without ever decrypting
them**, so privacy is guaranteed by mathematics rather than by policy.

---

## Repository layout

| Path | Contents |
|---|---|
| [`server/`](server/) | FastAPI backend — authentication, hospitals, training rounds, predictions, labelling queue, metrics, audit log. Talks to PostgreSQL. Never holds a secret key. |
| [`fl/`](fl/) | Flower federated learning. `client/` runs inside a hospital (local training + encrypt/decrypt); `server/` orchestrates rounds; `strategy/` is the custom encrypted-FedAvg `Strategy`. |
| [`frontend/`](frontend/) | React + TypeScript + Tailwind dashboard. Display only — it never touches raw patient data or encryption keys. |
| [`shared/`](shared/) | The `medshield` Python library imported by both `server/` and `fl/`: `data/`, `models/`, `crypto/`, `explain/`, `active/`, `config/`, `utils/`. |
| [`data/`](data/) | Local working data — `raw/`, `processed/`, `partitions/`, `checkpoints/`. **Contents are git-ignored**; only the folder skeleton is tracked. |
| [`docs/`](docs/) | Documentation, dataset licence notes, architecture decision records (`adr/`), and generated result tables and figures (`results/`). |

### Why the layout is split this way

The separation is a privacy control, not just tidiness. `shared/medshield/crypto` holds
the secret key and runs only inside `fl/client` — the hospital side. `server/` and the
Flower aggregator import the public context alone. Anything that crosses the network is
ciphertext plus public parameters. Keeping these in distinct packages makes an accidental
secret-key import visible in review rather than buried in a monolith.

---

## Technology stack

| Layer | Technology |
|---|---|
| Frontend | React.js, TypeScript, Tailwind CSS |
| Backend API | FastAPI (Python) |
| AI framework | PyTorch, `timm` — ViT-Base/16 |
| Federated learning | Flower (`flwr`), FedAvg |
| Encryption | TenSEAL — CKKS scheme |
| Database | PostgreSQL |
| Explainability | Grad-CAM adapted for ViT |

---

## Getting started

This repository is built task by task against
[`MedShieldFL_100_Tasks_Build_Guide.pdf`](MedShieldFL_100_Tasks_Build_Guide.pdf) —
100 tasks across 10 levels, each with explicit acceptance criteria.

**Current status: Task 10 complete** — dataset download and verification script ready.

Setup instructions land in [`CONTRIBUTING.md`](CONTRIBUTING.md) and are filled in as the
corresponding tasks complete:

- Task 2 — Python environment and `requirements.txt`
- Task 3 — frontend scaffold
- Task 4 — PostgreSQL via Docker
- Task 7 — full stack via `docker-compose up`

---

## Build roadmap

| Level | Tasks | Scope |
|---|---|---|
| 0 | 1–8 | Foundations and environment setup |
| 1 | 9–20 | Data engineering — brain-MRI pipeline |
| 2 | 21–34 | Vision Transformer model and local training |
| 3 | 35–46 | Homomorphic encryption — TenSEAL / CKKS |
| 4 | 47–60 | Federated orchestration — Flower + encrypted FedAvg |
| 5 | 61–68 | Active learning — smart labelling assistant |
| 6 | 69–76 | Explainable AI — Grad-CAM for ViT |
| 7 | 77–86 | Backend services and database |
| 8 | 87–96 | Frontend dashboard |
| 9 | 97–100 | Integration, testing, security, deployment |

---

## Data and privacy

No real patient data or PHI is used. The dataset is a de-identified, publicly released
research benchmark, and its licence terms are recorded in `docs/` at Task 9.

Two rules hold throughout the codebase:

1. **Secret keys never leave the hospital client.** Only public contexts and ciphertexts
   cross the network. Enforced in code and asserted by tests at Tasks 45–46.
2. **No raw imaging is committed.** `data/` contents are git-ignored; the database stores
   references and metadata, never the images themselves.

---

## Licence

MIT — see [`LICENSE`](LICENSE). The licence covers this source code only. The MRI dataset
carries its own separate terms, documented in `docs/`.
