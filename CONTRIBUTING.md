# Contributing to MedShield-FL

This project is built task by task against `MedShieldFL_100_Tasks_Build_Guide.pdf`
(100 tasks, 10 levels). Each task carries explicit acceptance criteria, and a task is
finished only when **every** criterion is demonstrably true.

---

## Setup

Setup steps are added here as the tasks that create them complete. Anything marked
*pending* does not exist yet — do not attempt it.

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git
- Docker and Docker Compose

### 1. Clone

```bash
git clone <repository-url>
cd MEDSHIELD-FL
```

The clone contains the full folder skeleton. `data/` and `docs/results/` are empty by
design — their contents are git-ignored and regenerated locally.

### 2. Python environment

Requires **Python 3.10 or higher**.

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install the shared library in editable mode first
pip install -e shared/

# Install all runtime dependencies
pip install -r requirements.txt

# (Optional) Install dev / test tools
pip install -r requirements-dev.txt
```

Copy the environment template and fill in your values:

```bash
cp .env.example .env
# Then edit .env with your local database URL, secret key, etc.
```

### 3. Frontend

Requires **Node.js 18 or higher**.

```bash
cd frontend
npm install
npm run dev
```

### 4. Database — *pending Task 4*

### 5. Full stack via Docker — *pending Task 7*

---

## Working a task

1. Read the task and its acceptance criteria in the build guide.
2. Create a branch: `git checkout -b task-NN-short-name`.
3. Implement only that task. Do not pull work forward from later tasks — the levels are
   ordered by dependency, and skipping ahead breaks the runnable-at-every-step property.
4. Verify each acceptance criterion yourself before opening a PR.
5. Open a PR whose description lists the acceptance criteria as a checklist, ticked.
6. Merge only when CI is green.

Commit after each task. One task, one commit — it keeps the history aligned with the
build guide and makes a broken step easy to isolate.

---

## Code standards

Tooling is configured at Task 6; until then, follow these by hand.

**Python**
- Formatted with `black`, linted with `ruff`.
- Type hints on public functions.
- Modules live under `shared/medshield/` when used by more than one of `server/` or `fl/`.

**TypeScript**
- Formatted with `prettier`, linted with `eslint`.
- No `any` in API response types — the generated client types are the source of truth.

**Comments**
- Explain *why*, not *what*. A comment justifying a non-obvious cryptographic or
  numerical choice is valuable; one restating the line below it is noise.

---

## Privacy rules — non-negotiable

These are correctness requirements, not style preferences. A violation invalidates the
project's central claim.

1. **A CKKS secret key must never reach `server/` or the Flower aggregator.**
   Only public contexts and ciphertexts cross the network. Tasks 45–46 add automated
   tests that fail if a secret key would be serialised into an outbound message.

2. **No raw patient imaging is committed.** `data/` contents are git-ignored. The
   database stores references and metadata only, never image bytes.

3. **The server never decrypts.** Aggregation is homomorphic end to end. Any code path
   that would call `decrypt()` on the server side is a bug, however convenient.

4. **No secrets in source.** Credentials, keys, and connection strings come from
   environment variables. `.env` is git-ignored; `.env.example` carries placeholders only.

Before pushing, check what you staged:

```bash
git status
git diff --cached
```

If a `.key`, `.ctx`, `.env`, or anything under `data/` appears, stop and remove it from
the index rather than committing and reverting later.

---

## Reporting an issue

Include the task number you were working on, the acceptance criterion that failed, and
the exact command and output. Reproducibility matters here: seeds, config, and library
versions are part of any bug report involving training or encryption.
