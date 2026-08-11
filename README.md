# AI-Powered Insurance Claim Fraud Detection (Health/Motor)

B.Tech project: a decision-support system that flags potentially fraudulent
health and motor insurance claims by combining document forensics, graph
relationship analytics, and an explainable ML risk score.

See `.claude/SESSION_STATE.md`, `.claude/WORK_LOG.md`, and
`.claude/NEXT_TASKS.md` for current project status.

## Components

- `src/data/` — dataset loading, schema validation, preprocessing
- `src/forensics/` — ELA, copy-move detection, metadata analysis
- `src/graph/` — claim entity graph construction, fraud-ring features
- `src/features/` — feature fusion (structured + forensic + graph signals)
- `src/models/` — XGBoost training and inference
- `src/explainability/` — SHAP-based explanations
- `src/pipeline/` — end-to-end inference orchestration
- `app/` — Streamlit investigator UI
- `tests/` — unit and integration tests

## Status

Real Indian claims data is not used. The system is developed and evaluated on
publicly available datasets adapted to represent Indian TPA-style claims,
with controlled/synthetic fraud-ring relationships and tampered images added
for evaluation purposes only. See `.claude/SESSION_STATE.md` for current
implementation status.

## Setup

```bash
pip install -r requirements.txt
```
