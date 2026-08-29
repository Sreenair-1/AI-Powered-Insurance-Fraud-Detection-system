# AI-Powered Insurance Claim Fraud Detection (Health/Motor)

B.Tech project: a decision-support system that flags potentially fraudulent
health and motor insurance claims by combining **document forensics**,
**graph relationship analytics**, and an **explainable XGBoost risk score**
(SHAP), served through a Streamlit investigator UI.

This is an academic decision-support system, not an autonomous fraud
adjudication system — it never claims to have "proven" fraud. Outputs are
phrased as risk categories ("low"/"medium"/"high") and SHAP-based evidence.

Real Indian claims data was not available for this student project. The
system is built and evaluated on public Kaggle datasets adapted to represent
Indian TPA-style claims, with controlled synthetic fraud-ring relationships
added for evaluation (see [Datasets and synthetic data](#datasets-and-synthetic-data)).


## Architecture

```
data/raw/{motor,health,demographics}      src/data/
        (Kaggle CSVs, local-only)    ->   loaders, validation, preprocessing,
                                           splits, synthetic augmentation
                                                |
        +-----------------------------+--------+------------------------+
        |                             |                                 |
  src/forensics/                src/graph/                     src/data/preprocessing
  ELA, copy-move,          construction (NetworkX graph)        structured claim/policy
  metadata analysis        + features (centrality, community,   features (per domain)
  (per-document)             entity-sharing "fraud-ring"
                              indicators)
        |                             |                                 |
        +-----------------------------+--------+------------------------+
                                       |
                              src/features/fusion.py
                        (claim_key-joined structured + graph
                              + forensic feature matrix)
                                       |
                        src/models/{health,motor}.py
                          (XGBoost, class-imbalance handled
                             via scale_pos_weight)
                                       |
                    +------------------+------------------+
                    |                                     |
        src/models/evaluation.py               src/explainability/{health,motor}.py
        Accuracy/Precision/Recall/F1/ROC-AUC          SHAP per-claim explanations
                    |                                     |
                    +------------------+------------------+
                                       |
                          src/pipeline/inference.py
                    (per-claim risk assessment: probability,
                      risk category, top SHAP factors)
                                       |
                                    app/
                          (Streamlit investigator UI)
```

### Components

| Path | Responsibility | Docs |
| --- | --- | --- |
| `src/data/` | Dataset loading, schema validation, preprocessing, leakage-safe train/val/test splits, synthetic entity/fraud-ring augmentation | `docs/data_contract.md`, `docs/synthetic_augmentation.md` |
| `src/forensics/` | Error Level Analysis, copy-move (ORB keypoint) detection, EXIF & PDF metadata analysis on claim document images and PDFs | `docs/forensics_ela.md`, `docs/forensics_copy_move.md`, `docs/forensics_metadata.md`, `docs/forensics_pdf.md` |
| `src/graph/` | Builds a `networkx` entity graph per claim (policyholder/provider/physician/garage/bank account) and derives centrality, community, and shared-entity "fraud-ring" features | `docs/graph.md` |
| `src/features/` | Fuses structured + graph + forensic signals into one `claim_key`-indexed feature matrix, `claim_key`-aligned so no source silently drops rows | `docs/feature_fusion.md`, `docs/feature_forensics.md` |
| `src/models/` | `HealthFraudModel`/`MotorFraudModel` (XGBoost), shared evaluation metrics, JSON-based model persistence (no `pickle`) | `docs/model_health.md`, `docs/model_motor.md`, `docs/model_evaluation.md`, `docs/model_persistence.md`, `docs/model_evaluation_report.md` |
| `src/explainability/` | SHAP `TreeExplainer`-based global/local explanations, per domain plus a shared model-agnostic layer | `docs/shap.md`, `docs/shap_health.md`, `docs/shap_motor.md` |
| `src/pipeline/` | Wires fusion + model + evaluation + SHAP into one per-claim risk assessment call | `docs/pipeline.md` |
| `app/` | Streamlit investigator UI: browse a claim, see forensic/graph/risk/SHAP evidence, upload a document (image/PDF) for re-scoring | `docs/app_ui.md` |
| `scripts/train_models.py` | Offline training entry point that produces the model artifacts `app/` loads | `docs/app_ui.md` |
| `tests/` | Unit + integration tests | `docs/running.md` |

## Datasets and synthetic data

| Domain | Kaggle source | License | Target |
| --- | --- | --- | --- |
| Motor | `shivamb/vehicle-claim-fraud-detection` | CC0 | `FraudFound_P` — real, claim-level |
| Health | `rohitrox/healthcare-provider-fraud-detection-analysis` | CC0 | `PotentialFraud` — real, but **provider-level**, not claim-level (see below) |
| Demographics reference | `patelparth3399/indian-insurance-premium-prediction-dataset` | CC0 | none (used only to sample Indian-context demographic attributes) |

Full schema/column-level detail: `docs/data_contract.md`.

Two known, deliberate approximations, both documented in code and never
presented as ground truth:

- **Health fraud label is provider-level.** Kaggle's health dataset only
  labels *providers* as `PotentialFraud`, not individual claims. A claim-
  level target is derived (`claim_fraud_label_provider_derived`, in
  `src/data/preprocessing.py`) and named that way everywhere in code — never
  bare `fraud` — precisely so nobody downstream mistakes it for a verified
  per-claim label.
- **Motor's graph entities are synthetic.** The raw motor dataset has no
  `GarageID`/`BankAccountID` columns at all. `src/data/synthetic.py`
  generates them (seeded, deterministic) and can inject controlled
  "fraud rings" (claims sharing a garage/bank account) so the graph-analytics
  stage has something to detect and can be validated against a known ground
  truth (`synthetic_fraud_ring_label`) — kept strictly separate from the real
  `FraudFound_P` label at every stage. See `docs/synthetic_augmentation.md`.

No claim-level document → image mapping exists in any source dataset, so
forensic features default to an "unavailable" placeholder for the bulk
dataset; the Streamlit app's document-upload panel is what actually
exercises the forensics pipeline against a real image or PDF document, per
claim, live.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

Use `./.venv/Scripts/python.exe` for every command below (not a bare
`python`) so you're running against the project's own virtual environment.

Place the raw Kaggle CSVs under `data/raw/{motor,health,demographics}/` per
the file mapping in `docs/data_contract.md` §2 — `data/` is gitignored
(generated/local-only, not committed), and every loader in `src/data/`
raises an actionable `DatasetFileMissingError` naming the exact expected
path if a file isn't there.

## Training

The Streamlit app never fits a model in-process — it only loads an artifact
already saved to disk. Produce one with:

```bash
./.venv/Scripts/python.exe scripts/train_models.py --domain both
```

This samples the raw data — by default the **full 15,420-claim motor
dataset** and a **28,000-claim health sample** (8,000 inpatient + 20,000
outpatient, out of ~558K available); see [Model performance](#model-performance)
for why these are the recommended defaults and
`docs/model_training_experiment.md` for the experiment that established
them. Both domains together train in well under two minutes on ordinary
hardware. The script builds the fused feature matrix (forensics-unavailable
+ graph + structured features), does a leakage-safe train/validation split
(row-stratified for motor, provider-grouped for health so no provider's
claims span both splits), fits an `XGBClassifier`-backed model with
`scale_pos_weight` computed from the split's own class balance, evaluates on
the held-out validation split, and saves the model
(`model.json`/`metadata.json`, XGBoost's own JSON format — no `pickle`) plus
`eval_metrics.json`/`confusion_matrix.json` into `models/<domain>/`.

Pass smaller `--*-sample-size` flags for a faster, weaker model (see
`--help` for the full list):

```bash
./.venv/Scripts/python.exe scripts/train_models.py --domain motor --motor-sample-size 5000
./.venv/Scripts/python.exe scripts/train_models.py --domain health \
    --health-inpatient-sample-size 400 --health-outpatient-sample-size 900
```

`models/` is gitignored; every teammate/grader needs to run this once
locally (or receive the artifact out-of-band) before the app can load a
model.

## Model performance

The `models/health`/`models/motor` artifacts shipped for the demo are
trained at the defaults above (full motor dataset; 28,000-claim health
sample), promoted from a larger offline training experiment documented in
full in `docs/model_training_experiment.md` — including the smaller-sample
numbers they replaced and why more data helped. Metrics are on each
domain's own held-out validation split, threshold 0.5:

| Metric | Motor (n=15,420) | Health (n=28,000) |
|---|---|---|
| Accuracy | 0.881 | 0.795 |
| Precision | 0.113 | 0.770 |
| Recall | 0.144 | 0.759 |
| F1 | 0.127 | 0.765 |
| ROC-AUC | 0.724 | 0.883 |
| Validation claims | 2,313 (139 positive) | 4,160 (1,824 positive) |

Motor's precision/recall stay low at the default 0.5 threshold — this
reflects a genuinely weak fraud signal in the available motor claim features
(5.99% base fraud rate), not a training-size artifact; ROC-AUC is the more
informative number for this domain. Health's claim-level label is a
provider-level approximation (`claim_fraud_label_provider_derived`, see
[Limitations](#limitations)), so its stronger numbers should be read as "the
model separates the two classes well," not as ground-truth claim-level
accuracy.

## Running the Streamlit app

```bash
./.venv/Scripts/python.exe -m streamlit run app/main.py
```

Requires both `data/raw/` (for the claim browser) and a trained model
artifact under `models/` (for scoring) — the app shows an actionable
`st.error` naming the missing piece and stops, rather than a raw traceback,
if either is absent.

## Demo workflow

1. Install dependencies and place the raw datasets (Setup, above).
2. `./.venv/Scripts/python.exe scripts/train_models.py --domain both` — trains
   and saves both models (a few minutes; prints validation metrics per
   domain as it goes).
3. `./.venv/Scripts/python.exe -m streamlit run app/main.py` — opens the UI.
4. Pick a domain (health/motor) in the sidebar, adjust the sample-size
   slider if wanted, and select a claim from the list.
5. Review, for the selected claim: the raw claim fields, the graph-derived
   relationship signals (shared entities, community membership, centrality),
   the model's fraud probability and risk category, and the SHAP factors
   driving that score.
6. Optionally upload a document image in the "Document forensics" panel —
   the app runs ELA/copy-move/metadata analysis on it and re-fuses +
   re-scores that one claim with the real forensic signal, so the updated
   score genuinely reflects the uploaded document, not a placeholder.

## Testing

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Per-module test commands and a requirements breakdown: `docs/running.md`.
CI (`.github/workflows/ci.yml`) runs the full suite on every push/PR against
synthetic in-memory fixtures only — it does not require `data/raw/`.

## Limitations

- **Provider-level health label, not claim-level** — see
  [Datasets and synthetic data](#datasets-and-synthetic-data) above.
  `claim_fraud_label_provider_derived` is a documented approximation.
- **Synthetic graph entities for motor** — `GarageID`/`BankAccountID` and any
  injected fraud rings are synthetic, generated to exercise and validate the
  graph-analytics stage, not observed real-world relationships.
- **No real claim-document mapping** — forensic features are "unavailable"
  by default for the bulk dataset; only a claim scored through the app's
  upload panel gets a real forensic signal.
- **Interactive app sample size, still bounded** — the *Streamlit app*
  builds features over a bounded, seeded sample of the raw data (500 motor /
  150+350 health claims by default, adjustable via the sidebar sliders), not
  the full 15,420 motor / ~558k health claims, so claim-selection stays
  responsive in an interactive session. Training is a separate, offline step
  and is no longer similarly bounded — `scripts/train_models.py`'s defaults
  now use the full motor dataset and a 28,000-claim health sample (see
  [Model performance](#model-performance)), since `src/graph/features.py`'s
  centrality now auto-switches to an approximate/cached computation above
  1,500 graph nodes (see `docs/graph.md`) and both domains train in well
  under two minutes combined. The app's own sample-size defaults were left
  unchanged regardless — interactive responsiveness, not centrality cost, is
  what motivates that bound. See `docs/app_ui.md`.
- **Graph signals are evidence, not verdicts** — a claim is never labeled
  fraudulent solely for having high centrality or a shared entity; these are
  model input features, weighed by XGBoost alongside everything else.
- **SHAP explains the model, not fraud** — SHAP attributions describe what
  drove *this model's* prediction, not proof that the claim is fraudulent.
- **Single-process local demo** — no authentication or multi-user session
  isolation; this is an academic decision-support demo, not a production
  service.

## Status

Full stack implemented end-to-end: data ingestion → forensics → graph
analytics → feature fusion → XGBoost models → evaluation → SHAP
explainability → inference pipeline → model persistence → Streamlit UI, with
real trained artifacts producible via `scripts/train_models.py`. The final evaluation
report (motor/health metrics, confusion matrices, ROC-AUC/F1 comparison,
findings, and limitations) is `docs/model_evaluation_report.md`.
