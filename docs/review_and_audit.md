# Investigator review, audit records and workload planning

These pieces cover the "documented, audited process" part of the problem
statement, plus the path from a decision-support prototype to real labels.

## Investigator review (`api/reviews.py`)

On each case file a reviewer records one of three decisions, with a rationale
and an optional name:

| Decision | Retraining label |
| --- | --- |
| Confirmed suspicious | 1 |
| Cleared | 0 |
| Needs more info | none until decided |

Decisions are appended to `data/reviews/reviews.jsonl` on the API host
(gitignored; override with `FRAUD_APP_REVIEWS_PATH`). Nothing is overwritten,
so each claim keeps its full decision history. Each line stores what the
reviewer saw:

- the model probability and risk band
- the network-risk level
- the model version: the first 12 hex characters of the SHA-256 of `model.json`

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/domains/{domain}/claims/{claim_key}/review` | Current decision and history |
| POST | `/domains/{domain}/claims/{claim_key}/review` | Record a decision: `{"outcome", "note", "reviewer"}` |
| GET | `/domains/{domain}/reviews` | Counts per outcome, usable labels, recent decisions |

Model insights shows the counts and offers the labels as CSV
(`claim_key,label,recorded_at,model_version`).

### From decisions to retraining

This step is not automated yet. To use the labels:

1. Take the latest confirmed or cleared decision per claim (the CSV).
2. Join the labels to that claim's features with the same builders
   `scripts/train_models.py` uses. Keep the row-disjoint (motor) or
   provider-disjoint (health) splits so no claim or provider crosses splits.
3. Train a candidate and compare it with the current model on held-out
   *reviewed* claims, selecting the threshold on validation only, as today.
4. Replace the artifact only if it is better; the model version in later reviews
   then changes.

**Selection bias.** Labels exist only for claims someone chose to review, so
they over-represent high scores. Report it, and consider reviewing a small
random sample of low-scored claims so the labels also cover what the model
misses.

## Audit record (`services.audit_report`, `api/audit_html.py`)

`GET /domains/{domain}/claims/{claim_key}/audit-report` returns a JSON
download. `POST` to the same path with a document runs forensics on that file
server-side and includes the findings. `?format=html` returns a printable
page (the browser's "Save as PDF").

The case file's **Audit record (JSON)** and **Printable report (PDF)** buttons
call these endpoints. After a document upload, the buttons send that file too.

A record contains:

- report id and UTC timestamp
- model version: artifact SHA-256 and modified time, risk-band cut-offs, and
  the validation operating threshold
- the assessment: probability, band and top SHAP factors
- claim attributes (labels are never included)
- network risk with its evidence, plus every linked entity and how many
  sample claims share it (synthetic entities are marked)
- document findings when a file is attached: file SHA-256, verdict,
  indicators, ELA/copy-move/metadata features, and the score before and after
  the documented rule-based adjustment. Images are not included.
- benchmark context for both signals
- the review decision and history
- limitations and a decision-support disclaimer
- `integrity.digest`: the SHA-256 of every other field as canonical JSON
  (sorted keys, no whitespace), so a later edit to the record is detectable

## Workload planner (Model insights)

A single threshold's precision (motor: about 18% on test) undersells a triage
model. The planner answers a capacity question: if investigators can review
the top N% of claims, what share of positive-label claims do they reach?
It is derived from the saved validation ROC curve:

```
share reviewed = TPR × prevalence + FPR × (1 − prevalence)
share caught   = TPR
```

Points between saved ROC vertices are interpolated, which is equivalent to
reviewing tied scores in random order.

| Domain (validation) | Review top 10% | Top 20% | Top 30% |
| --- | --- | --- | --- |
| Motor | 39% caught (3.9× random) | 65% (3.3×) | 81% (2.7×) |
| Health (provider proxy) | 12% (1.2×) | 24% (1.2×) | 33% (1.1×) |

Motor ranks well enough to plan around. Health barely beats random review,
consistent with its provider-level proxy label.
