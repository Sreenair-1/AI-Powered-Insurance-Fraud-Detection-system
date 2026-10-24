# Entity graph and fraud-ring detection

## Graph construction (`src/graph/construction.py`)

Each claim becomes a node linked to the entities on it. Node ids are prefixed
with their type (`claim:…`, `garage:…`) so ids never collide across types, and
building a graph never mutates the input table.

| Domain | Entity nodes | Source |
| --- | --- | --- |
| Motor | `garage`, `bank_account` | Synthetic: `src.data.synthetic.assign_synthetic_entities` + `inject_fraud_rings` (the motor dataset has neither column) |
| Health | `provider`, `physician` (attending), `beneficiary` | Real columns of the Kaggle health data |

The motor claim node id is the dataframe row index, because the motor dataset
has no claim identifier.

## Graph features (`src/graph/features.py`)

`extract_claim_features` joins, per claim node:

- **Centrality**: degree, betweenness, closeness, eigenvector. Above
  `LARGE_GRAPH_NODE_THRESHOLD` (1,500 nodes), betweenness uses 200 sampled
  sources and closeness uses 200 landmarks instead of exact all-pairs
  computation. Approximate results are cached on the graph object.
  Eigenvector centrality falls back to `NaN` when power iteration does not
  converge.
- **Community**: greedy modularity community id.
- **Entity sharing**: number of linked entities, and the largest and mean
  number of *other* claims reachable through one shared entity
  (`has_shared_entity` when that is above 0).

These features feed `src.features.fusion`. Neither saved model uses them: the
motor validation comparison found that they lowered average precision (see the
README's model performance and component evaluation sections).

## Candidate fraud rings (`src/graph/rings.py`)

Sharing one entity is normal, because a busy garage or hospital serves
thousands of honest claims. The detector therefore looks for *unexpected
co-sharing*:

1. **Resolve entity values** (`resolve_entity_values`), using one of three modes:
   `raw` (exact text), `normalized` (case, spaces and punctuation removed) or
   `fuzzy` (normalised values within one edit, i.e. a substitution, insertion,
   deletion or adjacent swap, are merged; only for values of 6+ characters).
   Fuzzy candidates come from shared one-deletion variants, so the cost grows
   linearly with the number of values.
2. **Count co-occurrence**: for every pair of entities of *different* types,
   `c` is the number of claims using both.
3. **Test it**: under independent assignment the expected count is
   `deg(e1) × deg(e2) / n_claims`. The pair is significant when the Poisson
   tail `P(X ≥ c)` is at most `alpha` (default `1e-3`).
4. **Group**: claims joined by significant pairs are linked. Connected groups of
   at least `min_ring_size` claims (default 3) are candidate rings, and each
   keeps its significant pairs as evidence.

Known blind spots:

- Evidence always needs two entity types, so claims that share only one
  entity (e.g. a mule account at garages everyone uses) are not flagged.
- Households sharing a bank account and phone produce candidate rings too.

### In the apps

`api/state.py` runs the detector once per loaded sample, with `raw` matching
because graph ids are exact. Motor uses garage + bank account. Health uses
physician + beneficiary: providers are left out because every physician
belongs to a provider, so that pair co-occurs by structure. Each case file
shows the claim's level (`candidate_ring`, `linked` or `none`), the evidence
pairs and the other ring members, beside the model score rather than inside
it. The benchmark results are in [component_evaluation.md](component_evaluation.md).
