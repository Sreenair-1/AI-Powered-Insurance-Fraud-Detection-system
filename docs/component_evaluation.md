# Component evaluation without labeled claims

There is no labeled Indian claims data, and no source dataset links claims to
documents or records real fraud rings. Document forensics and ring detection
are therefore measured on generated data where the right answer is known by
construction. Run:

```bash
./.venv/Scripts/python.exe scripts/run_benchmarks.py
```

This writes `models/benchmarks/tampering_benchmark.json` and
`models/benchmarks/ring_benchmark.json`. The web app's Model insights page
renders both, and each case file quotes the matching row next to that
claim's network and document risk. The numbers below come from the committed
run: 40 bills and 3 ring populations per setting, seed 42.

## Document tampering (`src/forensics/benchmark.py`)

**Clean bills.** `render_synthetic_bill` draws a hospital or garage bill:
letterhead, GSTIN, bill number, 5 to 8 line items, a total and a PAID stamp. It then
applies capture realism (slight blur, uneven lighting, sensor noise) and saves
it as a phone JPEG (quality 88, camera Make/Model and matching
DateTime/DateTimeOriginal). `--images-dir` uses your own bill images instead;
their edit regions are random blocks in the lower-right of the page.

**Untouched variants** (should not be flagged): the original capture, a
gallery re-save at quality 75, a 70% scanner-style downscale, and a PNG
screenshot.

**Edits** (should be flagged):

| Edit | What changes |
| --- | --- |
| `copy_move` | One whole line-item row is cloned over another row |
| `splice` | The total is replaced by a patch from another bill, compressed at quality 60 |
| `amount_edit` | The total is painted over and re-typed; camera metadata kept |
| `amount_edit_in_editor` | Same edit, saved with an editor Software tag and a new DateTime |

Every file is scored twice: as submitted, and after a chat-app style re-share
(75% size, JPEG quality 70, metadata stripped). Each file goes through the
production `compute_ela`, `compute_copy_move`, `analyze_metadata` and
`evaluate_document_tampering`.

**Metrics.** These are the detection rate per edit type, the false-positive rate per untouched
variant, precision, ROC-AUC of each check's continuous signal, and the best
detection reachable at no more than 10% false positives.

### Results

| Check | Edited flagged (as submitted / re-shared) | Untouched flagged (as submitted / re-shared) | ROC-AUC (as submitted) |
| --- | --- | --- | --- |
| ELA | 0% / 0% | 0% / 0% | – |
| Copy-move | 100% / 100% | 100% / 100% | 0.55 |
| Metadata | 25% / 0% | 0% / 0% | 0.62 |
| Combined verdict | 100% / 100% | 100% / 100% | 0.64 |

**The checks do not separate edited bills from untouched ones:**

- Copy-move flags every bill. Printed text repeats the same glyphs at fixed
  row spacing. The ORB ratio test then discards exactly the duplicate matches
  a cloned row produces, because each glyph has many near-identical
  neighbours.
- ELA's whole-image statistics (`high_error_pixel_ratio > 3%`) do not move for
  an edit covering a few percent of the page.
- Only the editor Software tag is reliable, and re-sharing strips it.

The app therefore labels document risk as a prompt to inspect the file. The
likely next step is a text-level check: OCR the amounts, verify the
arithmetic, and match bill numbers across claims.

## Fraud-ring recovery (`src/graph/ring_benchmark.py`)

**Populations.** Each population has 15,420 claims (the motor dataset's size)
with three entities per claim:

- Garage names from 400 garages with log-normal popularity.
- 12-digit bank accounts and 10-digit phones, unique per claimant.
- Households: 3% of claims head a household of 2 to 5 that shares an account
  and phone. These are legitimate, and a source of false alarms.
- 40 rings of 3 to 12 members (1.9% of claims). A ring shares one garage (40%
  of the time an existing busy one), one mule account and one phone. Each
  member uses each with probability 0.7, 0.8 and 0.6, and every member uses
  at least one.
- **Noise**: each recorded value is, with that probability, reformatted
  (case, spacing, hyphens) or given a one-character typo.

**Metrics.** These are claim-level precision and recall of ring membership, and ring recovery. A
true ring counts as recovered when a candidate ring has Jaccard ≥ 0.5 with
it. The baseline flags any claim that shares any entity, which is the
existing `has_shared_entity` feature used as a detector.

### Results (mean of 3 populations)

| Noise | Rings found: exact / normalised / typo-tolerant | Precision: exact / normalised / typo-tolerant | Baseline precision |
| --- | --- | --- | --- |
| 0% | 88% / 88% / 88% | 20% / 20% / 20% | 1.9% |
| 10% | 73% / 82% / 88% | 24% / 22% / 20% | 2.0% |
| 20% | 59% / 78% / 88% | 29% / 25% / 20% | 2.0% |
| 30% | 35% / 67% / 87% | 32% / 26% / 20% | 2.1% |
| 40% | 16% / 55% / 88% | 41% / 29% / 21% | 2.2% |

| Minimum ring size (typo-tolerant, 20% noise) | Rings found | Precision | Recall |
| --- | --- | --- | --- |
| 3 (app default) | 88% | 20% | 77% |
| 4 | 73% | 29% | 71% |
| 5 | 60% | 43% | 63% |

**Findings.** Typo-tolerant matching keeps recovery flat as details drift,
while exact matching collapses. Precision is about 10 times the baseline but
limited by households. Alpha between `1e-2` and `1e-6` changes little.
Claim-level recall tops out near 77%, because a member who shares only one
ring entity cannot be linked by a pair test.

## Limitations

- Generated bills and rings encode assumptions. Real forgeries (re-typed or
  reprinted bills) and real rings (rotating accounts, several garages, several
  insurers) can look different. These results bound the components on these
  patterns only.
- The tampering set is 50% edited, so its precision overstates what the
  checks would achieve at real tampering rates.
- Motor entities in the apps come from the seeded augmentation, not from this
  benchmark's generator.
