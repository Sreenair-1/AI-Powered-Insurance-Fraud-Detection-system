import { Users } from "lucide-react";
import { useId, useMemo, useState } from "react";
import { fmtInt, fmtMetric, fmtPct } from "@/lib/format";
import type { Domain, WorkloadCurve } from "@/lib/types";
import { CurveChart } from "./charts";
import { EmptyState } from "./common";
import { Card, CardBody, CardHeader } from "./ui/card";

const PER = 1000;

/** Linear interpolation between saved ROC-derived points (ties are reviewed in random order). */
function at(curve: WorkloadCurve, rate: number) {
  const pts = [...curve.points].sort((a, b) => a.review_rate - b.review_rate);
  let hi = pts.findIndex((p) => p.review_rate >= rate);
  if (hi === -1) hi = pts.length - 1;
  const lo = Math.max(0, hi - 1);
  const a = pts[lo];
  const b = pts[hi];
  const span = b.review_rate - a.review_rate;
  const w = span > 0 ? (rate - a.review_rate) / span : 0;
  const recall = a.recall + w * (b.recall - a.recall);
  const precision = rate > 0 ? (recall * curve.prevalence) / rate : null;
  return { recall, precision, lift: precision != null ? precision / curve.prevalence : null, threshold: b.threshold };
}

/**
 * "If investigators review the top N% of claims, how many positive-label
 * claims do they reach?" Turns a low precision at one threshold into a
 * capacity-planning view.
 */
export function WorkloadSection({ curve, domain }: { curve: WorkloadCurve | null; domain: Domain }) {
  const sliderId = useId();
  const [percent, setPercent] = useState(10);
  const rate = percent / 100;
  const point = useMemo(() => (curve ? at(curve, rate) : null), [curve, rate]);
  const positives = domain === "health" ? "positive-label (provider proxy) claims" : "positive-label claims";

  return (
    <Card>
      <CardHeader
        title="Review workload planner"
        description={
          curve
            ? `Validation split · ${fmtInt(curve.n_samples)} claims, ${fmtPct(curve.prevalence)} positive-label · claims reviewed in descending risk-score order`
            : undefined
        }
        icon={<Users />}
      />
      <CardBody>
        {curve && point ? (
          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <div className="space-y-4">
              <div className="space-y-2">
                <label htmlFor={sliderId} className="text-sm font-medium text-text">
                  Investigators can review the top <span className="tabular">{percent}%</span> of claims
                </label>
                <input
                  id={sliderId}
                  type="range"
                  min={1}
                  max={60}
                  value={percent}
                  onChange={(e) => setPercent(Number(e.target.value))}
                  className="w-full accent-primary"
                />
                <p className="tabular text-xs text-text-muted">
                  ≈ {fmtInt(Math.round(rate * PER))} of every {fmtInt(PER)} claims
                </p>
              </div>
              <dl className="tabular grid grid-cols-3 gap-3">
                <div className="rounded-lg border border-border px-3 py-2.5">
                  <dt className="text-2xs text-text-muted">Caught</dt>
                  <dd className="text-xl font-semibold text-text">{fmtPct(point.recall, 0)}</dd>
                </div>
                <div className="rounded-lg border border-border px-3 py-2.5">
                  <dt className="text-2xs text-text-muted">Of reviewed, positive</dt>
                  <dd className="text-xl font-semibold text-text">{fmtPct(point.precision, 0)}</dd>
                </div>
                <div className="rounded-lg border border-border px-3 py-2.5">
                  <dt className="text-2xs text-text-muted">Lift over random</dt>
                  <dd className="text-xl font-semibold text-text">{point.lift == null ? "—" : `${point.lift.toFixed(1)}×`}</dd>
                </div>
              </dl>
              <p className="text-sm text-text">
                Reviewing the top {percent}% reaches about {fmtPct(point.recall, 0)} of {positives}; a random {percent}% would reach
                about {percent}%.
                {point.threshold != null ? ` That is roughly a risk-score cut-off of ${fmtMetric(point.threshold, 2)}.` : ""}
              </p>
              <p className="text-2xs text-text-subtle">
                Derived from the saved validation ROC curve (share reviewed = TPR × prevalence + FPR × (1 − prevalence));
                values between saved points are interpolated. The live triage bands are separate.
              </p>
            </div>
            <div className="pt-6">
              <CurveChart
                kind="gain"
                x={curve.points.map((p) => p.review_rate)}
                y={curve.points.map((p) => p.recall)}
                areaLabel={`At ${percent}% reviewed, ${fmtPct(point.recall, 0)} caught; dashed line is random review`}
                markerX={rate}
              />
            </div>
          </div>
        ) : (
          <EmptyState title="No ROC curve saved">Run scripts/train_models.py to generate evaluation sidecars.</EmptyState>
        )}
      </CardBody>
    </Card>
  );
}
