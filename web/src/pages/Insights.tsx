import { AlertTriangle, FlaskConical, Gauge, ShieldQuestion, Target } from "lucide-react";
import type { ReactNode } from "react";
import { ConfusionMatrix, CurveChart } from "@/components/charts";
import { Callout, EmptyState, ErrorState, PageHeader, SkeletonRows } from "@/components/common";
import { useDomain } from "@/lib/domain";
import { Badge } from "@/components/ui/badge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Tooltip } from "@/components/ui/tooltip";
import { useInsights } from "@/lib/api";
import { DOMAIN_LABEL, fmtInt, fmtMetric, fmtPct, label } from "@/lib/format";
import type { InsightsResponse, MetricSet } from "@/lib/types";

const METRICS: { key: string; name: string; help: string }[] = [
  { key: "roc_auc", name: "ROC-AUC", help: "Ranking quality across every threshold (0.5 = random)." },
  { key: "average_precision", name: "Average precision", help: "Area under the precision–recall curve; compare with prevalence." },
  { key: "precision", name: "Precision", help: "Share of flagged claims whose label is positive." },
  { key: "recall", name: "Recall", help: "Share of positive-label claims that were flagged." },
  { key: "f1", name: "F1", help: "Harmonic mean of precision and recall." },
  { key: "specificity", name: "Specificity", help: "Share of negative-label claims not flagged." },
  { key: "balanced_accuracy", name: "Balanced accuracy", help: "Mean of recall and specificity." },
  { key: "accuracy", name: "Accuracy", help: "Share of all claims classified correctly; misleading when positives are rare." },
];

function MetricGrid({ metrics }: { metrics: MetricSet }) {
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {METRICS.filter((m) => metrics[m.key] != null).map((m) => (
        <Tooltip key={m.key} content={m.help}>
          <div tabIndex={0} className="cursor-help rounded-lg border border-border bg-surface px-3 py-2.5">
            <dt className="text-2xs font-medium text-text-muted">{m.name}</dt>
            <dd className="tabular mt-0.5 text-xl font-semibold text-text" title={String(metrics[m.key])}>
              {fmtMetric(metrics[m.key])}
            </dd>
          </div>
        </Tooltip>
      ))}
    </dl>
  );
}

function SampleLine({ metrics, threshold, unit = "claims" }: { metrics: MetricSet; threshold?: number | null; unit?: string }) {
  const n = metrics.n_samples ?? null;
  const pos = metrics.n_positive ?? null;
  return (
    <p className="tabular mt-3 text-xs text-text-muted">
      {fmtInt(n)} {unit} · {fmtInt(pos)} positive-label ({n && pos != null ? fmtPct(pos / n) : "—"} prevalence) · threshold{" "}
      {fmtMetric(threshold ?? metrics.threshold)} · {fmtInt(metrics.review_volume)} flagged ({fmtPct(metrics.review_rate)} review rate)
    </p>
  );
}

function Section({ title, description, icon, children }: { title: string; description?: ReactNode; icon?: ReactNode; children: ReactNode }) {
  return (
    <Card>
      <CardHeader title={title} description={description} icon={icon} />
      <CardBody>{children}</CardBody>
    </Card>
  );
}

export default function InsightsPage() {
  const domain = useDomain();
  const { data, error, refetch, isLoading } = useInsights(domain);

  return (
    <>
      <PageHeader
        eyebrow={`${DOMAIN_LABEL[domain]} model`}
        title="Model insights"
        description="Training-time evaluation read directly from the saved artifacts in models/. Validation results guided threshold selection; test results come from a separate held-out split scored after choices were frozen. Nothing on this page is recomputed."
      />
      {error ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading || !data ? (
        <Card className="p-5">
          <SkeletonRows rows={10} />
        </Card>
      ) : (
        <InsightsBody data={data} />
      )}
    </>
  );
}

function InsightsBody({ data }: { data: InsightsResponse }) {
  const domain = data.domain;
  const val = data.eval_metrics;
  const test = data.test_metrics?.selected_threshold_metrics;
  const positiveLabel = domain === "health" ? "positive (provider proxy)" : "positive";
  const prevalence = val?.n_samples && val.n_positive != null ? val.n_positive / val.n_samples : undefined;
  const providerGrouped = data.provider_grouped_test_metrics;
  const comparison = (data.data_audit?.synthetic_graph_representation_comparison ?? null) as Record<
    string,
    { validation_average_precision?: number }
  > | null;
  const apStructured = comparison?.structured_only?.validation_average_precision;
  const apGraph = comparison?.structured_plus_split_graph?.validation_average_precision;
  const graphHelped = apStructured != null && apGraph != null ? apGraph > apStructured : null;

  return (
    <div className="space-y-4">
      <section aria-label="Known limitations" className="grid gap-3 md:grid-cols-2">
        {data.limitations.map((t) => (
          <Callout key={t} tone="warning" title="Known limitation">
            {t}
          </Callout>
        ))}
      </section>

      <Card className="flex flex-wrap items-center gap-x-6 gap-y-2 p-4 text-xs text-text-muted">
        <span>
          Artifact: <code className="font-mono text-text">{data.model_dir}</code>
        </span>
        <span>{data.feature_names.length} model features</span>
        <span>
          Graph features used by the model:{" "}
          <Badge tone={data.uses_graph_features ? "primary" : "neutral"}>{data.uses_graph_features ? "Yes" : "No"}</Badge>
        </span>
        <span className="tabular">
          Live triage bands: medium ≥ {fmtPct(data.risk_bands.medium, 0)}, high ≥ {fmtPct(data.risk_bands.high, 0)} (separate from the
          operating threshold below)
        </span>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Section
          title="Validation results"
          description="Used to choose model settings and the operating threshold"
          icon={<FlaskConical />}
        >
          {val ? (
            <>
              {data.threshold_selection ? (
                <div className="mb-3 flex flex-wrap gap-2 text-xs">
                  <Badge tone="primary">
                    <Target aria-hidden="true" /> Threshold {fmtMetric(data.threshold_selection.threshold)}
                  </Badge>
                  <Badge tone="neutral">Objective: {data.threshold_selection.objective}</Badge>
                  <Badge tone="neutral">
                    Constraints:{" "}
                    {Object.keys(data.threshold_selection.constraints ?? {}).length
                      ? JSON.stringify(data.threshold_selection.constraints)
                      : "none"}
                  </Badge>
                </div>
              ) : null}
              <MetricGrid metrics={val} />
              <SampleLine metrics={val} threshold={data.threshold_selection?.threshold} />
            </>
          ) : (
            <EmptyState title="No validation metrics saved">Run scripts/train_models.py to generate evaluation sidecars.</EmptyState>
          )}
        </Section>

        <Section
          title="Untouched test results"
          description="Held-out split, scored once after all choices were frozen"
          icon={<ShieldQuestion />}
        >
          {test ? (
            <>
              <div className="mb-3 flex flex-wrap gap-2 text-xs">
                <Badge tone="primary">
                  <Target aria-hidden="true" /> Frozen threshold {fmtMetric(data.test_metrics?.threshold_frozen_from_validation)}
                </Badge>
                <Badge tone="neutral">Split: {data.test_metrics?.split ?? "test"}</Badge>
              </div>
              <MetricGrid metrics={test} />
              <SampleLine metrics={test} threshold={data.test_metrics?.threshold_frozen_from_validation} />
            </>
          ) : (
            <EmptyState title="No test metrics saved" />
          )}
        </Section>
      </div>

      {providerGrouped?.metrics ? (
        <Section
          title="Provider-grouped test results"
          description={`${providerGrouped.aggregation ?? "Aggregated per provider"} · ${fmtInt(providerGrouped.n_providers)} held-out providers. The label is provider-level, so this view matches its granularity.`}
          icon={<Gauge />}
        >
          <MetricGrid metrics={providerGrouped.metrics} />
          <SampleLine metrics={providerGrouped.metrics} unit="providers" />
        </Section>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <Section title="ROC curve" description={`Validation split · ROC-AUC ${fmtMetric(val?.roc_auc)}`}>
          {data.roc_curve ? (
            <div className="pt-8">
              <CurveChart
                kind="roc"
                x={data.roc_curve.fpr}
                y={data.roc_curve.tpr}
                thresholds={data.roc_curve.thresholds}
                areaLabel={`ROC-AUC ${fmtMetric(val?.roc_auc)}`}
              />
            </div>
          ) : (
            <EmptyState title="No ROC curve saved" />
          )}
        </Section>
        <Section
          title="Precision–recall curve"
          description={`Validation split · average precision ${fmtMetric(val?.average_precision)} · dashed line = prevalence`}
        >
          {data.pr_curve ? (
            <div className="pt-8">
              <CurveChart
                kind="pr"
                x={data.pr_curve.recall}
                y={data.pr_curve.precision}
                thresholds={data.pr_curve.thresholds}
                baseline={prevalence}
                areaLabel={`Average precision ${fmtMetric(val?.average_precision)}`}
              />
            </div>
          ) : (
            <EmptyState title="No PR curve saved" />
          )}
        </Section>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Section
          title="Validation confusion matrix"
          description={data.confusion_matrix ? `At threshold ${fmtMetric(data.confusion_matrix.threshold)}` : undefined}
        >
          {data.confusion_matrix ? (
            <ConfusionMatrix cm={data.confusion_matrix} positiveLabel={positiveLabel} />
          ) : (
            <EmptyState title="No confusion matrix saved" />
          )}
        </Section>
        <Section
          title="Test confusion matrix"
          description={data.test_metrics?.confusion_matrix ? `At threshold ${fmtMetric(data.test_metrics.confusion_matrix.threshold)}` : undefined}
        >
          {data.test_metrics?.confusion_matrix ? (
            <ConfusionMatrix cm={data.test_metrics.confusion_matrix} positiveLabel={positiveLabel} />
          ) : (
            <EmptyState title="No test confusion matrix saved" />
          )}
        </Section>
      </div>

      {comparison ? (
        <Section
          title="Graph feature representation comparison"
          description="Validation average precision for each feature representation (data_audit.json)"
          icon={<AlertTriangle />}
        >
          <table className="tabular w-full max-w-lg text-sm">
            <thead className="text-left text-xs text-text-muted">
              <tr>
                <th scope="col" className="py-1.5 font-medium">
                  Representation
                </th>
                <th scope="col" className="py-1.5 text-right font-medium">
                  Validation AP
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison).map(([k, v]) => (
                <tr key={k} className="border-t border-border">
                  <td className="py-1.5 text-text">{label(k)}</td>
                  <td className="py-1.5 text-right font-medium text-text" title={String(v.validation_average_precision)}>
                    {fmtMetric(v.validation_average_precision)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-text-muted">
            Selected representation: {String(data.data_audit?.selected_features ?? "—")}.{" "}
            {graphHelped == null
              ? null
              : graphHelped
                ? "Adding the synthetic graph features raised validation average precision."
                : "Adding the synthetic graph features did not raise validation average precision."}
          </p>
        </Section>
      ) : null}

      <Section title="Model features" description="The exact columns the saved model was trained on">
        <ul className="flex flex-wrap gap-1.5">
          {data.feature_names.map((f) => (
            <li key={f}>
              <Badge tone="neutral" title={f}>
                {label(f)}
              </Badge>
            </li>
          ))}
        </ul>
      </Section>

      <p className="text-2xs text-text-subtle">
        {domain === "motor"
          ? "Positive-label claims are rare in the motor data, so accuracy and precision at a single threshold can look weak; ranking metrics (ROC-AUC, average precision) are the primary measures for triage prioritisation."
          : "Health metrics measure how well claim scores separate the provider-level proxy label; the provider-grouped view matches the label's granularity."}
      </p>
    </div>
  );
}
