import { motion } from "framer-motion";
import { AlertOctagon, AlertTriangle, ArrowRight, ClipboardList, Layers, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { ImportanceChart, ProbabilityHistogram, RiskDistributionChart, SegmentChart } from "@/components/charts";
import { DecisionBanner, EmptyState, ErrorState, KpiCard, PageHeader, RiskBadge, SkeletonRows } from "@/components/common";
import { useDomain } from "@/lib/domain";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useOverview } from "@/lib/api";
import { DOMAIN_LABEL, fmtInt, fmtPct, fmtValue, label, shortClaim } from "@/lib/format";
import { usePrefersReducedMotion } from "@/lib/utils";

export default function OverviewPage() {
  const domain = useDomain();
  const { data, isLoading, error, refetch } = useOverview(domain);
  const reduced = usePrefersReducedMotion();

  const header = (
    <PageHeader
      eyebrow={`${DOMAIN_LABEL[domain]} insurance`}
      title="Investigation overview"
      description="Model risk scores across the currently loaded claim sample. These are live scores for triage, kept separate from the training-time evaluation in Model insights."
      actions={
        <Button asChild variant="outline">
          <Link to={`/${domain}/claims`}>
            Open claims workspace <ArrowRight aria-hidden="true" />
          </Link>
        </Button>
      }
    />
  );

  if (error) {
    return (
      <>
        {header}
        <ErrorState error={error} onRetry={() => refetch()} />
      </>
    );
  }

  const n = data?.total_claims ?? 0;
  const share = (k: number) => (n ? `${fmtPct(k / n)} of sample` : "");
  const sampleText = data
    ? domain === "motor"
      ? `up to ${fmtInt(data.sample.sample_size)} requested`
      : `up to ${fmtInt(data.sample.inpatient_sample_size)} inpatient + ${fmtInt(data.sample.outpatient_sample_size)} outpatient`
    : "";

  return (
    <>
      {header}
      <DecisionBanner className="mb-6" />

      <section aria-label="Key figures" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {isLoading || !data ? (
          Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-[7.25rem]" />)
        ) : (
          <>
            <KpiCard label="Claims in sample" value={n} hint={`Seeded sample · ${sampleText}`} icon={<ClipboardList />} />
            <KpiCard label="High risk" value={data.risk_counts.high} tone="high" hint={`${share(data.risk_counts.high)} · priority 1`} icon={<AlertOctagon />} />
            <KpiCard label="Medium risk" value={data.risk_counts.medium} tone="medium" hint={`${share(data.risk_counts.medium)} · priority 2`} icon={<AlertTriangle />} />
            <KpiCard label="Low risk" value={data.risk_counts.low} tone="low" hint={share(data.risk_counts.low)} icon={<ShieldCheck />} />
            <KpiCard
              label="Review rate"
              value={(data.review_rate ?? 0) * 100}
              format={(v) => `${v.toFixed(1)}%`}
              hint={`${fmtInt(data.review_count)} claims flagged for review (high + medium)`}
              icon={<Layers />}
            />
          </>
        )}
      </section>

      <div className="mt-6 grid gap-4 xl:grid-cols-[1fr_1.25fr]">
        <Card>
          <CardHeader
            title="Risk distribution"
            description={data ? `Bands: low < ${fmtPct(data.risk_bands.medium, 0)} ≤ medium < ${fmtPct(data.risk_bands.high, 0)} ≤ high` : "Claims per risk level"}
          />
          <CardBody>
            {data ? (
              <>
                <RiskDistributionChart counts={data.risk_counts} total={n} />
                <div className="mt-6 rounded-lg bg-surface-sunken p-4">
                  <p className="text-xs font-medium text-text-muted">Review queue</p>
                  <p className="tabular mt-1 text-sm text-text">
                    <span className="text-lg font-semibold">{fmtInt(data.review_count)}</span> of {fmtInt(n)} claims are flagged
                    for human review ({fmtPct(data.review_rate)}). {fmtInt(data.risk_counts.high)} high-risk claims form the first
                    priority.
                  </p>
                  <p className="mt-2 text-2xs text-text-subtle">
                    Bands are triage conventions applied to model scores, not fitted operating points or outcome rates.
                  </p>
                </div>
              </>
            ) : (
              <SkeletonRows rows={3} />
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Risk score distribution" description="Claims per 10-point score range; dashed lines mark the band cut-offs" />
          <CardBody className="pt-12">
            {data ? <ProbabilityHistogram bins={data.probability_histogram} bands={data.risk_bands} /> : <Skeleton className="h-60" />}
          </CardBody>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.25fr_1fr]">
        <Card>
          <CardHeader
            title="Highest-risk claims"
            description="High risk first, then medium, ordered by score"
            action={
              <Button asChild variant="link" size="sm">
                <Link to={`/${domain}/claims?risk=high`}>View all</Link>
              </Button>
            }
          />
          <CardBody className="px-2 pb-2 pt-3">
            {!data ? (
              <SkeletonRows rows={8} className="px-3" />
            ) : data.top_claims.length === 0 ? (
              <EmptyState title="No claims in this sample" />
            ) : (
              <ol>
                {data.top_claims.map((c, i) => {
                  const detailFields = (domain === "motor" ? ["Make", "PolicyType", "AccidentArea"] : ["claim_type", "Provider", "InscClaimAmtReimbursed"])
                    .filter((f) => c.attributes[f] != null)
                    .map((f) => `${label(f)}: ${fmtValue(c.attributes[f])}`)
                    .join(" · ");
                  return (
                    <motion.li
                      key={c.claim_key}
                      initial={reduced ? false : { opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: reduced ? 0 : i * 0.035, duration: 0.25 }}
                    >
                      <Link
                        to={`/${domain}/claims/${encodeURIComponent(c.claim_key)}`}
                        className="group grid grid-cols-[1.5rem_1fr_auto] items-center gap-3 rounded-md px-3 py-2.5 hover:bg-surface-sunken"
                      >
                        <span className="tabular text-xs text-text-subtle">{i + 1}</span>
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-medium text-text group-hover:text-primary">
                            Claim {shortClaim(c.claim_key)}
                          </span>
                          <span className="block truncate text-xs text-text-muted">{detailFields}</span>
                        </span>
                        <span className="flex items-center gap-3">
                          <span className="tabular text-sm font-semibold text-text">{fmtPct(c.fraud_probability)}</span>
                          <RiskBadge risk={c.risk_category} className="hidden sm:inline-flex" />
                        </span>
                      </Link>
                    </motion.li>
                  );
                })}
              </ol>
            )}
          </CardBody>
        </Card>
        <div className="space-y-4">
          {data?.segment_field && data.segments.length ? (
            <Card>
              <CardHeader title={`Risk by ${label(data.segment_field).toLowerCase()}`} description="Claim volume split by risk level" />
              <CardBody className="pt-12">
                <SegmentChart field={data.segment_field} segments={data.segments} />
              </CardBody>
            </Card>
          ) : null}
          <Card>
            <CardHeader title="Top model drivers" description="Global XGBoost feature importance (gain), learned at training time" />
            <CardBody className="pt-12">
              {data ? (
                data.global_importance.length ? (
                  <ImportanceChart items={data.global_importance} />
                ) : (
                  <EmptyState title="Feature importance unavailable" />
                )
              ) : (
                <Skeleton className="h-48" />
              )}
            </CardBody>
          </Card>
        </div>
      </div>
    </>
  );
}
