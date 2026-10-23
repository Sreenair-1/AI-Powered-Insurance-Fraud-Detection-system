import { ArrowLeft, FileSearch, FolderOpen, Network, NotebookPen, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Callout, DecisionBanner, ErrorState, ProbabilityMeter, RiskBadge, SkeletonRows } from "@/components/common";
import { EntityGraph } from "@/components/entity-graph";
import { DocumentRescore } from "@/components/document-rescore";
import { useDomain } from "@/lib/domain";
import { ShapChart } from "@/components/shap-chart";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Label, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, useClaimDetail, useClaimGraph } from "@/lib/api";
import { DOMAIN_LABEL, fmtPct, fmtValue, label, shortClaim } from "@/lib/format";
import type { ClaimDetail, Domain } from "@/lib/types";

export default function ClaimDetailPage() {
  const domain = useDomain();
  const { claimKey = "" } = useParams();
  const key = decodeURIComponent(claimKey);
  const detail = useClaimDetail(domain, key);
  const graph = useClaimGraph(domain, key);

  const back = (
    <Button asChild variant="ghost" size="sm" className="-ml-2 mb-3">
      <Link to={`/${domain}/claims`}>
        <ArrowLeft aria-hidden="true" /> All claims
      </Link>
    </Button>
  );

  if (detail.error) {
    const notFound = detail.error instanceof ApiError && detail.error.status === 404;
    return (
      <>
        {back}
        <ErrorState error={detail.error} onRetry={notFound ? undefined : () => detail.refetch()} />
        {notFound ? (
          <p className="mt-3 text-center text-xs text-text-muted">
            This claim is not in the currently loaded {DOMAIN_LABEL[domain].toLowerCase()} sample. It may have been
            excluded when the sample size changed.
          </p>
        ) : null}
      </>
    );
  }

  const d = detail.data;
  return (
    <>
      {back}
      <CaseHeader detail={d} claimKey={key} domain={domain} />

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4">
          <Card>
            <CardHeader
              title="Why this score"
              description="Top SHAP factors for this claim. Red bars raise the risk score, green bars lower it."
              icon={<Sparkles />}
            />
            <CardBody>
              {d ? <ShapChart factors={d.assessment.top_factors} baseValue={d.assessment.base_value} /> : <SkeletonRows rows={8} />}
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title="Entity network"
              description="The claim, its linked entities and other sample claims that share them. Click a node or pick it from the list."
              icon={<Network />}
            />
            <CardBody>
              {graph.error ? (
                <ErrorState error={graph.error} onRetry={() => graph.refetch()} />
              ) : graph.data ? (
                <EntityGraph data={graph.data} domain={domain} />
              ) : (
                <Skeleton className="h-[420px]" />
              )}
              {d && !d.model_uses_graph_features ? (
                <Callout className="mt-4">
                  The saved {DOMAIN_LABEL[domain].toLowerCase()} model does not use graph features, so this network is
                  investigative context and does not change the score.
                  {domain === "motor" ? " Garage and bank-account links are synthetic (seeded augmentation)." : ""}
                </Callout>
              ) : null}
            </CardBody>
          </Card>
        </div>

        <div className="min-w-0 space-y-4">
          <Card>
            <CardHeader title="Claim attributes" description="As recorded in the source dataset" icon={<FolderOpen />} />
            <CardBody className="space-y-5">
              {d ? (
                d.attribute_groups.map((g) => (
                  <section key={g.name} aria-labelledby={`grp-${g.name}`}>
                    <h3 id={`grp-${g.name}`} className="mb-2 text-2xs font-semibold uppercase tracking-wide text-text-subtle">
                      {g.name}
                    </h3>
                    <dl className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
                      {g.fields.map((f) => (
                        <div key={f} className="min-w-0 border-b border-dashed border-border pb-1.5">
                          <dt className="text-2xs text-text-muted">{label(f)}</dt>
                          <dd className="tabular break-words text-sm font-medium text-text">{fmtValue(d.attributes[f])}</dd>
                        </div>
                      ))}
                    </dl>
                  </section>
                ))
              ) : (
                <SkeletonRows rows={8} />
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Graph signals" description="Entity-sharing and centrality features from the loaded sample graph" icon={<Network />} />
            <CardBody>
              {d ? (
                <dl className="tabular grid grid-cols-2 gap-3">
                  {Object.entries(d.graph_signals).map(([k, v]) => (
                    <div key={k} className="rounded-md bg-surface-sunken px-3 py-2">
                      <dt className="text-2xs text-text-muted">{label(k)}</dt>
                      <dd className="mt-0.5 text-sm font-semibold text-text">
                        {k === "has_shared_entity" ? (
                          v ? (
                            <span className="text-accent">Yes</span>
                          ) : (
                            "No"
                          )
                        ) : typeof v === "number" && !Number.isInteger(v) ? (
                          <span title={String(v)}>{v.toFixed(4)}</span>
                        ) : (
                          fmtValue(v)
                        )}
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <SkeletonRows rows={3} />
              )}
            </CardBody>
          </Card>

          <InvestigatorNotes key={`${domain}:${key}`} claimKey={key} domain={domain} />
        </div>
      </div>

      <section className="mt-6" aria-labelledby="doc-heading">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 id="doc-heading" className="text-lg font-semibold tracking-tight text-text">
              Supporting document
            </h2>
            <p className="text-sm text-text-muted">
              Upload an invoice, repair estimate, police report or bill to run forensics and re-score this claim.
            </p>
          </div>
          <Button asChild variant="link" size="sm">
            <Link to={`/${domain}/forensics?claim=${encodeURIComponent(key)}`}>
              <FileSearch aria-hidden="true" /> Open in forensics workspace
            </Link>
          </Button>
        </div>
        <DocumentRescore key={key} domain={domain} claimKey={key} />
      </section>
    </>
  );
}

function CaseHeader({ detail, claimKey, domain }: { detail?: ClaimDetail; claimKey: string; domain: Domain }) {
  return (
    <Card className="overflow-hidden">
      <div className="grid gap-6 p-5 md:grid-cols-[1fr_minmax(16rem,22rem)] md:p-6">
        <div className="min-w-0">
          <p className="text-2xs font-semibold uppercase tracking-[0.12em] text-accent">
            Case file · {DOMAIN_LABEL[domain]} claim
          </p>
          <h1 className="mt-1 break-all text-2xl font-semibold tracking-tight text-text">Claim {shortClaim(claimKey)}</h1>
          {detail ? (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <RiskBadge risk={detail.assessment.risk_category} size="md" />
              <span className="text-sm text-text-muted">{detail.assessment.risk_category_description}</span>
            </div>
          ) : (
            <Skeleton className="mt-3 h-7 w-64" />
          )}
          <DecisionBanner className="mt-4" />
        </div>
        <div className="rounded-lg border border-border bg-surface-sunken p-4">
          <p className="text-xs font-medium text-text-muted">Risk score (model probability)</p>
          {detail ? (
            <>
              <p className="tabular mt-1 text-4xl font-semibold tracking-tight text-text" title={String(detail.assessment.fraud_probability)}>
                {fmtPct(detail.assessment.fraud_probability)}
              </p>
              <ProbabilityMeter value={detail.assessment.fraud_probability} bands={detail.risk_bands} className="mt-3" />
            </>
          ) : (
            <Skeleton className="mt-2 h-16" />
          )}
        </div>
      </div>
    </Card>
  );
}

const REVIEW_STATUSES = ["Unreviewed", "Under investigation", "Referred to SIU", "Cleared after review"];

function InvestigatorNotes({ claimKey, domain }: { claimKey: string; domain: Domain }) {
  const storageKey = `insuresight.notes.${domain}.${claimKey}`;
  const [state, setState] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(storageKey) ?? "") as { status: string; note: string };
    } catch {
      return { status: REVIEW_STATUSES[0], note: "" };
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch {
      /* storage unavailable */
    }
  }, [state, storageKey]);
  return (
    <Card>
      <CardHeader
        title="Investigator notes"
        description="Saved in this browser only. Never written back to the dataset or the model."
        icon={<NotebookPen />}
      />
      <CardBody className="space-y-3">
        <div className="space-y-1.5">
          <Label htmlFor="review-status">Review status</Label>
          <Select id="review-status" value={state.status} onChange={(e) => setState({ ...state, status: e.target.value })}>
            {REVIEW_STATUSES.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="review-note">Notes</Label>
          <textarea
            id="review-note"
            rows={3}
            value={state.note}
            onChange={(e) => setState({ ...state, note: e.target.value })}
            placeholder="Rationale, document checks, next steps…"
            className="w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-sm text-text shadow-xs placeholder:text-text-subtle"
          />
        </div>
      </CardBody>
    </Card>
  );
}
