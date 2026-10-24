import { ArrowRight, Check, FileSearch, Search } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Callout, DecisionBanner, EmptyState, ErrorState, PageHeader, RiskBadge, SkeletonRows } from "@/components/common";
import { DocumentRescore } from "@/components/document-rescore";
import { useDomain } from "@/lib/domain";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import { useClaimDetail, useClaims } from "@/lib/api";
import { DOMAIN_LABEL, fmtPct, shortClaim } from "@/lib/format";
import { cn, useDebounced } from "@/lib/utils";

export default function ForensicsPage() {
  const domain = useDomain();
  const [sp, setSp] = useSearchParams();
  const selected = sp.get("claim");
  const [query, setQuery] = useState("");
  const q = useDebounced(query, 250);
  const results = useClaims(domain, {
    search: q,
    risk: [],
    minProbability: 0,
    segment: [],
    sort: "fraud_probability",
    order: "desc",
    page: 1,
    pageSize: 8,
  });

  return (
    <>
      <PageHeader
        eyebrow={`${DOMAIN_LABEL[domain]} insurance`}
        title="Document forensics"
        description="Attach a claim document to a loaded claim. Error level analysis, ORB copy-move detection and EXIF/PDF metadata checks run on the file, then the claim is re-scored with that evidence."
        actions={
          <Button asChild variant="outline">
            <Link to={`/${domain}/assess`}>
              Assess a new claim instead <ArrowRight aria-hidden="true" />
            </Link>
          </Button>
        }
      />
      <DecisionBanner compact className="mb-4" />

      <div className="grid gap-4 lg:grid-cols-[20rem_1fr]">
        <Card className="self-start">
          <CardHeader title="1. Choose the claim" description="Highest-risk claims are listed first" />
          <CardBody className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="claim-picker">Find a claim</Label>
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-text-subtle" aria-hidden="true" />
                <Input
                  id="claim-picker"
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Claim, provider or make"
                  className="pl-9"
                />
              </div>
            </div>
            {results.error ? (
              <ErrorState error={results.error} onRetry={() => results.refetch()} />
            ) : !results.data ? (
              <SkeletonRows rows={6} />
            ) : results.data.items.length === 0 ? (
              <p className="text-xs text-text-muted">No claims match “{q}”.</p>
            ) : (
              <ul className="-mx-1 space-y-0.5" aria-label="Matching claims">
                {results.data.items.map((c) => {
                  const on = c.claim_key === selected;
                  return (
                    <li key={c.claim_key}>
                      <button
                        type="button"
                        aria-pressed={on}
                        onClick={() => setSp({ claim: c.claim_key }, { replace: true })}
                        className={cn(
                          "flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-left text-sm hover:bg-surface-sunken",
                          on && "bg-primary-soft",
                        )}
                      >
                        {on ? <Check className="size-4 text-primary" aria-hidden="true" /> : <span className="size-4" />}
                        <span className={cn("min-w-0 flex-1 truncate font-medium", on ? "text-primary" : "text-text")}>
                          {shortClaim(c.claim_key)}
                        </span>
                        <span className="tabular text-xs text-text-muted">{fmtPct(c.fraud_probability)}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </CardBody>
        </Card>

        <div className="min-w-0 space-y-4">
          {selected ? (
            <>
              <SelectedClaim claimKey={selected} />
              <div>
                <h2 className="mb-2 text-sm font-semibold text-text">2. Upload and analyse the document</h2>
                <DocumentRescore key={`${domain}:${selected}`} domain={domain} claimKey={selected} />
              </div>
            </>
          ) : (
            <Card>
              <EmptyState title="Select a claim to begin" icon={<FileSearch />}>
                Forensics run on uploaded files only — the bulk datasets contain no documents. Pick a claim on the left,
                then drop its supporting document here.
              </EmptyState>
            </Card>
          )}
          <Callout>
            The bulk dataset has no linked documents, so forensic features are “unavailable” for every claim until a
            file is uploaded in this session. Nothing uploaded here is stored.
          </Callout>
        </div>
      </div>
    </>
  );
}

function SelectedClaim({ claimKey }: { claimKey: string }) {
  const domain = useDomain();
  const { data, error, refetch } = useClaimDetail(domain, claimKey);
  if (error) return <ErrorState error={error} onRetry={() => refetch()} />;
  return (
    <Card className="flex flex-wrap items-center gap-4 p-4">
      <div className="min-w-0 flex-1">
        <p className="text-2xs font-semibold uppercase tracking-wide text-text-subtle">Selected claim</p>
        <p className="truncate text-base font-semibold text-text">Claim {shortClaim(claimKey)}</p>
      </div>
      {data ? (
        <>
          <div className="text-right">
            <p className="text-2xs text-text-muted">Current risk score</p>
            <p className="tabular text-xl font-semibold text-text">{fmtPct(data.assessment.fraud_probability)}</p>
          </div>
          <RiskBadge risk={data.assessment.risk_category} size="md" />
        </>
      ) : (
        <SkeletonRows rows={1} className="w-40" />
      )}
      <Button asChild variant="ghost" size="sm">
        <Link to={`/${domain}/claims/${encodeURIComponent(claimKey)}`}>Open case file</Link>
      </Button>
    </Card>
  );
}
