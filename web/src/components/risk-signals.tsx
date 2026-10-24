import { FileWarning, Network } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { enc } from "@/lib/api";
import { fmtPct, label, shortClaim } from "@/lib/format";
import type { ClaimDetail, DocumentRescoreResponse, Domain, NetworkRisk } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "./ui/badge";
import { Card, CardBody, CardHeader } from "./ui/card";

const NETWORK_LEVEL: Record<NetworkRisk["level"], { text: string; tone: "high" | "medium" | "neutral" }> = {
  candidate_ring: { text: "Candidate ring", tone: "high" },
  linked: { text: "Unusual shared link", tone: "medium" },
  none: { text: "No unusual sharing", tone: "neutral" },
};

const DOCUMENT_TONE: Record<string, "high" | "medium" | "low"> = { tampered: "high", suspicious: "medium", authentic: "low" };

function SignalTile({ title, method, icon, children }: { title: string; method: string; icon: ReactNode; children: ReactNode }) {
  return (
    <Card className="flex flex-col p-4">
      <div className="flex items-center gap-2 text-text-subtle [&_svg]:size-4">
        <span aria-hidden="true">{icon}</span>
        <p className="text-xs font-medium text-text-muted">{title}</p>
      </div>
      <p className="mt-0.5 text-2xs text-text-subtle">{method}</p>
      <div className="mt-3 flex flex-1 flex-col gap-2">{children}</div>
    </Card>
  );
}

function Context({ children, warn }: { children: ReactNode; warn?: boolean }) {
  return <p className={cn("mt-auto pt-1 text-2xs leading-relaxed", warn ? "text-risk-medium" : "text-text-muted")}>{children}</p>;
}

/**
 * Network and document risk, shown beside the model score in the case
 * header. They are deliberately not merged into it: only the model score sets
 * the triage band, and each signal carries its own benchmark result so it can
 * be judged on its own.
 */
export function RiskSignals({ detail, document }: { detail: ClaimDetail; document: DocumentRescoreResponse | null }) {
  return (
    <section aria-label="Network and document risk" className="mt-3">
      <div className="grid gap-3 md:grid-cols-2">
        <NetworkTile risk={detail.network_risk} />
        <DocumentTile detail={detail} document={document} />
      </div>
      <p className="mt-2 text-2xs text-text-subtle">
        Shown separately from the model risk score above, which alone sets the triage band. Neither saved model uses these
        signals as features.
      </p>
    </section>
  );
}

function NetworkTile({ risk }: { risk: NetworkRisk }) {
  const level = NETWORK_LEVEL[risk.level];
  const bench = risk.benchmark;
  return (
    <SignalTile title="Network risk" method={`Unexpected sharing of ${risk.entity_types.map((t) => label(t).toLowerCase()).join(" + ")}`} icon={<Network />}>
      <div className="flex flex-wrap items-baseline gap-2">
        <Badge tone={level.tone} className="text-sm">
          {level.text}
        </Badge>
        {risk.level === "candidate_ring" ? (
          <span className="tabular text-sm text-text">{risk.ring_size} claims</span>
        ) : null}
      </div>
      <p className="tabular text-xs text-text-muted">
        {risk.sample_candidate_rings} candidate ring{risk.sample_candidate_rings === 1 ? "" : "s"} ({risk.sample_flagged_claims} of{" "}
        {risk.sample_claims} claims) in the loaded sample.
      </p>
      <Context>
        {bench
          ? `Ring benchmark (generated populations, same settings): finds ${fmtPct(bench.ring_recovery_rate, 0)} of rings; ${fmtPct(bench.precision, 0)} of flagged claims are true members (any-shared-entity rule: ${fmtPct(bench.naive_precision, 0)}).`
          : "Run scripts/run_benchmarks.py to show how this rule performs on generated rings."}
      </Context>
    </SignalTile>
  );
}

function DocumentTile({ detail, document }: { detail: ClaimDetail; document: DocumentRescoreResponse | null }) {
  const bench = detail.document_benchmark?.as_submitted;
  const unreliable = bench?.false_positive_rate != null && bench.false_positive_rate >= 0.5;
  const verdict = document?.forensics.verdict;
  return (
    <SignalTile title="Document risk" method="ELA, copy-move and metadata checks on an uploaded file" icon={<FileWarning />}>
      {verdict ? (
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="tabular text-2xl font-semibold text-text" title={String(verdict.tampering_score)}>
            {fmtPct(verdict.tampering_score, 0)}
          </span>
          <Badge tone={DOCUMENT_TONE[verdict.status] ?? "neutral"}>{label(verdict.status)}</Badge>
        </div>
      ) : (
        <p className="text-sm text-text-muted">No document analysed yet. Upload one below.</p>
      )}
      <Context warn={unreliable}>
        {bench
          ? `Tampering benchmark: the verdict flagged ${fmtPct(bench.detection_rate, 0)} of edited bills and ${fmtPct(bench.false_positive_rate, 0)} of untouched ones${unreliable ? ", so treat it as a prompt to inspect the file, not as evidence." : "."}`
          : "Run scripts/run_benchmarks.py to show how these checks perform on known edits."}
      </Context>
    </SignalTile>
  );
}

const fmtP = (p: number) => (p < 1e-4 ? p.toExponential(1) : p.toFixed(4));
const fmtExpected = (x: number) => (x < 0.01 ? "< 0.01" : x.toFixed(2));

/** Why the claim is in a candidate ring: the significant shared pairs and the other members. */
export function RingEvidenceCard({ risk, domain }: { risk: NetworkRisk; domain: Domain }) {
  return (
    <Card>
      <CardHeader
        title="Candidate ring evidence"
        description={risk.method}
        icon={<Network />}
      />
      <CardBody className="space-y-4">
        {risk.level === "candidate_ring" ? (
          <>
            <ul className="space-y-2">
              {risk.evidence.map((e) => (
                <li key={e.entities.map((x) => `${x.type}:${x.value}`).join("|")} className="rounded-md bg-surface-sunken px-3 py-2 text-xs">
                  <p className="font-medium text-text">
                    {e.entities.map((x) => `${label(x.type)} ${x.value}`).join(" + ")}
                  </p>
                  <p className="tabular mt-0.5 text-text-muted">
                    Used together on {e.claims} claims; chance alone predicts {fmtExpected(e.expected)} (p = {fmtP(e.p_value)})
                    {e.includes_this_claim ? "" : " · links other ring members"}
                  </p>
                </li>
              ))}
            </ul>
            <div>
              <p className="mb-1.5 text-2xs font-semibold uppercase tracking-wide text-text-subtle">
                Other claims in {risk.ring_id}
              </p>
              <ul className="flex flex-wrap gap-1.5">
                {risk.ring_claims.map((c) => (
                  <li key={c}>
                    <Link
                      to={`/${domain}/claims/${enc(c)}`}
                      className="tabular rounded-md border border-border px-2 py-0.5 text-xs text-text hover:border-border-strong hover:bg-surface-sunken"
                    >
                      {shortClaim(c)}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </>
        ) : (
          <p className="text-sm text-text-muted">
            {risk.level === "linked"
              ? "This claim shares an unusual pair of entities with one other claim, but the group is smaller than a candidate ring."
              : "This claim's entities are not shared with other claims more than their popularity explains."}
          </p>
        )}
        <p className="text-2xs text-text-subtle">{risk.note} A candidate ring is a pattern to review, not proof of collusion.</p>
      </CardBody>
    </Card>
  );
}
