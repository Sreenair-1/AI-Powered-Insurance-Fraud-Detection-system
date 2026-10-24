import { FileWarning, Network } from "lucide-react";
import type { ReactNode } from "react";
import { fmtInt, fmtMetric, fmtPct, label } from "@/lib/format";
import type { ForensicMethod, RingBenchmark, TamperingBenchmark } from "@/lib/types";
import { Callout, EmptyState } from "./common";
import { Card, CardBody, CardHeader } from "./ui/card";

const METHOD_LABEL: Record<ForensicMethod, string> = {
  ela: "Error level analysis",
  copy_move: "Copy-move",
  metadata: "Metadata",
  combined_verdict: "Combined verdict",
};
const METHODS = Object.keys(METHOD_LABEL) as ForensicMethod[];
const CONDITION_LABEL: Record<string, string> = { as_submitted: "As submitted", messaging_app: "After chat re-share" };
const MATCHING_LABEL: Record<string, string> = { raw: "Exact", normalized: "Normalised", fuzzy: "Typo-tolerant" };

function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return (
    <th scope="col" className={`px-2 py-1.5 font-medium ${right ? "text-right" : ""}`}>
      {children}
    </th>
  );
}

function Td({ children, right, strong, title }: { children: ReactNode; right?: boolean; strong?: boolean; title?: string }) {
  return (
    <td className={`px-2 py-1.5 ${right ? "text-right" : ""} ${strong ? "font-medium text-text" : "text-text-muted"}`} title={title}>
      {children}
    </td>
  );
}

function Table({ head, children, caption }: { head: ReactNode; children: ReactNode; caption: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[32rem] text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead className="text-left text-xs text-text-muted">
          <tr>{head}</tr>
        </thead>
        <tbody className="[&_tr]:border-t [&_tr]:border-border">{children}</tbody>
      </table>
    </div>
  );
}

const pct = (v: number | null | undefined) => fmtPct(v, 0);
// Ring counts are means over several generated populations.
const count = (v: number | null | undefined) => (v == null ? "—" : fmtInt(Math.round(v)));
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function TamperingBenchmarkSection({ data }: { data: TamperingBenchmark | null }) {
  if (!data) {
    return (
      <Card>
        <CardHeader title="Document tampering benchmark" icon={<FileWarning />} />
        <CardBody>
          <EmptyState title="No tampering benchmark saved">Run scripts/run_benchmarks.py to generate models/benchmarks/.</EmptyState>
        </CardBody>
      </Card>
    );
  }
  const conditions = Object.keys(data.results);
  const first = data.results[conditions[0]];
  const verdictFpr = first?.methods.combined_verdict.false_positive_rate;
  return (
    <Card>
      <CardHeader
        title="Document tampering benchmark"
        description={`${fmtInt(data.n_bills)} ${data.source === "folder" ? "supplied" : "rendered"} bills, each scored untouched (${data.untouched_variants.length} variants) and with ${data.edit_types.length} known edits · ${fmtInt(data.n_samples)} files`}
        icon={<FileWarning />}
      />
      <CardBody className="space-y-5">
        {verdictFpr != null && verdictFpr >= 0.5 ? (
          <Callout tone="warning" title="The current checks do not separate edited from untouched bills">
            The combined verdict flagged {pct(verdictFpr)} of untouched bills. Printed text repeats the same characters, which
            the copy-move check reads as cloned regions, and global ELA statistics do not move for a small edited area.
            Treat document risk as a prompt to inspect the file, not as evidence.
          </Callout>
        ) : null}

        <Table
          caption="Detection and false-positive rate per forensic check"
          head={
            <>
              <Th>Check</Th>
              {conditions.map((c) => (
                <Th key={c} right>
                  {CONDITION_LABEL[c] ?? label(c)}: detected / false alarms
                </Th>
              ))}
              <Th right>ROC-AUC</Th>
              <Th right>Best detection at ≤10% false alarms</Th>
            </>
          }
        >
          {METHODS.map((m) => (
            <tr key={m}>
              <Td strong>{METHOD_LABEL[m]}</Td>
              {conditions.map((c) => (
                <Td key={c} right>
                  {pct(data.results[c].methods[m].detection_rate)} / {pct(data.results[c].methods[m].false_positive_rate)}
                </Td>
              ))}
              <Td right title="As submitted; continuous signal">
                {fmtMetric(first.methods[m].roc_auc_continuous, 2)}
              </Td>
              <Td right title="As submitted">
                {pct(first.methods[m].best_detection_at_fpr_10.detection_rate)}
              </Td>
            </tr>
          ))}
        </Table>

        <Table
          caption="Detection rate per edit type, as submitted"
          head={
            <>
              <Th>Edit (as submitted)</Th>
              {METHODS.map((m) => (
                <Th key={m} right>
                  {METHOD_LABEL[m]}
                </Th>
              ))}
            </>
          }
        >
          {data.edit_types.map((edit) => (
            <tr key={edit}>
              <Td strong>{label(edit)}</Td>
              {METHODS.map((m) => (
                <Td key={m} right>
                  {pct(first.detection_rate_by_edit[edit]?.[m])}
                </Td>
              ))}
            </tr>
          ))}
        </Table>

        <ul className="list-disc space-y-0.5 pl-4 text-2xs text-text-subtle">
          {Object.entries(data.conditions).map(([k, v]) => (
            <li key={k}>
              {CONDITION_LABEL[k] ?? label(k)}: {v}.
            </li>
          ))}
          {data.limitations.map((l) => (
            <li key={l}>{l}.</li>
          ))}
        </ul>
      </CardBody>
    </Card>
  );
}

export function RingBenchmarkSection({ data }: { data: RingBenchmark | null }) {
  if (!data) {
    return (
      <Card>
        <CardHeader title="Fraud-ring benchmark" icon={<Network />} />
        <CardBody>
          <EmptyState title="No ring benchmark saved">Run scripts/run_benchmarks.py to generate models/benchmarks/.</EmptyState>
        </CardBody>
      </Card>
    );
  }
  const modes = ["raw", "normalized", "fuzzy"] as const;
  const points = data.operating_points.rows.filter((r) => r.alpha === data.detector.alpha);
  return (
    <Card>
      <CardHeader
        title="Fraud-ring benchmark"
        description={`${fmtInt(data.scenario.n_claims)} generated claims with ${fmtInt(data.scenario.n_rings)} rings (${fmtPct(data.ring_member_prevalence)} of claims), averaged over ${data.seeds.length} populations per row`}
        icon={<Network />}
      />
      <CardBody className="space-y-5">
        <p className="text-xs text-text-muted">{sentence(data.detector.method)}. Noise is the share of recorded values reformatted or carrying a one-character typo.</p>
        <Table
          caption="Ring recovery and precision by noise level and matching mode"
          head={
            <>
              <Th>Noise</Th>
              {modes.map((m) => (
                <Th key={m} right>
                  {MATCHING_LABEL[m]}: rings found / precision
                </Th>
              ))}
              <Th right>Any shared entity: precision</Th>
            </>
          }
        >
          {data.noise_sweep.map((row) => (
            <tr key={row.noise}>
              <Td strong>{pct(row.noise)}</Td>
              {modes.map((m) => (
                <Td key={m} right>
                  {pct(row[m].ring_recovery_rate)} / {pct(row[m].precision)}
                </Td>
              ))}
              <Td right>{pct(row.naive_any_shared_entity.precision)}</Td>
            </tr>
          ))}
        </Table>

        {points.length ? (
          <Table
            caption="Effect of the minimum ring size"
            head={
              <>
                <Th>Minimum ring size</Th>
                <Th right>Rings found</Th>
                <Th right>Precision</Th>
                <Th right>Recall</Th>
                <Th right>Candidate rings</Th>
                <Th right>Of which not a ring</Th>
              </>
            }
          >
            {points.map((r) => (
              <tr key={r.min_ring_size}>
                <Td strong>{r.min_ring_size}</Td>
                <Td right>{pct(r.ring_recovery_rate)}</Td>
                <Td right>{pct(r.precision)}</Td>
                <Td right>{pct(r.recall)}</Td>
                <Td right>{count(r.n_candidate_rings)}</Td>
                <Td right>{count(r.n_false_candidate_rings)}</Td>
              </tr>
            ))}
          </Table>
        ) : null}

        <ul className="list-disc space-y-0.5 pl-4 text-2xs text-text-subtle">
          <li>
            Minimum ring size table: {MATCHING_LABEL[data.operating_points.matching] ?? data.operating_points.matching} matching at{" "}
            {pct(data.operating_points.noise)} noise. The live app uses size {data.detector.min_ring_size}.
          </li>
          <li>{sentence(data.recovery_rule)}.</li>
          {data.limitations.map((l) => (
            <li key={l}>{l}.</li>
          ))}
        </ul>
      </CardBody>
    </Card>
  );
}
