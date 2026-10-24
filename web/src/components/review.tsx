import { Download, FileText, History, NotebookPen } from "lucide-react";
import { useId, useState } from "react";
import { toast } from "sonner";
import { fetchAuditReport, useClaimReview, useRecordReview, useReviewSummary } from "@/lib/api";
import { fmtPct } from "@/lib/format";
import { useSampleParams } from "@/lib/settings-store";
import type { Domain, ReviewOutcome, ReviewRecord } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ErrorState, SkeletonRows } from "./common";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardBody, CardHeader } from "./ui/card";
import { Input, Label } from "./ui/input";

const OUTCOMES: { value: ReviewOutcome; label: string; tone: "high" | "low" | "medium"; help: string }[] = [
  { value: "confirmed_suspicious", label: "Confirmed suspicious", tone: "high", help: "Becomes a positive training label" },
  { value: "cleared", label: "Cleared", tone: "low", help: "Becomes a negative training label" },
  { value: "needs_more_info", label: "Needs more info", tone: "medium", help: "Not a label until decided" },
];
const OUTCOME_LABEL = Object.fromEntries(OUTCOMES.map((o) => [o.value, o])) as Record<
  ReviewOutcome,
  (typeof OUTCOMES)[number]
>;

const REVIEWER_KEY = "insuresight.reviewer";
const legacyNoteKey = (domain: Domain, claimKey: string) => `insuresight.notes.${domain}.${claimKey}`;

function readLocal(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function writeLocal(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
}

/** A note typed into the old browser-only notes card, offered as a starting point. */
function legacyNote(domain: Domain, claimKey: string): string {
  try {
    return (JSON.parse(readLocal(legacyNoteKey(domain, claimKey)) || "{}") as { note?: string }).note ?? "";
  } catch {
    return "";
  }
}

function fmtTime(iso: string) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function HistoryItem({ r }: { r: ReviewRecord }) {
  const o = OUTCOME_LABEL[r.outcome];
  return (
    <li className="border-l-2 border-border pl-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge tone={o.tone}>{o.label}</Badge>
        <span className="text-text-muted">
          {fmtTime(r.recorded_at)}
          {r.reviewer ? ` · ${r.reviewer}` : ""}
        </span>
      </div>
      <p className="tabular mt-0.5 text-2xs text-text-subtle">
        Score at review {fmtPct(r.fraud_probability_at_review)} ({r.risk_category_at_review ?? "—"}) · model {r.model_version ?? "—"}
      </p>
      {r.note ? <p className="mt-1 whitespace-pre-wrap text-xs text-text">{r.note}</p> : null}
    </li>
  );
}

/**
 * Investigator decision for this claim, stored on the API host as an
 * append-only log. Confirmed/cleared decisions are the project's only route
 * to real labels (see README: "Investigator feedback and retraining").
 */
export function ReviewPanel({ domain, claimKey }: { domain: Domain; claimKey: string }) {
  const ids = { note: useId(), reviewer: useId() };
  const review = useClaimReview(domain, claimKey);
  const record = useRecordReview(domain, claimKey);
  const [outcome, setOutcome] = useState<ReviewOutcome | null>(null);
  const [note, setNote] = useState(() => legacyNote(domain, claimKey));
  const [reviewer, setReviewer] = useState(() => readLocal(REVIEWER_KEY));
  const current = review.data?.current;
  const history = review.data?.history ?? [];

  const submit = () => {
    if (!outcome) return;
    writeLocal(REVIEWER_KEY, reviewer.trim());
    record.mutate(
      { outcome, note, reviewer },
      {
        onSuccess: () => {
          toast.success("Decision recorded", { description: OUTCOME_LABEL[outcome].label });
          setOutcome(null);
          setNote("");
        },
        onError: (e) => toast.error("Could not record the decision", { description: e.message }),
      },
    );
  };

  return (
    <Card>
      <CardHeader
        title="Investigator review"
        description="Saved on the API host as an append-only log with the score you saw. Confirmed and cleared decisions become training labels."
        icon={<NotebookPen />}
      />
      <CardBody className="space-y-4">
        {review.error ? (
          <ErrorState error={review.error} onRetry={() => review.refetch()} />
        ) : review.isLoading ? (
          <SkeletonRows rows={2} />
        ) : (
          <p className="text-sm text-text">
            Current decision:{" "}
            {current ? (
              <Badge tone={OUTCOME_LABEL[current.outcome].tone}>{OUTCOME_LABEL[current.outcome].label}</Badge>
            ) : (
              <span className="text-text-muted">not reviewed yet</span>
            )}
          </p>
        )}

        <fieldset className="space-y-1.5">
          <legend className="mb-1.5 text-xs font-medium text-text">New decision</legend>
          <div className="grid gap-1.5 sm:grid-cols-3">
            {OUTCOMES.map((o) => (
              <label
                key={o.value}
                className={cn(
                  "cursor-pointer rounded-md border px-2.5 py-2 text-xs has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-primary",
                  outcome === o.value ? "border-primary bg-primary-soft text-text" : "border-border text-text-muted hover:border-border-strong",
                )}
              >
                <input
                  type="radio"
                  name={`outcome-${claimKey}`}
                  value={o.value}
                  checked={outcome === o.value}
                  onChange={() => setOutcome(o.value)}
                  className="sr-only"
                />
                <span className="block font-medium text-text">{o.label}</span>
                <span className="block text-2xs">{o.help}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <div className="space-y-1.5">
          <Label htmlFor={ids.note}>Rationale</Label>
          <textarea
            id={ids.note}
            rows={3}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            maxLength={4000}
            placeholder="What you checked and why…"
            className="w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-sm text-text shadow-xs placeholder:text-text-subtle"
          />
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <div className="min-w-0 flex-1 space-y-1.5">
            <Label htmlFor={ids.reviewer}>Reviewer (optional)</Label>
            <Input id={ids.reviewer} value={reviewer} maxLength={120} onChange={(e) => setReviewer(e.target.value)} placeholder="Name or ID" />
          </div>
          <Button onClick={submit} disabled={!outcome || record.isPending}>
            {record.isPending ? "Saving…" : "Record decision"}
          </Button>
        </div>

        {history.length ? (
          <details className="group">
            <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-text-muted">
              <History className="size-3.5" aria-hidden="true" /> History ({history.length})
            </summary>
            <ol className="mt-3 space-y-3">
              {[...history].reverse().map((r) => (
                <HistoryItem key={r.recorded_at + r.outcome + r.note} r={r} />
              ))}
            </ol>
          </details>
        ) : null}
      </CardBody>
    </Card>
  );
}

/** Model insights: how many decisions exist and how they would become training labels. */
export function ReviewSummaryCard({ domain }: { domain: Domain }) {
  const summary = useReviewSummary(domain);
  const s = summary.data;
  const labels = s?.labels ?? [];
  const positives = labels.filter((l) => l.label === 1).length;

  const download = () => {
    const rows = ["claim_key,label,recorded_at,model_version", ...labels.map((l) => `${l.claim_key},${l.label},${l.recorded_at},${l.model_version ?? ""}`)];
    save(new Blob([rows.join("\n") + "\n"], { type: "text/csv" }), `review-labels-${domain}.csv`);
  };

  return (
    <Card>
      <CardHeader
        title="Investigator labels"
        description="Decisions recorded on case files. They are the route to real, claim-level labels."
        icon={<NotebookPen />}
        action={
          labels.length ? (
            <Button variant="outline" size="sm" onClick={download}>
              <Download aria-hidden="true" /> Labels CSV
            </Button>
          ) : undefined
        }
      />
      <CardBody className="space-y-4">
        {summary.error ? (
          <ErrorState error={summary.error} onRetry={() => summary.refetch()} />
        ) : !s ? (
          <SkeletonRows rows={2} />
        ) : (
          <>
            <dl className="tabular grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-lg border border-border px-3 py-2.5">
                <dt className="text-2xs text-text-muted">Claims reviewed</dt>
                <dd className="text-xl font-semibold text-text">{s.reviewed_claims}</dd>
              </div>
              {OUTCOMES.map((o) => (
                <div key={o.value} className="rounded-lg border border-border px-3 py-2.5">
                  <dt className="text-2xs text-text-muted">{o.label}</dt>
                  <dd className="text-xl font-semibold text-text">{s.counts[o.value] ?? 0}</dd>
                </div>
              ))}
            </dl>
            <p className="text-sm text-text">
              {labels.length
                ? `${labels.length} usable label${labels.length === 1 ? "" : "s"} (${positives} confirmed, ${labels.length - positives} cleared).`
                : "No confirmed or cleared decisions yet."}
            </p>
            <ol className="list-decimal space-y-1 pl-4 text-xs text-text-muted">
              <li>Each decision is appended to the review log with the score, risk band and model version the reviewer saw.</li>
              <li>The latest confirmed or cleared decision per claim becomes a label (1 or 0); “needs more info” waits.</li>
              <li>
                To retrain (not automated yet): join these labels to the claims' features, keep the provider- or
                row-disjoint splits, and compare with the current model on held-out reviewed claims before replacing it.
              </li>
              <li>Labels come only from claims someone chose to review, so they over-represent high scores; report that bias.</li>
            </ol>
          </>
        )}
      </CardBody>
    </Card>
  );
}

function save(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Download the claim's audit record as JSON, or open a printable page
 * ("Save as PDF"). An uploaded document is re-analysed server-side so its
 * findings are recorded too.
 */
export function AuditExport({
  domain,
  claimKey,
  document: doc,
}: {
  domain: Domain;
  claimKey: string;
  document: { file: File; pageIndex: number } | null;
}) {
  const sample = useSampleParams();
  const [busy, setBusy] = useState<"json" | "html" | null>(null);

  const run = async (format: "json" | "html") => {
    // Open the tab synchronously so the browser does not treat it as a popup.
    const tab = format === "html" ? window.open("", "_blank") : null;
    setBusy(format);
    try {
      const blob = await fetchAuditReport(domain, claimKey, sample, format, doc);
      if (format === "json") {
        save(blob, `audit-${domain}-${claimKey.replace(":", "_")}.json`);
      } else if (tab) {
        tab.location.href = URL.createObjectURL(blob);
      }
    } catch (e) {
      tab?.close();
      toast.error("Could not build the audit report", { description: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button variant="outline" size="sm" onClick={() => run("json")} disabled={busy !== null}>
        <Download aria-hidden="true" /> {busy === "json" ? "Building…" : "Audit record (JSON)"}
      </Button>
      <Button variant="outline" size="sm" onClick={() => run("html")} disabled={busy !== null}>
        <FileText aria-hidden="true" /> {busy === "html" ? "Building…" : "Printable report (PDF)"}
      </Button>
      {doc ? <span className="text-2xs text-text-subtle">Includes forensics for {doc.file.name}</span> : null}
    </div>
  );
}
