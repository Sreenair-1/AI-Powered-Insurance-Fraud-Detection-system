import { motion } from "framer-motion";
import {
  ArrowRight,
  Copy,
  FileImage,
  FileText,
  Flag,
  Layers,
  ScanSearch,
  UploadCloud,
  X,
} from "lucide-react";
import { useId, useRef, useState, type DragEvent } from "react";
import { TAMPER_LABEL, fmtPct, fmtSigned, fmtValue, label } from "@/lib/format";
import type { Assessment, ForensicReport } from "@/lib/types";
import { ACCEPTED_TYPES, validateUpload } from "@/lib/upload";
import { cn, usePrefersReducedMotion } from "@/lib/utils";
import { Callout, RiskBadge } from "./common";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardBody, CardHeader } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

// ---------------------------------------------------------------------------
// Dropzone (keyboard: the hidden input is focusable via its label)
// ---------------------------------------------------------------------------

export function Dropzone({
  file,
  onFile,
  disabled,
  compact = false,
}: {
  file: File | null;
  onFile: (file: File | null) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const id = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accept = (f: File | undefined | null) => {
    if (!f) return;
    const problem = validateUpload(f);
    setError(problem);
    onFile(problem ? null : f);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (!disabled) accept(e.dataTransfer.files?.[0]);
  };

  if (file) {
    const Icon = file.name.toLowerCase().endsWith(".pdf") ? FileText : FileImage;
    return (
      <div className="flex items-center gap-3 rounded-lg border border-border bg-surface-sunken px-3 py-2.5">
        <Icon className="size-5 shrink-0 text-primary" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-text">{file.name}</p>
          <p className="tabular text-xs text-text-muted">{(file.size / 1024).toFixed(1)} KB</p>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => {
            onFile(null);
            if (inputRef.current) inputRef.current.value = "";
          }}
          disabled={disabled}
          aria-label={`Remove ${file.name}`}
        >
          <X aria-hidden="true" />
        </Button>
      </div>
    );
  }

  return (
    <div>
      <label
        htmlFor={id}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 text-center transition-colors duration-fast has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-ring",
          compact ? "py-6" : "py-10",
          drag ? "border-primary bg-primary-soft" : "border-border-strong bg-surface hover:border-primary/60 hover:bg-surface-sunken",
          disabled && "pointer-events-none opacity-50",
        )}
      >
        <UploadCloud className="size-7 text-primary" aria-hidden="true" />
        <span className="text-sm font-medium text-text">
          Drop a claim document here, or <span className="text-primary underline underline-offset-2">browse</span>
        </span>
        <span className="text-xs text-text-muted">PNG, JPEG, TIFF, BMP or PDF · up to 20 MB</span>
        <input
          id={id}
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES}
          className="sr-only"
          disabled={disabled}
          onChange={(e) => accept(e.target.files?.[0])}
          aria-describedby={error ? `${id}-error` : undefined}
        />
      </label>
      {error ? (
        <p id={`${id}-error`} role="alert" className="mt-2 text-xs font-medium text-risk-high">
          {error}
        </p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tamper gauge
// ---------------------------------------------------------------------------

export function TamperGauge({ score, status }: { score: number; status: string }) {
  const reduced = usePrefersReducedMotion();
  const tone = status === "tampered" ? "high" : status === "suspicious" ? "medium" : "low";
  const strokeClass = { high: "stroke-risk-high-solid", medium: "stroke-risk-medium-solid", low: "stroke-risk-low-solid" }[tone];
  const arc = Math.PI * 52;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 120 70" className="w-48" role="img" aria-label={`Tamper score ${fmtPct(score, 0)}: ${TAMPER_LABEL[status] ?? status}`}>
        <path d="M8 62 A52 52 0 0 1 112 62" fill="none" className="stroke-surface-sunken" strokeWidth="10" strokeLinecap="round" />
        <motion.path
          d="M8 62 A52 52 0 0 1 112 62"
          fill="none"
          className={strokeClass}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={arc}
          initial={{ strokeDashoffset: reduced ? arc * (1 - score) : arc }}
          animate={{ strokeDashoffset: arc * (1 - score) }}
          transition={{ duration: reduced ? 0 : 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
        <text x="60" y="56" textAnchor="middle" className="fill-text text-[20px] font-semibold tabular">
          {fmtPct(score, 0)}
        </text>
      </svg>
      <Badge tone={tone} className="mt-1">
        <Flag aria-hidden="true" />
        {TAMPER_LABEL[status] ?? status}
      </Badge>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Image comparison + copy-move overlay
// ---------------------------------------------------------------------------

function Overlay({ report, showPairs }: { report: ForensicReport; showPairs: boolean }) {
  const w = report.image_width ?? 1;
  const h = report.image_height ?? 1;
  const stroke = Math.max(w, h) / 300;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="pointer-events-none absolute inset-0 size-full" aria-hidden="true">
      {showPairs
        ? report.copy_move_pairs.map(([x1, y1, x2, y2], i) => (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className="stroke-accent" strokeWidth={stroke * 0.8} opacity={0.55} />
          ))
        : null}
      {report.copy_move_regions.map((r, i) => (
        <g key={i}>
          {[r.source, r.target].map(([x0, y0, x1, y1], j) => (
            <rect
              key={j}
              x={x0}
              y={y0}
              width={x1 - x0}
              height={y1 - y0}
              fill="none"
              className="stroke-accent"
              strokeWidth={stroke * 1.6}
              strokeDasharray={j ? `${stroke * 4} ${stroke * 2}` : undefined}
            />
          ))}
          <text x={r.source[0]} y={r.source[1] - stroke * 2} className="fill-accent" fontSize={stroke * 9} fontWeight={600}>
            Region {i + 1} · {r.matches} matches
          </text>
        </g>
      ))}
    </svg>
  );
}

function Frame({ src, alt, children }: { src: string | null; alt: string; children?: React.ReactNode }) {
  if (!src) {
    return (
      <div className="flex aspect-[4/3] items-center justify-center rounded-md border border-dashed border-border-strong text-xs text-text-muted">
        Not available for this file
      </div>
    );
  }
  return (
    <div className="relative overflow-hidden rounded-md border border-border bg-surface-sunken">
      <img src={src} alt={alt} className="block h-auto w-full" />
      {children}
    </div>
  );
}

export function ImageCompare({ report }: { report: ForensicReport }) {
  const [showPairs, setShowPairs] = useState(true);
  const regions = report.copy_move_regions.length;
  return (
    <Tabs defaultValue="ela">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <TabsList aria-label="Image views">
          <TabsTrigger value="ela">
            <Layers aria-hidden="true" /> Original vs ELA
          </TabsTrigger>
          <TabsTrigger value="copy">
            <Copy aria-hidden="true" /> Copy-move regions
          </TabsTrigger>
          <TabsTrigger value="viz">
            <ScanSearch aria-hidden="true" /> Keypoint matches
          </TabsTrigger>
        </TabsList>
        {report.is_pdf ? (
          <Badge tone="neutral">
            PDF page {report.page_index + 1} of {report.page_count}
          </Badge>
        ) : null}
      </div>
      <TabsContent value="ela" className="mt-3">
        <div className="grid gap-3 md:grid-cols-2">
          <figure>
            <Frame src={report.original_image} alt={`Analysed page of ${report.filename}`} />
            <figcaption className="mt-1.5 text-xs text-text-muted">Original (as analysed)</figcaption>
          </figure>
          <figure>
            <Frame src={report.ela_image} alt="Error level analysis heatmap" />
            <figcaption className="mt-1.5 text-xs text-text-muted">
              ELA heatmap — brighter areas recompress differently from the rest of the image
            </figcaption>
          </figure>
        </div>
      </TabsContent>
      <TabsContent value="copy" className="mt-3">
        <div className="mb-2 flex flex-wrap items-center gap-3 text-xs text-text-muted">
          <span>
            {regions
              ? `${regions} candidate ${regions === 1 ? "region pair" : "region pairs"} (solid = source, dashed = copy)`
              : report.copy_move_pairs.length
                ? "Isolated keypoint matches only; no group of matches shares an offset"
                : "No matched keypoint pairs"}
          </span>
          {report.copy_move_pairs.length ? (
            <label className="ml-auto inline-flex cursor-pointer items-center gap-1.5">
              <input type="checkbox" checked={showPairs} onChange={(e) => setShowPairs(e.target.checked)} className="accent-[rgb(var(--accent))]" />
              Show match lines
            </label>
          ) : null}
        </div>
        <div className="mx-auto max-w-3xl">
          <Frame src={report.original_image} alt={`Copy-move overlay on ${report.filename}`}>
            <Overlay report={report} showPairs={showPairs} />
          </Frame>
        </div>
      </TabsContent>
      <TabsContent value="viz" className="mt-3">
        <div className="mx-auto max-w-3xl">
          <Frame src={report.copy_move_image} alt="ORB keypoint match visualisation" />
          <p className="mt-1.5 text-xs text-text-muted">Rendered by the copy-move module: matched ORB keypoint pairs joined by lines.</p>
        </div>
      </TabsContent>
    </Tabs>
  );
}

// ---------------------------------------------------------------------------
// Metadata, ELA/copy-move statistics
// ---------------------------------------------------------------------------

const HIDDEN_META = new Set(["exif_fields", "raw_metadata"]);

export function MetadataPanel({ report }: { report: ForensicReport }) {
  const meta = report.metadata;
  if (!meta) return <p className="text-sm text-text-muted">Metadata could not be read from this file.</p>;
  const flags = new Map(report.metadata_flags.map((f) => [f.field, f.reason]));
  const entries = Object.entries(meta.structured).filter(([k]) => !HIDDEN_META.has(k));
  const rawKey = "exif_fields" in meta.structured ? "exif_fields" : "raw_metadata";
  const raw = (meta.structured[rawKey] ?? {}) as Record<string, unknown>;
  return (
    <div className="space-y-3">
      <dl className="divide-y divide-border rounded-md border border-border">
        {entries.map(([k, v]) => {
          const reason = flags.get(k);
          return (
            <div key={k} className={cn("grid grid-cols-[9rem_1fr] gap-3 px-3 py-2 text-xs", reason && "bg-risk-medium-soft")}>
              <dt className="text-text-muted">{label(k)}</dt>
              <dd className="min-w-0 break-words font-medium text-text">
                {fmtValue(v)}
                {reason ? (
                  <span className="mt-0.5 flex items-center gap-1 font-normal text-risk-medium">
                    <Flag className="size-3" aria-hidden="true" />
                    <span>
                      <span className="sr-only">Flagged: </span>
                      {reason}
                    </span>
                  </span>
                ) : null}
              </dd>
            </div>
          );
        })}
      </dl>
      {Object.keys(raw).length ? (
        <details className="rounded-md border border-border text-xs">
          <summary className="cursor-pointer px-3 py-2 font-medium text-text-muted">
            {Object.keys(raw).length} raw {rawKey === "exif_fields" ? "EXIF" : "PDF"} fields
          </summary>
          <dl className="max-h-56 overflow-auto border-t border-border px-3 py-2">
            {Object.entries(raw).map(([k, v]) => (
              <div key={k} className="grid grid-cols-[9rem_1fr] gap-3 py-0.5">
                <dt className="truncate text-text-muted">{k}</dt>
                <dd className="break-all text-text">{fmtValue(v)}</dd>
              </div>
            ))}
          </dl>
        </details>
      ) : null}
      {meta.indicators.length ? (
        <ul className="space-y-1 text-xs text-text-muted">
          {meta.indicators.map((t) => (
            <li key={t} className="flex gap-1.5">
              <span aria-hidden="true">•</span>
              {t}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function StatList({ stats }: { stats: Record<string, unknown> }) {
  return (
    <dl className="tabular grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
      {Object.entries(stats).map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2 border-b border-dashed border-border py-0.5">
          <dt className="text-text-muted">{label(k)}</dt>
          <dd className="font-medium text-text">{fmtValue(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ForensicReportView({ report }: { report: ForensicReport }) {
  const v = report.verdict;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[17rem_1fr]">
        <Card>
          <CardHeader title="Tamper score" description="Combined ELA, copy-move and metadata indicators" as="h3" />
          <CardBody className="flex flex-col items-center gap-3">
            <TamperGauge score={v.tampering_score} status={v.status} />
            <p className="text-center text-xs text-text-muted">{v.summary}</p>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Indicators" description={`${v.indicators.length} raised by the forensic checks`} as="h3" />
          <CardBody>
            {v.indicators.length ? (
              <ul className="space-y-2">
                {v.indicators.map((t) => (
                  <li key={t} className="flex gap-2 text-sm text-text">
                    <Flag className="mt-0.5 size-4 shrink-0 text-risk-medium" aria-hidden="true" />
                    {t}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">No forensic indicators were raised for this document.</p>
            )}
            <Callout className="mt-4">
              Forensic signals are indicators for an investigator to examine. Scanning, re-saving and PDF tools can
              trigger them on genuine documents.
            </Callout>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Visual analysis" description={report.filename} as="h3" />
        <CardBody>
          <ImageCompare report={report} />
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title={report.is_pdf ? "PDF metadata" : "EXIF metadata"} description="Suspicious fields are highlighted and flagged" as="h3" />
          <CardBody>
            <MetadataPanel report={report} />
          </CardBody>
        </Card>
        <div className="space-y-4">
          <Card>
            <CardHeader title="Error level analysis" description="Recompression error statistics" as="h3" />
            <CardBody>{report.ela ? <StatList stats={report.ela} /> : <p className="text-sm text-text-muted">ELA could not run.</p>}</CardBody>
          </Card>
          <Card>
            <CardHeader title="Copy-move detection" description="ORB keypoint self-matching" as="h3" />
            <CardBody>
              {report.copy_move ? <StatList stats={report.copy_move} /> : <p className="text-sm text-text-muted">Copy-move could not run.</p>}
            </CardBody>
          </Card>
        </div>
      </div>

      {report.extracted_text ? (
        <Card>
          <CardHeader title="Extracted text" description={`Page ${report.page_index + 1}`} as="h3" />
          <CardBody>
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-surface-sunken p-3 font-mono text-xs text-text">
              {report.extracted_text}
            </pre>
          </CardBody>
        </Card>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Before / after re-score
// ---------------------------------------------------------------------------

function ScoreTile({ title, assessment, highlight }: { title: string; assessment: Assessment; highlight?: boolean }) {
  return (
    <div className={cn("rounded-lg border p-4", highlight ? "border-primary/40 bg-primary-soft" : "border-border")}>
      <p className="text-xs font-medium text-text-muted">{title}</p>
      <p className="tabular mt-1 text-3xl font-semibold tracking-tight text-text">{fmtPct(assessment.fraud_probability)}</p>
      <RiskBadge risk={assessment.risk_category} className="mt-2" />
    </div>
  );
}

export function BeforeAfter({ before, after, note }: { before: Assessment; after: Assessment; note: string }) {
  const delta = after.fraud_probability - before.fraud_probability;
  return (
    <div className="space-y-3">
      <div className="grid items-center gap-3 sm:grid-cols-[1fr_auto_1fr]">
        <ScoreTile title="Before document" assessment={before} />
        <div className="flex justify-center" aria-hidden="true">
          <ArrowRight className="size-5 rotate-90 text-text-subtle sm:rotate-0" />
        </div>
        <ScoreTile title="After document" assessment={after} highlight />
      </div>
      <p className="tabular text-sm text-text">
        Change: <span className="font-semibold">{fmtSigned(delta * 100, 1)} percentage points</span>
        {before.risk_category !== after.risk_category ? (
          <> — risk level moved from {before.risk_category} to {after.risk_category}.</>
        ) : (
          <> — risk level unchanged.</>
        )}
      </p>
      <Callout title="How the re-score is computed">{note}</Callout>
    </div>
  );
}
