import { BarChart3, Table2 } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { RISK_LABEL, fmtInt, fmtMetric, fmtPct, label } from "@/lib/format";
import type { RiskBands, RiskCategory } from "@/lib/types";
import { cn, usePrefersReducedMotion, useTokenColors, type TokenColors } from "@/lib/utils";
import { RiskBadge } from "./common";
import { Button } from "./ui/button";

const RISKS: RiskCategory[] = ["high", "medium", "low"];

function riskFill(c: TokenColors, r: RiskCategory) {
  return c[`risk-${r}-solid` as const];
}

// ---------------------------------------------------------------------------
// Frame: summary for screen readers + a chart/table toggle
// ---------------------------------------------------------------------------

export function ChartFrame({
  summary,
  table,
  children,
  height = 240,
}: {
  summary: string;
  table?: { headers: string[]; rows: (string | number)[][] };
  children: ReactNode;
  height?: number;
}) {
  const [asTable, setAsTable] = useState(false);
  return (
    <figure className="relative">
      {table ? (
        <div className="absolute -top-11 right-0">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setAsTable((v) => !v)}
            aria-pressed={asTable}
            aria-label={asTable ? "Show chart" : "Show data as table"}
            title={asTable ? "Show chart" : "Show data as table"}
          >
            {asTable ? <BarChart3 aria-hidden="true" /> : <Table2 aria-hidden="true" />}
          </Button>
        </div>
      ) : null}
      <figcaption className="sr-only">{summary}</figcaption>
      {asTable && table ? (
        <div className="max-h-[320px] overflow-auto rounded-md border border-border" style={{ minHeight: height }}>
          <table className="tabular w-full text-xs">
            <thead className="sticky top-0 bg-surface-sunken text-left text-text-muted">
              <tr>
                {table.headers.map((h) => (
                  <th key={h} scope="col" className="px-3 py-2 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i} className="border-t border-border">
                  {row.map((cell, j) => (
                    <td key={j} className="px-3 py-1.5 text-text">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ height }} aria-hidden="true">
          {children}
        </div>
      )}
    </figure>
  );
}

function TooltipBox({ title, rows }: { title?: ReactNode; rows: [ReactNode, ReactNode][] }) {
  return (
    <div className="tabular rounded-md border border-border bg-surface-raised px-3 py-2 text-xs shadow-md">
      {title ? <p className="mb-1 font-semibold text-text">{title}</p> : null}
      {rows.map(([k, v], i) => (
        <p key={i} className="flex justify-between gap-4 text-text-muted">
          <span>{k}</span>
          <span className="font-medium text-text">{v}</span>
        </p>
      ))}
    </div>
  );
}

function axisProps(c: TokenColors) {
  return {
    stroke: c["chart-grid"],
    tick: { fill: c["chart-axis"], fontSize: 11 },
    tickLine: false,
  } as const;
}

// ---------------------------------------------------------------------------
// Risk distribution (counts per band) - horizontal bars, direct labels
// ---------------------------------------------------------------------------

export function RiskDistributionChart({ counts, total }: { counts: Record<RiskCategory, number>; total: number }) {
  const reduced = usePrefersReducedMotion();
  const rows = RISKS.map((r) => ({ risk: r, count: counts[r] ?? 0, share: total ? (counts[r] ?? 0) / total : 0 }));
  return (
    <div className="space-y-3" role="list" aria-label="Claims per risk level">
      {rows.map((row) => (
        <div key={row.risk} role="listitem" className="grid grid-cols-[7.5rem_1fr_5.5rem] items-center gap-3">
          <RiskBadge risk={row.risk} className="justify-self-start" />
          <div className="h-3 overflow-hidden rounded-full bg-surface-sunken">
            <div
              className={cn(
                "h-full rounded-full",
                { high: "bg-risk-high-solid", medium: "bg-risk-medium-solid", low: "bg-risk-low-solid" }[row.risk],
                !reduced && "transition-[width] duration-slow ease-out",
              )}
              style={{ width: `${row.share * 100}%` }}
            />
          </div>
          <p className="tabular text-right text-xs">
            <span className="font-semibold text-text">{fmtInt(row.count)}</span>{" "}
            <span className="text-text-muted">({fmtPct(row.share)})</span>
          </p>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Probability histogram with band cut-offs
// ---------------------------------------------------------------------------

export function ProbabilityHistogram({
  bins,
  bands,
}: {
  bins: { start: number; end: number; count: number }[];
  bands: RiskBands;
}) {
  const c = useTokenColors();
  const reduced = usePrefersReducedMotion();
  const data = bins.map((b) => ({ ...b, mid: (b.start + b.end) / 2, name: `${fmtPct(b.start, 0)}–${fmtPct(b.end, 0)}` }));
  const peak = data.reduce((m, d) => (d.count > m.count ? d : m), data[0] ?? { name: "", count: 0 });
  return (
    <ChartFrame
      summary={`Histogram of risk scores across the loaded sample. The most common range is ${peak.name} with ${peak.count} claims.`}
      table={{ headers: ["Score range", "Claims"], rows: data.map((d) => [d.name, d.count]) }}
    >
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 4, left: -16 }} barCategoryGap={2}>
          <CartesianGrid vertical={false} stroke={c["chart-grid"]} />
          <XAxis dataKey="mid" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tickFormatter={(v) => fmtPct(v, 0)} {...axisProps(c)} />
          <YAxis allowDecimals={false} {...axisProps(c)} axisLine={false} />
          <RTooltip
            cursor={{ fill: c["chart-grid"], opacity: 0.5 }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox title={payload[0].payload.name} rows={[["Claims", fmtInt(payload[0].payload.count)]]} />
              ) : null
            }
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} isAnimationActive={!reduced} stroke={c.surface} strokeWidth={2}>
            {data.map((d) => (
              <Cell
                key={d.start}
                fill={d.start >= bands.high ? riskFill(c, "high") : d.start >= bands.medium ? riskFill(c, "medium") : riskFill(c, "low")}
              />
            ))}
          </Bar>
          {[
            [bands.medium, `Medium ≥ ${fmtPct(bands.medium, 0)}`],
            [bands.high, `High ≥ ${fmtPct(bands.high, 0)}`],
          ].map(([x, text]) => (
            <ReferenceLine
              key={String(x)}
              x={x as number}
              stroke={c["text-subtle"]}
              strokeDasharray="4 3"
              label={{ value: text as string, position: "top", fill: c["text-muted"], fontSize: 11 }}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// Segment breakdown - stacked horizontal bars with 2px surface gaps
// ---------------------------------------------------------------------------

export function SegmentChart({
  field,
  segments,
}: {
  field: string;
  segments: { segment: string; low: number; medium: number; high: number }[];
}) {
  const c = useTokenColors();
  const reduced = usePrefersReducedMotion();
  const height = Math.max(140, segments.length * 44 + 40);
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-2" aria-hidden="true">
        {RISKS.map((r) => (
          <RiskBadge key={r} risk={r} />
        ))}
      </div>
      <ChartFrame
        height={height}
        summary={`Claims per ${label(field)}, split by risk level.`}
        table={{
          headers: [label(field), "High", "Medium", "Low"],
          rows: segments.map((s) => [s.segment, s.high, s.medium, s.low]),
        }}
      >
        <ResponsiveContainer>
          <BarChart data={segments} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }} barSize={22}>
            <CartesianGrid horizontal={false} stroke={c["chart-grid"]} />
            <XAxis type="number" allowDecimals={false} {...axisProps(c)} />
            <YAxis type="category" dataKey="segment" width={84} {...axisProps(c)} axisLine={false} />
            <RTooltip
              cursor={{ fill: c["chart-grid"], opacity: 0.4 }}
              content={({ active, payload }) =>
                active && payload?.length ? (
                  <TooltipBox
                    title={payload[0].payload.segment}
                    rows={RISKS.map((r) => [RISK_LABEL[r], fmtInt(payload[0].payload[r])] as [string, string])}
                  />
                ) : null
              }
            />
            {(["low", "medium", "high"] as RiskCategory[]).map((r, i, arr) => (
              <Bar
                key={r}
                dataKey={r}
                stackId="risk"
                fill={riskFill(c, r)}
                stroke={c.surface}
                strokeWidth={2}
                radius={i === arr.length - 1 ? [0, 4, 4, 0] : 0}
                isAnimationActive={!reduced}
                name={RISK_LABEL[r]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Global feature importance (single series)
// ---------------------------------------------------------------------------

export function ImportanceChart({ items }: { items: { feature: string; importance: number }[] }) {
  const c = useTokenColors();
  const reduced = usePrefersReducedMotion();
  const data = items.map((i) => ({ ...i, name: label(i.feature) }));
  return (
    <ChartFrame
      height={Math.max(160, data.length * 30 + 20)}
      summary={`Model-wide feature importance (XGBoost gain). Highest: ${data[0]?.name ?? "none"}.`}
      table={{ headers: ["Feature", "Importance"], rows: data.map((d) => [d.name, d.importance]) }}
    >
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, bottom: 0, left: 8 }} barSize={14}>
          <XAxis type="number" hide domain={[0, "dataMax"]} />
          <YAxis type="category" dataKey="name" width={150} {...axisProps(c)} axisLine={false} />
          <RTooltip
            cursor={{ fill: c["chart-grid"], opacity: 0.4 }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox title={payload[0].payload.name} rows={[["Importance", String(payload[0].payload.importance)]]} />
              ) : null
            }
          />
          <Bar dataKey="importance" fill={c["chart-1"]} radius={[0, 4, 4, 0]} isAnimationActive={!reduced}>
            <LabelList dataKey="importance" position="right" fill={c["text-muted"]} fontSize={11} formatter={(v: number) => v.toFixed(3)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// ROC / PR curves (saved artifacts, thinned for drawing only)
// ---------------------------------------------------------------------------

/** Keeps ~maxPoints evenly spaced points plus both ends. Drawing aid only - the
 * table view and every metric still use the saved values. */
function thin<T>(points: T[], maxPoints = 500): T[] {
  if (points.length <= maxPoints) return points;
  const step = (points.length - 1) / (maxPoints - 1);
  const out: T[] = [];
  for (let i = 0; i < maxPoints; i++) out.push(points[Math.round(i * step)]);
  return out;
}

const CURVE_NAMES = {
  roc: { xName: "False positive rate", yName: "True positive rate", title: "ROC" },
  pr: { xName: "Recall", yName: "Precision", title: "Precision-recall" },
  gain: { xName: "Share of claims reviewed", yName: "Positive-label claims caught", title: "Cumulative gain" },
} as const;

export function CurveChart({
  kind,
  x,
  y,
  thresholds,
  baseline,
  areaLabel,
  markerX,
}: {
  kind: "roc" | "pr" | "gain";
  x: number[];
  y: number[];
  thresholds?: number[];
  baseline?: number;
  areaLabel: string;
  /** Vertical marker, e.g. the review share picked on a slider. */
  markerX?: number;
}) {
  const c = useTokenColors();
  const reduced = usePrefersReducedMotion();
  const points = useMemo(
    () => x.map((xv, i) => ({ x: xv, y: y[i], t: thresholds?.[i] })).sort((a, b) => a.x - b.x || a.y - b.y),
    [x, y, thresholds],
  );
  const drawn = useMemo(() => thin(points), [points]);
  const { xName, yName, title } = CURVE_NAMES[kind];
  const diagonal = kind !== "pr";
  return (
    <ChartFrame
      height={260}
      summary={`${title} curve from the saved validation artifact, ${points.length} points. ${areaLabel}.`}
      table={{
        headers: [xName, yName, "Threshold"],
        rows: thin(points, 200).map((p) => [fmtMetric(p.x), fmtMetric(p.y), p.t == null ? "—" : fmtMetric(p.t)]),
      }}
    >
      <ResponsiveContainer>
        <LineChart data={drawn} margin={{ top: 8, right: 12, bottom: 18, left: -8 }}>
          <CartesianGrid stroke={c["chart-grid"]} />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            {...axisProps(c)}
            label={{ value: xName, position: "insideBottom", offset: -10, fill: c["text-muted"], fontSize: 11 }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            {...axisProps(c)}
            label={{ value: yName, angle: -90, position: "insideLeft", offset: 18, fill: c["text-muted"], fontSize: 11 }}
          />
          {markerX != null ? <ReferenceLine x={markerX} stroke={c["chart-2"]} strokeWidth={1.5} /> : null}
          {diagonal ? (
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={c["text-subtle"]} strokeDasharray="4 3" />
          ) : baseline != null ? (
            <ReferenceLine
              y={baseline}
              stroke={c["text-subtle"]}
              strokeDasharray="4 3"
              label={{ value: `No-skill baseline ${fmtMetric(baseline)}`, position: "insideTopRight", fill: c["text-muted"], fontSize: 11 }}
            />
          ) : null}
          <RTooltip
            cursor={{ stroke: c["text-subtle"], strokeDasharray: "3 3" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox
                  rows={[
                    [xName, fmtMetric(payload[0].payload.x)],
                    [yName, fmtMetric(payload[0].payload.y)],
                    ...(payload[0].payload.t != null ? [["Threshold", fmtMetric(payload[0].payload.t)] as [string, string]] : []),
                  ]}
                />
              ) : null
            }
          />
          <Line
            type={kind === "pr" ? "stepAfter" : "linear"}
            dataKey="y"
            stroke={c["chart-1"]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={!reduced}
            activeDot={{ r: 4, stroke: c.surface, strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------------
// Confusion matrix (single-hue intensity + numbers in every cell)
// ---------------------------------------------------------------------------

export function ConfusionMatrix({ cm, positiveLabel }: { cm: Record<string, number>; positiveLabel: string }) {
  const cells = [
    { key: "true_negative", name: "True negative", actual: 0, pred: 0 },
    { key: "false_positive", name: "False positive", actual: 0, pred: 1 },
    { key: "false_negative", name: "False negative", actual: 1, pred: 0 },
    { key: "true_positive", name: "True positive", actual: 1, pred: 1 },
  ];
  const max = Math.max(...cells.map((x) => cm[x.key] ?? 0), 1);
  const total = cells.reduce((s, x) => s + (cm[x.key] ?? 0), 0);
  return (
    <table className="tabular w-full border-separate border-spacing-1 text-xs">
      <caption className="sr-only">
        Confusion matrix at threshold {cm.threshold}. Rows are actual labels, columns are model predictions.
      </caption>
      <thead>
        <tr>
          <td />
          <th scope="col" className="pb-1 font-medium text-text-muted">
            Predicted: not flagged
          </th>
          <th scope="col" className="pb-1 font-medium text-text-muted">
            Predicted: flagged
          </th>
        </tr>
      </thead>
      <tbody>
        {[0, 1].map((actual) => (
          <tr key={actual}>
            <th scope="row" className="w-28 pr-2 text-left font-medium text-text-muted">
              Actual: {actual ? positiveLabel : "negative"}
            </th>
            {cells
              .filter((x) => x.actual === actual)
              .map((x) => {
                const v = cm[x.key] ?? 0;
                const strength = 0.08 + 0.62 * (v / max);
                return (
                  <td
                    key={x.key}
                    className="rounded-md px-3 py-3 text-center"
                    style={{ background: `rgb(var(--primary) / ${strength})` }}
                  >
                    <span className={cn("block text-lg font-semibold", strength > 0.45 ? "text-primary-on" : "text-text")}>
                      {fmtInt(v)}
                    </span>
                    <span className={cn("block text-2xs", strength > 0.45 ? "text-primary-on" : "text-text-muted")}>
                      {x.name} · {fmtPct(total ? v / total : 0)}
                    </span>
                  </td>
                );
              })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
