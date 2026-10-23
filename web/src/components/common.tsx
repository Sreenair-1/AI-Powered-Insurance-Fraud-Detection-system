import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { AlertOctagon, AlertTriangle, Inbox, Info, RefreshCw, ShieldCheck, Scale } from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { ApiError } from "@/lib/api";
import { RISK_LABEL, fmtPct } from "@/lib/format";
import type { RiskBands, RiskCategory } from "@/lib/types";
import { cn, usePrefersReducedMotion } from "@/lib/utils";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Skeleton } from "./ui/skeleton";

const RISK_ICON = { high: AlertOctagon, medium: AlertTriangle, low: ShieldCheck } as const;

/** Risk level: icon + text + colour, never colour alone. */
export function RiskBadge({ risk, className, size = "sm" }: { risk: RiskCategory; className?: string; size?: "sm" | "md" }) {
  const Icon = RISK_ICON[risk];
  return (
    <Badge tone={risk} className={cn(size === "md" && "px-2.5 py-1 text-sm [&_svg]:size-4", className)}>
      <Icon aria-hidden="true" />
      {RISK_LABEL[risk]}
    </Badge>
  );
}

export function DecisionBanner({ className, compact = false }: { className?: string; compact?: boolean }) {
  return (
    <div
      role="note"
      className={cn(
        "flex items-start gap-2.5 rounded-lg border border-primary/20 bg-primary-soft px-3.5 py-2.5 text-primary",
        className,
      )}
    >
      <Scale className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <p className="text-xs leading-relaxed">
        <span className="font-semibold">Decision support only – requires human review.</span>
        {compact ? null : (
          <>
            {" "}
            Scores are model probabilities used to prioritise investigation. A high score means a claim is flagged
            for review, not that wrongdoing has been established.
          </>
        )}
      </p>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow ? (
          <p className="text-2xs font-semibold uppercase tracking-[0.12em] text-accent">{eyebrow}</p>
        ) : null}
        <h1 className="mt-1 text-balance text-2xl font-semibold tracking-tight text-text">{title}</h1>
        {description ? <p className="mt-1.5 max-w-3xl text-sm text-text-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function AnimatedNumber({
  value,
  format = (n) => Math.round(n).toLocaleString("en-IN"),
}: {
  value: number;
  format?: (n: number) => string;
}) {
  const reduced = usePrefersReducedMotion();
  const mv = useMotionValue(reduced ? value : 0);
  const text = useTransform(mv, (n) => format(n));
  useEffect(() => {
    if (reduced) {
      mv.set(value);
      return;
    }
    const controls = animate(mv, value, { duration: 0.9, ease: [0.22, 1, 0.36, 1] });
    return () => controls.stop();
  }, [value, reduced, mv]);
  return (
    <>
      <motion.span aria-hidden="true">{text}</motion.span>
      <span className="sr-only">{format(value)}</span>
    </>
  );
}

export function KpiCard({
  label,
  value,
  format,
  hint,
  tone,
  icon,
}: {
  label: string;
  value: number;
  format?: (n: number) => string;
  hint?: ReactNode;
  tone?: RiskCategory | "primary";
  icon?: ReactNode;
}) {
  const accent = {
    high: "before:bg-risk-high-solid",
    medium: "before:bg-risk-medium-solid",
    low: "before:bg-risk-low-solid",
    primary: "before:bg-primary",
  }[tone ?? "primary"];
  return (
    <Card
      className={cn(
        "relative overflow-hidden p-4 before:absolute before:inset-y-0 before:left-0 before:w-1",
        accent,
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium text-text-muted">{label}</p>
        {icon ? <span className="text-text-subtle [&_svg]:size-4" aria-hidden="true">{icon}</span> : null}
      </div>
      <p className="tabular mt-2 text-[1.75rem] font-semibold leading-none tracking-tight text-text">
        <AnimatedNumber value={value} format={format} />
      </p>
      {hint ? <p className="mt-2 text-xs text-text-muted">{hint}</p> : null}
    </Card>
  );
}

export function ErrorState({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  const err = error instanceof ApiError ? error : null;
  const title =
    err?.status === 404
      ? "Not found in this sample"
      : err?.status === 503
        ? "Service not ready"
        : err?.status === 0
          ? "API unreachable"
          : "Something went wrong";
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border-strong px-6 py-10 text-center",
        className,
      )}
    >
      <AlertTriangle className="size-6 text-risk-medium" aria-hidden="true" />
      <p className="text-sm font-semibold text-text">{title}</p>
      <p className="max-w-md text-xs text-text-muted">{error instanceof Error ? error.message : String(error)}</p>
      {onRetry ? (
        <Button variant="outline" size="sm" className="mt-2" onClick={onRetry}>
          <RefreshCw aria-hidden="true" /> Try again
        </Button>
      ) : null}
    </div>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <span className="text-text-subtle [&_svg]:size-6" aria-hidden="true">
        {icon ?? <Inbox />}
      </span>
      <p className="text-sm font-semibold text-text">{title}</p>
      {children ? <div className="max-w-md text-xs text-text-muted">{children}</div> : null}
    </div>
  );
}

export function Callout({
  title,
  children,
  tone = "info",
  className,
}: {
  title?: string;
  children: ReactNode;
  tone?: "info" | "warning";
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex gap-2.5 rounded-lg border px-3.5 py-3 text-xs leading-relaxed",
        tone === "warning"
          ? "border-risk-medium-border bg-risk-medium-soft text-risk-medium"
          : "border-border bg-surface-sunken text-text-muted",
        className,
      )}
    >
      {tone === "warning" ? (
        <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      ) : (
        <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      )}
      <div>
        {title ? <p className="font-semibold">{title}</p> : null}
        <div className={cn(title && "mt-0.5")}>{children}</div>
      </div>
    </div>
  );
}

/** Horizontal probability meter with the medium/high band cut-offs marked. */
export function ProbabilityMeter({ value, bands, className }: { value: number; bands: RiskBands; className?: string }) {
  const reduced = usePrefersReducedMotion();
  const fill = value >= bands.high ? "bg-risk-high-solid" : value >= bands.medium ? "bg-risk-medium-solid" : "bg-risk-low-solid";
  return (
    <div className={cn("w-full", className)}>
      <div
        className="relative h-2.5 w-full overflow-hidden rounded-full bg-surface-sunken"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value}
        aria-valuetext={`Risk score ${fmtPct(value)}`}
        aria-label="Risk score"
      >
        <motion.div
          className={cn("h-full rounded-full", fill)}
          initial={{ width: reduced ? `${value * 100}%` : 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: reduced ? 0 : 0.7, ease: [0.22, 1, 0.36, 1] }}
        />
        {[bands.medium, bands.high].map((b) => (
          <span key={b} className="absolute inset-y-0 w-0.5 bg-surface" style={{ left: `${b * 100}%` }} aria-hidden="true" />
        ))}
      </div>
      <div className="tabular relative mt-1 h-4 text-2xs text-text-subtle" aria-hidden="true">
        <span className="absolute left-0">0%</span>
        <span className="absolute -translate-x-1/2" style={{ left: `${bands.medium * 100}%` }}>
          {fmtPct(bands.medium, 0)}
        </span>
        <span className="absolute -translate-x-1/2" style={{ left: `${bands.high * 100}%` }}>
          {fmtPct(bands.high, 0)}
        </span>
        <span className="absolute right-0">100%</span>
      </div>
    </div>
  );
}

export function SkeletonRows({ rows = 6, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
      ))}
    </div>
  );
}
