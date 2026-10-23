import { motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, FileWarning } from "lucide-react";
import { explainFactor, fmtSigned, fmtValue, label } from "@/lib/format";
import type { Factor } from "@/lib/types";
import { cn, usePrefersReducedMotion } from "@/lib/utils";
import { Tooltip } from "./ui/tooltip";

/**
 * Diverging bar list of per-claim contributions. Bars extend right (raises
 * risk) or left (lowers risk) from a shared zero line; direction is also
 * carried by an arrow icon and the signed value, so colour is never the only cue.
 */
export function ShapChart({ factors, baseValue }: { factors: Factor[]; baseValue?: number }) {
  const reduced = usePrefersReducedMotion();
  if (!factors.length) {
    return <p className="text-sm text-text-muted">No contributing factors were returned for this claim.</p>;
  }
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.shap_value)), 1e-9);
  const raising = factors.filter((f) => f.shap_value > 0).length;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-sm bg-risk-high-solid" aria-hidden="true" />
          <ArrowUpRight className="size-3.5 text-risk-high" aria-hidden="true" /> Raises risk
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-sm bg-risk-low-solid" aria-hidden="true" />
          <ArrowDownRight className="size-3.5 text-risk-low" aria-hidden="true" /> Lowers risk
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-sm border border-dashed border-text-subtle" aria-hidden="true" />
          Document rule weight (not SHAP)
        </span>
        {baseValue != null ? (
          <span className="tabular ml-auto">Model baseline (log-odds): {baseValue.toFixed(3)}</span>
        ) : null}
      </div>

      <ol
        className="space-y-1"
        aria-label={`Top ${factors.length} factors: ${raising} raise the risk score and ${factors.length - raising} lower it.`}
      >
        {factors.map((f, i) => {
          const up = f.shap_value > 0;
          const width = (Math.abs(f.shap_value) / maxAbs) * 50;
          const heuristic = f.source === "document_heuristic";
          const Icon = up ? ArrowUpRight : ArrowDownRight;
          return (
            <li key={`${f.feature}-${i}`}>
              <Tooltip content={explainFactor(f)} side="top">
                <div
                  tabIndex={0}
                  className="grid cursor-default grid-cols-[minmax(8rem,13rem)_1fr_4.5rem] items-center gap-3 rounded-md px-2 py-1.5 hover:bg-surface-sunken focus-visible:bg-surface-sunken"
                  aria-label={explainFactor(f)}
                >
                  <div className="min-w-0">
                    <p className="flex items-start gap-1 text-xs font-medium text-text">
                      {heuristic ? <FileWarning className="mt-px size-3.5 shrink-0 text-accent" aria-hidden="true" /> : null}
                      <span className="line-clamp-2">{label(f.feature)}</span>
                    </p>
                    <p className="truncate text-2xs text-text-muted">{fmtValue(f.feature_value)}</p>
                  </div>
                  <div className="relative h-5" aria-hidden="true">
                    <span className="absolute inset-y-0 left-1/2 w-px bg-border-strong" />
                    <motion.span
                      className={cn(
                        "absolute top-1/2 h-3.5 -translate-y-1/2",
                        up ? "left-1/2 rounded-r" : "right-1/2 rounded-l",
                        heuristic
                          ? "border border-dashed border-accent bg-accent-soft"
                          : up
                            ? "bg-risk-high-solid"
                            : "bg-risk-low-solid",
                      )}
                      initial={{ width: reduced ? `${width}%` : 0 }}
                      animate={{ width: `${width}%` }}
                      transition={{ duration: reduced ? 0 : 0.5, delay: reduced ? 0 : i * 0.035, ease: [0.22, 1, 0.36, 1] }}
                    />
                  </div>
                  <p
                    className={cn(
                      "tabular flex items-center justify-end gap-0.5 text-xs font-semibold",
                      up ? "text-risk-high" : "text-risk-low",
                    )}
                  >
                    <Icon className="size-3.5" aria-hidden="true" />
                    {fmtSigned(f.shap_value)}
                  </p>
                </div>
              </Tooltip>
            </li>
          );
        })}
      </ol>
      <p className="mt-3 text-2xs text-text-subtle">
        SHAP values are additive log-odds contributions from the trained XGBoost model (TreeExplainer). They describe
        how the model reached this score, not evidence about the claim itself.
      </p>
    </div>
  );
}
