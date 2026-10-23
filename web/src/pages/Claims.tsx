import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, FilterX, Network, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { DecisionBanner, EmptyState, ErrorState, PageHeader, RiskBadge, SkeletonRows } from "@/components/common";
import { useDomain } from "@/lib/domain";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { useClaims, type ClaimsFilters } from "@/lib/api";
import { DOMAIN_LABEL, RISK_LABEL, fmtInt, fmtPct, fmtValue, label, shortClaim } from "@/lib/format";
import type { RiskCategory } from "@/lib/types";
import { cn, useDebounced } from "@/lib/utils";

const RISKS: RiskCategory[] = ["high", "medium", "low"];
const SORTS = ["fraud_probability", "claim_key", "risk_category"] as const;
const HIDDEN_COLUMNS = new Set(["community_id", "degree_centrality", "max_entity_claim_degree"]);

function readFilters(sp: URLSearchParams): ClaimsFilters {
  const sort = sp.get("sort");
  const pageSize = Number(sp.get("size"));
  return {
    search: sp.get("q") ?? "",
    risk: sp.getAll("risk").filter((r): r is RiskCategory => RISKS.includes(r as RiskCategory)),
    minProbability: Math.min(1, Math.max(0, Number(sp.get("min")) || 0)),
    segment: sp.getAll("segment"),
    sort: (SORTS as readonly string[]).includes(sort ?? "") ? (sort as ClaimsFilters["sort"]) : "fraud_probability",
    order: sp.get("order") === "asc" ? "asc" : "desc",
    page: Math.max(1, Number(sp.get("page")) || 1),
    pageSize: [10, 25, 50, 100].includes(pageSize) ? pageSize : 25,
  };
}

function writeFilters(f: ClaimsFilters): URLSearchParams {
  const sp = new URLSearchParams();
  if (f.search) sp.set("q", f.search);
  f.risk.forEach((r) => sp.append("risk", r));
  if (f.minProbability) sp.set("min", String(f.minProbability));
  f.segment.forEach((s) => sp.append("segment", s));
  if (f.sort !== "fraud_probability") sp.set("sort", f.sort);
  if (f.order !== "desc") sp.set("order", f.order);
  if (f.page > 1) sp.set("page", String(f.page));
  if (f.pageSize !== 25) sp.set("size", String(f.pageSize));
  return sp;
}

export default function ClaimsPage() {
  const domain = useDomain();
  const navigate = useNavigate();
  const [sp, setSp] = useSearchParams();
  const filters = useMemo(() => readFilters(sp), [sp]);
  const [searchDraft, setSearchDraft] = useState(filters.search);
  const debouncedSearch = useDebounced(searchDraft, 300);

  const update = (patch: Partial<ClaimsFilters>, resetPage = true) =>
    setSp(writeFilters({ ...filters, ...patch, ...(resetPage ? { page: 1 } : {}) }), { replace: true });

  useEffect(() => {
    if (debouncedSearch !== filters.search) update({ search: debouncedSearch });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only react to the debounced draft
  }, [debouncedSearch]);

  const { data, isLoading, isFetching, error, refetch } = useClaims(domain, filters);
  const columns = (data?.columns ?? []).filter((c) => !HIDDEN_COLUMNS.has(c));
  const activeFilters = filters.search || filters.risk.length || filters.minProbability || filters.segment.length;

  const toggleSort = (col: ClaimsFilters["sort"]) => {
    if (filters.sort === col) update({ order: filters.order === "desc" ? "asc" : "desc" }, false);
    else update({ sort: col, order: col === "claim_key" ? "asc" : "desc" }, false);
  };

  const sortHeader = (col: ClaimsFilters["sort"], text: string, className?: string) => {
    const active = filters.sort === col;
    const Icon = !active ? ArrowUpDown : filters.order === "asc" ? ArrowUp : ArrowDown;
    return (
      <th
        scope="col"
        aria-sort={active ? (filters.order === "asc" ? "ascending" : "descending") : "none"}
        className={cn("px-3 py-2.5 font-medium", className)}
      >
        <button
          type="button"
          onClick={() => toggleSort(col)}
          className={cn("inline-flex cursor-pointer items-center gap-1 rounded hover:text-text", active && "text-text")}
        >
          {text}
          <Icon className="size-3.5" aria-hidden="true" />
        </button>
      </th>
    );
  };

  return (
    <>
      <PageHeader
        eyebrow={`${DOMAIN_LABEL[domain]} insurance`}
        title="Claims investigation"
        description="Search, filter and sort the scored sample. Select a claim to open its case file with SHAP factors, the entity network and document forensics."
      />
      <DecisionBanner compact className="mb-4" />

      <Card className="mb-4 p-4">
        <div className="grid gap-4 md:grid-cols-[minmax(14rem,1.4fr)_auto_minmax(10rem,1fr)] md:items-end">
          <div className="space-y-1.5">
            <Label htmlFor="claim-search">Search</Label>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-text-subtle" aria-hidden="true" />
              <Input
                id="claim-search"
                type="search"
                value={searchDraft}
                onChange={(e) => setSearchDraft(e.target.value)}
                placeholder={domain === "motor" ? "Claim number or vehicle make" : "Claim, provider or beneficiary ID"}
                className="pl-9"
              />
            </div>
          </div>
          <fieldset className="space-y-1.5">
            <legend className="mb-1.5 text-xs font-medium text-text-muted">Risk level</legend>
            <div className="flex flex-wrap gap-1.5">
              {RISKS.map((r) => {
                const on = filters.risk.includes(r);
                return (
                  <button
                    key={r}
                    type="button"
                    aria-pressed={on}
                    onClick={() => update({ risk: on ? filters.risk.filter((x) => x !== r) : [...filters.risk, r] })}
                    className={cn(
                      "cursor-pointer rounded-full transition-opacity duration-fast",
                      filters.risk.length && !on && "opacity-45 hover:opacity-80",
                    )}
                  >
                    <RiskBadge risk={r} className={cn("h-8 px-3", on && "ring-2 ring-ring ring-offset-1 ring-offset-surface")} />
                    <span className="sr-only">{on ? "(filter on)" : ""}</span>
                  </button>
                );
              })}
            </div>
          </fieldset>
          <div className="space-y-1.5">
            <div className="flex justify-between">
              <Label htmlFor="min-prob">Minimum risk score</Label>
              <span className="tabular text-xs font-medium text-text">{fmtPct(filters.minProbability, 0)}</span>
            </div>
            <input
              id="min-prob"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={filters.minProbability}
              onChange={(e) => update({ minProbability: Number(e.target.value) })}
              className="h-10 w-full cursor-pointer accent-[rgb(var(--primary))]"
            />
          </div>
        </div>
        {data?.segment_field && data.segment_options.length ? (
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-3">
            <span className="text-xs font-medium text-text-muted">{label(data.segment_field)}:</span>
            {data.segment_options.map((s) => {
              const on = filters.segment.includes(s);
              return (
                <button
                  key={s}
                  type="button"
                  aria-pressed={on}
                  onClick={() => update({ segment: on ? filters.segment.filter((x) => x !== s) : [...filters.segment, s] })}
                  className={cn(
                    "h-7 cursor-pointer rounded-full border px-3 text-xs font-medium transition-colors duration-fast",
                    on ? "border-primary bg-primary text-primary-on" : "border-border-strong text-text-muted hover:text-text",
                  )}
                >
                  {s}
                </button>
              );
            })}
            {activeFilters ? (
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto"
                onClick={() => {
                  setSearchDraft("");
                  setSp(new URLSearchParams(), { replace: true });
                }}
              >
                <FilterX aria-hidden="true" /> Clear filters
              </Button>
            ) : null}
          </div>
        ) : null}
      </Card>

      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2.5">
          <p className="tabular text-xs text-text-muted" aria-live="polite">
            {data ? (
              <>
                <span className="font-semibold text-text">{fmtInt(data.total)}</span> matching claims
                {isFetching ? " · updating…" : ""}
              </>
            ) : (
              "Loading claims…"
            )}
          </p>
          <div className="flex items-center gap-2">
            <Label htmlFor="page-size" className="whitespace-nowrap">
              Rows
            </Label>
            <Select
              id="page-size"
              value={filters.pageSize}
              onChange={(e) => update({ pageSize: Number(e.target.value) })}
              className="h-8 w-20 text-xs"
            >
              {[10, 25, 50, 100].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          </div>
        </div>

        {error ? (
          <ErrorState error={error} onRetry={() => refetch()} className="m-4" />
        ) : isLoading || !data ? (
          <SkeletonRows rows={10} className="p-4" />
        ) : data.items.length === 0 ? (
          <EmptyState title="No claims match these filters" icon={<FilterX />}>
            Widen the risk levels or lower the minimum score.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className={cn("tabular w-full text-sm transition-opacity", isFetching && "opacity-60")}>
              <caption className="sr-only">
                Claims page {data.page} of {data.pages}, sorted by {label(filters.sort)} {filters.order === "asc" ? "ascending" : "descending"}
              </caption>
              <thead className="bg-surface-sunken text-left text-xs text-text-muted">
                <tr>
                  {sortHeader("claim_key", "Claim")}
                  {sortHeader("risk_category", "Risk level")}
                  {sortHeader("fraud_probability", "Risk score", "text-right")}
                  {columns.map((c) => (
                    <th key={c} scope="col" className="whitespace-nowrap px-3 py-2.5 font-medium">
                      {label(c)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.items.map((row) => {
                  const href = `/${domain}/claims/${encodeURIComponent(row.claim_key)}`;
                  return (
                    <tr
                      key={row.claim_key}
                      onClick={(e) => {
                        if ((e.target as HTMLElement).closest("a")) return;
                        navigate(href);
                      }}
                      className="h-[var(--table-row-height)] cursor-pointer border-t border-border hover:bg-primary-soft/60"
                    >
                      <td className="whitespace-nowrap px-3">
                        <Link to={href} className="font-medium text-primary hover:underline">
                          {shortClaim(row.claim_key)}
                        </Link>
                      </td>
                      <td className="px-3">
                        <RiskBadge risk={row.risk_category} />
                      </td>
                      <td className="px-3 text-right">
                        <span className="inline-flex items-center gap-2">
                          <span className="hidden h-1.5 w-14 overflow-hidden rounded-full bg-surface-sunken sm:inline-block" aria-hidden="true">
                            <span
                              className={cn(
                                "block h-full rounded-full",
                                { high: "bg-risk-high-solid", medium: "bg-risk-medium-solid", low: "bg-risk-low-solid" }[row.risk_category],
                              )}
                              style={{ width: `${row.fraud_probability * 100}%` }}
                            />
                          </span>
                          <span className="font-semibold text-text" title={String(row.fraud_probability)}>
                            {fmtPct(row.fraud_probability)}
                          </span>
                        </span>
                      </td>
                      {columns.map((c) => {
                        const v = row.attributes[c];
                        if (c === "has_shared_entity") {
                          return (
                            <td key={c} className="px-3">
                              {v ? (
                                <Badge tone="accent">
                                  <Network aria-hidden="true" /> Shared
                                </Badge>
                              ) : (
                                <span className="text-xs text-text-subtle">None</span>
                              )}
                            </td>
                          );
                        }
                        return (
                          <td key={c} className="max-w-[12rem] truncate whitespace-nowrap px-3 text-text-muted">
                            {fmtValue(v)}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {data && data.items.length ? (
          <nav className="flex items-center justify-between gap-2 border-t border-border px-4 py-2.5" aria-label="Pagination">
            <p className="tabular text-xs text-text-muted">
              Page {data.page} of {data.pages}
            </p>
            <div className="flex gap-1.5">
              <Button variant="outline" size="sm" disabled={data.page <= 1} onClick={() => update({ page: data.page - 1 }, false)}>
                <ChevronLeft aria-hidden="true" /> Previous
              </Button>
              <Button variant="outline" size="sm" disabled={data.page >= data.pages} onClick={() => update({ page: data.page + 1 }, false)}>
                Next <ChevronRight aria-hidden="true" />
              </Button>
            </div>
          </nav>
        ) : null}
      </Card>
      <p className="mt-3 text-2xs text-text-subtle">
        {RISK_LABEL.high} and {RISK_LABEL.medium.toLowerCase()} claims are flagged for review. Bands are triage conventions, not fitted operating points.
      </p>
    </>
  );
}
