import { ExternalLink, Maximize2, Network } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods, type NodeObject } from "react-force-graph-2d";
import { Link } from "react-router-dom";
import { NODE_TYPE_LABEL, RISK_LABEL, fmtPct, shortClaim } from "@/lib/format";
import type { Domain, GraphNode, GraphResponse } from "@/lib/types";
import { cn, useTokenColors, type TokenColors } from "@/lib/utils";
import { RiskBadge } from "./common";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

type FGNode = NodeObject & GraphNode;

const TYPE_TOKEN: Record<string, keyof TokenColors> = {
  claim: "node-claim",
  provider: "node-provider",
  physician: "node-physician",
  beneficiary: "node-beneficiary",
  garage: "node-garage",
  bank_account: "node-bank",
};

function nodeColor(c: TokenColors, type: string) {
  return c[TYPE_TOKEN[type] ?? "text-subtle"];
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export function EntityGraph({ data, domain, height = 420 }: { data: GraphResponse; domain: Domain; height?: number }) {
  const c = useTokenColors();
  const fgRef = useRef<ForceGraphMethods<FGNode> | undefined>(undefined);
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const [selectedId, setSelectedId] = useState<string>(data.claim_key);

  const graphData = useMemo(
    () => ({
      nodes: data.nodes.map((n) => ({ ...n })) as FGNode[],
      links: data.edges.map((e) => ({ ...e })),
    }),
    [data],
  );
  const selected = data.nodes.find((n) => n.id === selectedId) ?? data.nodes[0];
  const neighbours = useMemo(() => {
    const set = new Set<string>();
    for (const e of data.edges) {
      if (e.source === selectedId) set.add(e.target);
      if (e.target === selectedId) set.add(e.source);
    }
    return set;
  }, [data.edges, selectedId]);

  const types = useMemo(() => Array.from(new Set(data.nodes.map((n) => n.type))), [data.nodes]);
  const sharedCount = data.nodes.filter((n) => n.shared).length;
  const otherClaims = data.nodes.filter((n) => n.type === "claim" && !n.is_focus).length;

  const draw = useCallback(
    (node: FGNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const isClaim = node.type === "claim";
      const r = node.is_focus ? 9 : isClaim ? 5.5 : 6.5;
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const dim = selectedId !== node.id && !neighbours.has(node.id) && selectedId !== data.claim_key;
      ctx.globalAlpha = dim ? 0.35 : 1;

      // Shared-entity halo (dashed) - the ring indicator.
      if (node.shared) {
        ctx.beginPath();
        ctx.setLineDash([3, 2]);
        ctx.arc(x, y, r + 5, 0, 2 * Math.PI);
        ctx.strokeStyle = c.accent;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // Shape by kind: circles for claims, rounded squares for entities.
      ctx.beginPath();
      if (isClaim) ctx.arc(x, y, r, 0, 2 * Math.PI);
      else ctx.roundRect(x - r, y - r, r * 2, r * 2, 2.5);
      ctx.fillStyle = nodeColor(c, node.type);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = c.surface;
      ctx.stroke();
      // Risk ring on claims in the loaded sample.
      if (isClaim && node.risk_category) {
        ctx.beginPath();
        ctx.arc(x, y, r + 2.2, 0, 2 * Math.PI);
        ctx.strokeStyle = c[`risk-${node.risk_category}-solid` as const];
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (node.id === selectedId) {
        ctx.beginPath();
        ctx.arc(x, y, r + (node.shared ? 8.5 : 5.5), 0, 2 * Math.PI);
        ctx.strokeStyle = c.text;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
      const showLabel = node.is_focus || !isClaim || scale > 2.2 || node.id === selectedId;
      if (showLabel) {
        const fontSize = Math.max(10 / scale, 2.5);
        ctx.font = `${node.is_focus ? 600 : 500} ${fontSize}px Inter Variable, Inter, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = c["text-muted"];
        ctx.fillText(node.is_focus ? `${node.label} (this claim)` : node.label, x, y + r + 4);
      }
      ctx.globalAlpha = 1;
    },
    [c, selectedId, neighbours, data.claim_key],
  );

  const fit = () => fgRef.current?.zoomToFit(400, 40);

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_15rem]">
      <div className="min-w-0">
        <div
          ref={wrapRef}
          className="relative overflow-hidden rounded-lg border border-border bg-surface-sunken"
          style={{ height }}
        >
          <div className="absolute left-3 top-3 z-10 flex flex-wrap gap-1.5">
            <Badge tone="neutral" className="bg-surface">
              <Network aria-hidden="true" /> {data.nodes.length} nodes
            </Badge>
            {sharedCount ? (
              <Badge tone="accent">{sharedCount} shared {sharedCount === 1 ? "entity" : "entities"}</Badge>
            ) : (
              <Badge tone="neutral" className="bg-surface">No shared entities</Badge>
            )}
          </div>
          <Button
            variant="outline"
            size="icon-sm"
            className="absolute right-3 top-3 z-10"
            onClick={fit}
            aria-label="Fit graph to view"
            title="Fit graph to view"
          >
            <Maximize2 aria-hidden="true" />
          </Button>
          <div aria-hidden="true">
            <ForceGraph2D<FGNode>
              ref={fgRef}
              graphData={graphData}
              width={width}
              height={height}
              backgroundColor="rgba(0,0,0,0)"
              nodeId="id"
              nodeCanvasObject={draw}
              nodePointerAreaPaint={(node, color, ctx) => {
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(node.x ?? 0, node.y ?? 0, 12, 0, 2 * Math.PI);
                ctx.fill();
              }}
              nodeLabel={(n) => `${NODE_TYPE_LABEL[n.type] ?? n.type}: ${n.label}`}
              linkColor={(l) => {
                const s = typeof l.source === "object" ? (l.source as FGNode).id : l.source;
                const t = typeof l.target === "object" ? (l.target as FGNode).id : l.target;
                return s === selectedId || t === selectedId ? c["text-subtle"] : c.border;
              }}
              linkWidth={1.2}
              cooldownTicks={120}
              onEngineStop={fit}
              onNodeClick={(n) => setSelectedId(String(n.id))}
              enableNodeDrag
            />
          </div>
        </div>
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-muted" aria-label="Graph legend">
          {types.map((t) => (
            <li key={t} className="inline-flex items-center gap-1.5">
              <span
                className={cn("inline-block size-2.5", t === "claim" ? "rounded-full" : "rounded-[2px]")}
                style={{ background: nodeColor(c, t) }}
                aria-hidden="true"
              />
              {NODE_TYPE_LABEL[t] ?? t}
              {data.synthetic_entities.includes(t) ? " (synthetic)" : ""}
            </li>
          ))}
          <li className="inline-flex items-center gap-1.5">
            <span className="inline-block size-2.5 rounded-full border border-dashed border-accent" aria-hidden="true" />
            Shared by 2+ claims
          </li>
          <li className="inline-flex items-center gap-1.5">
            <span className="inline-block size-2.5 rounded-full border-2 border-risk-high-solid" aria-hidden="true" />
            Ring = claim risk level
          </li>
        </ul>
        {data.truncated ? (
          <p className="mt-1 text-2xs text-text-subtle">Showing the first 80 nodes of this neighbourhood.</p>
        ) : null}
      </div>

      <aside className="flex min-h-0 flex-col rounded-lg border border-border lg:max-h-[var(--graph-panel-h)]" style={{ ["--graph-panel-h" as string]: `${height + 56}px` }} aria-label="Node details">
        {selected ? <NodeDetails node={selected} domain={domain} synthetic={data.synthetic_entities.includes(selected.type)} /> : null}
        <div className="border-t border-border px-3 py-2">
          <p className="text-2xs font-semibold uppercase tracking-wide text-text-subtle">
            Nodes ({otherClaims} linked {otherClaims === 1 ? "claim" : "claims"})
          </p>
        </div>
        <ul className="max-h-48 min-h-0 overflow-y-auto px-1.5 pb-1.5 lg:max-h-none lg:flex-1" aria-label="Select a node">
          {data.nodes.map((n) => (
            <li key={n.id}>
              <button
                type="button"
                onClick={() => setSelectedId(n.id)}
                aria-pressed={n.id === selectedId}
                className={cn(
                  "flex w-full cursor-pointer items-center gap-2 rounded px-2 py-1 text-left text-xs hover:bg-surface-sunken",
                  n.id === selectedId && "bg-primary-soft text-primary",
                )}
              >
                <span
                  className={cn("inline-block size-2 shrink-0", n.type === "claim" ? "rounded-full" : "rounded-[2px]")}
                  style={{ background: nodeColor(c, n.type) }}
                  aria-hidden="true"
                />
                <span className="truncate">{n.label}</span>
                <span className="ml-auto shrink-0 text-2xs text-text-subtle">
                  {n.is_focus ? "this claim" : NODE_TYPE_LABEL[n.type]}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}

function NodeDetails({ node, domain, synthetic }: { node: GraphNode; domain: Domain; synthetic: boolean }) {
  const isClaim = node.type === "claim";
  return (
    <div className="space-y-2 p-3" aria-live="polite">
      <p className="text-2xs font-semibold uppercase tracking-wide text-text-subtle">
        {NODE_TYPE_LABEL[node.type] ?? node.type}
        {synthetic ? " · synthetic" : ""}
      </p>
      <p className="break-all text-sm font-semibold text-text">{node.label}</p>
      {isClaim ? (
        <>
          {node.risk_category ? (
            <div className="flex items-center gap-2">
              <RiskBadge risk={node.risk_category} />
              <span className="tabular text-xs text-text-muted">{fmtPct(node.fraud_probability)}</span>
            </div>
          ) : (
            <p className="text-xs text-text-muted">Not in the loaded sample.</p>
          )}
          {node.community_id != null ? <p className="text-xs text-text-muted">Community #{node.community_id}</p> : null}
          {!node.is_focus && node.risk_category ? (
            <Button asChild variant="outline" size="sm" className="w-full">
              <Link to={`/${domain}/claims/${encodeURIComponent(node.id)}`}>
                Open claim {shortClaim(node.id)} <ExternalLink aria-hidden="true" />
              </Link>
            </Button>
          ) : null}
          {node.is_focus ? <p className="text-xs text-text-muted">The claim under review.</p> : null}
        </>
      ) : (
        <>
          <p className="tabular text-xs text-text-muted">
            Linked to <span className="font-semibold text-text">{node.linked_claims}</span>{" "}
            {node.linked_claims === 1 ? "claim" : "claims"} in the loaded sample
          </p>
          {node.shared ? (
            <Badge tone="accent">Shared entity — review linked claims</Badge>
          ) : (
            <Badge tone="neutral">Not shared</Badge>
          )}
        </>
      )}
      {isClaim && node.risk_category ? <span className="sr-only">{RISK_LABEL[node.risk_category]}</span> : null}
    </div>
  );
}
