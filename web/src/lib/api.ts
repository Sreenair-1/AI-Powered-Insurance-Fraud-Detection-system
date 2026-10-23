import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import type {
  AssessResponse,
  ClaimDetail,
  ClaimSchema,
  ClaimsPage,
  DocumentRescoreResponse,
  Domain,
  GraphResponse,
  HealthResponse,
  InsightsResponse,
  OverviewResponse,
  RiskCategory,
} from "./types";
import { useSampleParams, type SampleParams } from "./settings-store";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code?: string,
  ) {
    super(message);
  }
}

function detailMessage(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (typeof d === "object" && d && "msg" in d ? `${(d as { loc?: unknown[] }).loc?.slice(-1)[0] ?? ""}: ${(d as { msg: string }).msg}` : String(d)))
      .join("; ");
  }
  return "Request failed";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "Cannot reach the InsureSight API. Is it running on port 8000?");
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    let code: string | undefined;
    try {
      const body = await res.json();
      message = detailMessage(body.detail);
      code = body.code;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message, code);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | string[] | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) v.forEach((item) => sp.append(k, item));
    else sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

function sampleQuery(domain: Domain, p: SampleParams) {
  return domain === "motor"
    ? { sample_size: p.motorSampleSize }
    : { inpatient_sample_size: p.inpatientSampleSize, outpatient_sample_size: p.outpatientSampleSize };
}

export const enc = (claimKey: string) => encodeURIComponent(claimKey);

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => request<HealthResponse>("/health"),
    refetchInterval: 30_000,
    retry: 1,
  });
}

export function useOverview(domain: Domain) {
  const p = useSampleParams();
  const sq = sampleQuery(domain, p);
  return useQuery({
    queryKey: ["overview", domain, sq],
    queryFn: () => request<OverviewResponse>(`/domains/${domain}/overview${qs(sq)}`),
  });
}

export interface ClaimsFilters {
  search: string;
  risk: RiskCategory[];
  minProbability: number;
  segment: string[];
  sort: "fraud_probability" | "claim_key" | "risk_category";
  order: "asc" | "desc";
  page: number;
  pageSize: number;
}

export function useClaims(domain: Domain, f: ClaimsFilters) {
  const p = useSampleParams();
  const sq = sampleQuery(domain, p);
  const params = {
    ...sq,
    search: f.search,
    risk: f.risk,
    min_probability: f.minProbability || undefined,
    segment: f.segment,
    sort: f.sort,
    order: f.order,
    page: f.page,
    page_size: f.pageSize,
  };
  return useQuery({
    queryKey: ["claims", domain, params],
    queryFn: () => request<ClaimsPage>(`/domains/${domain}/claims${qs(params)}`),
    placeholderData: keepPreviousData,
  });
}

export function useClaimDetail(domain: Domain, claimKey: string) {
  const p = useSampleParams();
  const sq = sampleQuery(domain, p);
  return useQuery({
    queryKey: ["claim", domain, claimKey, sq],
    queryFn: () => request<ClaimDetail>(`/domains/${domain}/claims/${enc(claimKey)}${qs(sq)}`),
    retry: (count, err) => !(err instanceof ApiError && err.status === 404) && count < 2,
  });
}

export function useClaimGraph(domain: Domain, claimKey: string) {
  const p = useSampleParams();
  const sq = sampleQuery(domain, p);
  return useQuery({
    queryKey: ["graph", domain, claimKey, sq],
    queryFn: () => request<GraphResponse>(`/domains/${domain}/claims/${enc(claimKey)}/graph${qs(sq)}`),
    retry: (count, err) => !(err instanceof ApiError && err.status === 404) && count < 2,
  });
}

export function useSchema(domain: Domain) {
  return useQuery({
    queryKey: ["schema", domain],
    queryFn: () => request<ClaimSchema>(`/domains/${domain}/schema`),
    staleTime: Infinity,
  });
}

export function useInsights(domain: Domain) {
  return useQuery({
    queryKey: ["insights", domain],
    queryFn: () => request<InsightsResponse>(`/domains/${domain}/insights`),
    staleTime: Infinity,
  });
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export function useDocumentRescore(domain: Domain) {
  const p = useSampleParams();
  return useMutation({
    mutationFn: ({ claimKey, file, pageIndex }: { claimKey: string; file: File; pageIndex: number }) => {
      const body = new FormData();
      body.append("file", file);
      body.append("page_index", String(pageIndex));
      return request<DocumentRescoreResponse>(
        `/domains/${domain}/claims/${enc(claimKey)}/document${qs(sampleQuery(domain, p))}`,
        { method: "POST", body },
      );
    },
  });
}

export function useAssess(domain: Domain) {
  return useMutation({
    mutationFn: ({
      attributes,
      file,
      pageIndex,
    }: {
      attributes: Record<string, unknown>;
      file?: File | null;
      pageIndex: number;
    }) => {
      const body = new FormData();
      body.append("attributes", JSON.stringify(attributes));
      if (file) {
        body.append("file", file);
        body.append("page_index", String(pageIndex));
      }
      return request<AssessResponse>(`/domains/${domain}/assess`, { method: "POST", body });
    },
  });
}
