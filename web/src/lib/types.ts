// Mirrors api/schemas.py. Field names are kept identical to the pipeline's
// own (`fraud_probability`, `risk_category`, ...); the UI decides labels.

export type Domain = "health" | "motor";
export type RiskCategory = "low" | "medium" | "high";
export type Scalar = string | number | boolean | null;

export interface RiskBands {
  medium: number;
  high: number;
}

export interface DomainHealth {
  domain: Domain;
  model_loaded: boolean;
  model_dir: string;
  error: string | null;
  n_features: number | null;
  uses_graph_features: boolean | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  domains: DomainHealth[];
  risk_bands: RiskBands;
}

export interface Factor {
  feature: string;
  feature_value: Scalar;
  shap_value: number;
  source: "model_shap" | "document_heuristic";
}

export interface Assessment {
  claim_key: string | null;
  domain: Domain;
  fraud_probability: number;
  risk_category: RiskCategory;
  risk_category_description: string;
  base_value: number;
  top_factors: Factor[];
}

export interface ClaimSummary {
  claim_key: string;
  fraud_probability: number;
  risk_category: RiskCategory;
  attributes: Record<string, Scalar>;
}

export interface OverviewResponse {
  domain: Domain;
  sample: Record<string, number>;
  total_claims: number;
  risk_counts: Record<RiskCategory, number>;
  review_count: number;
  review_rate: number | null;
  risk_bands: RiskBands;
  probability_histogram: { start: number; end: number; count: number }[];
  segment_field: string | null;
  segments: { segment: string; low: number; medium: number; high: number }[];
  global_importance: { feature: string; importance: number }[];
  top_claims: ClaimSummary[];
}

export interface ClaimsPage {
  domain: Domain;
  sample: Record<string, number>;
  total: number;
  page: number;
  page_size: number;
  pages: number;
  columns: string[];
  segment_field: string | null;
  segment_options: string[];
  items: ClaimSummary[];
}

export interface ClaimDetail {
  domain: Domain;
  claim_key: string;
  assessment: Assessment;
  attributes: Record<string, Scalar>;
  attribute_groups: { name: string; fields: string[] }[];
  graph_signals: Record<string, Scalar>;
  model_uses_graph_features: boolean;
  risk_bands: RiskBands;
}

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  is_focus: boolean;
  linked_claims: number;
  shared: boolean;
  risk_category: RiskCategory | null;
  fraud_probability: number | null;
  community_id: number | null;
}

export interface GraphResponse {
  domain: Domain;
  claim_key: string;
  nodes: GraphNode[];
  edges: { source: string; target: string; type: string }[];
  truncated: boolean;
  synthetic_entities: string[];
}

export type Box = [number, number, number, number];

export interface ForensicReport {
  filename: string;
  is_pdf: boolean;
  page_index: number;
  page_count: number;
  image_width: number | null;
  image_height: number | null;
  original_image: string | null;
  ela_image: string | null;
  copy_move_image: string | null;
  ela: Record<string, number> | null;
  copy_move: (Record<string, number> & { potential_duplicate_regions_found: boolean }) | null;
  copy_move_pairs: Box[];
  copy_move_regions: { source: Box; target: Box; matches: number }[];
  metadata: {
    structured: Record<string, unknown>;
    features: Record<string, Scalar>;
    indicators: string[];
  } | null;
  metadata_flags: { field: string; reason: string }[];
  extracted_text: string | null;
  verdict: {
    status: "authentic" | "suspicious" | "tampered" | string;
    tampering_score: number;
    is_tampered: boolean;
    indicators: string[];
    summary: string;
    details: Record<string, Scalar>;
  };
}

export interface DocumentRescoreResponse {
  domain: Domain;
  claim_key: string;
  forensics: ForensicReport;
  before: Assessment;
  after: Assessment;
  fusion_note: string;
}

export interface FieldSpec {
  name: string;
  kind: "categorical" | "integer" | "number" | "boolean";
  options: string[] | null;
  default: Scalar;
  minimum: number | null;
  required: boolean;
}

export interface ClaimSchema {
  domain: Domain;
  fields: FieldSpec[];
  notes: string[];
}

export interface AssessResponse {
  domain: Domain;
  claim_key: string;
  assessment: Assessment;
  forensics: ForensicReport | null;
  notes: string[];
}

export type MetricSet = Record<string, number | null>;

export interface InsightsResponse {
  domain: Domain;
  model_dir: string;
  feature_names: string[];
  uses_graph_features: boolean;
  risk_bands: RiskBands;
  eval_metrics: MetricSet | null;
  validation_metrics: Record<string, unknown> | null;
  test_metrics: {
    split?: string;
    threshold_frozen_from_validation?: number;
    selected_threshold_metrics?: MetricSet;
    confusion_matrix?: Record<string, number>;
  } | null;
  threshold_selection: { threshold: number; objective: string; constraints: Record<string, unknown> } | null;
  confusion_matrix: Record<string, number> | null;
  roc_curve: { fpr: number[]; tpr: number[]; thresholds?: number[] } | null;
  pr_curve: { precision: number[]; recall: number[]; thresholds?: number[] } | null;
  probability_diagnostics: Record<
    string,
    { count: number; mean: number; histogram?: { bin_edges: number[]; counts: number[] } }
  > | null;
  data_audit: Record<string, unknown> | null;
  validation_experiment: Record<string, unknown> | null;
  provider_grouped_test_metrics: { metrics?: MetricSet; n_providers?: number; aggregation?: string } | null;
  limitations: string[];
}
