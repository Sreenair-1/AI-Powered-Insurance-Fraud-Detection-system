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

export interface RingEvidence {
  entities: { type: string; value: string }[];
  claims: number;
  expected: number;
  p_value: number;
  includes_this_claim: boolean;
}

export interface NetworkRisk {
  level: "candidate_ring" | "linked" | "none";
  entity_types: string[];
  method: string;
  note: string;
  sample_candidate_rings: number;
  sample_flagged_claims: number;
  sample_claims: number;
  ring_id: string | null;
  ring_size: number;
  ring_claims: string[];
  evidence: RingEvidence[];
  benchmark: {
    noise: number | null;
    precision: number | null;
    recall: number | null;
    ring_recovery_rate: number | null;
    naive_precision: number | null;
  } | null;
}

export interface RateSummary {
  detection_rate: number | null;
  false_positive_rate: number | null;
}

export interface DocumentBenchmarkSummary {
  source: string | null;
  n_bills: number | null;
  as_submitted?: RateSummary;
  messaging_app?: RateSummary;
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
  network_risk: NetworkRisk;
  document_benchmark: DocumentBenchmarkSummary | null;
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
  tampering_benchmark: TamperingBenchmark | null;
  ring_benchmark: RingBenchmark | null;
  workload_curve: WorkloadCurve | null;
}

export type ReviewOutcome = "confirmed_suspicious" | "cleared" | "needs_more_info";

export interface ReviewRecord {
  domain: Domain;
  claim_key: string;
  outcome: ReviewOutcome;
  retraining_label: 0 | 1 | null;
  note: string;
  reviewer: string | null;
  recorded_at: string;
  fraud_probability_at_review: number | null;
  risk_category_at_review: string | null;
  network_level_at_review: string | null;
  model_version: string | null;
}

export interface ClaimReview {
  current: ReviewRecord | null;
  history: ReviewRecord[];
}

export interface ReviewSummary {
  domain: Domain;
  reviewed_claims: number;
  counts: Record<ReviewOutcome, number>;
  labels: { claim_key: string; label: 0 | 1; recorded_at: string; model_version: string | null }[];
  recent: ReviewRecord[];
}

export interface WorkloadPoint {
  review_rate: number;
  recall: number;
  precision: number | null;
  lift: number | null;
  threshold: number | null;
}

export interface WorkloadCurve {
  split: string;
  prevalence: number;
  n_samples: number;
  n_positive: number;
  points: WorkloadPoint[];
}

// scripts/run_benchmarks.py output (models/benchmarks/*.json)
export type ForensicMethod = "ela" | "copy_move" | "metadata" | "combined_verdict";

export interface TamperingMethodResult {
  detection_rate: number | null;
  false_positive_rate: number | null;
  precision: number | null;
  roc_auc_continuous: number | null;
  best_detection_at_fpr_10: { threshold: number | null; detection_rate: number; false_positive_rate: number };
}

export interface TamperingConditionResult {
  n_tampered: number;
  n_untouched: number;
  methods: Record<ForensicMethod, TamperingMethodResult>;
  detection_rate_by_edit: Record<string, Record<ForensicMethod, number | null>>;
  false_positive_rate_by_variant: Record<string, Record<ForensicMethod, number | null>>;
}

export interface TamperingBenchmark {
  source: string;
  n_bills: number;
  n_samples: number;
  edit_types: string[];
  untouched_variants: string[];
  conditions: Record<string, string>;
  results: Record<string, TamperingConditionResult>;
  limitations: string[];
}

export interface RingDetectionMetrics {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  ring_recovery_rate?: number | null;
  n_candidate_rings?: number | null;
  n_false_candidate_rings?: number | null;
}

export interface RingBenchmark {
  seeds: number[];
  scenario: Record<string, unknown> & { n_claims: number; n_rings: number };
  ring_member_prevalence: number;
  detector: { method: string; alpha: number; min_ring_size: number; matching_modes: Record<string, string> };
  recovery_rule: string;
  noise_sweep: ({ noise: number; naive_any_shared_entity: RingDetectionMetrics } & Record<
    "raw" | "normalized" | "fuzzy",
    RingDetectionMetrics
  >)[];
  operating_points: { noise: number; matching: string; rows: (RingDetectionMetrics & { alpha: number; min_ring_size: number })[] };
  limitations: string[];
}
