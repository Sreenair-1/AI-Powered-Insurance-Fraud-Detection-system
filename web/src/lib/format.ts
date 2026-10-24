import type { Domain, Factor, RiskCategory, Scalar } from "./types";

// Display formatting only: values are shown as returned by the API, with the
// full-precision value available in a tooltip wherever a number is shortened.

const intFmt = new Intl.NumberFormat("en-IN");

export const fmtInt = (n: number | null | undefined) => (n == null ? "—" : intFmt.format(n));

export function fmtPct(p: number | null | undefined, digits = 1): string {
  if (p == null || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function fmtMetric(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtSigned(v: number, digits = 3): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
}

export function fmtValue(v: Scalar | unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "boolean") return v ? "Yes" : "No";
  // Raw attribute values: no digit grouping, so years and IDs read as recorded (1994, not 1,994).
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(4)));
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export const RISK_LABEL: Record<RiskCategory, string> = {
  high: "High risk",
  medium: "Medium risk",
  low: "Low risk",
};

export const DOMAIN_LABEL: Record<Domain, string> = {
  health: "Health",
  motor: "Motor",
};

// Human labels for model features and claim attributes. Unknown names fall
// back to a de-camel-cased version of the raw column name.
const LABELS: Record<string, string> = {
  claim_key: "Claim",
  fraud_probability: "Risk score",
  risk_category: "Risk level",
  PolicyType: "Policy type",
  BasePolicy: "Base policy",
  Deductible: "Deductible",
  Days_Policy_Accident: "Days: policy → accident",
  Days_Policy_Claim: "Days: policy → claim",
  Make: "Vehicle make",
  VehicleCategory: "Vehicle category",
  VehiclePrice: "Vehicle price band",
  AgeOfVehicle: "Vehicle age",
  NumberOfCars: "Cars on policy",
  AccidentArea: "Accident area",
  Fault: "At fault",
  PoliceReportFiled: "Police report filed",
  WitnessPresent: "Witness present",
  Month: "Accident month",
  WeekOfMonth: "Accident week of month",
  DayOfWeek: "Accident day",
  MonthClaimed: "Month claimed",
  WeekOfMonthClaimed: "Week of month claimed",
  DayOfWeekClaimed: "Day claimed",
  Year: "Year",
  Age: "Driver age",
  Sex: "Sex",
  MaritalStatus: "Marital status",
  DriverRating: "Driver rating",
  AgeOfPolicyHolder: "Policyholder age band",
  PastNumberOfClaims: "Past claims",
  AgentType: "Agent type",
  NumberOfSuppliments: "Supplements",
  AddressChange_Claim: "Address change before claim",
  GarageID: "Garage (synthetic)",
  BankAccountID: "Bank account (synthetic)",
  ClaimID: "Claim ID",
  claim_type: "Claim type",
  ClaimStartDt: "Claim start",
  ClaimEndDt: "Claim end",
  InscClaimAmtReimbursed: "Amount reimbursed",
  DeductibleAmtPaid: "Deductible paid",
  Provider: "Provider",
  AttendingPhysician: "Attending physician",
  OperatingPhysician: "Operating physician",
  OtherPhysician: "Other physician",
  BeneID: "Beneficiary",
  num_diagnosis_codes: "Diagnosis codes",
  num_procedure_codes: "Procedure codes",
  claim_duration_days: "Claim duration (days)",
  admission_to_claim_days: "Admission → claim (days)",
  is_deceased: "Beneficiary deceased",
  chronic_condition_count: "Chronic conditions",
  has_renal_disease: "Renal disease",
  has_shared_entity: "Shares an entity",
  max_entity_claim_degree: "Max claims via one entity",
  num_entities: "Linked entities",
  community_id: "Community",
  degree_centrality: "Degree centrality",
  betweenness_centrality: "Betweenness centrality",
};

export function label(name: string): string {
  if (LABELS[name]) return LABELS[name];
  if (name.startsWith("Document Forensics: ")) return name.replace("Document Forensics: ", "Document: ");
  return name
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/^./, (c) => c.toUpperCase());
}

export function shortClaim(key: string): string {
  return key.replace(/^claim:/, "");
}

/** One plain-language sentence for a SHAP or document-evidence factor. */
export function explainFactor(f: Factor): string {
  const value = fmtValue(f.feature_value);
  const raises = f.shap_value > 0;
  if (f.source === "document_heuristic") {
    return `${label(f.feature)} (${value}) added a rule-based document-evidence weight that ${raises ? "raises" : "lowers"} the score. This is a fixed rule, not a SHAP value.`;
  }
  const direction = raises ? "pushed this claim's risk score up" : "pulled this claim's risk score down";
  return `${label(f.feature)} = ${value} ${direction} relative to the model's average claim.`;
}

export const NODE_TYPE_LABEL: Record<string, string> = {
  claim: "Claim",
  provider: "Provider",
  physician: "Physician",
  beneficiary: "Beneficiary",
  garage: "Garage",
  bank_account: "Bank account",
};

export const TAMPER_LABEL: Record<string, string> = {
  authentic: "No significant indicators",
  suspicious: "Suspicious — review",
  tampered: "Strong tampering indicators",
};
