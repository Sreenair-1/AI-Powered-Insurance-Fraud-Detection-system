import { AlertCircle, ClipboardCheck, RotateCcw, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { toast } from "sonner";
import { Callout, DecisionBanner, ErrorState, PageHeader, ProbabilityMeter, RiskBadge, SkeletonRows } from "@/components/common";
import { Dropzone, ForensicReportView } from "@/components/forensics";
import { useDomain } from "@/lib/domain";
import { ShapChart } from "@/components/shap-chart";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { useAssess, useHealth, useSchema } from "@/lib/api";
import { DOMAIN_LABEL, fmtPct, label, shortClaim } from "@/lib/format";
import type { Domain, FieldSpec, Scalar } from "@/lib/types";

// Presentation grouping only; the field list itself comes from the model's
// saved feature names via GET /domains/{domain}/schema.
const GROUPS: Record<Domain, Record<string, string[]>> = {
  motor: {
    "Incident & claim timing": ["Month", "WeekOfMonth", "DayOfWeek", "MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed", "Year", "AccidentArea", "Fault", "PoliceReportFiled", "WitnessPresent"],
    Policy: ["PolicyType", "BasePolicy", "Deductible", "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "AgentType", "NumberOfSuppliments", "AddressChange_Claim"],
    Vehicle: ["Make", "VehicleCategory", "VehiclePrice", "AgeOfVehicle", "NumberOfCars"],
    Driver: ["Age", "Sex", "MaritalStatus", "DriverRating", "AgeOfPolicyHolder"],
  },
  health: {
    Claim: ["claim_type", "num_diagnosis_codes", "num_procedure_codes", "InscClaimAmtReimbursed", "DeductibleAmtPaid", "claim_duration_days", "admission_to_claim_days"],
    Beneficiary: ["is_deceased", "chronic_condition_count", "has_renal_disease"],
  },
};

type Values = Record<string, string | boolean>;

function initialValues(fields: FieldSpec[]): Values {
  const v: Values = {};
  for (const f of fields) {
    if (f.kind === "boolean") v[f.name] = Boolean(f.default);
    else v[f.name] = f.default == null ? "" : String(f.default);
  }
  return v;
}

function validate(f: FieldSpec, raw: string | boolean): string | null {
  if (f.kind === "boolean" || raw === "") return null;
  if (f.kind === "categorical") return f.options?.includes(String(raw)) ? null : "Choose one of the listed values.";
  const n = Number(raw);
  if (!Number.isFinite(n)) return "Enter a number.";
  if (f.kind === "integer" && !Number.isInteger(n)) return "Enter a whole number.";
  if (f.minimum != null && n < f.minimum) return `Must be ${f.minimum} or more.`;
  return null;
}

function toPayload(fields: FieldSpec[], values: Values): Record<string, Scalar> {
  const out: Record<string, Scalar> = {};
  for (const f of fields) {
    const raw = values[f.name];
    if (f.kind === "boolean") out[f.name] = Boolean(raw);
    else if (raw === "" || raw == null) continue;
    else if (f.kind === "categorical") out[f.name] = String(raw);
    else out[f.name] = Number(raw);
  }
  return out;
}

export default function AssessPage() {
  const domain = useDomain();
  const schema = useSchema(domain);
  const assess = useAssess(domain);
  const bands = useHealth().data?.risk_bands;
  const [values, setValues] = useState<Values>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitted, setSubmitted] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const summaryRef = useRef<HTMLDivElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const fields = useMemo(() => schema.data?.fields ?? [], [schema.data]);
  useEffect(() => {
    setValues(initialValues(fields));
    setTouched({});
    setSubmitted(false);
    assess.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reset only when the schema (domain) changes
  }, [fields]);

  const errors = useMemo(() => {
    const e: Record<string, string> = {};
    for (const f of fields) {
      const msg = validate(f, values[f.name] ?? "");
      if (msg) e[f.name] = msg;
    }
    return e;
  }, [fields, values]);

  const groups = useMemo(() => {
    const byName = new Map(fields.map((f) => [f.name, f]));
    const used = new Set<string>();
    const out: [string, FieldSpec[]][] = [];
    for (const [g, names] of Object.entries(GROUPS[domain])) {
      const fs = names.map((n) => byName.get(n)).filter((f): f is FieldSpec => !!f);
      fs.forEach((f) => used.add(f.name));
      if (fs.length) out.push([g, fs]);
    }
    const rest = fields.filter((f) => !used.has(f.name));
    if (rest.length) out.push(["Other model features", rest]);
    return out;
  }, [fields, domain]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (Object.keys(errors).length) {
      requestAnimationFrame(() => summaryRef.current?.focus());
      return;
    }
    const isPdf = !!file?.name.toLowerCase().endsWith(".pdf");
    assess.mutate(
      { attributes: toPayload(fields, values), file, pageIndex: isPdf ? pdfPage - 1 : 0 },
      {
        onSuccess: (r) => {
          toast.success("Assessment complete", { description: `Risk score ${fmtPct(r.assessment.fraud_probability)}` });
          requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
        },
        onError: (err) => toast.error("Assessment failed", { description: err.message }),
      },
    );
  };

  const showError = (name: string) => (submitted || touched[name]) && errors[name];
  const errorList = submitted ? Object.entries(errors) : [];

  return (
    <>
      <PageHeader
        eyebrow={`${DOMAIN_LABEL[domain]} insurance`}
        title="New claim assessment"
        description={`Score a claim that is not in the dataset. The form is generated from the ${schema.data ? fields.length : ""} features the saved ${DOMAIN_LABEL[domain].toLowerCase()} model was trained on.`}
      />
      <DecisionBanner compact className="mb-4" />

      {schema.error ? (
        <ErrorState error={schema.error} onRetry={() => schema.refetch()} />
      ) : !schema.data ? (
        <Card className="p-5">
          <SkeletonRows rows={8} />
        </Card>
      ) : (
        <form onSubmit={onSubmit} noValidate className="grid gap-4 xl:grid-cols-[1fr_22rem]">
          <div className="space-y-4">
            {errorList.length ? (
              <div
                ref={summaryRef}
                tabIndex={-1}
                role="alert"
                className="rounded-lg border border-risk-high-border bg-risk-high-soft p-4 text-sm text-risk-high"
              >
                <p className="flex items-center gap-2 font-semibold">
                  <AlertCircle className="size-4" aria-hidden="true" />
                  {errorList.length} {errorList.length === 1 ? "field needs" : "fields need"} attention
                </p>
                <ul className="mt-2 list-inside list-disc text-xs">
                  {errorList.map(([name, msg]) => (
                    <li key={name}>
                      <a href={`#field-${name}`} className="underline underline-offset-2">
                        {label(name)}
                      </a>
                      : {msg}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {groups.map(([group, fs]) => (
              <Card key={group}>
                <fieldset>
                  <legend className="sr-only">{group}</legend>
                  <CardHeader title={group} as="h2" />
                  <CardBody className="grid gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
                    {fs.map((f) => {
                      const id = `field-${f.name}`;
                      const err = showError(f.name);
                      const describedBy = err ? `${id}-error` : undefined;
                      const common = {
                        id,
                        "aria-invalid": err ? true : undefined,
                        "aria-describedby": describedBy,
                        onBlur: () => setTouched((t) => ({ ...t, [f.name]: true })),
                      };
                      return (
                        <div key={f.name} className="space-y-1.5">
                          {f.kind === "boolean" ? (
                            <label htmlFor={id} className="flex h-[var(--input-height)] cursor-pointer items-center gap-2.5 rounded-md border border-border-strong bg-surface px-3 text-sm text-text">
                              <input
                                type="checkbox"
                                checked={Boolean(values[f.name])}
                                onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.checked }))}
                                className="size-4 accent-[rgb(var(--primary))]"
                                {...common}
                              />
                              {label(f.name)}
                            </label>
                          ) : (
                            <>
                              <Label htmlFor={id}>{label(f.name)}</Label>
                              {f.kind === "categorical" ? (
                                <Select
                                  value={String(values[f.name] ?? "")}
                                  onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}
                                  {...common}
                                >
                                  <option value="">Not provided</option>
                                  {f.options?.map((o) => (
                                    <option key={o} value={o}>
                                      {o}
                                    </option>
                                  ))}
                                </Select>
                              ) : (
                                <Input
                                  type="number"
                                  inputMode={f.kind === "integer" ? "numeric" : "decimal"}
                                  step={f.kind === "integer" ? 1 : "any"}
                                  min={f.minimum ?? undefined}
                                  placeholder="Not provided"
                                  value={String(values[f.name] ?? "")}
                                  onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}
                                  className="tabular"
                                  {...common}
                                />
                              )}
                            </>
                          )}
                          {err ? (
                            <p id={`${id}-error`} className="text-xs font-medium text-risk-high">
                              {errors[f.name]}
                            </p>
                          ) : null}
                        </div>
                      );
                    })}
                  </CardBody>
                </fieldset>
              </Card>
            ))}
          </div>

          <div className="space-y-4 xl:sticky xl:top-[calc(var(--topbar-height)+1.5rem)] xl:self-start">
            <Card>
              <CardHeader title="Supporting document" description="Optional — adds document forensics to the assessment" as="h2" />
              <CardBody className="space-y-3">
                <Dropzone file={file} onFile={setFile} disabled={assess.isPending} compact />
                {file?.name.toLowerCase().endsWith(".pdf") ? (
                  <div className="flex items-center gap-2">
                    <Label htmlFor="assess-pdf-page">PDF page</Label>
                    <Input
                      id="assess-pdf-page"
                      type="number"
                      min={1}
                      value={pdfPage}
                      onChange={(e) => setPdfPage(Math.max(1, Number(e.target.value) || 1))}
                      className="w-20"
                    />
                  </div>
                ) : null}
              </CardBody>
            </Card>
            <Card className="p-4">
              <Button type="submit" size="lg" className="w-full" disabled={assess.isPending}>
                <ClipboardCheck aria-hidden="true" />
                {assess.isPending ? "Scoring…" : "Assess claim"}
              </Button>
              <Button
                type="button"
                variant="ghost"
                className="mt-2 w-full"
                onClick={() => {
                  setValues(initialValues(fields));
                  setTouched({});
                  setSubmitted(false);
                  setFile(null);
                  assess.reset();
                }}
              >
                <RotateCcw aria-hidden="true" /> Reset form
              </Button>
              <ul className="mt-3 space-y-1.5 text-2xs text-text-muted">
                {schema.data.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Card>
          </div>
        </form>
      )}

      <div ref={resultRef} className="scroll-mt-24">
        {assess.error ? <ErrorState error={assess.error} className="mt-6" /> : null}
        {assess.data ? (
          <section className="mt-8 space-y-4" aria-labelledby="result-heading">
            <h2 id="result-heading" className="text-lg font-semibold tracking-tight text-text">
              Assessment result
            </h2>
            <Card className="grid gap-6 p-5 md:grid-cols-[1fr_minmax(16rem,22rem)]">
              <div>
                <p className="text-2xs font-semibold uppercase tracking-[0.12em] text-accent">Session claim</p>
                <p className="mt-1 text-xl font-semibold text-text">{shortClaim(assess.data.claim_key)}</p>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <RiskBadge risk={assess.data.assessment.risk_category} size="md" />
                  <span className="text-sm text-text-muted">{assess.data.assessment.risk_category_description}</span>
                </div>
                <DecisionBanner className="mt-4" compact />
              </div>
              <div className="rounded-lg border border-border bg-surface-sunken p-4">
                <p className="text-xs font-medium text-text-muted">Risk score (model probability)</p>
                <p className="tabular mt-1 text-4xl font-semibold tracking-tight text-text">
                  {fmtPct(assess.data.assessment.fraud_probability)}
                </p>
                {bands ? (
                  <ProbabilityMeter value={assess.data.assessment.fraud_probability} bands={bands} className="mt-3" />
                ) : null}
              </div>
            </Card>
            <Card>
              <CardHeader title="Why this score" description="Top contributing factors" icon={<Sparkles />} />
              <CardBody>
                <ShapChart factors={assess.data.assessment.top_factors} baseValue={assess.data.assessment.base_value} />
              </CardBody>
            </Card>
            {assess.data.forensics ? <ForensicReportView report={assess.data.forensics} /> : null}
            <div className="grid gap-2">
              {assess.data.notes.map((n) => (
                <Callout key={n}>{n}</Callout>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </>
  );
}
