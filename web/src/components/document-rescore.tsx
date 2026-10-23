import { Upload } from "lucide-react";
import { useId, useState } from "react";
import { toast } from "sonner";
import { useDocumentRescore } from "@/lib/api";
import { fmtPct } from "@/lib/format";
import type { Domain } from "@/lib/types";
import { ErrorState } from "./common";
import { BeforeAfter, Dropzone, ForensicReportView } from "./forensics";
import { ShapChart } from "./shap-chart";
import { Button } from "./ui/button";
import { Card, CardBody, CardHeader } from "./ui/card";
import { Input, Label } from "./ui/input";
import { Skeleton } from "./ui/skeleton";

/**
 * Upload a document for a loaded claim, run forensics and show the
 * before/after re-score. Parents pass `key={claimKey}` to reset on claim change.
 */
export function DocumentRescore({ domain, claimKey }: { domain: Domain; claimKey: string }) {
  const pageId = useId();
  const [file, setFile] = useState<File | null>(null);
  const [page, setPage] = useState(1);
  const rescore = useDocumentRescore(domain);
  const isPdf = !!file?.name.toLowerCase().endsWith(".pdf");

  const run = () => {
    if (!file) return;
    rescore.mutate(
      { claimKey, file, pageIndex: isPdf ? page - 1 : 0 },
      {
        onSuccess: (r) =>
          toast.success("Document analysed", {
            description: `Risk score ${fmtPct(r.before.fraud_probability)} → ${fmtPct(r.after.fraud_probability)}`,
          }),
        onError: (e) => toast.error("Analysis failed", { description: e.message }),
      },
    );
  };

  return (
    <div>
      <Card className="p-4">
        <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-end">
          <Dropzone
            file={file}
            onFile={(f) => {
              setFile(f);
              setPage(1);
              rescore.reset();
            }}
            disabled={rescore.isPending}
            compact
          />
          <div className="flex items-end gap-2">
            {isPdf ? (
              <div className="space-y-1.5">
                <Label htmlFor={pageId}>PDF page</Label>
                <Input
                  id={pageId}
                  type="number"
                  min={1}
                  value={page}
                  onChange={(e) => setPage(Math.max(1, Number(e.target.value) || 1))}
                  className="w-20"
                />
              </div>
            ) : null}
            <Button onClick={run} disabled={!file || rescore.isPending} size="lg">
              <Upload aria-hidden="true" />
              {rescore.isPending ? "Analysing…" : "Analyse & re-score"}
            </Button>
          </div>
        </div>
        {rescore.isPending ? <Skeleton className="mt-4 h-40" /> : null}
        {rescore.error ? <ErrorState error={rescore.error} className="mt-4" /> : null}
      </Card>

      {rescore.data ? (
        <div className="mt-4 space-y-4">
          <Card>
            <CardHeader title="Re-scored with document evidence" description="Risk score before and after the uploaded document" />
            <CardBody>
              <BeforeAfter before={rescore.data.before} after={rescore.data.after} note={rescore.data.fusion_note} />
            </CardBody>
          </Card>
          <ForensicReportView report={rescore.data.forensics} />
          <Card>
            <CardHeader title="Factors after re-scoring" description="Model SHAP factors plus any document rule weights (dashed)" />
            <CardBody>
              <ShapChart factors={rescore.data.after.top_factors} />
            </CardBody>
          </Card>
        </div>
      ) : null}
    </div>
  );
}
