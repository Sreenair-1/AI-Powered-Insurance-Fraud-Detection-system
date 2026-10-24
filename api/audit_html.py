"""Printable HTML view of a claim audit report (`services.audit_report`).

A standalone page with print CSS, so a reviewer can use the browser's
"Save as PDF" without a PDF library on the server. Every value is escaped.
"""
from __future__ import annotations

from html import escape
from typing import Any, Iterable

STYLE = """
body{font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#0f172a;background:#fff;
     max-width:860px;margin:24px auto;padding:0 16px}
h1{font-size:20px;margin:0} h2{font-size:14px;margin:22px 0 6px;border-bottom:1px solid #cbd5e1;padding-bottom:3px}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:3px 8px 3px 0;vertical-align:top;border-bottom:1px solid #e2e8f0}
th{font-weight:600} .kv th{width:34%} .num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#475569;font-size:12px} .banner{border:1px solid #94a3b8;background:#f1f5f9;padding:8px 10px;margin:12px 0}
code{font-size:11px;word-break:break-all} .print{float:right;padding:6px 12px;font:inherit;cursor:pointer}
@media print{.print{display:none} body{margin:0}}
"""


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _kv(pairs: Iterable[tuple[str, Any]]) -> str:
    body = "".join(f"<tr><th>{escape(k)}</th><td>{escape(str(v))}</td></tr>" for k, v in pairs)
    return f"<table class=kv>{body}</table>"


def _table(headers: list[str], rows: Iterable[list[str]], numeric: frozenset[int] = frozenset()) -> str:
    def cell(tag: str, i: int, value: str) -> str:
        return f"<{tag}{' class=num' if i in numeric else ''}>{escape(value)}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _list(items: Iterable[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def _model_section(report: dict[str, Any]) -> str:
    a, m = report["assessment"], report["model"]
    bands = m["risk_bands"]
    return "<h2>Model assessment</h2>" + _kv([
        ("Risk score (model probability)", _pct(a["fraud_probability"])),
        ("Risk band", a["risk_category"]),
        ("Band cut-offs", f"medium ≥ {_pct(bands['medium'])}, high ≥ {_pct(bands['high'])}"),
        ("Model version", f"{m['version']} (model.json modified {m['artifact_modified_at'] or 'unknown'})"),
    ]) + "<h2>Top factors</h2>" + _table(
        ["Feature", "Value", "Contribution (log-odds)", "Source"],
        ([str(f["feature"]), str(f["feature_value"]), f"{f['shap_value']:+.3f}",
          "rule weight" if f.get("source") == "document_heuristic" else "SHAP"] for f in a["top_factors"]),
        numeric=frozenset({2}),
    )


def _network_section(report: dict[str, Any]) -> str:
    risk = report["network"]["risk"]
    level = escape(risk["level"].replace("_", " "))
    ring = f" ({escape(risk['ring_id'])}, {risk['ring_size']} claims)" if risk["ring_id"] else ""
    evidence = [
        " + ".join(f"{x['type']} {x['value']}" for x in ev["entities"])
        + f": {ev['claims']} claims, chance predicts {ev['expected']:.2f} (p = {ev['p_value']:.1e})"
        for ev in risk["evidence"]
    ]
    links = _table(
        ["Linked entity", "Identifier", "Claims in sample"],
        ([link["entity_type"], link["entity_id"] + (" (synthetic)" if link["synthetic"] else ""),
          str(link["claims_in_sample"])] for link in report["network"]["graph_links"]),
        numeric=frozenset({2}),
    )
    return (
        f"<h2>Network</h2><p>Network risk: <strong>{level}</strong>{ring}.</p>"
        + (_list(evidence) if evidence else "")
        + links
        + f"<p class=muted>{escape(risk['method'])} {escape(risk['note'])}</p>"
    )


def _document_section(report: dict[str, Any]) -> str:
    doc = report["document"]
    html = "<h2>Document forensics</h2>"
    if not doc:
        html += "<p>No document was attached to this report.</p>"
    else:
        verdict = doc["verdict"]
        html += _kv([
            ("File", doc["filename"]),
            ("SHA-256", doc["sha256"]),
            ("Verdict", verdict["status"]),
            ("Tampering score", _pct(verdict["tampering_score"])),
            ("Score before / after document",
             f"{_pct(doc['score_before_document'])} / {_pct(doc['score_after_document'])}"),
        ])
        html += "<p>Indicators:</p>" + _list(verdict["indicators"] or ["None"])
        html += f"<p class=muted>{escape(doc['fusion_note'])}</p>"
    bench = (report["benchmarks"]["document"] or {}).get("as_submitted")
    if bench:
        html += (
            f"<p class=muted>Tampering benchmark: the verdict flagged {_pct(bench['detection_rate'])} of edited and "
            f"{_pct(bench['false_positive_rate'])} of untouched bills.</p>"
        )
    return html


def _review_section(report: dict[str, Any]) -> str:
    current = report["review"]["current"]
    n = len(report["review"]["history"])
    if not current:
        body = "<p>Not reviewed yet.</p>"
    else:
        body = _kv([
            ("Decision", current["outcome"].replace("_", " ")),
            ("Reviewer", current.get("reviewer") or "—"),
            ("Recorded", current["recorded_at"]),
            ("Score at review", _pct(current.get("fraud_probability_at_review"))),
            ("Note", current.get("note") or "—"),
        ])
    return f"<h2>Investigator review</h2>{body}<p class=muted>{n} decision(s) on record for this claim.</p>"


def render_audit_html(report: dict[str, Any]) -> str:
    title = f"{report['domain'].title()} claim {report['claim_key']}"
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>Audit report {escape(report['claim_key'])}</title><style>{STYLE}</style></head><body>"
        "<button class=print onclick='window.print()'>Print / save as PDF</button>"
        "<h1>Claim risk audit report</h1>"
        f"<p class=muted>{escape(title)} · generated {escape(report['generated_at'])} · "
        f"report {escape(report['report_id'])}</p>"
        f"<div class=banner>{escape(report['disclaimer'])}</div>"
        + _model_section(report)
        + _network_section(report)
        + _document_section(report)
        + _review_section(report)
        + "<h2>Limitations</h2>" + _list(report["limitations"])
        + "<h2>Integrity</h2><p class=muted>SHA-256 of the JSON report: "
        f"<code>{escape(report['integrity']['digest'])}</code></p>"
        "</body></html>"
    )
