"""InsureSight AI — Insurance Intelligence & Fraud Investigation Platform.

Enterprise Streamlit interface for investigators.
Thin renderer over `app.model_loader`, `app.data_loader`, and `app.claim_view`.
Scoring and forensic methods are unchanged; this module organizes the decision-support workflow.
See `docs/app_ui.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
import html
import pandas as pd
import streamlit as st
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.claim_view import (
    ClaimViewError,
    assess_claim,
    assess_new_claim_document,
    build_investigation_table,
    claim_row,
    rescore_with_forensics,
    run_forensic_checks,
    score_loaded_sample,
)
from app.data_loader import (
    DatasetNotAvailableError,
    get_cached_health_dataset,
    get_cached_motor_dataset,
    health_claim_row,
    motor_claim_row,
)
from app.model_loader import (
    ModelNotAvailableError,
    get_health_model,
    get_motor_model,
    load_confusion_matrix,
    load_eval_metrics,
    load_pr_curve,
    load_test_metrics,
    load_threshold_selection,
    load_roc_curve,
)
from app.ui_components import (
    attribute_risk_chart,
    claims_to_review_table,
    feature_importance_chart,
    grouped_attribute_cards,
    probability_distribution_chart,
    render_assessment,
    render_confusion_matrix_card,
    render_document_summary,
    render_evaluation_curves,
    render_forensic_comparison,
    render_forensic_panels,
    render_graph_signals,
    render_kpi_row,
    render_preview,
    risk_distribution_chart,
)
from app.ui_theme import (
    RISK_COLORS,
    delta_badge,
    info_callout,
    inject_theme,
    page_header,
    risk_badge,
    sidebar_brand,
    sidebar_footer,
    sidebar_system_status,
    status_badge,
    triage_banner,
)
from src.features.structured import CLAIM_KEY_COL
from src.forensics import PDFLoadError, extract_pdf_text, get_pdf_info, render_pdf_page
from src.pipeline.inference import DEFAULT_RISK_THRESHOLDS, RISK_CATEGORY_DESCRIPTIONS

st.set_page_config(
    page_title="InsureSight AI — Insurance Intelligence",
    page_icon="IS",
    layout="wide",
    initial_sidebar_state="expanded",
)

MOTOR_DOMAIN = "motor"
HEALTH_DOMAIN = "health"
PAGES = (
    "Overview",
    "Claims Investigation",
    "Document Forensics",
    "Model Insights",
)

MOTOR_GROUPS = {
    "Policy": ("PolicyType", "BasePolicy", "Deductible", "Days_Policy_Accident", "Days_Policy_Claim"),
    "Vehicle": ("Make", "VehicleCategory", "VehiclePrice", "AgeOfVehicle", "NumberOfCars"),
    "Incident": ("AccidentArea", "Fault", "PoliceReportFiled", "WitnessPresent", "Month", "Year"),
    "Driver": ("Age", "Sex", "MaritalStatus", "DriverRating", "AgeOfPolicyHolder"),
}
HEALTH_GROUPS = {
    "Claim": ("ClaimID", "claim_type", "ClaimStartDt", "ClaimEndDt", "InscClaimAmtReimbursed"),
    "Provider": ("Provider", "AttendingPhysician", "OperatingPhysician", "OtherPhysician"),
    "Beneficiary": ("BeneID", "num_diagnosis_codes", "num_procedure_codes"),
}


def set_navigation_page(page: str) -> None:
    st.session_state["nav_page"] = page


def _sidebar_controls() -> tuple[str, str, dict]:
    sidebar_brand()
    page = st.sidebar.radio("Navigation", PAGES, key="nav_page")
    st.sidebar.markdown("---")
    domain = st.sidebar.radio(
        "Insurance domain",
        (MOTOR_DOMAIN, HEALTH_DOMAIN),
        format_func=lambda v: "Health Insurance" if v == HEALTH_DOMAIN else "Motor Insurance",
        key="domain_select",
    )
    params: dict = {}
    if domain == MOTOR_DOMAIN:
        params["sample_size"] = st.sidebar.slider(
            "Motor claims sample size",
            100,
            config.MAX_MOTOR_SAMPLE_SIZE,
            config.DEFAULT_MOTOR_SAMPLE_SIZE,
            step=100,
            help="Configures the number of motor claim records sampled in-memory.",
        )
    else:
        params["inpatient_sample_size"] = st.sidebar.slider(
            "Inpatient claims sample",
            20,
            config.MAX_HEALTH_INPATIENT_SAMPLE_SIZE,
            config.DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE,
            step=20,
        )
        params["outpatient_sample_size"] = st.sidebar.slider(
            "Outpatient claims sample",
            40,
            config.MAX_HEALTH_OUTPATIENT_SAMPLE_SIZE,
            config.DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE,
            step=40,
        )
    st.sidebar.caption(
        "Interactive dashboard uses a bounded, deterministic sample of real claims. "
        "See docs/app_ui.md."
    )
    return page, domain, params


def _load_model(domain: str):
    directory = config.motor_model_dir() if domain == MOTOR_DOMAIN else config.health_model_dir()
    getter = get_motor_model if domain == MOTOR_DOMAIN else get_health_model
    try:
        return getter(str(directory)), directory
    except ModelNotAvailableError as exc:
        sidebar_system_status(False, 0, directory.name, domain)
        sidebar_footer()
        st.error(str(exc))
        st.stop()
        raise


def _load_dataset(domain: str, params: dict):
    try:
        if domain == MOTOR_DOMAIN:
            return get_cached_motor_dataset(**params)
        return get_cached_health_dataset(**params)
    except DatasetNotAvailableError as exc:
        sidebar_system_status(False, 0, "unavailable", domain)
        sidebar_footer()
        st.error(
            f"{exc}\n\nThis app requires the datasets under `data/raw/` — see "
            "docs/data_contract.md for the expected file layout."
        )
        st.stop()
        raise


def _get_scores(domain: str, model, dataset, sampling_params: dict) -> pd.DataFrame:
    key = f"scores_{domain}_{sorted(sampling_params.items())}_{len(dataset.X)}"
    cached = st.session_state.get(key)
    if isinstance(cached, pd.DataFrame) and len(cached) == len(dataset.X):
        return cached
    with st.spinner("Scoring the loaded sample (predict_proba inference)..."):
        scores = score_loaded_sample(model, dataset.X, DEFAULT_RISK_THRESHOLDS)
    st.session_state[key] = scores
    return scores


def _claim_info(domain: str, dataset, claim_key: str) -> pd.Series:
    if domain == MOTOR_DOMAIN:
        return motor_claim_row(dataset, claim_key)
    return health_claim_row(dataset, claim_key)


def _is_pdf(uploaded: Any) -> bool:
    name = str(getattr(uploaded, "name", "")).lower()
    return name.endswith(".pdf") or getattr(uploaded, "type", "") == "application/pdf"


def _prepare_upload(uploaded: Any, page_key: str) -> tuple[bool, int, Optional[Image.Image], str]:
    is_pdf = _is_pdf(uploaded)
    page_index = 0
    preview_image = None
    extracted_text = ""
    if is_pdf:
        pdf_info = get_pdf_info(uploaded)
        if pdf_info.page_count > 1:
            page_selection = st.selectbox(
                "Select PDF page to analyze",
                options=list(range(1, pdf_info.page_count + 1)),
                format_func=lambda p, n=pdf_info.page_count: f"Page {p} of {n}",
                key=page_key,
            )
            page_index = page_selection - 1
        preview_image = render_pdf_page(uploaded, page_index=page_index, scale=1.5)
        extracted_text = extract_pdf_text(uploaded, page_index=page_index)
    else:
        preview_image = Image.open(uploaded)
    return is_pdf, page_index, preview_image, extracted_text


# ---------------------------------------------------------------------------
# Destination 1: Overview Dashboard
# ---------------------------------------------------------------------------

def _inspect_selected_claim(chosen) -> None:
    st.session_state["selected_claim_key"] = chosen
    st.session_state["nav_page"] = "Claims Investigation"


def _render_overview(domain: str, model, table: pd.DataFrame) -> None:
    page_header(
        "Overview",
        "Investigation overview",
        "KPI metrics and charts below represent model risk inference across the "
        "currently loaded sample, clearly separated from training-time evaluation benchmarks.",
    )
    if table.empty:
        st.warning("No claims loaded for this sample size. Adjust the sidebar sample controls.")
        return

    n = len(table)
    high = int((table["risk_category"] == "high").sum())
    medium = int((table["risk_category"] == "medium").sum())
    low = int((table["risk_category"] == "low").sum())
    review = high + medium

    render_kpi_row(
        [
            ("Loaded Sample", f"{n:,}", "Claims analyzed in memory", None),
            ("High Risk", f"{high:,}", f"{high / max(1, n):.1%} of sample · Priority 1", RISK_COLORS["high"]),
            ("Medium Risk", f"{medium:,}", f"{medium / max(1, n):.1%} of sample · Priority 2", RISK_COLORS["medium"]),
            ("Low Risk", f"{low:,}", f"{low / max(1, n):.1%} of sample · Straight-through", RISK_COLORS["low"]),
        ]
    )
    triage_banner(review, n, high, medium)

    st.caption(
        "Risk cutoffs use decision-support thresholds: Low < 30%, Medium 30%–69%, High ≥ 70%. "
        "They represent triage prioritization bands, not confirmed fraud rates."
    )

    st.markdown("##### Portfolio Risk Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk-Category Distribution**")
        risk_distribution_chart(table)
    with c2:
        st.markdown("**Claim Volume & Risk Breakdown**")
        attribute_risk_chart(table, domain)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Fraud Probability Score Distribution**")
        probability_distribution_chart(table)
        st.caption("Dotted lines indicate Medium (30%) and High (70%) decision thresholds.")
    with c4:
        st.markdown("**Top Model Risk Drivers (Global Feature Importance)**")
        feature_importance_chart(model, top_n=8)
        st.caption("Key predictive features learned by the XGBoost fraud detection model.")

    st.markdown("---")
    st.markdown("##### Priority Claims Requiring Investigation")
    st.caption("Sorted with high-risk priority first, followed by medium-risk, in stable order.")
    review_df = claims_to_review_table(table, limit=15)

    show_cols = [
        c
        for c in [
            CLAIM_KEY_COL,
            "domain",
            "risk_category",
            "fraud_probability",
            "Make",
            "AccidentArea",
            "claim_type",
            "Provider",
            "has_shared_entity",
        ]
        if c in review_df.columns
    ]

    display_df = review_df[show_cols].copy()
    if "fraud_probability" in display_df.columns:
        display_df["fraud_probability"] = display_df["fraud_probability"].map(lambda p: f"{float(p):.1%}")

    st.dataframe(display_df, hide_index=True, width="stretch")

    
    options = list(review_df[CLAIM_KEY_COL])

    if options:
        col_sel, col_btn = st.columns([3, 1])

        with col_sel:
            chosen = st.selectbox(
                "Select a priority claim to inspect in detail",
                ["—"] + options,
                key="overview_selected_claim",
            )

        with col_btn:
            st.write("")
            st.write("")

            st.button(
                "Inspect selected claim",
                type="primary",
                disabled=(chosen == "—"),
                on_click=_inspect_selected_claim,
                args=(chosen,),
            )



# ---------------------------------------------------------------------------
# Destination 2: Claims Investigation Workspace
# ---------------------------------------------------------------------------


def _filter_table(table: pd.DataFrame, domain: str) -> pd.DataFrame:
    c_search, c_risk, c_domain, c_prob = st.columns([2, 1.5, 1.5, 1.5])
    with c_search:
        search = st.text_input("Search claim identifier", placeholder="e.g. claim:12 or claim:CLM4…")
    with c_risk:
        risk_opts = st.multiselect(
            "Risk tier",
            ["high", "medium", "low"],
            default=["high", "medium", "low"],
            format_func=lambda r: f"{r.title()} Risk",
        )
    with c_prob:
        min_p = st.slider("Min probability", 0.0, 1.0, 0.0, step=0.05)

    filtered = table[table["risk_category"].isin(risk_opts)] if risk_opts else table.iloc[0:0]
    if min_p > 0.0:
        filtered = filtered[filtered["fraud_probability"] >= min_p]

    with c_domain:
        if domain == HEALTH_DOMAIN and "claim_type" in filtered.columns:
            types = sorted(filtered["claim_type"].dropna().astype(str).unique().tolist())
            chosen_types = st.multiselect("Claim type", types, default=types)
            if chosen_types:
                filtered = filtered[filtered["claim_type"].astype(str).isin(chosen_types)]
        elif domain == MOTOR_DOMAIN and "AccidentArea" in filtered.columns:
            areas = sorted(filtered["AccidentArea"].dropna().astype(str).unique().tolist())
            chosen_areas = st.multiselect("Accident area", areas, default=areas)
            if chosen_areas:
                filtered = filtered[filtered["AccidentArea"].astype(str).isin(chosen_areas)]
        else:
            st.caption("No sub-category filter available.")

    if search.strip():
        needle = search.strip().lower()
        filtered = filtered[filtered[CLAIM_KEY_COL].astype(str).str.lower().str.contains(needle, regex=False)]

    return filtered


def _render_claim_detail(domain: str, model, dataset, claim_key: str) -> None:
    c_back, c_space = st.columns([1, 4])
    with c_back:
        if st.button("← Back to claims list"):
            st.session_state["selected_claim_key"] = None
            st.rerun()

    st.markdown(f"#### Case Investigation: `{claim_key}`")
    try:
        matches = dataset.X[dataset.X[CLAIM_KEY_COL] == claim_key]
        if matches.empty:
            st.error(
                f"Selected claim '{claim_key}' was not found in the currently loaded dataset sample. "
                "It may have been excluded when dataset sampling parameters were modified."
            )
            return
        info = _claim_info(domain, dataset, claim_key)
        fused = matches.iloc[0]
        row = claim_row(dataset, claim_key)
        assessment = assess_claim(domain, model, row, DEFAULT_RISK_THRESHOLDS)
    except (ClaimViewError, DatasetNotAvailableError, KeyError, IndexError) as exc:
        st.error(f"Unable to load claim details: {exc}")
        return

    # Section A: Claim Summary & Section B: Risk Assessment
    render_assessment(assessment, heading="Claim Risk Assessment & Scoring")

    # Section D: Structured Claim Attributes
    st.markdown("##### Structured Claim Attributes")
    grouped_attribute_cards(info, MOTOR_GROUPS if domain == MOTOR_DOMAIN else HEALTH_GROUPS)

    # Section E: Graph Relationship Signals
    render_graph_signals(fused, dataset.graph, claim_key)

    # Session-Only Investigation Notes
    st.markdown("##### Investigation Notes & Triage Status")
    info_callout(
        "Notes and review statuses entered here are session-only and will not overwrite "
        "or mutate the underlying historical database.",
        title="Session Working State",
        variant="amber",
    )
    col_stat, col_notes = st.columns([1, 2])
    with col_stat:
        st.selectbox(
            "Review status (session-only)",
            ("Unreviewed", "Under Active Investigation", "Special Investigation Unit (SIU) Referral", "Cleared / No Fraud Indicated"),
            key=f"review_status_{claim_key}",
        )
    with col_notes:
        st.text_area(
            "Investigator notes (session-only)",
            key=f"review_note_{claim_key}",
            placeholder="Record rationale, forensic flags, or next investigation steps...",
            height=90,
        )

    # Attached Document Forensics & Reassessment
    st.markdown("---")
    st.markdown("##### Attached Document Forensics for this Claim")
    st.caption("Upload a supporting claim document (invoice, repair estimate, police report, medical bill).")
    _render_attached_forensics(domain, model, dataset, claim_key, assessment)


def _render_attached_forensics(domain: str, model, dataset, claim_key: str, original) -> None:
    uploaded = st.file_uploader(
        "Upload a claim document (PNG, JPEG, TIFF, BMP, or PDF)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
        key=f"forensics_upload_{claim_key}",
    )
    if uploaded is None:
        st.info("No document uploaded — baseline forensic features remain unavailable for this claim.")
        return
    try:
        is_pdf, page_index, preview, text = _prepare_upload(uploaded, f"pdf_page_{claim_key}")
        forensic = run_forensic_checks(uploaded, page_index=page_index)
    except PDFLoadError as exc:
        st.error(f"Could not render or parse PDF document: {exc}")
        return
    except OSError as exc:
        st.error(f"Could not read the uploaded document file: {exc}")
        return
    except Exception as exc:
        st.error(f"An unexpected error occurred while analyzing the document: {exc}")
        return

    if not forensic.any_available:
        st.warning("Could not execute forensic checks on this file; features remain unavailable.")
        return

    render_document_summary(uploaded, forensic, is_pdf)
    render_preview(preview, text, getattr(uploaded, "name", "document"))
    render_forensic_panels(forensic, is_pdf)

    try:
        rescored = rescore_with_forensics(domain, model, dataset, claim_key, forensic)
    except ClaimViewError as exc:
        st.error(f"Could not re-score claim: {exc}")
        return

    render_forensic_comparison(original, rescored)


def _render_claims(domain: str, model, dataset, table: pd.DataFrame) -> None:
    selected = st.session_state.get("selected_claim_key")
    if selected:
        if selected not in dataset.X[CLAIM_KEY_COL].values:
            st.session_state["selected_claim_key"] = None
            st.warning(
                f"Selected claim '{selected}' is no longer present in the active dataset sample "
                "(the sample size or active domain was changed). Displaying full claims list."
            )
        else:
            _render_claim_detail(domain, model, dataset, selected)
            return

    page_header(
        "Claims investigation",
        "Claims workspace",
        "Filter and browse the loaded claims sample, then select an individual claim for deep-dive investigation. "
        "SHAP explanations are computed on-demand for the inspected claim.",
    )
    if table.empty:
        st.warning("No claims loaded for this sample size. Adjust the sidebar sample controls.")
        return

    filtered = _filter_table(table, domain)

    c_sort, c_order, c_size = st.columns([2, 1, 1])
    with c_sort:
        sort_col = st.selectbox(
            "Sort claims by",
            [c for c in ["fraud_probability", CLAIM_KEY_COL, "risk_category"] if c in filtered.columns],
            format_func=lambda c: "Fraud Probability" if c == "fraud_probability" else ("Claim ID" if c == CLAIM_KEY_COL else "Risk Category"),
        )
    with c_order:
        descending = st.checkbox("Descending", value=True)
    with c_size:
        page_size = st.selectbox("Rows per page", [10, 25, 50], index=1)

    filtered = filtered.sort_values(sort_col, ascending=not descending)
    n = len(filtered)
    pages = max(1, (n + page_size - 1) // page_size)

    c_page_num, c_page_info = st.columns([1, 3])
    with c_page_num:
        page_i = st.number_input("Page", min_value=1, max_value=pages, value=1)
    with c_page_info:
        st.write("")
        st.caption(f"Showing page {page_i} of {pages} ({n:,} matching claims)")

    start = (page_i - 1) * page_size
    page_df = filtered.iloc[start : start + page_size].copy()

    show_cols = [
        c
        for c in [
            CLAIM_KEY_COL,
            "domain",
            "risk_category",
            "fraud_probability",
            "Make",
            "AccidentArea",
            "Age",
            "PolicyType",
            "claim_type",
            "Provider",
            "BeneID",
            "InscClaimAmtReimbursed",
            "has_shared_entity",
        ]
        if c in page_df.columns
    ]

    display_page_df = page_df[show_cols].copy()
    if "fraud_probability" in display_page_df.columns:
        display_page_df["fraud_probability"] = display_page_df["fraud_probability"].map(lambda p: f"{float(p):.1%}")

    st.dataframe(display_page_df, hide_index=True, width="stretch")

    keys = list(page_df[CLAIM_KEY_COL])
    if not keys:
        st.info("No claims match the active search and filter criteria. Clear filters to proceed.")
        return

    col_pick, col_act = st.columns([3, 1])
    with col_pick:
        pick = st.selectbox("Selected claim", keys)
    with col_act:
        st.write("")
        st.write("")
        if st.button("Inspect claim", type="primary"):
            st.session_state["selected_claim_key"] = pick
            st.rerun()


# ---------------------------------------------------------------------------
# Destination 3: Document Forensics Workspace
# ---------------------------------------------------------------------------


def _new_claim_overrides(domain: str) -> dict[str, Any]:
    st.markdown("**Claim Baseline Attributes**")
    st.caption(
        "Uploaded documents do not contain a structured claim schema. Specify the "
        "structured parameters required by the feature fusion pipeline."
    )
    values: dict[str, Any] = {}
    if domain == MOTOR_DOMAIN:
        c1, c2, c3 = st.columns(3)
        values["Make"] = c1.text_input("Vehicle Make", "Honda")
        values["AccidentArea"] = c2.selectbox("Accident Area", ["Urban", "Rural"])
        values["Age"] = c3.number_input("Driver Age", min_value=16, max_value=100, value=35)
        c1, c2, c3 = st.columns(3)
        values["PolicyType"] = c1.text_input("Policy Type", "Sedan - All Perils")
        values["VehicleCategory"] = c2.text_input("Vehicle Category", "Sedan")
        values["PoliceReportFiled"] = c3.selectbox("Police Report Filed", ["No", "Yes"])
    else:
        c1, c2, c3 = st.columns(3)
        values["claim_type"] = c1.selectbox("Claim Type", ["inpatient", "outpatient"])
        values["num_diagnosis_codes"] = c2.number_input("Diagnosis Codes", min_value=0, value=2)
        values["num_procedure_codes"] = c3.number_input("Procedure Codes", min_value=0, value=1)
        c1, c2 = st.columns(2)
        values["chronic_condition_count"] = c1.number_input("Chronic Condition Count", min_value=0, value=1)
        values["has_renal_disease"] = c2.checkbox("Renal Disease Indicator")
    return values


def _render_forensics_page(domain: str, model, dataset) -> None:
    page_header(
        "Document forensics",
        "Document workspace",
        "Analyze claim documents using multi-spectral Error Level Analysis (ELA), "
        "ORB Copy-Move forgery detection, and EXIF/PDF catalog metadata parsing.",
    )
    tab_existing, tab_new = st.tabs(["Attach to Loaded Claim", "New-Claim Document Assessment"])

    with tab_existing:
        keys = list(dataset.X[CLAIM_KEY_COL])
        if not keys:
            st.warning("No claims loaded in memory for this sample.")
        else:
            claim_key = st.selectbox("Select claim to attach document to", keys, key="forensic_claim_pick")
            row = claim_row(dataset, claim_key)
            original = assess_claim(domain, model, row, DEFAULT_RISK_THRESHOLDS)
            _render_attached_forensics(domain, model, dataset, claim_key, original)

    with tab_new:
        st.markdown("##### Standalone Document Assessment (No Pre-selected Claim)")
        uploaded = st.file_uploader(
            "Upload a claim document for standalone forensic triage (PDF, PNG, JPG, TIFF, BMP)",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="new_claim_uploader",
        )
        if uploaded is None:
            st.session_state.pop("new_claim_result", None)
            st.session_state.pop("new_claim_doc_sig", None)
            st.info("Upload an invoice, hospital bill, repair receipt, or claim form to start a standalone triage.")
            return

        try:
            is_pdf, page_index, preview, text = _prepare_upload(uploaded, "new_claim_pdf_page")
        except PDFLoadError as exc:
            st.session_state.pop("new_claim_result", None)
            st.session_state.pop("new_claim_doc_sig", None)
            st.error(f"Could not process PDF document: {exc}")
            return
        except OSError as exc:
            st.session_state.pop("new_claim_result", None)
            st.session_state.pop("new_claim_doc_sig", None)
            st.error(f"Could not read the uploaded document: {exc}")
            return
        except Exception as exc:
            st.session_state.pop("new_claim_result", None)
            st.session_state.pop("new_claim_doc_sig", None)
            st.error(f"An unexpected error occurred while loading the document: {exc}")
            return

        # Invalidate previous assessment if file or selected page changed
        doc_sig = (
            getattr(uploaded, "name", "document"),
            getattr(uploaded, "size", None),
            getattr(uploaded, "file_id", None),
            page_index,
            domain,
        )
        if st.session_state.get("new_claim_doc_sig") != doc_sig:
            st.session_state.pop("new_claim_result", None)

        render_preview(preview, text, getattr(uploaded, "name", "document"))

        with st.form("new_claim_form"):
            overrides = _new_claim_overrides(domain)
            submitted = st.form_submit_button("Run Document Assessment", type="primary")

        if not submitted and "new_claim_result" not in st.session_state:
            st.caption("Submit the form above to execute forensics and multimodal fraud risk scoring.")
            return

        if submitted:
            try:
                result = assess_new_claim_document(
                    domain=domain,
                    model=model,
                    document=uploaded,
                    page_index=page_index,
                    overrides=overrides,
                )
                st.session_state["new_claim_result"] = result
                st.session_state["new_claim_doc_sig"] = doc_sig
            except Exception as exc:
                st.error(f"Assessment failed: {exc}")
                return

        result = st.session_state.get("new_claim_result")
        if not result:
            return

        new_claim_id, forensic, verdict, assessment = result
        st.success(f"Generated Session Claim Identifier: `{new_claim_id}`")

        # Document Tampering Verdict Summary Card
        v_status = verdict.status
        v_variant = "danger" if v_status == "tampered" else ("warning" if v_status == "suspicious" else "success")
        st.markdown(
            f'<div class="is-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">'
            f'<div style="font-weight:700;font-size:1.05rem;">Document Tampering Verdict</div>'
            f'<div>{status_badge(v_status.title(), v_variant)}</div>'
            f'</div>'
            f'<div style="font-size:0.88rem;color:#334155;margin-bottom:0.4rem;">{html.escape(verdict.summary)}</div>'
            f'<div style="font-size:0.8rem;color:#64748B;">Tampering Anomaly Score: <b>{verdict.tampering_score:.0%}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if verdict.tampering_indicators:
            st.markdown("**Tampering Flags Identified**")
            for flag in verdict.tampering_indicators:
                st.markdown(f"- {flag}")

        render_document_summary(uploaded, forensic, is_pdf)
        render_forensic_panels(forensic, is_pdf)
        render_assessment(assessment, heading="Combined Multimodal Risk Assessment (Document + Attributes)")


# ---------------------------------------------------------------------------
# Destination 4: Model Insights & Transparency
# ---------------------------------------------------------------------------


def _render_insights(domain: str, active_model_dir: Path) -> None:
    page_header(
        "Model insights",
        "Transparency and evaluation",
        "Validation results guide threshold selection. Final test results come from a separate "
        "held-out split and are scored after model choices are frozen.",
    )

    tab_motor, tab_health = st.tabs(["Motor Fraud Model", "Health Fraud Model"])

    for tab, tab_domain, m_dir in (
        (tab_motor, MOTOR_DOMAIN, config.motor_model_dir()),
        (tab_health, HEALTH_DOMAIN, config.health_model_dir()),
    ):
        with tab:
            st.markdown(f"#### {tab_domain.title()} Insurance Risk Model")
            st.caption(f"Model Artifact Location: `{m_dir}`")

            metrics = load_eval_metrics(m_dir)
            test_results = load_test_metrics(m_dir)
            threshold_info = load_threshold_selection(m_dir)
            confusion = load_confusion_matrix(m_dir)
            roc_data = load_roc_curve(m_dir)
            pr_data = load_pr_curve(m_dir)

            if metrics is None:
                st.info(
                    "No evaluation metrics sidecar found for this artifact. "
                    "Run `scripts/train_models.py` to generate evaluation sidecars."
                )
            else:
                st.markdown("##### Validation results (threshold selection split)")
                if threshold_info:
                    st.caption(
                        f"Selected threshold: {threshold_info['threshold']:.3f} · "
                        f"Objective: {threshold_info['objective']} · "
                        f"Constraints: {threshold_info.get('constraints') or 'none'}"
                    )
                m_cols = st.columns(6)
                m_cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="Ranking performance across all probability thresholds")
                m_cols[1].metric("Average precision", f"{metrics.get('average_precision', float('nan')):.3f}", help="Area under the precision-recall curve summary")
                m_cols[2].metric("Precision", f"{metrics['precision']:.3f}", help="Proportion of flagged claims that are positive")
                m_cols[3].metric("Recall", f"{metrics['recall']:.3f}", help="Proportion of true positive claims detected")
                m_cols[4].metric("Specificity", f"{metrics.get('specificity', float('nan')):.3f}")
                m_cols[5].metric("Balanced accuracy", f"{metrics.get('balanced_accuracy', float('nan')):.3f}")

                n_samp = metrics.get("n_samples", "—")
                n_pos = metrics.get("n_positive", "—")
                prev = (
                    f"{metrics.get('n_positive', 0) / max(1, metrics.get('n_samples', 1)):.1%}"
                    if isinstance(n_samp, int) and n_samp > 0
                    else "—"
                )
                st.markdown(
                    f'<div class="is-kv-item" style="display:flex;justify-content:space-between;align-items:center;margin:0.5rem 0 1rem 0;">'
                    f'<div><b>F1-Score (Harmonic Mean):</b> <code>{metrics["f1"]:.3f}</code></div>'
                    f'<div style="font-size:0.8rem;color:#64748B;">Evaluated on <b>{n_samp:,}</b> validation claims ({n_pos:,} fraud cases · {prev} prevalence) at threshold <b>{metrics.get("threshold", 0.5):.2f}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Validation review volume: {metrics.get('review_volume', '—')} of "
                    f"{metrics.get('n_samples', '—')} claims ({metrics.get('review_rate', 0):.1%})."
                )

                if test_results is not None:
                    final_metrics = test_results.get("selected_threshold_metrics", {})
                    st.markdown("##### Final test results (untouched during fitting and selection)")
                    test_cols = st.columns(4)
                    test_cols[0].metric("ROC-AUC", f"{final_metrics.get('roc_auc', float('nan')):.3f}")
                    test_cols[1].metric("Average precision", f"{final_metrics.get('average_precision', float('nan')):.3f}")
                    test_cols[2].metric("Precision", f"{final_metrics.get('precision', float('nan')):.3f}")
                    test_cols[3].metric("Recall", f"{final_metrics.get('recall', float('nan')):.3f}")
                    st.caption(
                        f"Frozen validation threshold: {test_results.get('threshold_frozen_from_validation', 0.5):.3f} · "
                        f"Test review volume: {final_metrics.get('review_volume', '—')} of {final_metrics.get('n_samples', '—')} claims"
                    )
                    if test_results.get("confusion_matrix"):
                        st.markdown("**Final test confusion matrix**")
                        render_confusion_matrix_card(test_results["confusion_matrix"])

            if confusion is not None:
                st.markdown("##### Validation confusion matrix")
                render_confusion_matrix_card(confusion)

            render_evaluation_curves(roc_data=roc_data, pr_data=pr_data, roc_auc=metrics.get("roc_auc") if metrics else None)

            st.markdown("##### Methodological Transparency & Limitations")
            if tab_domain == HEALTH_DOMAIN:
                info_callout(
                    "The health target is `claim_fraud_label_provider_derived`: each claim inherits its provider's "
                    "PotentialFraud label. High validation performance reflects provider-level class separation, "
                    "not ground-truth claim fraud adjudication.",
                    title="Health Domain Label Derivation",
                    variant="amber",
                )
            else:
                info_callout(
                    "Motor model training uses structured claim attributes. Synthetic garage and bank account "
                    "relationships are excluded because they are not observed claims data.",
                    title="Motor Domain Feature Scope",
                    variant="amber",
                )

            st.caption(
                "When fraud classes are rare (e.g. ~5.99% in motor claims), standard accuracy and 0.50-threshold precision "
                "can appear low. ROC-AUC is the primary ranking metric for triage prioritization."
            )


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    inject_theme()
    page, domain, sampling_params = _sidebar_controls()

    # Reset selection when switching domain
    if st.session_state.get("_last_domain") != domain:
        st.session_state["selected_claim_key"] = None
        st.session_state["_last_domain"] = domain
        st.session_state.pop("new_claim_result", None)

    model, model_dir = _load_model(domain)
    dataset = _load_dataset(domain, sampling_params)
    scores = _get_scores(domain, model, dataset, sampling_params)
    table = build_investigation_table(domain, dataset, scores)

    # Validate that selected_claim_key remains valid after sampling or domain changes
    selected = st.session_state.get("selected_claim_key")
    if selected is not None and selected not in dataset.X[CLAIM_KEY_COL].values:
        st.session_state["selected_claim_key"] = None

    # Render system status in sidebar
    sidebar_system_status(True, len(table), model_dir.name, domain)
    sidebar_footer()

    # Route navigation
    if page == "Overview":
        _render_overview(domain, model, table)
    elif page == "Claims Investigation":
        _render_claims(domain, model, dataset, table)
    elif page == "Document Forensics":
        _render_forensics_page(domain, model, dataset)
    else:
        _render_insights(domain, model_dir)


if __name__ == "__main__":
    main()
