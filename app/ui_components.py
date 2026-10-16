"""Enterprise UI components and visualizations for InsureSight AI.

These functions render already-computed results. They do not score claims,
run forensics, or change risk-threshold semantics. See `docs/app_ui.md`.
"""
from __future__ import annotations

import html
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from app.claim_view import ForensicUploadResult
from app.ui_theme import (
    NAVY_PRIMARY,
    RISK_COLORS,
    SLATE_MUTED,
    SLATE_TEXT,
    TEAL_ACCENT,
    info_callout,
    kpi_card,
    risk_badge,
    status_badge,
)
from src.features.structured import CLAIM_KEY_COL
from src.pipeline.inference import DEFAULT_RISK_THRESHOLDS, RISK_CATEGORY_DESCRIPTIONS

try:
    import altair as alt
except ImportError:  # pragma: no cover
    alt = None


# ---------------------------------------------------------------------------
# KPI Rows
# ---------------------------------------------------------------------------


def render_kpi_row(items: list[tuple[str, str, str, Optional[str]]]) -> None:
    """Render a responsive horizontal row of enterprise KPI cards."""
    cols = st.columns(len(items))
    for col, (label, value, hint, accent) in zip(cols, items):
        with col:
            kpi_card(label, value, hint, accent)


# ---------------------------------------------------------------------------
# Analytics Visualizations (Overview Dashboard)
# ---------------------------------------------------------------------------


def risk_distribution_chart(table: pd.DataFrame) -> None:
    """Render an interactive risk category distribution bar chart with brand risk colors."""
    if table.empty or "risk_category" not in table.columns:
        st.info("No risk distribution data available.")
        return

    counts = (
        table["risk_category"]
        .value_counts()
        .reindex(["low", "medium", "high"])
        .fillna(0)
        .astype(int)
    )
    total = len(table)
    pcts = [(c / total * 100.0) if total > 0 else 0.0 for c in counts]

    df = pd.DataFrame({
        "category": ["Low Risk", "Medium Risk", "High Risk"],
        "key": ["low", "medium", "high"],
        "count": counts.values,
        "percentage": pcts,
        "label": [f"{c:,} ({p:.1f}%)" for c, p in zip(counts.values, pcts)],
    })

    if alt is not None:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadius=5)
            .encode(
                x=alt.X(
                    "category:N",
                    sort=["Low Risk", "Medium Risk", "High Risk"],
                    title=None,
                    axis=alt.Axis(labelAngle=0, labelFontWeight="bold"),
                ),
                y=alt.Y("count:Q", title="Number of Claims"),
                color=alt.Color(
                    "key:N",
                    scale=alt.Scale(
                        domain=["low", "medium", "high"],
                        range=[RISK_COLORS["low"], RISK_COLORS["medium"], RISK_COLORS["high"]],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("count:Q", title="Claims", format=","),
                    alt.Tooltip("percentage:Q", title="Share (%)", format=".1f"),
                ],
            )
            .properties(height=230)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.bar_chart(counts)


def attribute_risk_chart(table: pd.DataFrame, domain: str) -> None:
    """Render claim volume and risk breakdown across domain-relevant attributes."""
    if table.empty or "risk_category" not in table.columns:
        st.info("No claim volume breakdown available.")
        return

    # Choose the most meaningful categorical dimension available
    col_name = None
    title = ""
    if domain == "motor":
        if "AccidentArea" in table.columns:
            col_name = "AccidentArea"
            title = "Accident Area"
        elif "VehicleCategory" in table.columns:
            col_name = "VehicleCategory"
            title = "Vehicle Category"
        elif "Make" in table.columns:
            col_name = "Make"
            title = "Vehicle Make"
    else:
        if "claim_type" in table.columns:
            col_name = "claim_type"
            title = "Claim Type"
        elif "Provider" in table.columns:
            col_name = "Provider"
            title = "Top Providers"

    if not col_name or col_name not in table.columns:
        st.info("No secondary categorical attribute available on this sample.")
        return

    # Take top categories if high cardinality
    top_cats = table[col_name].astype(str).value_counts().head(8).index.tolist()
    filtered = table[table[col_name].astype(str).isin(top_cats)].copy()
    filtered["attr_val"] = filtered[col_name].astype(str)

    if alt is not None:
        chart = (
            alt.Chart(filtered)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X(
                    "attr_val:N",
                    sort="-y",
                    title=title,
                    axis=alt.Axis(labelAngle=-20, labelFontWeight="bold"),
                ),
                y=alt.Y("count():Q", title="Claims Count"),
                color=alt.Color(
                    "risk_category:N",
                    scale=alt.Scale(
                        domain=["low", "medium", "high"],
                        range=[RISK_COLORS["low"], RISK_COLORS["medium"], RISK_COLORS["high"]],
                    ),
                    legend=alt.Legend(
                        title="Risk Tier",
                        orient="top",
                        symbolType="circle",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("attr_val:N", title=title),
                    alt.Tooltip("risk_category:N", title="Risk Band"),
                    alt.Tooltip("count():Q", title="Claims"),
                ],
            )
            .properties(height=230)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.bar_chart(filtered["attr_val"].value_counts())


def probability_distribution_chart(table: pd.DataFrame) -> None:
    """Render the distribution of predicted fraud probabilities across the sample."""
    if table.empty or "fraud_probability" not in table.columns:
        return

    if alt is not None:
        df = table[["fraud_probability"]].copy()
        hist = (
            alt.Chart(df)
            .mark_bar(cornerRadius=3, color=TEAL_ACCENT, opacity=0.85)
            .encode(
                x=alt.X(
                    "fraud_probability:Q",
                    bin=alt.Bin(maxbins=18),
                    title="Estimated Fraud Probability Score",
                    scale=alt.Scale(domain=[0.0, 1.0]),
                ),
                y=alt.Y("count():Q", title="Claims Count"),
                tooltip=[
                    alt.Tooltip("count():Q", title="Claims in Bin"),
                ],
            )
            .properties(height=230)
        )

        rule_data = pd.DataFrame([
            {"cutoff": DEFAULT_RISK_THRESHOLDS.medium, "tier": "Medium (0.30)", "color": RISK_COLORS["medium"]},
            {"cutoff": DEFAULT_RISK_THRESHOLDS.high, "tier": "High (0.70)", "color": RISK_COLORS["high"]},
        ])
        rules = (
            alt.Chart(rule_data)
            .mark_rule(strokeDash=[4, 4], strokeWidth=2)
            .encode(
                x="cutoff:Q",
                color=alt.Color("color:N", scale=None),
            )
        )
        st.altair_chart(hist + rules, width="stretch")
    else:
        st.line_chart(table["fraud_probability"].sort_values())


def extract_model_feature_importances(model: Any, top_n: int = 8) -> list[dict[str, Any]]:
    """Safely extract global feature importances from HealthFraudModel or MotorFraudModel."""
    try:
        names: Optional[list[str]] = None
        importances: Optional[np.ndarray] = None

        if hasattr(model, "feature_names_") and hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
            names = list(model.feature_names_)
            importances = model.model.feature_importances_
        elif hasattr(model, "feature_columns") and hasattr(model, "_model") and hasattr(model._model, "feature_importances_"):
            cols = model.feature_columns
            names = list(cols) if cols else []
            importances = model._model.feature_importances_

        if names and importances is not None and len(names) == len(importances):
            pairs = sorted(zip(names, [float(x) for x in importances]), key=lambda x: x[1], reverse=True)
            return [{"feature": f, "importance": round(imp, 4)} for f, imp in pairs[:top_n] if imp > 0.0]
    except Exception:
        pass
    return []


def feature_importance_chart(model: Any, top_n: int = 8) -> None:
    """Render a horizontal bar chart of the top model-wide feature importances."""
    factors = extract_model_feature_importances(model, top_n=top_n)
    if not factors:
        st.info("Global feature importance is not available for this model.")
        return

    df = pd.DataFrame(factors)
    if alt is not None:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadius=4, color=NAVY_PRIMARY)
            .encode(
                x=alt.X("importance:Q", title="Global Feature Importance (Gini / Gain)"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("importance:Q", title="Weight", format=".4f"),
                ],
            )
            .properties(height=max(180, 26 * len(df)))
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.dataframe(df, hide_index=True)


# ---------------------------------------------------------------------------
# Explainable AI (SHAP Diverging Bar Chart)
# ---------------------------------------------------------------------------


def shap_chart(top_factors: list[dict[str, Any]]) -> None:
    """Render an enterprise horizontal diverging bar chart for SHAP factor contributions."""
    if not top_factors:
        st.info("No claim-specific SHAP factors are available for this assessment.")
        return

    rows = [
        {
            "feature": str(f.get("feature", "")),
            "feature_value": str(f.get("feature_value", "")),
            "shap_value": float(f.get("shap_value", 0.0)),
        }
        for f in top_factors
    ]
    df = pd.DataFrame(rows)

    # Tabular presentation
    display = df.copy()
    display["Direction"] = display["shap_value"].map(
        lambda v: "Increases risk (+)" if v >= 0 else "Decreases risk (-)"
    )
    display["SHAP impact"] = display["shap_value"].map(lambda v: f"{v:+.3f}")
    display = display.rename(columns={"feature": "Feature Factor", "feature_value": "Claim Value"})
    st.dataframe(
        display[["Feature Factor", "Claim Value", "SHAP impact", "Direction"]],
        hide_index=True,
        width="stretch",
    )

    if alt is None:
        return

    chart_df = df.sort_values("shap_value")
    chart_df["direction"] = chart_df["shap_value"].map(
        lambda v: "Increases risk" if v >= 0 else "Decreases risk"
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("shap_value:Q", title="SHAP Contribution to Prediction Margin (Log-Odds)"),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Increases risk", "Decreases risk"],
                    range=[RISK_COLORS["high"], RISK_COLORS["low"]],
                ),
                legend=alt.Legend(title="Factor Impact", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("feature_value:N", title="Value"),
                alt.Tooltip("shap_value:Q", title="SHAP Value", format="+.4f"),
            ],
        )
        .properties(height=max(190, 28 * len(chart_df)))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Positive SHAP values push the model toward higher fraud risk; negative "
        "values push it lower. This explains this model's prediction mechanics, not a legal finding of fraud."
    )


# ---------------------------------------------------------------------------
# Hero Assessment & Detail Views
# ---------------------------------------------------------------------------


def render_assessment(assessment: Any, *, heading: str = "Risk assessment") -> None:
    """Render the hero risk assessment card with meter, badges, and SHAP factors."""
    summary = assessment.to_ui_summary()
    category = str(summary.get("risk_category", "")).lower()
    probability = float(summary.get("fraud_probability", 0.0))

    cat_color = RISK_COLORS.get(category, NAVY_PRIMARY)
    interpretation = {
        "high": "Elevated risk flagged for review — requires forensic and policy validation.",
        "medium": "Moderate risk — secondary verification and cross-check suggested.",
        "low": "Low risk profile — standard straight-through processing indicated.",
    }.get(category, "Model risk score calculated.")

    st.markdown(f"##### {heading}")

    # Enterprise Hero Risk Card
    st.markdown(
        f'<div class="is-hero-risk">'
        f'<div class="is-hero-risk-header">'
        f'<div>'
        f'<div style="font-size:0.75rem;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.04em;">Assessed Probability</div>'
        f'<div class="is-hero-score-val" style="color:{cat_color};">{probability:.1%}</div>'
        f'</div>'
        f'<div>{risk_badge(category)}</div>'
        f'</div>'
        f'<div style="font-size:0.86rem;color:#334155;margin-bottom:0.75rem;"><b>Interpretation:</b> {html.escape(interpretation)}</div>'
        f'<div class="is-risk-meter-container">'
        f'<div style="background:#E2E8F0;height:8px;border-radius:4px;overflow:hidden;position:relative;">'
        f'<div style="background:{cat_color};height:100%;width:{min(max(probability, 0.0), 1.0) * 100:.1f}%;border-radius:4px;"></div>'
        f'</div>'
        f'<div class="is-risk-meter-labels">'
        f'<span>0% Low Risk</span>'
        f'<span>30% Medium Threshold</span>'
        f'<span>70% High Threshold</span>'
        f'<span>100%</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Provide explicit metric component for AppTest compatibility
    metric_cols = st.columns([1, 1, 2])
    metric_cols[0].metric("Model fraud probability", f"{probability:.1%}")
    metric_cols[1].metric("Risk Category", category.title())
    metric_cols[2].caption(
        f"Decision-support thresholds: Low < {DEFAULT_RISK_THRESHOLDS.medium:.0%}, "
        f"Medium ≥ {DEFAULT_RISK_THRESHOLDS.medium:.0%}, High ≥ {DEFAULT_RISK_THRESHOLDS.high:.0%}. "
        "A high score flags a claim for investigator review; it is not definitive fraud proof."
    )

    st.markdown("##### Explainable AI (SHAP Factor Contributions)")
    shap_chart(list(summary.get("top_factors") or []))


def grouped_attribute_cards(row: pd.Series, groups: dict[str, tuple[str, ...]]) -> None:
    """Render structured key-value attribute cards organized into domain groups."""
    used: set[str] = set()
    for title, keys in groups.items():
        present = [k for k in keys if k in row.index]
        if not present:
            continue
        used.update(present)
        st.markdown(f"**{title} Attributes**")
        cols = st.columns(min(3, len(present)))
        for i, key in enumerate(present):
            val = str(row[key])
            cols[i % len(cols)].markdown(
                f'<div class="is-kv-item">'
                f'<div class="is-kv-label">{html.escape(key)}</div>'
                f'<div class="is-kv-value">{html.escape(val)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    remaining = [k for k in row.index if k not in used]
    if remaining:
        with st.expander("Inspect all recorded attributes (raw fields)"):
            detail = pd.DataFrame(
                {"Attribute": [str(k) for k in remaining], "Value": [str(row[k]) for k in remaining]}
            )
            st.dataframe(detail, hide_index=True, width="stretch")


def render_graph_signals(fused_row: pd.Series) -> None:
    """Render graph relationship indicators and entity network signals."""
    st.markdown("##### Graph Relationship Signals")
    cols = st.columns(4)
    shared = bool(fused_row.get("has_shared_entity")) if "has_shared_entity" in fused_row.index else False
    max_deg = int(fused_row.get("max_entity_claim_degree", 0) or 0)
    comm_id = int(fused_row.get("community_id", -1) if pd.notna(fused_row.get("community_id", -1)) else -1)
    centrality = float(fused_row.get("degree_centrality", 0.0) or 0.0)

    cols[0].metric("Shared Entities", "Yes (Detected)" if shared else "No")
    cols[1].metric("Max Entity Claim Degree", max_deg)
    cols[2].metric("Community ID", comm_id if comm_id != -1 else "None")
    cols[3].metric("Degree Centrality", f"{centrality:.4f}")

    st.caption(
        "Shared entities indicate that this claim is linked to other claims through a "
        "shared provider, garage, bank account, or attending physician. These relationships "
        "warrant examination, but do not automatically indicate a confirmed fraud ring."
    )


# ---------------------------------------------------------------------------
# Document Forensics Panels
# ---------------------------------------------------------------------------


def render_document_summary(uploaded: Any, forensic: ForensicUploadResult, is_pdf: bool) -> None:
    """Render a metadata summary card for the uploaded claim document."""
    meta = forensic.metadata.structured_metadata if forensic.metadata else {}
    size_kb = len(uploaded.getvalue()) / 1024.0 if hasattr(uploaded, "getvalue") else 0.0
    page_count = meta.get("page_count", 1)
    fmt = f"PDF Document ({page_count} page{'s' if page_count > 1 else ''})" if is_pdf else str(meta.get("format", "Image"))

    software = "Unavailable"
    if forensic.metadata is not None:
        software = (
            "Editing tool indicated"
            if forensic.metadata.features.software_editing_indicator
            else (str(meta.get("software")) if meta.get("software") else "None recorded")
        )

    cols = st.columns(4)
    cols[0].metric("File Name", getattr(uploaded, "name", "document"))
    cols[1].metric("File Size", f"{size_kb:.1f} KB")
    cols[2].metric("Format", fmt)
    cols[3].metric("Software Tag", software)


def render_forensic_panels(forensic: ForensicUploadResult, is_pdf: bool) -> None:
    """Render 3 distinct, organized forensic investigation panels."""
    col1, col2, col3 = st.columns(3)

    # Panel 1: Error Level Analysis (ELA)
    with col1:
        st.markdown("**Error Level Analysis (ELA)**")
        st.caption("Analyzes digital recompression error to highlight spliced or pasted regions.")
        if forensic.ela is None:
            st.markdown(status_badge("Unavailable", "neutral"), unsafe_allow_html=True)
            st.caption("ELA could not run on this file format.")
        else:
            st.image(forensic.ela.ela_image, width="stretch")
            ratio = forensic.ela.features.high_error_pixel_ratio
            is_anomaly = ratio > 0.03 or forensic.ela.features.p95_error > 45.0
            st.markdown(
                status_badge("Potential Anomaly" if is_anomaly else "No Anomaly Detected", "danger" if is_anomaly else "success"),
                unsafe_allow_html=True,
            )
            st.metric("High-Error Pixel Ratio", f"{ratio:.3f}")
            st.caption("Numerical indicators are forensic measurements, not an automated fraud ruling.")

    # Panel 2: Copy-Move Forgery Detection
    with col2:
        st.markdown("**Copy-Move Forgery Detection**")
        st.caption("ORB keypoint matching identifies duplicated stamps, signatures, or figures.")
        if forensic.copy_move is None:
            st.markdown(status_badge("Unavailable", "neutral"), unsafe_allow_html=True)
            st.caption("Copy-move check could not execute on this file.")
        else:
            if forensic.copy_move.visualization is not None:
                st.image(forensic.copy_move.visualization, width="stretch")
            found = forensic.copy_move.potential_duplicate_regions_found
            st.markdown(
                status_badge("Potential Anomaly" if found else "No Anomaly Detected", "danger" if found else "success"),
                unsafe_allow_html=True,
            )
            raw_matches = forensic.copy_move.features.num_raw_matches
            filtered_matches = forensic.copy_move.features.num_filtered_matches
            st.metric("Filtered Keypoint Matches", f"{filtered_matches} / {raw_matches} raw")
            st.caption(f"Cluster density: {forensic.copy_move.features.max_cluster_size} keypoints.")

    # Panel 3: Metadata and Revision Analysis
    with col3:
        st.markdown("**Metadata & Revision Analysis**")
        st.caption("Inspects EXIF tags, creation/modification timestamps, and author signatures.")
        if forensic.metadata is None:
            st.markdown(status_badge("Unavailable", "neutral"), unsafe_allow_html=True)
            st.caption("No readable metadata found.")
            return

        meta = forensic.metadata.structured_metadata or {}
        has_edit = forensic.metadata.features.software_editing_indicator
        has_mismatch = forensic.metadata.features.datetime_original_modified_mismatch
        has_anomaly = has_edit or has_mismatch

        st.markdown(
            status_badge("Potential Anomaly" if has_anomaly else "No Anomaly Detected", "danger" if has_anomaly else "success"),
            unsafe_allow_html=True,
        )

        if is_pdf:
            for label, key in (
                ("Author", "author"),
                ("Software", "software"),
                ("Created", "datetime_original"),
                ("Modified", "datetime_modified"),
            ):
                if meta.get(key):
                    st.write(f"**{label}:** `{meta.get(key)}`")
        else:
            st.write(f"**Dimensions:** `{meta.get('width', 0)} × {meta.get('height', 0)} px`")
            cam = " ".join(filter(None, [meta.get("camera_make"), meta.get("camera_model")]))
            if cam:
                st.write(f"**Camera:** `{cam}`")
            for label, key in (
                ("Software", "software"),
                ("Created", "datetime_original"),
                ("Modified", "datetime_modified"),
            ):
                if meta.get(key):
                    st.write(f"**{label}:** `{meta.get(key)}`")

        indicators = forensic.metadata.analysis.get("indicators", [])
        if indicators:
            st.markdown("**Forensic Indicators**")
            for item in indicators:
                st.write(f"- {item}")
        else:
            st.caption("No suspicious metadata flags recorded.")
        st.caption(str(forensic.metadata.analysis.get("note", "")))


def render_preview(preview_image: Optional[Image.Image], extracted_text: str, caption: str) -> None:
    """Render an expandable preview of the document with extracted text."""
    with st.expander("Document visual preview & OCR text", expanded=True):
        if preview_image is not None:
            st.image(preview_image, caption=caption, width="stretch")
        if extracted_text.strip():
            st.text_area("Extracted Document Text", extracted_text, height=130, disabled=True)
        elif preview_image is None:
            st.info("No preview is available for this file.")


# ---------------------------------------------------------------------------
# Overview Helpers
# ---------------------------------------------------------------------------


def claims_to_review_table(table: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Produce a stable high-then-medium priority review table for investigators."""
    if table.empty:
        return table
    rank = {"high": 0, "medium": 1, "low": 2}
    out = table.copy()
    out["_rank"] = out["risk_category"].map(lambda c: rank.get(str(c).lower(), 9))
    out = out.sort_values(["_rank", "fraud_probability", CLAIM_KEY_COL], ascending=[True, False, True])
    return out.drop(columns=["_rank"]).head(limit)


def render_confusion_matrix_card(confusion: dict[str, Any]) -> None:
    """Render a clean, formatted confusion matrix table with metrics."""
    tn = confusion.get("true_negative", 0)
    fp = confusion.get("false_positive", 0)
    fn = confusion.get("false_negative", 0)
    tp = confusion.get("true_positive", 0)
    total = tn + fp + fn + tp

    cm_df = pd.DataFrame(
        [
            [f"{tn:,}", f"{fp:,}", f"{(tn + fp):,}"],
            [f"{fn:,}", f"{tp:,}", f"{(fn + tp):,}"],
            [f"{(tn + fn):,}", f"{(fp + tp):,}", f"{total:,}"],
        ],
        index=["Actual Legitimate (0)", "Actual Fraud / Elevated (1)", "Total Predicted"],
        columns=["Predicted Legitimate (0)", "Predicted Fraud (1)", "Total Actual"],
    )
    st.dataframe(cm_df, width="stretch")

    c1, c2, c3 = st.columns(3)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    c1.metric("Specificity (True Neg Rate)", f"{spec:.1%}")
    c2.metric("Sensitivity / Recall (True Pos)", f"{sens:.1%}")
    c3.metric("Decision Threshold", f"{confusion.get('threshold', 0.5):.2f}")
