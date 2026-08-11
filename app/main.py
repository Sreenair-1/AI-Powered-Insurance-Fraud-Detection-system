"""Streamlit decision-support UI for motor and health claim fraud risk assessment.

Academic decision-support tool, not an autonomous fraud adjudication system
(CLAUDE.md section 9). Every risk score, SHAP factor, and graph signal shown
here describes model behavior or claim relationships — never a proof of
fraud. Language throughout uses "risk" / "flagged for investigation" /
"elevated" rather than any claim of confirmed fraud.

Run with:
    streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.services import (
    HealthAppData,
    MotorAppData,
    N_DEMO_HEALTH_PROVIDERS,
    N_DEMO_MOTOR_CLAIMS,
    claim_row,
    graph_claim_metrics,
    load_health_app_data,
    load_motor_app_data,
    reassess_health_with_forensics,
    reassess_motor_with_forensics,
)
from src.explainability.common import ExplainabilityError
from src.features.fusion import ForensicResultBundle
from src.features.structured import CLAIM_KEY_COL
from src.forensics import (
    CopyMoveImageLoadError,
    ImageLoadError,
    MetadataImageLoadError,
    analyze_metadata,
    compute_copy_move,
    compute_ela,
)
from src.models.motor import MotorModelError
from src.pipeline.inference import (
    ClaimRiskAssessment,
    RiskThresholds,
    categorize_risk,
)

st.set_page_config(page_title="Claim Fraud Risk Assessment", layout="wide")

RISK_COLORS = {"low": "#2e7d32", "medium": "#e6a700", "high": "#c62828"}


def _risk_badge(risk_category: str, fraud_probability: float) -> None:
    color = RISK_COLORS.get(risk_category, "#888")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:0.5rem 1rem;border-radius:0.5rem;
                    background:{color};color:white;font-weight:600;font-size:1.1rem;">
            {risk_category.upper()} RISK &nbsp;·&nbsp; fraud probability {fraud_probability:.1%}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _shap_chart(assessment: ClaimRiskAssessment) -> None:
    factors = assessment.top_factors
    if not factors:
        st.info("No SHAP factors available for this claim.")
        return
    labels = [f["feature"] for f in factors][::-1]
    values = [f["shap_value"] for f in factors][::-1]
    colors = ["#c62828" if v > 0 else "#1565c0" for v in values]

    fig, ax = plt.subplots(figsize=(6, max(2, 0.35 * len(factors))))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to log-odds; red = raises risk, blue = lowers risk)")
    fig.tight_layout()
    st.pyplot(fig)

    table = pd.DataFrame(
        [
            {
                "Feature": f["feature"],
                "Value in claim": str(f["feature_value"]),
                "SHAP contribution": f["shap_value"],
            }
            for f in factors
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")


def _forensic_bundle_from_upload(uploaded_file) -> tuple[ForensicResultBundle, dict[str, object]]:
    image = Image.open(uploaded_file)
    image.load()

    results: dict[str, object] = {}
    ela_result = None
    try:
        ela_result = compute_ela(image)
        results["ela"] = ela_result
    except ImageLoadError as exc:
        st.error(f"ELA failed: {exc}")

    copy_move_result = None
    try:
        copy_move_result = compute_copy_move(image)
        results["copy_move"] = copy_move_result
    except CopyMoveImageLoadError as exc:
        st.error(f"Copy-move detection failed: {exc}")

    metadata_result = None
    try:
        metadata_result = analyze_metadata(image)
        results["metadata"] = metadata_result
    except MetadataImageLoadError as exc:
        st.error(f"Metadata analysis failed: {exc}")

    bundle = ForensicResultBundle(ela=ela_result, copy_move=copy_move_result, metadata=metadata_result)
    return bundle, results


def _render_forensics_tab(domain: str, claim_key: str, thresholds: RiskThresholds) -> None:
    st.caption(
        "Attach a claim document (e.g. a photographed bill or repair estimate) to run "
        "Error Level Analysis, copy-move detection, and metadata analysis, and see how the "
        "risk score changes when this evidence is included. Without an attached document, "
        "this claim's risk score already reflects forensic features as 'unavailable' — that "
        "is the default for claims with no linked document."
    )
    uploaded = st.file_uploader("Claim document image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if uploaded is None:
        return

    try:
        bundle, results = _forensic_bundle_from_upload(uploaded)
    except UnidentifiedImageError:
        st.error("Could not read this file as an image.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded, caption="Uploaded document", width="stretch")

    ela_result = results.get("ela")
    if ela_result is not None:
        with col2:
            st.image(ela_result.ela_image, caption="Error Level Analysis (brighter = higher recompression error)", width="stretch")
        f = ela_result.features
        st.markdown("**ELA features**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"mean error": f.mean_error, "std error": f.std_error, "max error": f.max_error,
                     "p95 error": f.p95_error, "high-error pixel ratio": f.high_error_pixel_ratio}
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    copy_move_result = results.get("copy_move")
    if copy_move_result is not None:
        st.markdown("**Copy-move detection**")
        st.write(
            "Potential duplicated regions found: "
            f"**{copy_move_result.potential_duplicate_regions_found}** "
            "(evidence to review, not a tampering determination)"
        )
        if copy_move_result.visualization is not None:
            st.image(copy_move_result.visualization, caption="Matched keypoint pairs", width="stretch")
        cm = copy_move_result.features
        st.dataframe(
            pd.DataFrame(
                [
                    {"keypoints": cm.num_keypoints, "raw matches": cm.num_raw_matches,
                     "filtered matches": cm.num_filtered_matches, "match ratio": cm.match_ratio}
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    metadata_result = results.get("metadata")
    if metadata_result is not None:
        st.markdown("**Metadata analysis**")
        indicators = metadata_result.analysis.get("indicators", [])
        if indicators:
            for indicator in indicators:
                st.write(f"- {indicator}")
        else:
            st.write("No notable metadata indicators.")
        note = metadata_result.analysis.get("note")
        if note:
            st.caption(note)

    st.divider()
    if st.button("Re-run risk assessment with this document"):
        try:
            if domain == "motor":
                data: MotorAppData = st.session_state["motor_data"]
                assessment = reassess_motor_with_forensics(data, claim_key, bundle, thresholds=thresholds)
            else:
                data: HealthAppData = st.session_state["health_data"]
                assessment = reassess_health_with_forensics(data, claim_key, bundle, thresholds=thresholds)
        except (MotorModelError, ExplainabilityError, ValueError) as exc:
            st.error(f"Could not re-assess this claim: {exc}")
            return
        st.session_state[f"override_{domain}_{claim_key}"] = assessment
        st.success("Risk assessment updated with this document's forensic evidence — see the Risk & Explanation tab.")


def _render_claim_info(domain: str, claim_key: str, row: pd.Series, label: object) -> None:
    st.subheader(f"Claim `{claim_key}`")
    display = row.drop(labels=[c for c in row.index if c in {CLAIM_KEY_COL}], errors="ignore")
    display = display.dropna()
    st.dataframe(display.astype(str).rename("value"), width="stretch")
    if label is not None:
        st.caption(
            f"Recorded outcome in this dataset (reference only, a historical/derived label — "
            f"see CLAUDE.md section 9): **{'flagged' if int(label) == 1 else 'not flagged'}**"
        )


def _render_graph_tab(metrics: dict[str, object] | None) -> None:
    if metrics is None:
        st.info("This claim has no graph relationships recorded.")
        return
    st.write(
        f"Shares an entity (garage/bank account/provider/beneficiary/physician) with other claims: "
        f"**{bool(metrics['has_shared_entity'])}**"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Distinct linked entities": metrics["num_entities"],
                    "Max claims sharing one entity": metrics["max_entity_claim_degree"],
                    "Mean claims sharing an entity": round(float(metrics["mean_entity_claim_degree"]), 2),
                    "Community id": metrics["community_id"],
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Centrality within the claim entity graph**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Degree": round(float(metrics["degree_centrality"]), 4),
                    "Betweenness": round(float(metrics["betweenness_centrality"]), 4),
                    "Closeness": round(float(metrics["closeness_centrality"]), 4),
                    "Eigenvector": (
                        round(float(metrics["eigenvector_centrality"]), 4)
                        if pd.notna(metrics["eigenvector_centrality"])
                        else "n/a"
                    ),
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Higher centrality/entity-sharing means this claim's linked entities recur across many "
        "claims — evidence for investigators to weigh, not proof of a fraud ring (CLAUDE.md section 9)."
    )


def _render_domain(domain: str, data, thresholds: RiskThresholds) -> None:
    claim_keys = data.claim_key_order
    labels = {
        key: (
            f"{data.assessments[key].risk_category.upper()} · {key}"
            if key in data.assessments
            else key
        )
        for key in claim_keys
    }
    selected_key = st.sidebar.selectbox(
        "Claim", claim_keys, format_func=lambda k: labels[k], key=f"claim_select_{domain}"
    )

    override = st.session_state.get(f"override_{domain}_{selected_key}")
    assessment = override if override is not None else data.assessments.get(selected_key)

    if assessment is None:
        st.warning("No assessment available for this claim.")
        return

    risk_category = categorize_risk(assessment.fraud_probability, thresholds)

    row = claim_row(selected_key, data.claim_key_order, data.claims_holdout)
    label = data.y_holdout.iloc[data.claim_key_order.index(selected_key)]

    tab_overview, tab_graph, tab_forensics, tab_explain, tab_model = st.tabs(
        ["Claim & Risk", "Graph Relationships", "Document Forensics", "SHAP Explanation", "Model & Evaluation"]
    )

    with tab_overview:
        _risk_badge(risk_category, assessment.fraud_probability)
        st.write("")
        _render_claim_info(domain, selected_key, row, label)
        if override is not None:
            st.info("This assessment includes an uploaded document's forensic evidence.")

    with tab_graph:
        metrics = graph_claim_metrics(data.graph_features_holdout, selected_key)
        _render_graph_tab(metrics)

    with tab_forensics:
        _render_forensics_tab(domain, selected_key, thresholds)

    with tab_explain:
        st.write(
            f"Model baseline (log-odds before this claim's features): **{assessment.base_value:.3f}**"
        )
        st.caption(
            "SHAP values explain what drove this model's prediction for this claim — they are "
            "not evidence of fraud on their own (CLAUDE.md section 9)."
        )
        _shap_chart(assessment)

    with tab_model:
        m = data.metrics
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{m.accuracy:.3f}")
        cols[1].metric("Precision", f"{m.precision:.3f}")
        cols[2].metric("Recall", f"{m.recall:.3f}")
        cols[3].metric("F1", f"{m.f1:.3f}")
        cols[4].metric("ROC-AUC", f"{m.roc_auc:.3f}")
        st.caption(
            f"Computed on a held-out demo-scale sample: {m.n_samples} claims "
            f"({m.n_positive} flagged in this dataset's history), trained on {data.n_train} claims."
        )
        if domain == "health":
            st.caption(
                f"Demo sample: {data.n_providers_sampled} of {5410} providers "
                "(stratified, capped claims/provider) — see app/services.py for why."
            )
        else:
            st.caption(
                f"Demo sample: {N_DEMO_MOTOR_CLAIMS} of 15,420 motor claims — see app/services.py for why."
            )


def main() -> None:
    st.title("AI-Powered Insurance Claim Fraud Risk Assessment")
    st.caption(
        "Academic decision-support demo for motor and health insurance claims. Risk scores and "
        "flags describe model behavior, not proof of fraud — see CLAUDE.md section 9."
    )

    domain_label = st.sidebar.radio("Insurance domain", ["Motor", "Health"])
    domain = "motor" if domain_label == "Motor" else "health"

    medium = st.sidebar.slider("Medium-risk threshold", 0.0, 1.0, 0.3, 0.05)
    high = st.sidebar.slider("High-risk threshold", medium, 1.0, max(0.7, medium + 0.05), 0.05)
    thresholds = RiskThresholds(medium=medium, high=high)

    if domain == "motor":
        data = load_motor_app_data()
        st.session_state["motor_data"] = data
    else:
        data = load_health_app_data()
        st.session_state["health_data"] = data

    _render_domain(domain, data, thresholds)


main()
