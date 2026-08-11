"""Streamlit decision-support UI for AI-powered insurance claim fraud detection.

Thin renderer over `app.model_loader` (persisted models), `app.data_loader`
(sampled, fused claim data), and `app.claim_view` (assessment + optional
forensic re-scoring) -- see `docs/app_ui.md` for the full architecture and
`docs/pipeline.md` for what `src/pipeline/inference.py` actually computes.

"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from app import config
from app.claim_view import (
    ClaimViewError,
    assess_claim,
    claim_row,
    rescore_with_forensics,
    run_forensic_checks,
)
from app.data_loader import (
    DatasetNotAvailableError,
    get_cached_health_dataset,
    get_cached_motor_dataset,
    health_claim_row,
    motor_claim_row,
)
from app.model_loader import ModelNotAvailableError, get_health_model, get_motor_model, load_eval_metrics
from src.features.structured import CLAIM_KEY_COL
from src.pipeline.inference import DEFAULT_RISK_THRESHOLDS

st.set_page_config(page_title="Insurance Claim Fraud Risk Review", layout="wide")

MOTOR_DOMAIN = "motor"
HEALTH_DOMAIN = "health"

_RISK_ICON = {"low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F534"}


def _render_header() -> None:
    st.title("Insurance Claim Fraud Risk Review")
    st.caption(
        "Decision-support only. Risk scores and SHAP explanations describe model "
        "behavior, not proof of fraud -- a \"high\" result means **flagged for "
        "investigation**, never a determination"
    )


def _sidebar_domain_and_sampling() -> tuple[str, dict]:
    st.sidebar.header("Dataset")
    domain = st.sidebar.radio("Claim type", (MOTOR_DOMAIN, HEALTH_DOMAIN), format_func=str.title)
    st.sidebar.caption(
        "Claims are loaded from a bounded, seeded sample of the real dataset: "
        "exact graph centrality (`src/graph/features.py`) does not scale to the "
        "full dataset within an interactive session. See docs/app_ui.md."
    )
    params: dict = {}
    if domain == MOTOR_DOMAIN:
        params["sample_size"] = st.sidebar.slider(
            "Motor claims to load",
            100,
            config.MAX_MOTOR_SAMPLE_SIZE,
            config.DEFAULT_MOTOR_SAMPLE_SIZE,
            step=100,
        )
    else:
        params["inpatient_sample_size"] = st.sidebar.slider(
            "Inpatient claims to load",
            20,
            config.MAX_HEALTH_INPATIENT_SAMPLE_SIZE,
            config.DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE,
            step=20,
        )
        params["outpatient_sample_size"] = st.sidebar.slider(
            "Outpatient claims to load",
            40,
            config.MAX_HEALTH_OUTPATIENT_SAMPLE_SIZE,
            config.DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE,
            step=40,
        )
    return domain, params


def _load_model(domain: str):
    directory = config.motor_model_dir() if domain == MOTOR_DOMAIN else config.health_model_dir()
    getter = get_motor_model if domain == MOTOR_DOMAIN else get_health_model
    try:
        return getter(str(directory)), directory
    except ModelNotAvailableError as exc:
        st.error(str(exc))
        st.stop()
        raise  # unreachable, keeps type-checkers happy


def _load_dataset(domain: str, params: dict):
    try:
        if domain == MOTOR_DOMAIN:
            return get_cached_motor_dataset(**params)
        return get_cached_health_dataset(**params)
    except DatasetNotAvailableError as exc:
        st.error(
            f"{exc}\n\nThis app needs the raw datasets under `data/raw/` -- see "
            "docs/data_contract.md for the expected local file layout."
        )
        st.stop()
        raise  # unreachable, keeps type-checkers happy


def _render_claim_info(domain: str, dataset, claim_key: str) -> None:
    row = motor_claim_row(dataset, claim_key) if domain == MOTOR_DOMAIN else health_claim_row(dataset, claim_key)
    st.subheader("Claim information")
    st.dataframe(row.to_frame(name="value"), use_container_width=True)


def _render_graph_panel(dataset, claim_key: str) -> None:
    row = dataset.X[dataset.X[CLAIM_KEY_COL] == claim_key].iloc[0]
    st.subheader("Graph relationship signals")
    cols = st.columns(4)
    cols[0].metric("Shared entities", "Yes" if bool(row.get("has_shared_entity")) else "No")
    cols[1].metric("Max entity claim degree", int(row.get("max_entity_claim_degree", 0)))
    cols[2].metric("Community ID", int(row.get("community_id", -1)))
    cols[3].metric("Degree centrality", f"{float(row.get('degree_centrality', 0.0)):.4f}")
    st.caption(
        "A high 'shared entities'/centrality value means this claim's "
        "provider/garage/bank-account etc. is also linked to other claims -- "
        "evidence to weigh, not proof of a fraud ring."
    )


def _render_assessment(assessment) -> None:
    summary = assessment.to_ui_summary()
    icon = _RISK_ICON.get(summary["risk_category"], "")
    cols = st.columns(2)
    cols[0].metric("Fraud probability", f"{summary['fraud_probability']:.1%}")
    cols[1].markdown(f"### {icon} {summary['risk_category_description']}")

    st.markdown("**Top contributing factors (SHAP)**")
    factors_df = pd.DataFrame(summary["top_factors"])
    if not factors_df.empty:
        factors_df = factors_df.rename(
            columns={"feature": "Feature", "feature_value": "Claim value", "shap_value": "SHAP contribution"}
        ).set_index("Feature")
        st.dataframe(factors_df, use_container_width=True)
        st.bar_chart(factors_df["SHAP contribution"])
    st.caption(
        "Positive SHAP contributions push the prediction toward higher fraud "
        "risk; negative contributions push it lower. This explains the "
        "model's behavior, not a finding of fraud."
    )


def _render_forensics_panel(domain: str, model, dataset, claim_key: str) -> None:
    st.subheader("Document forensics (optional)")
    uploaded = st.file_uploader(
        "Upload a claim document image to check for tampering (ELA / copy-move / metadata)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )
    if uploaded is None:
        st.caption("No document uploaded for this claim -- forensic features are treated as unavailable.")
        return

    try:
        image = Image.open(uploaded)
    except Exception:
        st.warning("Could not read the uploaded file as an image.")
        return

    forensic = run_forensic_checks(image)
    if not forensic.any_available:
        st.warning("Could not analyze the uploaded file as an image; forensic features remain unavailable.")
        return

    col1, col2, col3 = st.columns(3)
    if forensic.ela is not None:
        col1.markdown("**Error Level Analysis**")
        col1.image(forensic.ela.ela_image, use_container_width=True)
        col1.write(f"High-error pixel ratio: {forensic.ela.features.high_error_pixel_ratio:.3f}")
    else:
        col1.info("ELA unavailable for this file.")

    if forensic.copy_move is not None:
        col2.markdown("**Copy-Move Detection**")
        if forensic.copy_move.visualization is not None:
            col2.image(forensic.copy_move.visualization, use_container_width=True)
        col2.write(
            "Potential duplicated regions: "
            + ("Yes" if forensic.copy_move.potential_duplicate_regions_found else "No")
        )
    else:
        col2.info("Copy-move detection unavailable for this file.")

    if forensic.metadata is not None:
        col3.markdown("**Metadata Analysis**")
        for indicator in forensic.metadata.analysis.get("indicators", []):
            col3.write(f"- {indicator}")
        col3.caption(forensic.metadata.analysis.get("note", ""))
    else:
        col3.info("Metadata analysis unavailable for this file.")

    try:
        rescored = rescore_with_forensics(domain, model, dataset, claim_key, forensic)
    except ClaimViewError as exc:
        st.error(str(exc))
        return
    st.markdown("**Risk assessment re-scored with this document's forensic evidence:**")
    _render_assessment(rescored)


def _render_eval_metrics(model_dir) -> None:
    with st.expander("Model evaluation (held-out validation split, from training time)"):
        metrics = load_eval_metrics(model_dir)
        if metrics is None:
            st.info(
                "No evaluation metrics recorded for this artifact. Re-run "
                "`scripts/train_models.py` to generate one."
            )
            return
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[1].metric("Precision", f"{metrics['precision']:.3f}")
        cols[2].metric("Recall", f"{metrics['recall']:.3f}")
        cols[3].metric("F1", f"{metrics['f1']:.3f}")
        cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        st.caption(f"{metrics['n_samples']} validation claims ({metrics['n_positive']} positive).")


def main() -> None:
    _render_header()

    domain, sampling_params = _sidebar_domain_and_sampling()
    model, model_dir = _load_model(domain)
    dataset = _load_dataset(domain, sampling_params)

    claim_keys = list(dataset.X[CLAIM_KEY_COL])
    if not claim_keys:
        st.warning("No claims loaded for this sample size -- try increasing it in the sidebar.")
        return

    claim_key = st.selectbox("Select a claim to review", claim_keys)

    _render_claim_info(domain, dataset, claim_key)
    _render_graph_panel(dataset, claim_key)

    row = claim_row(dataset, claim_key)
    assessment = assess_claim(domain, model, row, DEFAULT_RISK_THRESHOLDS)
    st.subheader("Fraud risk assessment")
    _render_assessment(assessment)

    _render_forensics_panel(domain, model, dataset, claim_key)
    _render_eval_metrics(model_dir)


main()
