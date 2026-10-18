"""InsureSight visual identity: palette, CSS, and enterprise presentational helpers.

Presentation only. Risk thresholds, scoring, and forensic methods live in
`src/` and `app/claim_view.py`.
"""
from __future__ import annotations

import html
from typing import Optional

import streamlit as st

# Enterprise Brand Palette
NAVY_PRIMARY = "#0B192C"
NAVY_DARK = "#070F1E"
SLATE_DARK = "#1E293B"
SLATE_TEXT = "#334155"
SLATE_MUTED = "#64748B"
SLATE_LIGHT = "#94A3B8"
BORDER_COLOR = "#E2E8F0"
SURFACE_WHITE = "#FFFFFF"
APP_BG = "#F8FAFC"

TEAL_ACCENT = "#0D9488"
TEAL_DEEP = "#0F766E"
TEAL_LIGHT = "#14B8A6"
TEAL_BG = "#F0FDFA"

# Risk tokens (Reserved strictly for meaningful risk states)
RISK_COLORS = {
    "low": "#059669",      # Emerald Green
    "medium": "#D97706",   # Amber / Orange
    "high": "#DC2626",     # Crimson Red
}

RISK_BG = {
    "low": "#ECFDF5",
    "medium": "#FFFBEB",
    "high": "#FEF2F2",
}

RISK_BORDER = {
    "low": "#A7F3D0",
    "medium": "#FCD34D",
    "high": "#FCA5A5",
}

# Legacy aliases for backward compatibility with existing imports
NAVY = NAVY_PRIMARY
SLATE = SLATE_TEXT
TEAL = TEAL_DEEP
SURFACE = SURFACE_WHITE

_CSS = """
<style>
    /* Global App Background & Base Typography */
    .stApp {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1E293B;
    }

    /* Enterprise Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B192C;
        border-right: 1px solid rgba(226, 232, 240, 0.1);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(226, 232, 240, 0.12);
        margin: 1rem 0;
    }
    /* Streamlit versions render radio options as nested markdown, spans, or
       labels. Style the full option subtree so none inherit a dark text color. */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label *,
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] * {
        color: #E2E8F0 !important;
        font-size: 0.88rem;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: #CBD5E1 !important;
        font-size: 0.82rem;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {
        color: #94A3B8 !important;
    }

    /* Brand Header in Sidebar */
    .is-brand-card {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(226, 232, 240, 0.12);
        margin-bottom: 1.1rem;
    }
    .is-brand-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .is-brand-logo {
        width: 38px;
        height: 38px;
        border-radius: 9px;
        background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
        box-shadow: 0 2px 6px rgba(13, 148, 136, 0.35);
        color: #FFFFFF;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        flex-shrink: 0;
    }
    .is-brand-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #F8FAFC !important;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }
    .is-brand-subtitle {
        font-size: 0.72rem;
        color: #94A3B8 !important;
        margin-top: 0.2rem;
        line-height: 1.35;
        font-weight: 400;
    }

    /* System Status Widget in Sidebar */
    .is-system-status-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(226, 232, 240, 0.14);
        border-radius: 9px;
        padding: 0.75rem 0.85rem;
        margin: 0.5rem 0;
    }
    .is-status-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
        color: #E2E8F0;
    }
    .is-status-row:last-child {
        margin-bottom: 0;
    }
    .is-status-dot-active {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.7);
        display: inline-block;
        flex-shrink: 0;
    }
    .is-status-dot-warning {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #F59E0B;
        box-shadow: 0 0 6px rgba(245, 158, 11, 0.7);
        display: inline-block;
        flex-shrink: 0;
    }
    .is-status-dot-error {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #EF4444;
        box-shadow: 0 0 6px rgba(239, 68, 68, 0.7);
        display: inline-block;
        flex-shrink: 0;
    }

    /* Page Header */
    .is-page-header-container {
        margin-bottom: 1.3rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .is-page-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #0D9488;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .is-page-title {
        font-size: 1.65rem;
        font-weight: 750;
        color: #0B192C;
        letter-spacing: -0.02em;
        margin: 0.1rem 0 0.3rem 0;
    }
    .is-page-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.45;
        max-width: 900px;
        margin: 0;
    }

    /* Enterprise KPI Cards */
    .is-kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.95rem 1.05rem;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02);
        min-height: 6.8rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .is-kpi-card:hover {
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
        transform: translateY(-1px);
    }
    .is-kpi-accent-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .is-kpi-label {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .is-kpi-value {
        font-size: 1.65rem;
        font-weight: 750;
        color: #0B192C;
        line-height: 1.15;
        margin: 0.25rem 0 0.15rem 0;
        letter-spacing: -0.02em;
    }
    .is-kpi-hint {
        font-size: 0.74rem;
        color: #64748B;
        line-height: 1.35;
    }

    /* General Enterprise Card */
    .is-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .is-card-title {
        font-size: 0.95rem;
        font-weight: 650;
        color: #0B192C;
        margin-bottom: 0.4rem;
    }

    /* Badges */
    .is-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        line-height: 1.2;
    }
    .is-badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Key-Value Display Grid */
    .is-kv-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
    }
    .is-kv-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .is-kv-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0B192C;
        word-break: break-word;
    }

    /* Hero Risk Assessment Card */
    .is-hero-risk {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px -1px rgba(15, 23, 42, 0.06);
    }
    .is-hero-risk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .is-hero-score-val {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .is-risk-meter-container {
        margin: 0.85rem 0 0.5rem 0;
    }
    .is-risk-meter-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 600;
        margin-top: 0.35rem;
    }

    /* Callout & Notice Boxes */
    .is-notice {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        font-size: 0.82rem;
        line-height: 1.45;
        border-left: 4px solid;
    }
    .is-notice-info {
        background: #F0FDF4;
        border-color: #059669;
        color: #065F46;
    }
    .is-notice-amber {
        background: #FFFBEB;
        border-color: #D97706;
        color: #92400E;
    }
    .is-notice-teal {
        background: #F0FDFA;
        border-color: #0D9488;
        color: #115E59;
    }

    /* Streamlit Dataframe / Table tweaks */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Heatmap Confusion Matrix */
    .is-cm-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .is-cm-grid {
        display: grid;
        grid-template-columns: 140px 1fr 1fr 110px;
        gap: 8px;
        align-items: stretch;
        margin: 1rem 0;
    }
    .is-cm-header-cell {
        font-size: 0.75rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 0.4rem;
    }
    .is-cm-row-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #1E293B;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0.5rem;
    }
    .is-cm-row-sub {
        font-size: 0.7rem;
        color: #64748B;
        font-weight: 400;
    }
    .is-cm-tile {
        border-radius: 8px;
        padding: 0.9rem 0.75rem;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.15s ease;
    }
    .is-cm-tile:hover {
        transform: translateY(-1px);
    }
    .is-cm-tile-tn {
        background: #F0FDF4;
        border: 1.5px solid #86EFAC;
        color: #14532D;
    }
    .is-cm-tile-fp {
        background: #FEF2F2;
        border: 1.5px solid #FCA5A5;
        color: #991B1B;
    }
    .is-cm-tile-fn {
        background: #FFFBEB;
        border: 1.5px solid #FDE68A;
        color: #92400E;
    }
    .is-cm-tile-tp {
        background: #F0FDFA;
        border: 1.5px solid #99F6E4;
        color: #0F766E;
    }
    .is-cm-tile-val {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .is-cm-tile-sub {
        font-size: 0.74rem;
        font-weight: 600;
        margin-top: 0.2rem;
        opacity: 0.85;
    }
    .is-cm-total-cell {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 0.5rem;
        font-size: 0.88rem;
        font-weight: 700;
        color: #334155;
    }

    /* Triage Review Queue Banner */
    .is-triage-banner {
        background: linear-gradient(135deg, #0B192C 0%, #1E293B 100%);
        border: 1px solid rgba(226, 232, 240, 0.15);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin: 0.8rem 0 1.2rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        color: #FFFFFF;
    }
    .is-triage-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .is-triage-sub {
        font-size: 0.78rem;
        color: #CBD5E1;
        margin-top: 0.15rem;
    }
    .is-triage-badge-group {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }

    /* Delta Score Badges */
    .is-delta-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .is-delta-pos {
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }
    .is-delta-neg {
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .is-delta-flat {
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }

    /* Footer in Sidebar */
    .is-sidebar-footer {
        font-size: 0.7rem;
        color: #94A3B8 !important;
        margin-top: 1.2rem;
        line-height: 1.4;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(226, 232, 240, 0.1);
    }
</style>
"""


def inject_theme() -> None:
    """Inject scoped enterprise CSS styling into the Streamlit app."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str) -> None:
    """Render a polished, standardized enterprise page header."""
    st.markdown(
        f'<div class="is-page-header-container">'
        f'<div class="is-page-kicker">{html.escape(kicker)}</div>'
        f'<div class="is-page-title">{html.escape(title)}</div>'
        f'<div class="is-page-desc">{html.escape(description)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, hint: str = "", accent: Optional[str] = None) -> None:
    """Render a clean, responsive KPI card with top accent bar."""
    bar_color = accent or TEAL_ACCENT
    st.markdown(
        f'<div class="is-kpi-card">'
        f'<div class="is-kpi-accent-bar" style="background-color: {html.escape(bar_color)};"></div>'
        f'<div class="is-kpi-label">{html.escape(label)}</div>'
        f'<div class="is-kpi-value" style="color: {html.escape(accent or NAVY_PRIMARY)};">{html.escape(value)}</div>'
        f'<div class="is-kpi-hint">{html.escape(hint)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> str:
    """Generate an enterprise risk badge with consistent risk colors."""
    cat = (category or "").lower().strip()
    color = RISK_COLORS.get(cat, SLATE_MUTED)
    bg = RISK_BG.get(cat, "#F1F5F9")
    border = RISK_BORDER.get(cat, "#CBD5E1")
    label = {
        "low": "Low Risk",
        "medium": "Medium Risk",
        "high": "High Risk",
    }.get(cat, category.title() if category else "Unassessed")
    return (
        f'<span class="is-badge" style="background:{bg};color:{color};border:1px solid {border}">'
        f'<span class="is-badge-dot" style="background:{color}"></span>'
        f'{html.escape(label)}'
        f'</span>'
    )


def status_badge(
    label: str,
    variant: str = "neutral",
) -> str:
    """Render a status badge for forensic indicators or domain tags."""
    variants = {
        "success": ("#059669", "#ECFDF5", "#A7F3D0"),
        "warning": ("#D97706", "#FFFBEB", "#FCD34D"),
        "danger": ("#DC2626", "#FEF2F2", "#FCA5A5"),
        "teal": ("#0D9488", "#F0FDFA", "#99F6E4"),
        "neutral": ("#475569", "#F8FAFC", "#E2E8F0"),
    }
    color, bg, border = variants.get(variant, variants["neutral"])
    return (
        f'<span class="is-badge" style="background:{bg};color:{color};border:1px solid {border}">'
        f'<span class="is-badge-dot" style="background:{color}"></span>'
        f'{html.escape(label)}'
        f'</span>'
    )


def info_callout(text: str, title: Optional[str] = None, variant: str = "teal") -> None:
    """Render a clean informational callout banner."""
    classes = {
        "teal": "is-notice is-notice-teal",
        "amber": "is-notice is-notice-amber",
        "info": "is-notice is-notice-info",
    }
    cls_name = classes.get(variant, "is-notice is-notice-teal")
    title_html = f"<b>{html.escape(title)}</b><br>" if title else ""
    st.markdown(
        f'<div class="{cls_name}">{title_html}{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    """Render the official InsureSight AI branding in the sidebar."""
    st.sidebar.markdown(
        '<div class="is-brand-card">'
        '<div class="is-brand-header">'
        '<div class="is-brand-logo">IS</div>'
        '<div>'
        '<div class="is-brand-title">InsureSight AI</div>'
        '<div class="is-brand-subtitle">Insurance Intelligence &amp; Fraud Investigation Platform</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def sidebar_system_status(
    model_ok: bool,
    n_claims: int,
    model_name: str,
    domain: str,
) -> None:
    """Render a compact, high-contrast system status card in the sidebar."""
    st.sidebar.markdown("**System status**")
    if model_ok:
        status_dot = '<span class="is-status-dot-active"></span>'
        status_text = "Operational · Ready"
        model_info = f"Model: <code>{html.escape(model_name)}</code>"
        data_info = f"Dataset: <code>{n_claims:,}</code> claims ({html.escape(domain).title()})"
    else:
        status_dot = '<span class="is-status-dot-error"></span>'
        status_text = "Model or dataset unavailable"
        model_info = "Model: Offline"
        data_info = "Dataset: Not loaded"

    st.sidebar.markdown(
        f'<div class="is-system-status-box">'
        f'<div class="is-status-row">{status_dot} <b>{status_text}</b></div>'
        f'<div class="is-status-row" style="color:#94A3B8;">{model_info}</div>'
        f'<div class="is-status-row" style="color:#94A3B8;">{data_info}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    """Render the official restrained investigation disclaimer footer."""
    st.sidebar.markdown(
        '<div class="is-sidebar-footer">'
        '<b>AI Decision-Support System</b><br>'
        'Scores and forensic indicators support investigator triage. They do not '
        'adjudicate claims or prove fraud.'
        '</div>',
        unsafe_allow_html=True,
    )


def triage_banner(review_count: int, total_count: int, high_count: int, medium_count: int) -> None:
    """Render a prominent, high-impact investigation priority banner."""
    pct = (review_count / total_count * 100.0) if total_count > 0 else 0.0
    st.markdown(
        f'<div class="is-triage-banner">'
        f'<div>'
        f'<div class="is-triage-title">Priority Investigation Queue · {review_count:,} Claims Requiring Review ({pct:.1f}%)</div>'
        f'<div class="is-triage-sub">Actionable claims exceeding decision thresholds ({high_count:,} High Risk · {medium_count:,} Medium Risk).</div>'
        f'</div>'
        f'<div class="is-triage-badge-group">'
        f'{status_badge(f"{high_count:,} High", "danger")}'
        f'{status_badge(f"{medium_count:,} Medium", "warning")}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def delta_badge(diff_prob: float) -> str:
    """Generate a stylized HTML badge showing score delta before vs after forensic evidence."""
    if diff_prob > 0.001:
        cls = "is-delta-pos"
        arrow = "▲"
        sign = "+"
    elif diff_prob < -0.001:
        cls = "is-delta-neg"
        arrow = "▼"
        sign = ""
    else:
        cls = "is-delta-flat"
        arrow = "●"
        sign = ""
    return (
        f'<span class="is-delta-badge {cls}">'
        f'{arrow} {sign}{diff_prob:.1%}'
        f'</span>'
    )

