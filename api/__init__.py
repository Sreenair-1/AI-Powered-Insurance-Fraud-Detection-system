"""InsureSight AI HTTP API: a thin FastAPI layer over `app/` and `src/`.

Nothing in this package fits a model, changes a score, or re-derives a
metric. It loads already-trained artifacts from `models/` once at startup and
delegates every computation to the same functions the Streamlit app uses
(`app.claim_view`, `app.data_loader`, `app.model_loader`, `src.forensics`).
"""
