"""
Pharmacogene Functional-Status Predictor
Web app built with Streamlit
"""

import io
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "scripts" / "data" / "processed" / "phase2_rf_model.joblib"

VEP_IMPACT_ORDER = {"MODIFIER": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}

EXAMPLE_CSV = """Name,GeneSymbol,most_severe_consequence,vep_impact,gnomad_af,phylop100way,gerp_score
NM_000777.5(CYP3A5):c.219-237A>G,CYP3A5,splice_acceptor_variant,HIGH,0.9018,-0.000969,1.15
NM_000463.3(UGT1A1):c.686C>A (p.Pro229Gln),UGT1A1,missense_variant,MODERATE,0.0001,0.715472,4.88
NM_000367.5(TPMT):c.460G>A (p.Ala154Thr),TPMT,missense_variant,MODERATE,,-0.150417,-0.479
NM_000367.2(TPMT):c.238G>C (p.Ala80Pro),TPMT,missense_variant,MODERATE,0.0002,2.07854,5.38
NM_000777.5(CYP3A5):c.1400C>T,CYP3A5,missense_variant,MODERATE,0.15,1.2,3.4
"""


CLASS_INFO = {
    "No function": {
        "color": "#C0392B",
        "bg": "#FDEDEC",
        "border": "#E74C3C",
        "description": "Enzyme is non-functional. Likely Poor Metabolizer phenotype."
    },
    "Decreased function": {
        "color": "#D35400",
        "bg": "#FEF5E7",
        "border": "#E67E22",
        "description": "Reduced enzyme activity. Likely Intermediate Metabolizer phenotype."
    },
    "Normal function": {
        "color": "#1E6B2E",
        "bg": "#EAFAF1",
        "border": "#27AE60",
        "description": "Normal enzyme activity. Likely Normal Metabolizer phenotype."
    },
}

GENE_PANEL = ["CYP2D6", "CYP2C19", "CYP2C9", "CYP2B6", "CYP3A5",
              "SLCO1B1", "TPMT", "NUDT15", "UGT1A1"]

# ── Brand palette (teal / sage) ────────────────────────────────────────────────
C_LIGHTEST = "#E8F4F1"   # page background
C_LIGHT    = "#B2D4CC"   # card borders, subtle accents
C_MID      = "#6FA89A"   # secondary elements, sidebar
C_DARK     = "#2E6B5E"   # header gradient start, buttons
C_DARKEST  = "#0D4438"   # header gradient end, strong accents

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PGx Functional Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Background */
    .stApp {{
        background-color: {C_LIGHTEST};
    }}
    .block-container {{
        padding-top: 4.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }}

/* Header banner */
    .pgx-header {{
        background: linear-gradient(135deg, {C_DARK} 0%, {C_DARKEST} 100%);
        border-radius: 12px;
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white !important;
    }}
    .pgx-header * {{
        color: #E8F4F1 !important;
    }}
    .pgx-header [data-testid="stHeadingWithActionElements"] h1 {{
        color: white !important;
        font-family: 'Georgia', serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        line-height: 1.3 !important;
    }}
    .stMarkdown .pgx-header p {{
        font-size: 1rem !important;
        opacity: 0.88 !important;
        margin: 0 !important;
        line-height: 1.6 !important;
        color: white !important;
    }}
   .stMarkdown .pgx-header .pgx-badge {{
        display: inline-block !important;
        background: rgba(255,255,255,0.15) !important;
        border: 2px solid rgba(255,255,255,0.5) !important;
        border-radius: 20px !important;
        padding: 4px 14px !important;
        font-size: 0.78rem !important;
        margin-top: 1rem !important;
        letter-spacing: 0.3px !important;
        color: white !important;
        line-height: 1.6 !important;
    }}
    .stMarkdown .pgx-header span.pgx-badge {{
        color: white !important;
        display: inline-block !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: white;
        border-radius: 10px;
        border: 1px solid {C_LIGHT} !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        padding: 0.4rem 0.2rem;
    }}

    /* Step labels */
    .step-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: {C_DARK};
        margin-bottom: 0.3rem;
    }}

    /* Result row cards */
    .result-row {{
        display: flex;
        align-items: center;
        background: white;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.5rem;
        border: 1px solid {C_LIGHT};
        gap: 1rem;
    }}
    .result-gene {{
        font-weight: 700;
        font-size: 0.9rem;
        color: {C_DARKEST};
        min-width: 90px;
    }}
    .result-name {{
        font-size: 0.8rem;
        color: #555;
        flex: 1;
        font-family: monospace;
    }}
    .result-class {{
        font-weight: 600;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 20px;
        min-width: 160px;
        text-align: center;
    }}
    .result-conf {{
        font-size: 0.82rem;
        color: #444;
        min-width: 70px;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }}

    /* Confidence bar */
    .conf-bar-wrap {{
        background: {C_LIGHTEST};
        border-radius: 4px;
        height: 6px;
        width: 80px;
        overflow: hidden;
        display: inline-block;
        vertical-align: middle;
        margin-right: 6px;
    }}
    .conf-bar-fill {{
        height: 100%;
        border-radius: 4px;
        background: {C_DARK};
    }}

    /* Gene panel badges */
    .gene-badge {{
        display: inline-block;
        background: {C_LIGHTEST};
        color: {C_DARKEST};
        border: 1px solid {C_LIGHT};
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 3px;
        font-family: monospace;
    }}

    /* Warning banner */
    .stMarkdown .pgx-warning {{
        background: #FFF8E7 !important;
        border-left: 4px solid #F39C12 !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.83rem !important;
        color: #7D5A00 !important;
        margin-bottom: 1rem !important;
    }}
    .stMarkdown .pgx-warning p {{
        color: #7D5A00 !important;
        margin: 0 !important;
    }}

    /* Summary metrics */
    .metric-box {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid {C_LIGHT};
    }}
    .metric-number {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.3rem;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {C_MID};
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown a {{
        color: {C_LIGHTEST} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.25) !important;
    }}
    section[data-testid="stSidebar"] code {{
        color: {C_DARKEST} !important;
        background-color: {C_LIGHTEST} !important;
        border-radius: 4px;
        padding: 1px 6px;
        font-weight: 600;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    [data-testid="stHeader"] {{
        background-color: {C_LIGHTEST};
    }}

    /* Button */
    .stButton > button {{
        background-color: {C_DARK};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
        width: 100%;
        transition: background 0.2s;
    }}
    .stButton > button:hover {{
        background-color: {C_DARKEST};
        color: white;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: white;
        border-radius: 10px;
        border: 2px dashed {C_LIGHT};
        padding: 1rem;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border-radius: 8px;
        overflow: hidden;
    }}
    /* Force light-mode readability for plain markdown text and tables,
       regardless of the visitor's device/browser dark-mode setting. */
    :root {{
        color-scheme: light;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {{
        color: #333333;
    }}
    .stMarkdown table {{
        background-color: white !important;
        border: 1px solid #B2D4CC !important;
    }}
    .stMarkdown table th {{
        background-color: #E8F4F1 !important;
        color: #0D4438 !important;
    }}
    .stMarkdown table td {{
        background-color: white !important;
        color: #333333 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(df: pd.DataFrame, model_data: dict):
    model = model_data["model"]
    preprocessor = model_data["preprocessor"]
    label_encoder = model_data["label_encoder"]

    df = df.copy()
    df["vep_impact_ordinal"] = df["vep_impact"].map(VEP_IMPACT_ORDER)

    feature_cols = ["most_severe_consequence", "vep_impact_ordinal",
                    "gnomad_af", "phylop100way", "gerp_score"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return pd.DataFrame()

    X = df[feature_cols].copy()
    X_proc = preprocessor.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_proc = X_proc.astype(np.float64)

    proba = model.predict_proba(X_proc)
    pred_idx = proba.argmax(axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)
    confidence = proba.max(axis=1)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc)
        feature_names = preprocessor.get_feature_names_out()
        if isinstance(shap_values, list):
            shap_for_pred = np.array([shap_values[c][i] for i, c in enumerate(pred_idx)])
        else:
            shap_for_pred = np.array([shap_values[i, :, c] for i, c in enumerate(pred_idx)])
        top_feat_idx = np.abs(shap_for_pred).argmax(axis=1)
        top_features = [feature_names[i].replace("cat__most_severe_consequence_", "")
                        .replace("num__", "") for i in top_feat_idx]
    except Exception:
        top_features = ["N/A"] * len(df)

    results = df.copy()
    results["_pred_class"] = pred_labels
    results["_confidence"] = confidence
    results["_top_feature"] = top_features
    return results, X_proc


def shap_bar_chart(model_data, X_proc):
    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    class_names = list(label_encoder.classes_)
    feature_names = model_data["preprocessor"].get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)

    if isinstance(shap_values, list):
        mean_abs = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])
    else:
        mean_abs = np.abs(shap_values).mean(axis=0).T

    overall = mean_abs.mean(axis=0)
    top_idx = overall.argsort()[::-1][:8]
    labels = [feature_names[i].replace("cat__most_severe_consequence_", "")
              .replace("num__", "") for i in top_idx]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = [C_DARKEST, C_MID, C_DARK]
    bottom = np.zeros(len(top_idx))
    for i, cls in enumerate(class_names):
        ax.barh(range(len(top_idx)), mean_abs[i][top_idx],
                left=bottom, label=cls, color=colors[i % len(colors)], alpha=0.9, height=0.6)
        bottom += mean_abs[i][top_idx]

    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### PGx Predictor")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Phase 2 semi-supervised Random Forest")
    st.markdown("---")

    model_data = load_model()
    if model_data:
        st.markdown("**Training data**")
        st.markdown(f"""
- {model_data['training_n_gold']} CPIC gold-standard variants
- {model_data['training_n_pseudo']} pseudo-labeled variants
- {model_data['training_n_total']} total training variants
        """)
        st.markdown("---")
        st.markdown("**Performance (honest CV)**")
        st.markdown("""
- Accuracy: 0.74
- Macro F1: 0.66
- F1 Decreased function: 0.53
        """)
    st.markdown("---")
    st.markdown("**Gene panel**")
    for g in GENE_PANEL:
        st.markdown(f"`{g}`")
    st.markdown("---")
    st.markdown("[GitHub repository](https://github.com/Dharshana-Karthikeyan/pharmacogene-ml)")
    st.markdown("Research use only. Not for clinical decisions.")

# ── Header ────────────────────────────────────────────────────────────────────
if model_data is None:
    st.error(f"Model file not found. Run step12_shap_phase2.py first.")
    st.stop()

st.markdown("""
<div class="pgx-header">
    <h1>Pharmacogene Functional-Status Predictor</h1>
    <p>Predict CPIC-style functional status for pharmacogene variants using an explainable<br>
    semi-supervised Random Forest model trained on curated star-allele annotations.</p>
    <span class="pgx-badge">Research use only &nbsp;|&nbsp; Not validated for clinical use</span>
</div>
""", unsafe_allow_html=True)

# ── Warning ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pgx-warning">
    Predictions are generated by a machine-learning model and have not been validated for clinical prescribing decisions.
    Always consult CPIC guidelines and a qualified pharmacogenomics specialist before making treatment decisions.
</div>
""", unsafe_allow_html=True)

# ── Step 1: Download example ──────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="step-label">Step 1 — Prepare your file</div>', unsafe_allow_html=True)
    st.markdown("""
Your input CSV must contain these columns — generated by running pipeline steps 5 and 6:

| Column | Description |
|---|---|
| `Name` | Variant name (HGVS or ClinVar name) |
| `GeneSymbol` | Gene symbol (e.g. CYP2D6) |
| `most_severe_consequence` | VEP consequence term |
| `vep_impact` | VEP impact category (HIGH / MODERATE / LOW / MODIFIER) |
| `gnomad_af` | gnomAD allele frequency (leave blank if unknown) |
| `phylop100way` | phyloP100way conservation score |
| `gerp_score` | GERP conservation score |
""")

    st.download_button(
        label="Download example input file",
        data=EXAMPLE_CSV,
        file_name="example_variants.csv",
        mime="text/csv"
    )

# ── Step 2: Upload ────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="step-label">Step 2 — Upload your variant file</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#FFF8E7;border-left:4px solid #F39C12;border-radius:0 8px 8px 0;
         padding:0.6rem 1rem;font-size:0.82rem;color:#7D5A00;margin-bottom:0.8rem;">
        📱 <strong>On mobile?</strong> This app is best experienced on a laptop or desktop.
        If you're on a phone and just want to try it out, use the <strong>Load example data</strong> button below; no file upload needed.
    </div>
    """, unsafe_allow_html=True)

    use_example = st.button("Load example data (mobile-friendly trial)")

    uploaded = st.file_uploader(
        "Select a CSV file",
        type=["csv"],
        label_visibility="collapsed"
    )

    if use_example:
        import io
        df = pd.read_csv(io.StringIO(EXAMPLE_CSV))
        st.markdown(f"**{len(df)} example variants loaded**")
        with st.expander("Preview example data"):
            st.dataframe(df.head(10), use_container_width=True)

    elif uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.markdown(f"**{len(df)} variants loaded** from `{uploaded.name}`")

            with st.expander("Preview uploaded data"):
                st.dataframe(df.head(10), use_container_width=True)

            required_cols = ["most_severe_consequence", "vep_impact",
                             "gnomad_af", "phylop100way", "gerp_score"]
            missing_cols = [c for c in required_cols if c not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}. Run pipeline steps 5-6 first to annotate your variants.")
                st.stop()

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        df = pd.DataFrame()

# ── Step 3: Predict ───────────────────────────────────────────────────────────
if (uploaded is not None or use_example) and not df.empty:
    with st.container(border=True):
        st.markdown('<div class="step-label">Step 3 — Run prediction</div>', unsafe_allow_html=True)
        run = st.button("Run Functional-Status Prediction")

    if run:
        with st.spinner("Running predictions..."):
            output = predict(df, model_data)

        if isinstance(output, tuple):
            results, X_proc = output
        else:
            st.stop()

        if results.empty:
            st.stop()

        st.markdown("---")

        # ── Summary metrics ───────────────────────────────────────────────────
        counts = results["_pred_class"].value_counts()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number" style="color:{C_DARKEST}">{len(results)}</div>
                <div class="metric-label">Variants analysed</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number" style="color:#C0392B">{counts.get('No function', 0)}</div>
                <div class="metric-label">No function</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number" style="color:#D35400">{counts.get('Decreased function', 0)}</div>
                <div class="metric-label">Decreased function</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number" style="color:#1E6B2E">{counts.get('Normal function', 0)}</div>
                <div class="metric-label">Normal function</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Results cards ─────────────────────────────────────────────────────
        st.markdown("#### Predictions")

        name_col = "Name" if "Name" in results.columns else None
        gene_col = "GeneSymbol" if "GeneSymbol" in results.columns else None

        for _, row in results.iterrows():
            cls = row["_pred_class"]
            conf = row["_confidence"]
            info = CLASS_INFO.get(cls, {"color": "#333", "bg": "#f5f5f5",
                                        "border": "#ccc", "description": ""})
            name = row[name_col] if name_col else "—"
            gene = row[gene_col] if gene_col else "—"
            bar_width = int(conf * 100)

            st.markdown(f"""
            <div class="result-row" style="border-left: 4px solid {info['border']};">
                <div class="result-gene">{gene}</div>
                <div class="result-name">{str(name)[:60]}{'...' if len(str(name)) > 60 else ''}</div>
                <div class="result-class" style="background:{info['bg']};color:{info['color']};">{cls}</div>
                <div class="result-conf">
                    <div style="font-size:0.75rem;color:#888;margin-bottom:2px;">
                        Confidence
                    </div>
                    <div>
                        <span class="conf-bar-wrap">
                            <span class="conf-bar-fill" style="width:{bar_width}%;background:{info['color']};"></span>
                        </span>
                        {conf:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Class legend ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Functional class reference**")
        leg_cols = st.columns(3)
        for i, (cls, info) in enumerate(CLASS_INFO.items()):
            with leg_cols[i]:
                st.markdown(f"""
                <div style="background:{info['bg']};border:1px solid {info['border']};
                     border-radius:8px;padding:0.8rem 1rem;">
                    <div style="font-weight:700;color:{info['color']};
                         font-size:0.85rem;margin-bottom:0.3rem;">{cls}</div>
                    <div style="font-size:0.78rem;color:#444;line-height:1.4;">
                        {info['description']}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP chart ────────────────────────────────────────────────────────
        st.markdown("#### Feature importance")
        st.markdown(
            "Which features drove predictions across all uploaded variants, "
            "ranked by mean absolute SHAP value."
        )
        try:
            fig = shap_bar_chart(model_data, X_proc)
            st.pyplot(fig, use_container_width=False)
            plt.close()
        except Exception as e:
            st.warning(f"Feature importance chart could not be generated: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown("#### Download results")
        out_df = results.rename(columns={
            "_pred_class": "Predicted Class",
            "_confidence": "Confidence",
            "_top_feature": "Top Feature"
        }).drop(columns=["vep_impact_ordinal"], errors="ignore")

        csv_out = out_df.to_csv(index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv_out,
            file_name="pgx_predictions.csv",
            mime="text/csv"
        )

else:
    if uploaded is None:
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:2rem;
             text-align:center;border:1px solid {C_LIGHT};color:#888;">
            Upload a variant CSV file above to begin.
        </div>""", unsafe_allow_html=True)
