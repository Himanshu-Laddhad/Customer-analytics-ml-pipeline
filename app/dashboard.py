import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KKBox Churn + CLV Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0F52BA"
GREEN   = "#1B5E20"
RED     = "#B71C1C"

ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data loaders  (all cached — zero computation at runtime)
# ---------------------------------------------------------------------------

@st.cache_data
def load_model_comparison():
    return pd.read_csv(OUTPUTS / "model_comparison.csv")

@st.cache_data
def load_segment_profiles():
    return pd.read_csv(OUTPUTS / "segment_profiles.csv")

@st.cache_data
def load_roi_summary():
    return pd.read_csv(OUTPUTS / "roi_summary.csv")

@st.cache_data
def load_clv_model_comparison():
    return pd.read_csv(OUTPUTS / "clv_model_comparison.csv")

@st.cache_data
def load_optimal_threshold():
    data = {}
    with open(OUTPUTS / "optimal_threshold.txt") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                data[k.strip()] = v.strip()
    return data

@st.cache_data
def load_image(fname):
    with open(OUTPUTS / fname, "rb") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
pages = [
    "📋 Overview",
    "👥 Behavioural Segments",
    "🎯 Churn Model",
    "🔍 SHAP Explainability",
    "💰 Business ROI",
]
page = st.sidebar.radio("Navigate", pages)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**KKBox Churn + CLV**\n\n"
    "Asia's largest music streaming platform. "
    "970k labelled customers, March 2017 churn labels.\n\n"
    "_Stack: DuckDB · XGBoost · LightGBM · Optuna · SHAP · Streamlit_"
)

# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "📋 Overview":
    st.title("KKBox Churn + CLV Prediction")
    st.markdown(
        "This pipeline predicts which KKBox subscribers will churn within 30 days "
        "and estimates their lifetime value — enabling the retention team to target "
        "high-value at-risk customers with a quantified expected return on each offer. "
        "Six models were evaluated (LR, RF, XGBoost, LightGBM, Stacking) on a "
        "temporally split dataset; XGBoost was selected as the final model "
        "(val ROC-AUC 0.9815). SHAP values identify behavioural engagement features "
        "as the primary churn drivers, and an EV-based ROI model quantifies the "
        "campaign value of each customer segment."
    )

    # KPI tiles
    thresh_data  = load_optimal_threshold()
    mc           = load_model_comparison()
    roi          = load_roi_summary()
    best_auc     = mc["Val ROC-AUC"].max()
    net_roi_row  = roi[roi["segment"] == "ALL (high-value target)"].iloc[0]
    net_roi_val  = net_roi_row["net_roi"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers Analysed", "970,960")
    k2.metric("Test Set Churn Rate", "8.5%")
    k3.metric("Best Model ROC-AUC", f"{best_auc:.4f}")
    k4.metric("Campaign Net ROI", f"${net_roi_val/1_000:.0f}k")

    st.markdown("---")

    # Model comparison bar chart
    st.subheader("Model Comparison — Validation ROC-AUC")
    colours = [GREEN if m == "XGBoost" else PRIMARY for m in mc["Model"]]
    fig = go.Figure(go.Bar(
        x=mc["Model"],
        y=mc["Val ROC-AUC"],
        marker_color=colours,
        text=mc["Val ROC-AUC"].map(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0.88, 0.99], title="Val ROC-AUC"),
        xaxis_title="Model",
        plot_bgcolor="white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "XGBoost (highlighted) achieved the best individual-model AUC (0.9815); "
        "the Stacking Ensemble gained only +0.0002 AUC at 45× training cost, "
        "so XGBoost was selected as the production model."
    )

# ===========================================================================
# PAGE 2 — BEHAVIOURAL SEGMENTS
# ===========================================================================
elif page == "👥 Behavioural Segments":
    st.title("Behavioural Segments")
    st.markdown(
        "K-Means (k=4) on PCA-reduced engagement features identified four distinct "
        "customer archetypes. Segment assignment is the first step in the retention "
        "prioritisation framework: **Segment → Churn Risk → CLV → Campaign ROI**."
    )

    sp = load_segment_profiles()

    # Segment profile table
    st.subheader("Segment Profiles")
    display_cols = {
        "segment_label":         "Segment",
        "n_customers":           "Customers",
        "churn_rate_pct":        "Churn Rate (%)",
        "avg_daily_secs":        "Avg Daily Secs",
        "completion_rate":       "Completion Rate",
        "skip_rate":             "Skip Rate",
        "active_days":           "Active Days",
        "days_since_last_log":   "Days Since Last Log",
    }
    sp_display = sp[list(display_cols.keys())].rename(columns=display_cols)
    sp_display["Avg Daily Secs"]     = sp_display["Avg Daily Secs"].map("{:,.0f}".format)
    sp_display["Completion Rate"]    = sp_display["Completion Rate"].map("{:.2%}".format)
    sp_display["Skip Rate"]          = sp_display["Skip Rate"].map("{:.2%}".format)
    sp_display["Active Days"]        = sp_display["Active Days"].map("{:,.0f}".format)
    sp_display["Days Since Last Log"]= sp_display["Days Since Last Log"].map("{:,.0f}".format)
    sp_display["Customers"]          = sp_display["Customers"].map("{:,}".format)
    st.dataframe(sp_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Rate per Segment")
        fig = px.bar(
            sp, x="segment_label", y="churn_rate_pct",
            color="segment_label",
            color_discrete_sequence=[PRIMARY, GREEN, RED, "#FF8F00"],
            labels={"segment_label": "Segment", "churn_rate_pct": "Churn Rate (%)"},
            text="churn_rate_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Churn rates are surprisingly similar across segments; "
            "Power Users churn at nearly the same rate as Casual Listeners "
            "— but Power Users represent far higher revenue at risk."
        )

    with col_b:
        st.subheader("PCA Scatter — Segment Clusters")
        st.image(load_image("05_pca_scatter.png"), use_container_width=True)
        st.caption(
            "PCA projection of the engagement feature space. "
            "Dormant customers (near-zero activity) form a tight cluster bottom-left; "
            "Power Users spread across high-engagement zones top-right."
        )

# ===========================================================================
# PAGE 3 — CHURN MODEL
# ===========================================================================
elif page == "🎯 Churn Model":
    st.title("Churn Model Evaluation")
    thresh_data = load_optimal_threshold()
    opt_thr     = float(thresh_data["optimal_threshold"])
    f1          = float(thresh_data["f1"])
    precision   = float(thresh_data["precision"])
    recall      = float(thresh_data["recall"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimal Threshold", f"{opt_thr:.2f}")
    m2.metric("F1 @ Optimal", f"{f1:.4f}")
    m3.metric("Precision", f"{precision:.4f}")
    m4.metric("Recall (Sensitivity)", f"{recall:.4f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ROC Curves — All Models")
        st.image(load_image("10_roc_curves_all.png"), use_container_width=True)
        st.caption(
            "All tree models (RF, XGB, LGBM, Stacking) significantly outperform "
            "the Logistic Regression baseline; XGBoost and Stacking are nearly "
            "indistinguishable, justifying the simpler model choice."
        )

    with col_b:
        st.subheader("Precision-Recall Curve — XGBoost")
        st.image(load_image("10_pr_curve.png"), use_container_width=True)
        st.caption(
            "The PR curve is most informative for imbalanced datasets (8.5% churn). "
            "The model maintains >80% precision up to ~60% recall before the "
            "precision-recall trade-off becomes significant."
        )

    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Confusion Matrix at Optimal Threshold")
        st.image(load_image("10_confusion_matrix_comparison.png"), use_container_width=True)
        st.caption(
            "At threshold 0.45 the model reduces false negatives (missed churners) "
            "vs the 0.50 default — reducing the most expensive business error "
            f"(FN cost = ${thresh_data['fn_cost_per_unit']}/customer)."
        )

    with col_d:
        st.subheader("Threshold vs Business Cost")
        st.image(load_image("10_threshold_optimisation.png"), use_container_width=True)
        st.caption(
            "Business cost (FN×$120 + FP×$15) is minimised at threshold 0.45. "
            "Lowering the threshold below 0.35 rapidly increases false positives "
            "without meaningfully reducing false negatives."
        )

# ===========================================================================
# PAGE 4 — SHAP EXPLAINABILITY
# ===========================================================================
elif page == "🔍 SHAP Explainability":
    st.title("SHAP Explainability")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) decomposes each prediction into "
        "per-feature contributions, revealing which behavioural signals drive "
        "churn risk — and by how much — for every individual customer."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Global Feature Importance (Mean |SHAP|)")
        st.image(load_image("12_shap_bar.png"), use_container_width=True)
        st.caption(
            "Days since last log-in is the dominant churn driver — "
            "inactive customers are overwhelmingly the ones who churn. "
            "This gives the retention team a simple, actionable early-warning trigger."
        )

    with col_b:
        st.subheader("SHAP Beeswarm — 10k Test Sample")
        st.image(load_image("12_shap_beeswarm.png"), use_container_width=True)
        st.caption(
            "Each dot is one customer. Red = high feature value, blue = low. "
            "High days_since_last_log (red, right) strongly pushes toward churn, "
            "while high completion_rate (red, left) acts as a protective factor."
        )

    st.markdown("---")
    st.subheader("Individual Explanation — Highest Churn Risk Customer")
    st.image(load_image("12_waterfall_case_A.png"), use_container_width=True)
    st.caption(
        "The waterfall plot shows how each feature shifts this customer's prediction "
        "from the baseline (average churn rate) toward the final predicted probability. "
        "This customer is predicted to churn because multiple disengagement signals "
        "(high inactivity, low completion) compound each other."
    )

    st.markdown("---")
    st.subheader("SHAP — CLV Model Feature Importance")
    st.image(load_image("12_shap_clv_bar.png"), use_container_width=True)
    st.caption(
        "For the CLV model, payment-related features (total_paid, avg_payment) "
        "dominate — customers who have historically paid more are predicted to "
        "generate higher annual revenue going forward."
    )

# ===========================================================================
# PAGE 5 — BUSINESS ROI
# ===========================================================================
elif page == "💰 Business ROI":
    st.title("Business ROI Frame")
    st.markdown(
        "Expected value (EV) model: for each high-value predicted churner, "
        "**EV = P(churn) × predicted_CLV × retention_rate − offer_cost**. "
        "Assumptions: 25% retention rate, \\$8 offer cost per customer."
    )

    roi = load_roi_summary()
    overall = roi[roi["segment"] == "ALL (high-value target)"].iloc[0]
    seg_rows = roi[roi["segment"] != "ALL (high-value target)"].copy()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Target Segment Size",   f"{int(overall['n_predicted_churners']):,}")
    r2.metric("Mean CLV (Target)",      f"${overall['mean_clv']:,.0f}")
    r3.metric("Campaign Cost",          f"${overall['campaign_cost']:,.0f}")
    r4.metric("Net ROI",                f"${overall['net_roi']/1_000:.0f}k")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("ROI Sensitivity Heatmap")
        st.image(load_image("13_roi_sensitivity.png"), use_container_width=True)
        st.caption(
            "Net ROI ($k) across retention rate (10–40%) and offer cost ($4–$20). "
            "The campaign remains profitable under even conservative assumptions "
            "(15% retention, \\$12 offer cost)."
        )

    with col_b:
        st.subheader("Expected Net ROI by Segment")
        colours = [GREEN if v >= 0 else RED for v in seg_rows["net_roi"]]
        fig = go.Figure(go.Bar(
            x=seg_rows["segment"],
            y=seg_rows["net_roi"] / 1000,
            marker_color=colours,
            text=(seg_rows["net_roi"] / 1000).map(lambda x: f"${x:.0f}k"),
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Net ROI ($k)",
            xaxis_title="Segment",
            plot_bgcolor="white",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Casual Listeners yield the highest absolute ROI ($1.56M) due to volume; "
            "At-Risk customers have the highest churn probability (84%) making them "
            "the most urgent per-customer retention priority."
        )

    st.markdown("---")
    st.subheader("Segment ROI Detail")
    display = seg_rows[["segment", "n_predicted_churners", "mean_clv", "mean_churn_prob", "net_roi", "roi_pct"]].copy()
    display.columns = ["Segment", "Predicted Churners", "Mean CLV ($)", "Mean Churn Prob", "Net ROI ($)", "ROI (%)"]
    display["Mean CLV ($)"]        = display["Mean CLV ($)"].map("${:,.0f}".format)
    display["Mean Churn Prob"]     = display["Mean Churn Prob"].map("{:.1%}".format)
    display["Net ROI ($)"]         = display["Net ROI ($)"].map("${:,.0f}".format)
    display["ROI (%)"]             = display["ROI (%)"].map("{:,.0f}%".format)
    display["Predicted Churners"]  = display["Predicted Churners"].map("{:,}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)
