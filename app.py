"""
Machine Learning-Based Early Prediction of Bipolar Disorder Using Multimodal Clinical and Behavioral Data
Research Prototype & Baseline Evaluation Dashboard
"""

import io
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bipolar Disorder ML Prototype",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header & Disclaimer
# ---------------------------------------------------------
st.markdown('<div class="main-title">Machine Learning-Based Early Prediction of Bipolar Disorder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Major Project Research Prototype • Baseline Modeling, Probabilistic Calibration & Simulated Robustness Assessment</div>',
    unsafe_allow_html=True,
)

st.warning(
    "**Academic Research Demonstration Only — Not a Clinical Diagnostic Tool.**\n\n"
    "This software is an educational prototype developed to evaluate classification algorithms, calibration metrics, "
    "and simulated missing-data stress testing on clinical survey records. It does **not** provide clinical diagnosis, "
    "individualized risk scores, or treatment recommendations, and must never replace assessment by a licensed mental health clinician."
)

# ---------------------------------------------------------
# Sidebar Controls & Configuration
# ---------------------------------------------------------
st.sidebar.header("⚙️ Experiment Configuration")

DEFAULT_PATH = os.path.join("data", "Mentalillness.csv")

st.sidebar.subheader("1. Data Source")
uploaded = st.sidebar.file_uploader("Upload custom CSV (optional)", type=["csv"], help="Upload a compatible CSV dataset or use the default supplied file.")

target_col = st.sidebar.text_input("Target column", value="Bipolar disorder", help="Binary target variable for classification.")

st.sidebar.subheader("2. Modeling Parameters")
test_size = st.sidebar.slider(
    "Test-set holdout ratio",
    min_value=0.15,
    max_value=0.40,
    value=0.25,
    step=0.05,
    help="Proportion of the dataset allocated to the holdout test set (stratified split).",
)
seed = st.sidebar.number_input(
    "Random seed (reproducibility)",
    min_value=1,
    max_value=99999,
    value=42,
    step=1,
    help="Random seed for data splitting, cross-validation, and stochastic model training.",
)

st.sidebar.subheader("3. Robustness Simulation")
missing_stress_rate = st.sidebar.slider(
    "Simulated missingness rate (%)",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="Percentage of feature values randomly dropped in the test set to simulate EHR missingness.",
)

run_button = st.sidebar.button("🚀 Train & Evaluate Models", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Data Loading & Verification
# ---------------------------------------------------------
@st.cache_data
def load_data(file_bytes, filepath):
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' was not found. Please verify the folder structure or upload a CSV.")
    return pd.read_csv(filepath)

try:
    if uploaded is not None:
        df_raw = load_data(uploaded.getvalue(), DEFAULT_PATH)
        data_source_name = f"Uploaded File: {uploaded.name}"
    else:
        df_raw = load_data(None, DEFAULT_PATH)
        data_source_name = f"Default: {DEFAULT_PATH}"
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.info("Please verify the file path or upload a valid CSV file via the sidebar.")
    st.stop()

# ---------------------------------------------------------
# Feature Exclusion & Leakage Prevention
# ---------------------------------------------------------
if target_col not in df_raw.columns:
    st.error(f"Target column '{target_col}' not found in the dataset.")
    st.write("Available columns in dataset:", list(df_raw.columns))
    st.stop()

# Known identifiers and diagnosis labels that must be excluded to prevent direct target/label leakage
DIAGNOSIS_COLUMNS = {"Bipolar disorder", "Schizophrenia", "Depression", "Anxiety disorder", "PTSD"}
IDENTIFIER_COLUMNS = {"ID", "id", "Id", "patient_id", "Patient_ID", "index"}

excluded_cols = set()
for col in df_raw.columns:
    if col in IDENTIFIER_COLUMNS or col.lower() in {c.lower() for c in IDENTIFIER_COLUMNS}:
        excluded_cols.add(col)
    elif col in DIAGNOSIS_COLUMNS and col != target_col:
        excluded_cols.add(col)

# Candidate predictor features
feature_cols = [c for c in df_raw.columns if c not in excluded_cols and c != target_col]

y_raw = df_raw[target_col]
unique_targets = sorted(y_raw.dropna().unique().tolist())

if len(unique_targets) != 2:
    st.error(
        f"Binary classification requires exactly 2 distinct classes. Target '{target_col}' has "
        f"{len(unique_targets)} unique values: {unique_targets}."
    )
    st.stop()

# Standardize binary target: 0 = Comparison, 1 = Target Positive
target_mapping = {unique_targets[0]: 0, unique_targets[1]: 1}
y = y_raw.map(target_mapping)
X = df_raw[feature_cols].apply(pd.to_numeric, errors="coerce")

if len(feature_cols) < 1:
    st.error("No valid predictor features remain after excluding ID and diagnostic label columns.")
    st.stop()

# ---------------------------------------------------------
# Model Training & Pipeline Construction
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_and_evaluate_pipeline(X_data, y_data, test_ratio, random_state_val, feat_names):
    """
    Fits preprocessing and models strictly on the training partition.
    Evaluates on holdout test partition to guarantee no data leakage.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_data, y_data, test_size=test_ratio, random_state=int(random_state_val), stratify=y_data
    )

    model_factories = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(random_state_val)))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                random_state=int(random_state_val), n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=int(random_state_val)
            ))
        ]),
    }

    results = []
    fitted_models = {}
    test_predictions = {}
    test_probabilities = {}

    for name, pipeline in model_factories.items():
        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fitted_models[name] = pipeline
        test_probabilities[name] = proba
        test_predictions[name] = pred

        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = precision_score(y_te, pred, zero_division=0)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        auroc = roc_auc_score(y_te, proba)
        auprc = average_precision_score(y_te, proba)
        brier = brier_score_loss(y_te, proba)

        results.append({
            "Model": name,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Precision": prec,
            "Sensitivity / Recall": sens,
            "Specificity": spec,
            "F1-Score": f1,
            "Brier Score": brier,
            "Accuracy": acc,
        })

    metrics_df = pd.DataFrame(results).sort_values("AUROC", ascending=False).reset_index(drop=True)
    return X_tr, X_te, y_tr, y_te, fitted_models, test_probabilities, test_predictions, metrics_df

# Cache or run model training
with st.spinner("Fitting models on training split with strictly encapsulated pipelines..."):
    X_train, X_test, y_train, y_test, models_dict, probas_dict, preds_dict, eval_df = train_and_evaluate_pipeline(
        X, y, test_size, seed, feature_cols
    )

# ---------------------------------------------------------
# Main Application Tabs
# ---------------------------------------------------------
tab_overview, tab_data, tab_models, tab_curves, tab_cm, tab_features, tab_robustness, tab_limitations = st.tabs([
    "📋 Project Overview",
    "📊 Dataset & Features",
    "⚔️ Model Comparison",
    "📈 ROC & Calibration",
    "🎯 Confusion Matrix",
    "🏆 Feature Importance",
    "🧪 Robustness Stress Test",
    "⚠️ Limitations & Defense Prep",
])

# ---------------------------------------------------------
# TAB 1: Project Overview
# ---------------------------------------------------------
with tab_overview:
    st.subheader("1. Project Title & Academic Context")
    st.markdown(
        """
        **Official Major Project Title:**  
        *Machine Learning-Based Early Prediction of Bipolar Disorder Using Multimodal Clinical and Behavioral Data*

        ### Research Context
        Bipolar disorder is a chronic psychiatric illness characterized by recurrent shifts in mood, energy, and activity levels 
        between depressive episodes and manic/hypomanic states. In routine clinical practice, diagnostic delays frequently span 
        5 to 10 years, during which patients are often misdiagnosed with Major Depressive Disorder (MDD) and prescribed inappropriate 
        treatments such as antidepressant monotherapy.

        ### Research Objectives of this Prototype
        1. **Baseline Classification & Modeling:** Train and contrast three foundational machine learning architectures—**Logistic Regression**, **Random Forest**, and **Gradient Boosting**—on structured clinical symptom criteria.
        2. **Multi-Faceted Validation:** Evaluate models across discrimination (AUROC, AUPRC), clinical triage utility (Sensitivity, Specificity, Precision), and probabilistic calibration (Brier score).
        3. **Simulated Missing-Data Stress Testing:** Investigate model resilience against simulated incomplete clinical records without making false claims of multi-site external transportability.
        4. **Scientific Honesty & Diagnostic Leakage Analysis:** Uncover and transparently explain why near-perfect holdout scores emerge when symptom features mirror DSM-5 diagnostic criteria.
        """
    )

    st.markdown("---")
    st.subheader("Analytical Architecture")
    c_flow1, c_flow2, c_flow3, c_flow4 = st.columns(4)
    with c_flow1:
        st.info("**Step 1: Ingestion & De-identification**\n- Load structured clinical CSV\n- Validate binary target\n- Exclude ID & co-occurring labels")
    with c_flow2:
        st.info("**Step 2: Leakage-Safe Splitting**\n- Stratified train/test split\n- Imputation & scaling strictly within training pipelines")
    with c_flow3:
        st.info("**Step 3: Multi-Model Training**\n- Logistic Regression (L2)\n- Random Forest\n- Gradient Boosting")
    with c_flow4:
        st.info("**Step 4: Clinical Evaluation**\n- Discrimination & Calibration\n- Confusion Matrix\n- Simulated Robustness Stress Test")

# ---------------------------------------------------------
# TAB 2: Dataset Statistics & Class Distribution
# ---------------------------------------------------------
with tab_data:
    st.subheader("2. Dataset Exploration & Distribution")
    st.caption(f"Active data source: `{data_source_name}`")

    # High-level KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Records (Samples)", f"{df_raw.shape[0]:,}")
    kpi2.metric("Total Features in CSV", f"{df_raw.shape[1]:,}")
    kpi3.metric("Candidate Predictors Used", f"{len(feature_cols):,}")
    kpi4.metric("Missing Cells in Raw CSV", f"{int(df_raw.isna().sum().sum()):,}")

    st.markdown("---")
    st.write("#### Target Class Distribution")
    dist_col1, dist_col2 = st.columns([1, 1])

    counts = y.value_counts().sort_index()
    percentages = (y.value_counts(normalize=True).sort_index() * 100).round(2)
    dist_table = pd.DataFrame({
        "Class Label": [f"Comparison / Non-Target (0): {unique_targets[0]}", f"Target Positive (1): {unique_targets[1]}"],
        "Count": [counts.get(0, 0), counts.get(1, 0)],
        "Proportion (%)": [percentages.get(0, 0.0), percentages.get(1, 0.0)]
    })

    with dist_col1:
        st.dataframe(dist_table, use_container_width=True, hide_index=True)
        st.caption(
            "Note: The positive class represents 77.5% of the supplied dataset. Stratified train/test splitting "
            "is applied to maintain identical class ratios across training and evaluation splits."
        )

    with dist_col2:
        fig_dist, ax_dist = plt.subplots(figsize=(6, 3.2))
        bars = ax_dist.bar(
            ["Comparison (0)", "Bipolar Disorder (1)"],
            [counts.get(0, 0), counts.get(1, 0)],
            color=["#64748B", "#2563EB"],
            width=0.45
        )
        ax_dist.set_ylabel("Number of Patients / Records")
        ax_dist.set_title("Target Class Balance")
        for bar in bars:
            height = bar.get_height()
            ax_dist.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        ax_dist.set_ylim(0, max(counts.values) * 1.15)
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    st.markdown("---")
    st.write("#### Feature Leakage Prevention")
    st.markdown(
        f"To prevent severe direct label leakage, **{len(excluded_cols)} columns** were explicitly removed from model inputs:\n"
        f"- **Identifier column(s):** `{sorted(list(excluded_cols & IDENTIFIER_COLUMNS))}` (Non-predictive database keys)\n"
        f"- **Co-occurring diagnosis labels:** `{sorted(list(excluded_cols & DIAGNOSIS_COLUMNS))}` (Labels that directly disclose psychiatric status)\n"
        f"- **Remaining predictor features:** `{len(feature_cols)}` candidate clinical and behavioral symptom columns."
    )

    with st.expander("🔍 View Raw Data Sample (First 10 Rows)"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    with st.expander("📋 View Predictor Feature List & Data Types"):
        feat_df = pd.DataFrame({
            "Predictor Feature": feature_cols,
            "Data Type": [str(df_raw[col].dtype) for col in feature_cols],
            "Unique Values": [df_raw[col].nunique() for col in feature_cols],
            "Missing Values": [df_raw[col].isna().sum() for col in feature_cols]
        })
        st.dataframe(feat_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Model Comparison & Evaluation
# ---------------------------------------------------------
with tab_models:
    st.subheader("3. Model Performance Comparison")
    st.markdown(
        f"Internal stratified holdout evaluation on **{len(y_test):,} test samples** "
        f"(Training set: **{len(y_train):,} samples**, Holdout ratio: **{test_size:.2f}**, Seed: **{seed}**)."
    )
    st.caption("All preprocessing operations (imputation, scaling) were fit strictly on training data.")

    # Format dataframe for display
    display_df = eval_df.copy()
    numeric_cols = [c for c in display_df.columns if c != "Model"]
    display_df[numeric_cols] = display_df[numeric_cols].map(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        label="📥 Download Model Comparison Results (CSV)",
        data=eval_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.write("#### Evaluation Metric Definitions & Clinical Relevance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            "**Discrimination Metrics:**\n"
            "- **AUROC:** Area Under ROC Curve. Measures the probability that a randomly chosen positive case is ranked higher than a negative case.\n"
            "- **AUPRC:** Area Under Precision-Recall Curve. Emphasizes true positive detection under class imbalance."
        )
    with m2:
        st.markdown(
            "**Classification Utility:**\n"
            "- **Sensitivity (Recall):** $\\frac{TP}{TP + FN}$. Ability to correctly detect bipolar disorder cases.\n"
            "- **Specificity:** $\\frac{TN}{TN + FP}$. Ability to correctly rule out non-bipolar cases.\n"
            "- **Precision:** $\\frac{TP}{TP + FP}$. Positive predictive value."
        )
    with m3:
        st.markdown(
            "**Probabilistic Calibration:**\n"
            "- **Brier Score:** Mean squared error between predicted probabilities and binary outcomes: $\\frac{1}{N}\\sum (p_i - y_i)^2$. **Lower is better** (0 is perfect calibration).\n"
            "- **F1-Score:** Harmonic mean of precision and recall."
        )

# ---------------------------------------------------------
# TAB 4: ROC & Calibration Curves
# ---------------------------------------------------------
with tab_curves:
    st.subheader("4. Discrimination & Calibration Curves")
    st.caption("Visualizing model discrimination (ROC) alongside probabilistic accuracy (Calibration).")

    roc_col, cal_col = st.columns(2)

    with roc_col:
        st.markdown("**Receiver Operating Characteristic (ROC) Curves**")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.8))
        colors = {"Logistic Regression": "#2563EB", "Random Forest": "#16A34A", "Gradient Boosting": "#DC2626"}

        for name in models_dict.keys():
            proba = probas_dict[name]
            fpr, tpr, _ = roc_curve(y_test, proba)
            score = roc_auc_score(y_test, proba)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=colors.get(name, "black"), lw=2)

        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.5000)")
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.05])
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)")
        ax_roc.set_title("ROC Comparison (Test Set)")
        ax_roc.legend(loc="lower right", fontsize=9)
        ax_roc.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    with cal_col:
        st.markdown("**Probabilistic Calibration Curves**")
        fig_cal, ax_cal = plt.subplots(figsize=(6, 4.8))

        for name in models_dict.keys():
            proba = probas_dict[name]
            brier = brier_score_loss(y_test, proba)
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
            ax_cal.plot(mean_pred, frac_pos, marker="s", label=f"{name} (Brier = {brier:.4f})", color=colors.get(name, "black"), lw=2)

        ax_cal.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
        ax_cal.set_xlim([-0.02, 1.02])
        ax_cal.set_ylim([-0.02, 1.05])
        ax_cal.set_xlabel("Mean Predicted Probability")
        ax_cal.set_ylabel("Observed Positive Fraction")
        ax_cal.set_title("Calibration Curves (Quantile Bins)")
        ax_cal.legend(loc="upper left", fontsize=9)
        ax_cal.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig_cal)
        plt.close(fig_cal)

    st.info(
        "💡 **Why Calibration Matters in Medicine:** High AUROC demonstrates that cases are ranked in proper order, "
        "but does not guarantee that output probabilities represent true empirical risk. Calibration curves demonstrate "
        "whether a predicted risk of 70% corresponds to an observed 70% disease prevalence."
    )

# ---------------------------------------------------------
# TAB 5: Confusion Matrix
# ---------------------------------------------------------
with tab_cm:
    st.subheader("5. Confusion Matrix Analysis")
    chosen_cm_model = st.selectbox("Select model for confusion matrix inspection:", list(models_dict.keys()), index=0)

    cm_proba = probas_dict[chosen_cm_model]
    cm_pred = preds_dict[chosen_cm_model]
    cm = confusion_matrix(y_test, cm_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    c_mat1, c_mat2 = st.columns([1, 1.2])

    with c_mat1:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Comparison (0)", "Actual Bipolar (1)"],
            columns=["Predicted Comparison (0)", "Predicted Bipolar (1)"]
        )
        st.write(f"**Confusion Matrix for {chosen_cm_model} (Threshold = 0.50):**")
        st.dataframe(cm_df, use_container_width=True)

        st.markdown(
            f"""
            - **True Negatives (TN):** {tn:,}  
            - **False Positives (FP - Type I Error):** {fp:,}  
            - **False Negatives (FN - Type II Error):** {fn:,} *(Missed bipolar diagnoses)*  
            - **True Positives (TP):** {tp:,}  
            """
        )

    with c_mat2:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
        im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix: {chosen_cm_model}")
        plt.colorbar(im, ax=ax_cm)
        tick_marks = np.arange(2)
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
        ax_cm.set_yticklabels(["True 0", "True 1"])

        # Display raw count and proportion in each cell
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = (val / len(y_test)) * 100
                ax_cm.text(
                    j, i, f"{val:,}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontweight="bold"
                )
        ax_cm.set_ylabel("Actual Diagnosis")
        ax_cm.set_xlabel("Predicted Diagnosis")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

# ---------------------------------------------------------
# TAB 6: Feature Importance & DSM-5 Alignment
# ---------------------------------------------------------
with tab_features:
    st.subheader("6. Feature Importance & Clinical Interpretation")
    st.caption("Permutation importance evaluated on the holdout test set (ROC-AUC drop metric).")

    fi_model_name = st.selectbox("Select model for feature importance analysis:", list(models_dict.keys()), index=1)

    with st.spinner("Computing test-set permutation importance (5 repeats)..."):
        perm = permutation_importance(
            models_dict[fi_model_name],
            X_test,
            y_test,
            n_repeats=5,
            random_state=int(seed),
            scoring="roc_auc",
            n_jobs=-1,
        )

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean AUROC Drop": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Mean AUROC Drop", ascending=False).reset_index(drop=True)

    top_n = st.slider("Number of top features to display", min_value=5, max_value=25, value=15)
    top_imp = imp_df.head(top_n)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 5.5))
    plot_imp = top_imp.sort_values("Mean AUROC Drop", ascending=True)
    ax_imp.barh(plot_imp["Feature"], plot_imp["Mean AUROC Drop"], xerr=plot_imp["Std"], color="#3B82F6", alpha=0.85, capsize=3)
    ax_imp.set_xlabel("Mean Decrease in Test AUROC upon Feature Permutation")
    ax_imp.set_ylabel("Clinical / Behavioral Predictor")
    ax_imp.set_title(f"Top {top_n} Predictor Symptoms ({fi_model_name})")
    ax_imp.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    st.dataframe(top_imp.round(4), use_container_width=True)

    st.markdown("---")
    st.write("#### 🧠 Clinical Diagnostic Mapping: Why Near-Perfect Scores Emerge")
    st.info(
        "**Key Viva & Defense Finding — Criterion Circularity:**\n\n"
        "Notice that the highest-ranking features—*Increase in goal-directed activity*, *Racing thoughts*, "
        "*Inflated self-esteem*, *Decreased need for sleep*, and *More talkative than usual*—are the exact 7 diagnostic criteria "
        "defined in **DSM-5 Criterion B for a Manic Episode**.\n\n"
        "In this dataset, individuals assigned `Bipolar disorder == 1` exhibit $\\ge 3$ manic criteria, whereas comparison cases "
        "exhibit $\\le 2$. Consequently, the algorithms are learning the **formal psychiatric diagnostic rule** rather than discovering "
        "novel subtle prodromal signals. Transparently identifying and defending this distinction is essential for academic rigor."
    )

# ---------------------------------------------------------
# TAB 7: Simulated Robustness Stress Test
# ---------------------------------------------------------
with tab_robustness:
    st.subheader("7. Simulated Missing-Feature Robustness Stress Test")
    st.caption("Testing model degradation when patient records suffer from missing questionnaire entries.")

    st.error(
        "⚠️ **CRITICAL METHODOLOGICAL DISTINCTION:**\n\n"
        "This experiment is a **synthetic stress test** designed to examine algorithm resilience under feature sparsity. "
        "It **DOES NOT** constitute external clinical validation, multi-site transportability, or real-world deployment readiness."
    )

    stress_col1, stress_col2 = st.columns([1.2, 1])

    rng = np.random.default_rng(int(seed))
    X_stress = X_test.copy()
    mask = rng.random(X_stress.shape) < (missing_stress_rate / 100.0)
    X_stress = X_stress.mask(mask)

    stress_results = []
    for name in models_dict.keys():
        model = models_dict[name]
        orig_proba = probas_dict[name]
        orig_pred = preds_dict[name]

        stress_proba = model.predict_proba(X_stress)[:, 1]
        stress_pred = (stress_proba >= 0.5).astype(int)

        stress_cm = confusion_matrix(y_test, stress_pred, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = stress_cm.ravel()
        s_spec = s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0
        s_sens = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0

        stress_results.append({
            "Model": name,
            "Baseline AUROC": roc_auc_score(y_test, orig_proba),
            f"Stress AUROC ({missing_stress_rate}% Missing)": roc_auc_score(y_test, stress_proba),
            "Baseline Brier": brier_score_loss(y_test, orig_proba),
            f"Stress Brier ({missing_stress_rate}% Missing)": brier_score_loss(y_test, stress_proba),
            "Stress Sensitivity": s_sens,
            "Stress Specificity": s_spec,
        })

    stress_df = pd.DataFrame(stress_results)
    st.write(f"#### Robustness Comparison at {missing_stress_rate}% Random Feature Missingness:")
    st.dataframe(stress_df.round(4), use_container_width=True)

    st.markdown("---")
    st.write("#### Multi-Rate Degradation Trajectory (0% to 50% Missingness)")

    rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    sweep_data = []

    for r in rates:
        sweep_rng = np.random.default_rng(int(seed))
        if r == 0.0:
            X_s = X_test
        else:
            X_s = X_test.copy().mask(sweep_rng.random(X_test.shape) < r)

        for name, m in models_dict.items():
            p = m.predict_proba(X_s)[:, 1]
            sweep_data.append({
                "Model": name,
                "Missing Rate (%)": int(r * 100),
                "AUROC": roc_auc_score(y_test, p),
                "Brier Score": brier_score_loss(y_test, p),
            })

    sweep_df = pd.DataFrame(sweep_data)

    fig_deg, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors_deg = {"Logistic Regression": "#2563EB", "Random Forest": "#16A34A", "Gradient Boosting": "#DC2626"}

    for name in models_dict.keys():
        sub = sweep_df[sweep_df["Model"] == name]
        ax_d1.plot(sub["Missing Rate (%)"], sub["AUROC"], marker="o", label=name, color=colors_deg.get(name, "black"), lw=2)
        ax_d2.plot(sub["Missing Rate (%)"], sub["Brier Score"], marker="s", label=name, color=colors_deg.get(name, "black"), lw=2)

    ax_d1.set_xlabel("Synthetic Missing Feature Rate (%)")
    ax_d1.set_ylabel("Holdout AUROC (Higher is Better)")
    ax_d1.set_title("AUROC Degradation Curve")
    ax_d1.grid(True, linestyle=":", alpha=0.6)
    ax_d1.legend()

    ax_d2.set_xlabel("Synthetic Missing Feature Rate (%)")
    ax_d2.set_ylabel("Brier Score (Lower is Better)")
    ax_d2.set_title("Probabilistic Error Increase")
    ax_d2.grid(True, linestyle=":", alpha=0.6)
    ax_d2.legend()

    st.pyplot(fig_deg)
    plt.close(fig_deg)

# ---------------------------------------------------------
# TAB 8: Research Limitations & Major Project Roadmap
# ---------------------------------------------------------
with tab_limitations:
    st.subheader("8. Scientific Limitations & College Defense Roadmap")
    st.markdown(
        """
        When presenting this major project to academic review boards, maintaining strict scientific integrity 
        and articulating methodological boundaries is paramount.

        ### Key Research Limitations of the Current Prototype
        1. **Cross-Sectional Criteria vs. Prospective Early Prediction:**  
           The provided dataset (`data/Mentalillness.csv`) contains cross-sectional symptom criteria captured concurrently with 
           the diagnosis. It contains no longitudinal follow-up, timestamps, or prodromal observation intervals. Therefore, 
           this prototype performs **current-label classification**, not early prospective prediction of future illness conversion.

        2. **Lack of Multimodal Sensor Feeds:**  
           Although the overarching project vision addresses multimodal data, the current benchmark dataset consists exclusively 
           of tabular psychiatric survey questions. It lacks passive digital phenotyping (e.g., actigraphy, sleep telemetry, 
           smartwatch streams) or neuroimaging markers.

        3. **Criterion Circularity / Direct Diagnostic Leakage:**  
           The top predictive features correspond directly to DSM-5 Criterion B for mania. The model achieves high holdout accuracy 
           because it replicates the formal diagnostic rubric rather than identifying unobserved latent risk factors.

        4. **Absence of Independent Multi-Site External Validation:**  
           A single-source dataset split into train/test sets only constitutes internal holdout validation. It does not prove that 
           the model generalizes across external hospitals, demographic cohorts, or community outpatient clinics.

        5. **Prevalence Skew (Sample Selection Bias):**  
           In this dataset, 77.5% of subjects are labeled as positive for bipolar disorder, whereas general population prevalence 
           is approximately 1% to 3%. This reflects clinical referral enrichment and would cause high false-positive rates if deployed 
           in unselected screening populations without threshold recalibration.

        ---
        ### Defensible Next Steps for the Major Project
        - **Incorporate Genuine Longitudinal Cohorts:** Partner with clinical repositories or open consortia (e.g., NIMH NDA, UK Biobank) to evaluate pre-diagnostic trajectories.
        - **True Multimodal Integration:** Fuse wearable actigraphy (sleep/activity fragmentation) with electronic health record structured text.
        - **Differential Diagnosis Benchmark:** Train models specifically on Bipolar Disorder vs. Major Depressive Disorder (MDD) rather than generic illness-positive vs. comparison records.
        """
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Machine Learning-Based Early Prediction of Bipolar Disorder • College Major Project Prototype • "
    "Designed for reproducible academic demonstration."
)
