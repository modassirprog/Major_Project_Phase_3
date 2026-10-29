# Machine Learning-Based Early Prediction of Bipolar Disorder Using Multimodal Clinical and Behavioral Data

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow.svg)]()

> **Academic Research Prototype Disclaimer:**  
> This application is developed strictly as an educational and academic demonstration for a university major project. It does **not** provide a clinical diagnosis, calculate individual medical risk scores, or replace formal psychiatric evaluation by a licensed healthcare professional.

---

## 1. Project Overview & Scope

Bipolar disorder is a severe mood disorder marked by cyclical episodes of mania/hypomania and major depression. Diagnostic delay in clinical settings averages 5 to 10 years, frequently leading to misdiagnosis (most commonly unipolar depression) and inappropriate clinical interventions.

This project implements a reproducible machine learning benchmark and interactive research dashboard to:
1. Establish a supervised baseline using structured clinical criteria from `data/Mentalillness.csv`.
2. Compare linear and non-linear machine learning architectures across discrimination, screening utility, and probabilistic calibration.
3. Quantify algorithm resilience against missing clinical records via a controlled synthetic stress test.
4. Transparently demonstrate the phenomenon of **criterion circularity** in psychiatric machine learning.

### Scope & Methodological Transparency Note
While the overarching major project aims toward prospective early prediction using continuous multimodal streams (e.g., actigraphy, sleep telemetry, passive mobile sensors), the current benchmark dataset (`Mentalillness.csv`) consists of **cross-sectional, tabular symptom survey criteria**. 
- It does **not** contain longitudinal time-series, timestamps, or wearable sensors.
- It performs **current-state criteria classification**, not prospective prediction of future illness conversion.
- It does **not** represent external hospital validation, as data originates from a single source.

---

## 2. Methodology & Modeling Pipeline

### A. Data Preprocessing & Target Definition
- **Data File:** `data/Mentalillness.csv` (3,753 patient records, 59 total columns).
- **Target Variable:** `Bipolar disorder` (Binary: `0` = Comparison/Non-target, `1` = Bipolar disorder positive).
  - Class Distribution: `0`: 844 samples (22.49%), `1`: 2,909 samples (77.51%).
- **Leakage Prevention Exclusions:**
  - `ID`: Non-clinical database identifier.
  - Co-occurring diagnosis labels: `Schizophrenia`, `Depression`, `Anxiety disorder`, `PTSD` (excluded to eliminate direct label leakage).
  - **Predictor Features:** 53 candidate clinical and behavioral symptom columns.

### B. Leakage-Safe Pipeline Encapsulation
To ensure strict mathematical reproducibility and eliminate data leakage:
1. **Stratified Holdout Split:** A stratified train/test split (default 75% train, 25% test) preserves class ratios across splits under a configurable random seed (`random_state=42`).
2. **Encapsulated Preprocessing:** All preprocessing steps are packaged inside scikit-learn `Pipeline` objects and fitted **strictly on the training partition**:
   - `SimpleImputer(strategy="median")`: Median feature imputation fitted on training data only.
   - `StandardScaler()`: Feature z-score normalization fitted on training data only (applied to Logistic Regression).

### C. Model Architectures Evaluated
1. **Logistic Regression (L2 Regularized):**
   - Serves as the interpretable linear baseline.
   - Configured with `class_weight="balanced"`, `max_iter=2000`, and L2 penalization.
2. **Random Forest Classifier:**
   - Non-linear bagged ensemble of 200 decision trees.
   - Configured with `min_samples_leaf=2`, `class_weight="balanced"`, and out-of-fold feature bagging.
3. **Gradient Boosting Classifier:**
   - Sequential boosting ensemble optimizing deviance loss with shrinkage.
   - Configured with 100 boosting stages, `learning_rate=0.1`, and `max_depth=3`.

---

## 3. Evaluation Metrics

Models are evaluated comprehensively across three diagnostic dimensions:

| Category | Metric | Formula / Definition | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **Discrimination** | **AUROC** | Area Under Receiver Operating Characteristic Curve | Probability that a positive case is ranked higher than a negative case across all decision thresholds. |
| | **AUPRC** | Area Under Precision-Recall Curve | Summary of precision across recall thresholds; robust to class imbalance. |
| **Clinical Utility** | **Sensitivity (Recall)** | $\frac{TP}{TP + FN}$ | Ability to detect true bipolar cases (minimizing dangerous false negatives). |
| | **Specificity** | $\frac{TN}{TN + FP}$ | Ability to correctly identify comparison cases (minimizing false alarms). |
| | **Precision** | $\frac{TP}{TP + FP}$ | Positive predictive value of an assigned bipolar label. |
| | **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic balance between precision and recall. |
| | **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall proportion of correct classifications. |
| **Calibration** | **Brier Score** | $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$ | Mean squared error of predicted probabilities. **Lower is better** (0 = perfect calibration). |
| | **Calibration Curve** | Observed fraction vs. mean predicted probability | Visual check of whether a 70% risk estimate reflects 70% disease rate. |

---

## 4. Simulated Robustness Experiment (Missing EHR Features)

In real-world clinical environments, electronic health records (EHR) and psychiatric questionnaires frequently contain missing entries.

The prototype includes a **Simulated Robustness Stress Test**:
- Randomly drops a configurable fraction of test-set feature values (e.g., 10%, 20%, 30%, 40%, 50%).
- Passes degraded records through the pipeline's trained imputer without retraining.
- Tracks performance degradation curves across AUROC and Brier score.
- **Important Note:** This is explicitly labeled as a synthetic data degradation experiment, **not** external multi-center clinical validation.

---

## 5. Key Research Finding: Criterion Circularity

When inspecting permutation feature importance, the top predictive symptoms consistently emerge as:
1. *Increase in goal-directed activity*
2. *Racing thoughts*
3. *Inflated self-esteem*
4. *Decreased need for sleep*
5. *More talkative than usual*

**Clinical Context:** These 5 symptoms—along with distractibility and risky behavior—constitute the exact 7 symptoms in **DSM-5 Criterion B for a Manic Episode**. 

In `Mentalillness.csv`, every record with `Bipolar disorder == 1` has $\ge 3$ manic symptoms, while all comparison cases have $\le 2$. Consequently:
- The algorithms achieve near-perfect holdout performance because they reconstruct the deterministic diagnostic rubric encoded in the survey.
- High performance reflects **criterion circularity**, not early prediction or latent biomarker discovery.
- Articulating this distinction demonstrates scientific rigor during project viva and defense presentations.

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git (optional)

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd d:\Download-D\bipolar_ml_prototype
```

### Step 2: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

**Windows (Command Prompt `cmd`):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```
Or directly using the virtual environment executable:
```powershell
.venv\Scripts\streamlit.exe run app.py
```

### Step 5: Access the Dashboard
Streamlit will launch the web application at:
```
Local URL: http://localhost:8501
```

---

## 7. Project Structure

```
bipolar_ml_prototype/
│
├── data/
│   └── Mentalillness.csv        # Supplied clinical symptom dataset (3,753 rows x 59 cols)
│
├── app.py                       # Full-featured Streamlit research dashboard
├── requirements.txt             # Project dependencies (streamlit, scikit-learn, etc.)
├── README.md                    # Project documentation, methodology & user guide
├── PROJECT_SUMMARY.md           # Concise research scope and academic disclaimer
└── .gitignore                   # Standard Python gitignore rules
```

---

## 8. Limitations & Major Project Defense Roadmap

1. **Cross-Sectional Nature:** The data records existing diagnostic symptom checklists; it cannot predict prospective disease onset or transition from depression to bipolar disorder.
2. **No Continuous Multimodal Data:** Wearable telemetry, speech acoustics, and neuroimaging are not present in this tabular benchmark.
3. **Internal vs. External Validation:** A single-dataset split proves internal consistency, but cannot guarantee generalizability across distinct clinical institutions.
4. **Prevalence Enrichment:** The dataset has a 77.5% bipolar positive prevalence, reflecting clinical selection bias compared to the 1–3% general population prevalence.
5. **Ethical Bounds:** AI models in mental health should assist clinicians in triage and record organization, never replace human psychiatric interviews.

### Recommended Next Steps for Major Project Completion
- Benchmark Bipolar Disorder specifically against Major Depressive Disorder (MDD) rather than general psychiatric controls.
- Secure access to longitudinal open-access datasets (e.g., NIMH National Data Archive, UK Biobank) to evaluate pre-morbid biomarkers.
- Integrate wearable actigraphy streams for digital phenotyping of circadian sleep-wake disruption.

---

## 9. Academic Reference

This research prototype's methodology is motivated by:
> Shi et al., *"Data Diversity vs. Model Complexity in the Prediction of Pediatric Bipolar Disorder: Evidence from Academic and Community Clinical Samples."*  
> [PubMed Reference](https://pubmed.ncbi.nlm.nih.gov/41929287/)
