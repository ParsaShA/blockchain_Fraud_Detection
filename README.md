# Ethereum Fraud Detection

> A comprehensive machine learning project for detecting and analyzing potentially fraudulent Ethereum accounts through **data cleaning, supervised classification, unsupervised clustering, and anomaly detection**.

## 👥 Team Contributions

This project was developed collaboratively, with each team member responsible for a distinct part of the pipeline:

| Team Member | Main Contribution |
|---|---|
| **Mehraveh** | Data cleaning & preprocessing; exploratory data analysis; unsupervised learning and anomaly detection; clustering evaluation and visualization |
| **Arian** | Supervised learning pipeline; leakage-aware preprocessing; class-imbalance handling; hyperparameter tuning; final model comparison and evaluation |
| **Parsa** | Feature engineering; feature selection; UMAP-based dimensionality reduction; Optuna optimization; additional supervised and unsupervised experimentation; model/artifact preparation |

> **Note on overlapping work:** Some experiments were developed in more than one notebook during the project. The sections below identify the primary owner of the final workflow or analysis presented in each section.

---

## 📌 Project Overview

Cryptocurrency fraud detection is a challenging machine learning problem because fraudulent accounts form a minority of the dataset and their behavior can overlap substantially with legitimate activity.

Our project approaches Ethereum fraud detection from **two complementary perspectives**:

1. **Supervised learning** — use the known `FLAG` label to learn a classifier that predicts whether an account is fraudulent.
2. **Unsupervised learning** — ignore `FLAG` during model development and investigate whether fraudulent behavior naturally emerges as clusters or anomalies.

The overall workflow is:

```text
Raw Ethereum Dataset
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Feature Engineering & Redundancy Reduction
        │
        ├──────────────────────────────┐
        ▼                              ▼
Supervised Learning            Unsupervised Learning
        │                              │
        ▼                              ▼
Classification               Clustering + Anomaly Detection
        │                              │
        ▼                              ▼
Fraud Prediction             Fraud Pattern Discovery
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Comparative Analysis
```

---

# 1. Dataset

### Dataset Source — Parsa

The Ethereum fraud-detection dataset was downloaded from Kaggle and contains account-level transaction and ERC20-related behavioral features together with a `FLAG` indicating fraudulent activity.

The final cleaned dataset used in the modeling workflows contains:

- **9,295 account observations**
- **34 columns**
- **31 numerical features**
- **2 categorical features**
- **1 binary target (`FLAG`)**
- **7,639 legitimate accounts (82.18%)**
- **1,656 fraudulent accounts (17.82%)**

The class distribution makes this an imbalanced classification problem, so metrics such as **PR-AUC, precision, recall, and F1-score** are more informative than accuracy alone.

### Target Variable

```text
FLAG = 0 → Legitimate
FLAG = 1 → Fraudulent
```

---

# 2. Data Cleaning & Preprocessing

**Primary contributor: Mehraveh**

The data-cleaning stage was designed to produce a consistent modeling dataset while removing identifiers, invalid information, redundant variables, and severe feature skew.

The main pipeline was:

```text
Acquire
  ↓
Quality Check
  ↓
Remove Identifiers / Artifacts
  ↓
Check Address Duplicates
  ↓
Handle Invalid & Missing Values
  ↓
Feature Engineering
  ↓
Correlation-Based Redundancy Filtering
  ↓
Log1p Transformation
  ↓
Final Validation
  ↓
cleaned_transactions_dataset.csv
```

This cleaning workflow is documented as the project's dedicated preprocessing stage.

### Main preprocessing operations

#### Identifier and artifact removal

Columns such as:

- `Unnamed: 0`
- `Address`
- `Index`

were removed because they do not represent useful behavioral signals for the machine learning models. Additional zero-variance features were also identified and removed.

#### Missing values

Missing values were concentrated primarily in ERC20-related variables. The cleaning analysis showed that these missing values were associated with accounts having no ERC20 activity, making zero-imputation appropriate for those cases.

#### Feature transformations

Highly skewed numerical features were transformed using `log1p` during the cleaning stage. The cleaned dataset was then validated before being saved as:

```text
cleaned_transactions_dataset.csv
```

### Feature semantics

The dataset includes several behavioral groups:

- Transaction timing
- Sent and received transaction counts
- Address diversity
- Ether value statistics
- Contract interaction statistics
- Total Ether sent/received
- ERC20 transaction statistics
- Token identity information

For example, `Sent tnx` and `Received Tnx` represent the total number of sent and received transactions, while `Unique Sent To Addresses` and `Unique Received From Addresses` describe interaction diversity.

---

# 3. Exploratory Data Analysis

**Primary contributor: Mehraveh**

The exploratory analysis examined class balance, missingness, feature distributions, and relationships between behavioral variables and fraud.

A key observation was that legitimate and fraudulent accounts overlap considerably across individual features. However, fraudulent accounts tend to concentrate toward the **low-activity end** of several distributions, while legitimate accounts extend further into high-activity regions. This suggests that fraud cannot reliably be detected from a single variable and instead requires combinations of behavioral signals.

---

# 4. Feature Engineering

**Primary contributor: Parsa**

Additional behavioral features were engineered to capture transaction patterns that are not represented directly by the original columns.

The engineered features include:

### `balance_to_received_ratio`

Ratio of total Ether balance to total Ether received.

### `tx_velocity`

Transaction rate calculated from total transactions over the account's activity period.

### `sent_address_diversity`

Ratio of unique sent addresses to total sent transactions.

### `value_spike_ratio`

Ratio of maximum received value to average received value.

These features were intended to expose behavioral characteristics such as transaction intensity, interaction diversity, and unusually large value spikes.

---

# 5. Feature Selection

**Primary contributor: Parsa**

A `RandomForestClassifier` with `SelectFromModel` was used to identify informative features and reduce dimensionality.

The feature-selection stage reduced the representation from **88 features to 24 selected features**, removing features whose importance fell below the mean importance threshold.

This reduced feature space was subsequently used in downstream modeling experiments.

---

# 6. Dimensionality Reduction with UMAP

**Primary contributor: Parsa**

UMAP (Uniform Manifold Approximation and Projection) was applied to the selected feature representation.

The purposes were:

- Visualizing high-dimensional behavioral structure
- Investigating natural account groupings
- Providing a lower-dimensional representation for clustering experiments

The UMAP representation was used in both supervised and unsupervised experiments in the broader analysis notebook.

---

# 7. Supervised Learning

**Primary contributor: Arian**

The supervised pipeline treats `FLAG` as the target and evaluates multiple classification algorithms under a common, leakage-aware workflow.

Arian's final supervised pipeline evaluates:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

The primary model-selection metric is **PR-AUC**, because the dataset is imbalanced and the minority fraud class is the primary class of interest.

## Supervised Pipeline

```text
Cleaned Dataset
      │
      ▼
Stratified 80/20 Train-Test Split
      │
      ▼
Training Data
      │
      ▼
Leakage-Safe Preprocessing Pipeline
      │
      ├── Numerical Features
      │     └── StandardScaler for Logistic Regression
      │
      └── Categorical Features
            └── OneHotEncoder
                  └── Rare-category handling
      │
      ▼
Class-Imbalance Handling
      │
      ▼
Randomized Hyperparameter Search
      │
      └── Stratified 3-Fold CV
      │
      ▼
Best Configuration
      │
      ▼
Held-Out Test Evaluation
      │
      ├── Classification Metrics
      ├── ROC Curve
      ├── Precision-Recall Curve
      ├── Confusion Matrix
      └── Overfitting Analysis
      │
      ▼
Saved Inference Pipeline
```

The test set remains untouched during hyperparameter optimization, while stratified cross-validation is performed inside the training set.

## Class Imbalance

Class weighting was used as the common strategy across the final five-model comparison:

- Logistic Regression → `class_weight="balanced"`
- Random Forest → `class_weight="balanced_subsample"`
- LightGBM → `class_weight="balanced"`
- XGBoost → `scale_pos_weight`
- CatBoost → `auto_class_weights="Balanced"`

SMOTE was also experimentally compared with class weighting for Logistic Regression. It produced a slightly higher PR-AUC in that experiment, but the comparison was not extended to all five model families, so class weighting was retained as the common approach.

## Hyperparameter Optimization

Each classifier was optimized using:

```text
RandomizedSearchCV
        │
        ├── Stratified 3-Fold CV
        ├── Randomized Parameter Sampling
        ├── PR-AUC Scoring
        └── Refit Best Configuration
```

The best cross-validation PR-AUC scores were:

| Model | Best CV PR-AUC |
|---|---:|
| **LightGBM** | **0.9673** |
| CatBoost | 0.9667 |
| XGBoost | 0.9655 |
| Random Forest | 0.9568 |
| Logistic Regression | 0.8979 |

---

# 8. Supervised Results

**Primary contributor: Arian**

The final held-out test results from the dedicated supervised pipeline are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM** | **97.20%** | 93.19% | 90.94% | **92.05%** | **99.06%** | **97.39%** |
| CatBoost | 97.20% | **93.73%** | 90.33% | 92.00% | 99.05% | 97.19% |
| XGBoost | 96.50% | 88.44% | **92.45%** | 90.40% | 98.97% | 97.08% |
| Random Forest | 96.18% | 93.62% | 84.29% | 88.71% | 98.88% | 96.33% |
| Logistic Regression | 91.23% | 70.10% | 88.52% | 78.24% | 96.30% | 91.19% |

### 🏆 Selected Supervised Model: LightGBM

LightGBM achieved the strongest overall performance:

- **PR-AUC:** 97.39%
- **ROC-AUC:** 99.06%
- **Accuracy:** 97.20%
- **Precision:** 93.19%
- **Recall:** 90.94%
- **F1-score:** 92.05%

Its cross-validation PR-AUC was **96.73%**, providing a close comparison with its final held-out test performance.

An additional earlier supervised experiment by Parsa evaluated a broader set of models, including Gradient Boosting, Extra Trees, MLP, SVM, KNN, and AdaBoost. In that experiment, Gradient Boosting and XGBoost were among the strongest performers by PR-AUC.

---

# 9. Unsupervised Learning

**Primary contributor: Mehraveh**

The unsupervised branch investigates whether fraudulent behavior can be identified **without exposing the fraud label during training**.

The key principle is:

```text
X = all modeling features except FLAG
y = FLAG, used only after training for evaluation
```

Thus, `FLAG` is treated as an external ground-truth oracle and is not used for model fitting, hyperparameter tuning, or model selection.

The unsupervised analysis has two components:

1. **Clustering**
2. **Anomaly Detection**

---

## 9.1 Clustering

The clustering experiments investigate whether Ethereum accounts naturally form behavioral groups.

The models include:

- Gaussian Mixture Model (GMM)
- HDBSCAN
- DBSCAN
- Agglomerative Clustering
- K-Means

Model selection relies on internal clustering criteria rather than supervised fraud metrics during training.

Examples include:

- **DBCV**
- **Silhouette Score**
- **BIC** for GMM

The unsupervised README reports, for example, that GMM evaluated 2–10 components and selected 10 components using BIC, while HDBSCAN and DBSCAN were tuned using density-based criteria.

### Clustering Findings

Post-hoc analysis reintroduced `FLAG` only after clustering to measure fraud enrichment and coverage.

Key observations included:

- **GMM:** one cluster reached a fraud rate of **41.12%**, representing approximately **2.3×** the overall fraud baseline.
- **HDBSCAN:** produced **52 dense clusters** and labeled approximately **57%** of the dataset as noise; several small clusters reached **100% fraud rate**.
- **DBSCAN:** captured roughly **87.5% of total fraud** in its primary cluster while assigning **13.85%** of observations to noise.

These results demonstrate that unsupervised learning can uncover highly concentrated fraud patterns even without access to fraud labels during model development.

---

# 9.2 Anomaly Detection

**Primary contributor: Mehraveh**

Anomaly detection approaches the problem differently: rather than dividing accounts into predefined groups, it identifies observations that are unusual relative to the overall behavioral distribution.

The evaluated models were:

- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **One-Class SVM**

The models were trained exclusively on the processed feature matrix.

### Evaluation

Because anomaly detection produces rankings or anomaly scores rather than ordinary class predictions, the analysis uses:

- PR-AUC
- Top-K precision
- Fraud capture
- Lift at the top 1%, 2%, 5%, and 10%

The reported results show:

- **One-Class SVM:** highest PR-AUC at **0.190**
- **LOF:** strongest Top-1% lift, identifying **19 fraudulent accounts among the top 93 investigated**, corresponding to **20.43% precision**.

This highlights an important distinction: an anomaly detector does not necessarily need to classify the entire dataset accurately to be useful. Its value can lie in ranking the most suspicious accounts for investigation.

---

# 10. Unsupervised Visualization

**Primary contributor: Mehraveh**

UMAP was used to project the high-dimensional behavioral representation into two dimensions for visual inspection.

The visualization helps examine:

- Natural account distributions
- Cluster structure
- Dense regions
- Noise points
- Potentially anomalous account behavior

The final unsupervised analysis uses UMAP specifically as a visualization tool for understanding the mathematical structure discovered by the clustering and anomaly-detection methods.

---

# 11. Additional Optimization & Comparative Analysis

**Primary contributor: Parsa**

An additional modeling workflow explored Bayesian optimization with `skopt.gp_minimize` for UMAP and clustering parameters.

The optimization considered:

- UMAP dimensionality
- K-Means cluster count
- GMM cluster count
- HDBSCAN minimum cluster size

One reported optimization produced:

| Method | UMAP Dimensions | Main Parameter | ARI | CH Index | DB Index |
|---|---:|---:|---:|---:|---:|
| K-Means | 10 | K = 13 | 0.0549 | 4437.60 | 0.7440 |
| HDBSCAN | 2 | min cluster size = 55 | 0.0839 | 10316.46 | 0.4511 |

This branch also produced a direct supervised-versus-unsupervised comparison in which HDBSCAN achieved a reported recall of **84%** and F1-score of **0.77** under that evaluation setup, compared with higher overall scores from supervised models.

> **Interpretation:** These results should be understood as experiment-specific comparisons because the supervised and unsupervised workflows use different training objectives and evaluation procedures.

---

# 12. Evaluation Strategy

The project deliberately uses different evaluation principles for supervised and unsupervised learning.

## Supervised Learning

The primary metric is:

### **PR-AUC**

Additional metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Precision-Recall curve
- ROC curve
- Train/test generalization gap

PR-AUC is prioritized because the fraud class represents only about 18% of the dataset.

## Unsupervised Learning

During model development, the fraud label is excluded.

Clustering uses internal structural metrics such as:

- DBCV
- Silhouette
- Davies-Bouldin
- Calinski-Harabasz
- BIC for GMM

After training, `FLAG` is reintroduced strictly for post-hoc interpretation through:

- Fraud purity
- Fraud coverage
- Fraud enrichment
- PR-AUC for anomaly rankings
- Top-K precision
- Lift

This separation prevents the unsupervised models from indirectly learning the ground-truth fraud labels.

---

# 13. Interpretability

**Contributors: Parsa & Arian**

Interpretability was investigated through feature importance and SHAP-based analysis.

The broader analysis found that UMAP-derived features such as `umap1`, `umap6`, and `umap9` were influential in supervised predictions.

The final supervised pipeline also examined feature importance for the strongest tree-based model. The analysis suggested that numerical wallet-behavior features contributed more strongly than the high-cardinality token-type features under the current one-hot representation.

Future explainability work can extend this with:

- SHAP global explanations
- SHAP local explanations
- Individual wallet-level explanations
- Feature importance visualization

---

# 14. Model Generalization

**Primary contributor: Arian**

The final supervised pipeline compared training and test PR-AUC:

| Model | Train PR-AUC | Test PR-AUC | Gap |
|---|---:|---:|---:|
| Random Forest | 0.9956 | 0.9633 | 0.0324 |
| CatBoost | 0.9999 | 0.9719 | 0.0280 |
| LightGBM | 0.9997 | 0.9739 | 0.0258 |
| XGBoost | 0.9941 | 0.9708 | 0.0233 |
| Logistic Regression | 0.9147 | 0.9119 | 0.0029 |

The tree-based models show larger training/test gaps because of their higher capacity, but their cross-validation and held-out test results remain relatively close, providing evidence of reasonable generalization.

---

# 15. Business Perspective

**Primary contributor: Parsa**

One additional supervised experiment estimated the potential operational impact of fraud detection using XGBoost:

- **Fraud cases caught:** 389
- **Potential loss prevented:** $194,500
- **Manual review cost:** $780
- **Estimated net benefit:** $193,720

These values are experiment-specific estimates rather than measured real-world financial outcomes. They illustrate how model predictions could be connected to operational fraud-review decisions.

---

# 16. Saved Artifacts

The project produced multiple reusable models, preprocessing components, evaluation files, and visualizations.

### Final supervised pipeline

```text
best_model.joblib
```

The artifact contains the fitted preprocessing pipeline and LightGBM classifier, allowing downstream inference without manually reconstructing preprocessing.

### Additional supervised and unsupervised artifacts

The broader project also includes artifacts such as:

```text
fraud_model_xgboost.pkl
scaler.pkl
umap_reducer.pkl
kmeans_model.pkl
hdbscan_model.pkl
gmm_model.pkl
best_clustering_models_summary.json
```

These were produced during the additional modeling and clustering workflow.

### Evaluation outputs

```text
model_comparison.csv
overfitting_check.csv
classification_report_best_model.txt
roc_pr_curves.png
confusion_matrices.png
model_comparison_bars.png
feature_importance.png
```

---

# 17. Reproducibility

The final supervised pipeline uses:

```python
RANDOM_STATE = 42
```

This seed is applied to the relevant randomized components so that cross-validation, randomized hyperparameter search, and model configurations can be reproduced under the same dataset and software environment.

---

# 18. Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **CatBoost**
- **imbalanced-learn**
- **SciPy**
- **Matplotlib**
- **Seaborn**
- **UMAP**
- **HDBSCAN**
- **Scikit-Optimize**
- **Optuna**
- **Joblib**
- **Jupyter / Google Colab**

---

# 19. Limitations & Future Work

Several improvements remain possible.

### Supervised Learning

- Optimize the classification threshold according to false-positive and false-negative costs.
- Compare SMOTE across all supervised model families using leakage-safe pipelines.
- Evaluate native CatBoost categorical handling.
- Compare frequency, target, and native categorical encoding.
- Investigate temporal validation when transaction timestamps are available.
- Calibrate predicted probabilities.
- Expand SHAP-based explainability.

These are consistent with the future-work directions identified in the final supervised pipeline.

### Unsupervised Learning

- Explore additional density-based clustering approaches.
- Improve anomaly-ranking evaluation.
- Investigate graph-based representations of Ethereum transactions.
- Compare different dimensionality-reduction strategies.
- Study whether clusters correspond to distinct fraud mechanisms.
- Evaluate stability of discovered clusters across parameter settings and data subsets.

---

# 20. Key Takeaways

### 🏆 Best Supervised Model

**LightGBM**

- PR-AUC: **97.39%**
- ROC-AUC: **99.06%**
- F1-score: **92.05%**
- Recall: **90.94%**
- Precision: **93.19%**

### 🔎 Best Unsupervised Findings

The unsupervised experiments demonstrate that fraud-related structure can emerge without exposing `FLAG` to the models:

- GMM discovered clusters with substantial fraud enrichment.
- HDBSCAN identified small clusters with **100% fraud rates**.
- LOF achieved the strongest reported Top-1% lift among the anomaly detectors.
- One-Class SVM achieved the highest reported anomaly-detection PR-AUC.

### 🎯 Overall Conclusion

The project demonstrates that Ethereum fraud detection benefits from combining **supervised prediction with unsupervised discovery**.

Supervised learning provides the strongest predictive performance when reliable labels are available, with LightGBM achieving the best final test PR-AUC in the dedicated supervised pipeline.

Unsupervised learning provides a complementary capability: it can reveal unusual behavioral structures, dense fraud-enriched groups, and potential fraud patterns without requiring fraud labels during training.

Together, these approaches form a more complete fraud-analysis framework:

```text
                    Ethereum Accounts
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Supervised ML              Unsupervised ML
              │                         │
       Known fraud labels        No fraud labels
              │                         │
              ▼                         ▼
       Fraud Prediction        Pattern Discovery
              │                         │
              └────────────┬────────────┘
                           ▼
                 Fraud Investigation
```

---

## 📊 Project Status

| Component | Status | Primary Contributor |
|---|---|---|
| Data acquisition | ✅ Complete | Parsa |
| Data cleaning & preprocessing | ✅ Complete | Mehraveh |
| Exploratory analysis | ✅ Complete | Mehraveh |
| Feature engineering | ✅ Complete | Parsa |
| Feature selection | ✅ Complete | Parsa |
| UMAP analysis | ✅ Complete | Parsa |
| Supervised classification | ✅ Complete | Arian |
| Hyperparameter optimization | ✅ Complete | Arian |
| Unsupervised clustering | ✅ Complete | Mehraveh |
| Anomaly detection | ✅ Complete | Mehraveh |
| Comparative analysis | ✅ Complete | All team members |
| Model artifacts | ✅ Complete | Arian / Parsa / Mehraveh |

---

## 👩‍💻 Team

**Mehraveh** — Data Cleaning, EDA, Unsupervised Learning & Anomaly Detection  
**Arian** — Supervised Machine Learning & Final Model Evaluation  
**Parsa** — Feature Engineering, Feature Selection, UMAP & Optimization

---

## 📄 License

Add the project's chosen license here if applicable.
