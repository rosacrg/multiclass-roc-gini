This repository provides a Python implementation of a framework for multiclass ROC analysis using the Multidimensional Gini Index. This framework generates a single multiclass ROC curve for the multiclass case. The methodology supports robust interactive visualizations. It enables class-weighted aggregation of multiclass ROC curves and incorporates rigorously tested robustness analysis tools.

The multidimensional Gini-weighted multiclass ROC methodology is especially well-suited for imbalanced datasets because it prioritizes true discriminative power rather than sample frequency, making model evaluation more reliable and interpretable in settings where minority classes are critical.

Unlike traditional macro-averaging (which treats all classes equally regardless of their reliability) and micro-averaging (which lets majority classes dominate the metric), the multidimensional Gini index assigns each class a weight based on its actual discriminative contribution to model performance. 

Gini-based weighting highlights classes that the model separates most effectively, even if those classes have few samples. This prevents the evaluation from being biased by the over-representation of majority classes, and guards against inflated scores that can mask catastrophic failures in critical minority groups.

The ZCA whitening step ensures that differences in variance or scale among predicted probabilities across classes cannot distort ROC aggregates. This is especially important in imbalanced contexts, where rare classes may have highly skewed probability distributions.

Because the Gini-weighted ROC curve provides a single, interpretable metric (AUC or multidimensional Gini) that reflects the model’s overall class-separability, it is ideal for regulatory settings requiring independent assessment of risk across all classes, not just the majority. This facilitates compliance with frameworks like the EU AI Act, which demand explainability in high-risk applications and cannot tolerate metrics that obscure poor minority class performance.


**Features**
* Unified Multiclass ROC Curve computation using the multidimensional Gini index, overcoming limitations of micro- and macro-averaging.
* ZCA Whitening of predicted probabilities for scale invariance and increased stability.
* Interactive Plotly ROC Curves with real-time threshold selection and performance metrics (accuracy, precision, recall, F1-score).
* Robustness Analysis using SAFE AI RGR and RGA methods for model stability against input perturbations.
* Comprehensive comparison with traditional metrics: Macro-AUC, Micro-AUC, and Gini-weighted metrics.


**Installation:**
Clone the repository:
git clone https://github.com/rosacrg/multiclass-roc-gini.git
cd multiclass-roc-gini


**Dependencies include:**
* numpy
* pandas
* scikit-learn
* matplotlib
* plotly
* catboost
* xgboost
* torch
* SAFE AI package (for robustness functions)


**Getting Started**
1. Data Preparation → prepare your train/test datasets as Pandas DataFrames. Target values should be integer-encoded for multiclass problems.
2. Train a Model → train any compatible classifier (e.g., RandomForest, CatBoost, XGBoost, LogisticRegression).
3. Multiclass ROC Analysis → use the provided functions to:
* Whiten predicted probabilities (with ZCA correlation whitening).
* Compute aggregated ROC metrics using Gini weights.
* Visualize and interpret results with Plotly or Matplotlib.
  

**Key Modules**
* gini_whitening.py →	Gini mean difference and ZCA whitening.
* metrics_multi_roc.py	→ Multiclass ROC aggregation with Gini weights.
* multi_roc_analysis.py	→ End-to-end pipeline for ROC curve analysis.
* multi_roc_plotting.py →	Interactive and static ROC/PR visualization.
* proba_whitening.py → Numerical stabilization for probability whitening.
* * robustness.py	→ Perturbation-driven robustness analysis.
* utils.py → Data integrity checks and auxiliary functions. This function is forked from: https://github.com/GolnooshBabaei/safeaipackage
* check_robustness.py	RGR → calculations per variable and overall. This function is forked from: https://github.com/GolnooshBabaei/safeaipackage
* MulticlassCreditESG → is an example application of the package for a highly imbalanced credit scoring dataset.


**Example:**
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(X_train, y_train)
results = complete_roc_analysis(y_train, y_test, X_test, X_train, model)

# Visualize Interactive ROC
results['figures']['interactive_roc'].show()


**This implementation is based on my MSc. thesis:**
"Aggregating Multiclass ROC Curves with Applications to ESG and Credit Risk Management," Rosa Carolina Rosciano, University of Pavia, 2025.
Thesis available on demand: rc.rosciano@gmail.com
See the thesis for full mathematical exposition and regulatory context.


**Contributing:**
Contributions, bug reports, and feature requests are welcome! Please submit via Issues or pull requests. For major changes, open an issue first to discuss.

**License**
This project is licensed under the terms of the MIT License.

