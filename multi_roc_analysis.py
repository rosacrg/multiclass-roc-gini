# MIT License

# Copyright (c) 2025 [Rosa Carolina Rosciano]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Multidimensional Gini-weighted ROC analysis for multiclass classification. 
# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)
# This file contains a function to perform a complete multiclass ROC analysis.

import numpy as np
from sklearn.metrics import  roc_auc_score
from gini_whitening import multidim_gini
from proba_whitening import whitening_predicted_proba_stable
from multi_roc_plotting import plot_multiclass_roc, gini_weighted_multiclass_roc, plot_marginal_and_multi_roc
from metrics_multi_roc import metrics_multiroc, aggregate_AUC


def complete_roc_analysis (y_train, y_test, X_test, X_train, model, classifier_name="Classifier", key_results_display = None, multilabel=False):
    """Complete ROC analysis using your existing functions."""
    
    # 1. Whiten probabilities using your existing function
    proba_test_w, proba_train_w = whitening_predicted_proba_stable(X_test, X_train, model, y_test, multilabel=None)
    
    # 2. Get theoretical Gini and weights using your existing function
    AUC_test, AUC_train, weights_test, weights_train = aggregate_AUC(proba_test_w, proba_train_w)
    gm_test, _ = multidim_gini(proba_test_w)
    gm_train, _ = multidim_gini(proba_train_w)
    
    # 3. Get ROC metrics using your existing function
    fpr_test, tpr_test, thresh_test, auc_test, std_test = metrics_multiroc(
        y_test, proba_test_w, weights_test)
    fpr_train, tpr_train, thresh_train, auc_train, std_train = metrics_multiroc(
        y_train, proba_train_w, weights_train)
    
    # 4. Generate figures using your existing functions
    interactive_fig, metrics_dict = plot_multiclass_roc(proba_test_w, y_test, weights_test)
    
    comparison_fig = gini_weighted_multiclass_roc(
        y_test, proba_test_w, y_train, proba_train_w,
        weights_test, weights_train, classifier_name, plot=True)
    
    marginal_fig = plot_marginal_and_multi_roc(proba_test_w,
            y_test, X_test, weights_test, model, classifier_name)
    
    if key_results_display is True:
        proba_test = model.predict_proba(X_test)
        # Display key results
        print(f"\nKey Results:")
        print(f"├─ Theoretical Gini (test): {gm_test:.4f}")
        print(f"├─ Theoretical AUC (test): {AUC_test:.4f}")
        print(f"├─ Empirical AUC (test): {auc_test:.4f}")

        print(f"\nClass Weights (Gini-based):")
        for i, weight in enumerate(weights_test):
            print(f"  Class {i}: {weight:.4f}")

        # Compare with standard metrics
        standard_auc_macro = roc_auc_score(y_test, proba_test, multi_class='ovr', average='macro')
        standard_auc_micro = roc_auc_score(y_test, proba_test, multi_class='ovr', average='micro')

        print(f"\nComparison with Standard Metrics:")
        print(f"├─ Standard Macro AUC: {standard_auc_macro:.4f}")
        print(f"├─ Standard Micro AUC: {standard_auc_micro:.4f}")
        print(f"└─ Gini-weighted AUC: {auc_test:.4f}")

        # Show interactive ROC curve
        print(f"\nDisplaying interactive ROC curve...")
        interactive_fig.show()

        # Show train/test comparison
        if comparison_fig is not None:
            comparison_fig

        # Show marginal ROC curves
        if marginal_fig is not None:
            marginal_fig.show()

        max_f1 = []

        for thre, metric in metrics_dict.items():
            for key, value in metric.items():
                if key == 'f1_score':
                    max_f1.append((float(thre), value))

        max_f1 = sorted(max_f1, key=lambda x: x[1], reverse=True)
        print(f"\nBest F1 Score {max_f1[0][1]:.4f} is achieved at threshold {max_f1[0][0]:.4f}")   

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print("="*60)
    
    return {
        'gini_train': gm_train,
        'gini_test': gm_test,
        'theoretical_auc_train': AUC_train,
        'theoretical_auc_test': AUC_test,
        'empirical_auc_test': auc_test,
        'empirical_auc_train': auc_train,
        'weights_test': weights_test,
        'weights_train': weights_train,
        'weights': weights_test,
        'whitened_proba_test': proba_test_w,
        'whitened_proba_train': proba_train_w,
        'roc_test': {
            'fpr': fpr_test,
            'tpr': tpr_test,
            'thresholds': thresh_test,
            'auc': auc_test,
            'std': std_test
        },
        'roc_train': {
            'fpr': fpr_train,
            'tpr': tpr_train,
            'thresholds': thresh_train,
            'auc': auc_train,
            'std': std_train
        },
        'threshold_metrics': metrics_dict,
        'figures': {
            'interactive_roc': interactive_fig,
            'train_test_comparison': comparison_fig,
            'marginal_roc': marginal_fig,
        }
    }

# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025).