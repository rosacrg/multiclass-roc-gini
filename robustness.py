# Multidimensional Gini-weighted ROC analysis for multiclass classification. 
# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)
# This file contains functions to analyze perturbations in multiclass classification models.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from metrics_multi_roc import metrics_multiroc
from safeaipackage import core
from check_robustness import perturb # Function from Vasily github repository: https://github.com/koleso500/Thesis/blob/main/safeai_files/check_robustness.py  


def analyze_multiclass_perturbation(xtest, y_test, model, variables, 
                                   whitened_proba_orig, gini_weights_test,
                                   perturbation_percentage=0.05):
    """
    Compares original vs perturbed aggregate ROC metrics and computes:
    - Individual variable RGR values per threshold
    - Grouped RGR values (all variables perturbed) per threshold
    """
    n_classes = len(np.unique(y_test))
    
    # Store perturbation results
    perturbation_results = {}
    grouped_rgr_values = []
    
    fpr_orig, tpr_orig, thresholds, auct, std = metrics_multiroc(y_test, whitened_proba_orig, gini_weights_test)

    # Compute grouped RGR (all variables perturbed) for each threshold
    for thresh in thresholds:
        # Perturb all variables simultaneously
        xtest_pert_all = xtest.copy()
        for var in variables:
            xtest_pert_all = perturb(xtest_pert_all, var, perturbation_percentage)
        
        # Get perturbed predictions for grouped perturbation
        whitened_proba_pert_all = model.predict_proba(xtest_pert_all)
        
        # Compute RGR at this threshold for grouped perturbation
        orig_rank = whitened_proba_orig >= thresh
        pert_rank = whitened_proba_pert_all >= thresh
        grouped_rgr = core.rga(orig_rank.flatten(), pert_rank.flatten())
        grouped_rgr_values.append(grouped_rgr)
    
    # Compute individual variable RGRs
    for var in variables:
        # Perturb single variable
        xtest_pert = perturb(xtest.copy(), var, perturbation_percentage)
        whitened_proba_pert = model.predict_proba(xtest_pert)

        # Compute perturbed metrics
        fpr_pert, tpr_pert, agg_thresholds_test, agg_auc_test, stdp = metrics_multiroc(y_test, whitened_proba_pert, gini_weights_test)
        
        # Compute RGR at each threshold
        rgr_values = []
        for thresh in thresholds:
            orig_rank = whitened_proba_orig >= thresh
            pert_rank = whitened_proba_pert >= thresh
            rgr_values.append(core.rga(orig_rank.flatten(), pert_rank.flatten()))
        
        perturbation_results[var] = {
            'fpr': fpr_pert,
            'tpr': tpr_pert,
            'rgr': rgr_values
        }
    
    # Create summary with max RGR for grouped perturbation
    max_grouped_idx = np.nanargmax(grouped_rgr_values)
    max_grouped_rgr = grouped_rgr_values[max_grouped_idx]
    max_grouped_threshold = thresholds[max_grouped_idx]
    
    # Create variable summary
    summary_rows = []
    for var, results in perturbation_results.items():
        max_idx = np.nanargmax(results['rgr'])
        summary_rows.append({
            'Variable': var,
            'Max RGR': results['rgr'][max_idx],
            'Threshold': thresholds[max_idx]
        })
    
    # Add grouped RGR to summary
    summary_rows.append({
        'Variable': 'Grouped',
        'Max RGR': max_grouped_rgr,
        'Threshold': max_grouped_threshold
    })
    
    summary_df = pd.DataFrame(summary_rows)
    
    return perturbation_results, grouped_rgr_values, summary_df

def plot_perturbed_roc_comparison(whitened_proba_orig, y_test, gini_weights_test, pert_results, grouped_rgr_values, summary_df):
    n_classes = len(np.unique(y_test))
    fpr_orig, tpr_orig, thresholds, auct, std = metrics_multiroc(y_test, whitened_proba_orig, gini_weights_test)

    fig = go.Figure()
    
    # Original ROC
    fig.add_trace(go.Scatter(
        x=fpr_orig, y=tpr_orig,
        mode='lines', name='Original ROC',
        line=dict(color='black', width=3)
    ))
    
    # --- Prepare variable selection ---
    # If you want to exclude "Grouped", filter it out
    variable_summary = summary_df[~summary_df['Variable'].str.contains('Grouped')]
    # If you want to include "Grouped", comment the above line and use:
    # variable_summary = summary_df

    # Sort by Max RGR ascending (lowest first)
    variable_summary = variable_summary.sort_values('Max RGR').reset_index(drop=True)
    
    # Get the 3 variables with lowest Max RGR
    top_vars = variable_summary['Variable'].iloc[:3].tolist()

    # --- Plot perturbed ROCs for the 3 variables with lowest Max RGR ---
    for var in top_vars:
        results = pert_results[var]
        fig.add_trace(go.Scatter(
            x=results['fpr'], y=results['tpr'],
            mode='lines', name=f'{var} Perturbed (Low Max RGR)',
            line=dict(dash='dot')
        ))
    
    fig.update_layout(
        title='ROC Comparison: Top 3 Sensitive Variables (Lowest RGR)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=1000,
        height=600
    )
    
    return fig

# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)