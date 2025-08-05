# Multidimensional Gini-weighted ROC analysis for multiclass classification. 
# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)
# This file contains functions for plotting and visualizing the results of the multiclass ROC analysis.

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle
from metrics_multi_roc import metrics_multiroc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def plot_multiclass_roc(whitened_proba_test, y_test, gini_weights_test, classifier_type=None):
    """
    Plots a Multiclass ROC curve. The metrics are weighted by the Unidimensional Gini coefficients.

    Parameters:
    - aggregated_fpr: False Positive Rates for the multiclass ROC curve.
    - aggregated_tpr: True Positive Rates for the multiclass ROC curve.
    - aggregate_auc: Aggregate AUC value for the multiclass ROC curve.
    - whitened_proba_test: whitened predicted probabilities for the test set.
    - y_test: True labels for the test set.
    - gini_weights_test: Unidimensional Gini weights for the test set.

    Returns:
    A Plotly figure object with the multiclaass ROC curve with a slider to see metrics at different thresholds.
    """
    n_classes = len(np.unique(y_test))
        
    aggregated_fpr, aggregated_tpr, agg_thresholds_test, aggregate_auc, std_test = metrics_multiroc(y_test, whitened_proba_test, gini_weights_test, n_points=100)

    # Convert to numpy arrays if needed
    if hasattr(y_test, 'to_numpy'):
        y_test = y_test.to_numpy()
    if hasattr(whitened_proba_test, 'to_numpy'):
        whitened_proba_test = whitened_proba_test.to_numpy()
    
    # To handle cases of numerical imprecision, we normalize the weights to sum to 1
    if np.sum(gini_weights_test) != 1:
        gini_weights_test_clean = gini_weights_test/np.sum(gini_weights_test)
    else:
        gini_weights_test_clean = gini_weights_test

    # Auto-detect classifier type if not provided, classifiers that use the One-vs-All technique have a benchmark line of 45
    if classifier_type is None:
        # Check if probabilities sum to 1 (native multiclass) or not (OvR)
        prob_sums = np.sum(whitened_proba_test, axis=1)
        is_native_multiclass = np.allclose(prob_sums, 1.0, atol=1e-6)
        classifier_type = 'native' if is_native_multiclass else 'ovr'
    
    # Always use the benchmark line: y = x / n_classes
    baseline_slope = 1.0 / n_classes
    baseline_auc = 1 / n_classes
    baseline_label = f'Benchmark (AUC = {baseline_auc:.3f}), y = x / {n_classes}'
    
    # Determine optimal threshold index based on minimal Euclidean distance to (0,1)
    distances = np.sqrt(aggregated_fpr**2 + (1 - aggregated_tpr)**2)
    opt_index = np.argmin(distances)

    aggregate_gini = 2 * aggregate_auc - 1  

    # Calculate per-class metrics and then average them using weights
    n_thresholds = len(aggregated_fpr)
    thresholds_metrics = np.linspace(0, 1, n_thresholds)
    metrics_list = []
    
    # Use nested dictionary structure for metrics
    metrics_dict = {}
    
    for thresh in thresholds_metrics:
        # Initialize metrics for this threshold
        class_acc = []
        class_prec = []
        class_rec = []
        class_f1 = []
        
        # Calculate per-class metrics
        for i in range(n_classes):
            # Handle different input formats
            if y_test.ndim == 1:  # 1D array for multiclass (class indices)
                y_true_class = (y_test == i).astype(int)
                y_pred_scores_class = whitened_proba_test[:, i]
            else:  # 2D array for multilabel or one-hot encoded multiclass
                y_true_class = y_test[:, i]
                y_pred_scores_class = whitened_proba_test[:, i]
            
            # Make predictions for this threshold
            y_pred_class = (y_pred_scores_class >= thresh).astype(int)
            
            # Calculate metrics
            class_acc.append(accuracy_score(y_true_class, y_pred_class))
            class_prec.append(precision_score(y_true_class, y_pred_class, zero_division=0))
            class_rec.append(recall_score(y_true_class, y_pred_class, zero_division=0))
            class_f1.append(f1_score(y_true_class, y_pred_class, zero_division=0))
        
        # Weighted average of metrics across classes using gini_weights_test
        avg_acc = np.average(class_acc, weights=gini_weights_test_clean)
        avg_prec = np.average(class_prec, weights=gini_weights_test_clean)
        avg_rec = np.average(class_rec, weights=gini_weights_test_clean)
        avg_f1 = np.average(class_f1, weights=gini_weights_test_clean)
        
        metrics_list.append((thresh, avg_acc, avg_prec, avg_rec, avg_f1))
        
        # Create a nested dictionary for this threshold
        metrics_dict[str(thresh)] = {
            'accuracy': avg_acc,
            'precision': avg_prec,
            'recall': avg_rec,
            'f1_score': avg_f1
        }
    
    metrics_array_agg = np.array(metrics_list)
    
    # Trace for aggregated ROC curve
    trace0 = go.Scatter(
        x=aggregated_fpr,
        y=aggregated_tpr,
        mode='lines',
        name=f'Multiclass ROC (AUC={aggregate_auc:.2f}, Gini={aggregate_gini:.2f})', #Multidimensional Gini
        line=dict(color='black', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 80, 0.2)'
    )
    # Trace for the red marker (selected threshold)
    trace1 = go.Scatter(
        x=[aggregated_fpr[opt_index]],
        y=[aggregated_tpr[opt_index]],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Selected Threshold'
    )
    # Initial annotation with metrics using the averaged metrics
    initial_annotation = dict(
        x=aggregated_fpr[opt_index],
        y=aggregated_tpr[opt_index],
        text=(f"FPR: {aggregated_fpr[opt_index]:.2f}<br>TPR: {aggregated_tpr[opt_index]:.2f}<br>"
              f"Threshold: {metrics_array_agg[opt_index, 0]:.2f}<br>"
              f"Acc: {metrics_array_agg[opt_index, 1]:.2f}<br>"
              f"Prec: {metrics_array_agg[opt_index, 2]:.2f}<br>"
              f"Rec: {metrics_array_agg[opt_index, 3]:.2f}<br>"
              f"F1: {metrics_array_agg[opt_index, 4]:.2f}"),
        showarrow=True,
        arrowhead=2
    )
    
    fig = go.Figure(
        data=[trace0, trace1],
        layout=go.Layout(
            title={
                'text': "Multiclass ROC Curve",
                'font': {'size': 24},
                'xref': 'paper',
                'x': 0.5,
                'y': 0.95,
                'yanchor': 'top',
                'xanchor': 'center',
                'subtitle': {
                    'text': "with metrics weighted by Unidimensional Gini coefficients",
                    'font': {'size': 18, 'color': 'gray'},
                }
            },
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=700,
            height=700,
            annotations=[initial_annotation]
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # Create animation frames: update red marker and annotation per threshold index.
    frames = []
    for i in range(len(aggregated_fpr)):
        frame = go.Frame(
            data=[
                # Aggregated ROC curve remains constant
                go.Scatter(
                    x=aggregated_fpr,
                    y=aggregated_tpr,
                    mode='lines',
                    line=dict(color='black', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 100, 80, 0.2)'
                ),
                # Update red marker position
                go.Scatter(
                    x=[aggregated_fpr[i]],
                    y=[aggregated_tpr[i]],
                    mode='markers',
                    marker=dict(color='red', size=12)
                )
            ],
            layout=go.Layout(
                annotations=[dict(
                    x=aggregated_fpr[i],
                    y=aggregated_tpr[i],
                    text=(f"FPR: {aggregated_fpr[i]:.2f}<br>TPR: {aggregated_tpr[i]:.2f}<br>"
                          f"Threshold: {metrics_array_agg[i, 0]:.2f}<br>"
                          f"Acc: {metrics_array_agg[i, 1]:.2f}<br>"
                          f"Prec: {metrics_array_agg[i, 2]:.2f}<br>"
                          f"Rec: {metrics_array_agg[i, 3]:.2f}<br>"
                          f"F1: {metrics_array_agg[i, 4]:.2f}"),
                    showarrow=True,
                    arrowhead=2
                )]
            ),
            name=str(i)
        )
        frames.append(frame)
    fig.frames = frames
    
    slider_steps = [
        dict(
            label=str(i),
            method="animate",
            args=[[str(i)],
                  {"mode": "immediate",
                   "frame": {"duration": 50, "redraw": True},
                   "transition": {"duration": 0}}]
        )
        for i in range(len(aggregated_fpr))
    ]
    fig.update_layout(sliders=[dict(
        active=opt_index,
        currentvalue={"prefix": "Threshold Index: "},
        pad={"t": 50},
        steps=slider_steps
    )])
    
    # Add benchmark reference line: y = x / n_classes
    diag_line = go.Scatter(
        x=[0, 1],
        y=[0, baseline_slope],
        mode='lines',
        line=dict(color='gray', dash='dot'),
        name=baseline_label
    )
    fig.add_trace(diag_line)
    
    return fig, metrics_dict

    
def gini_weighted_multiclass_roc(y_test, whitened_proba_test, y_train, whitened_proba_train, gini_weights_test, gini_weights_train, classifier_name="", n_points=100, plot=True):
    """
    Plots a Multiclass ROC curve for both test and train sets.
    This Multiclass ROC curve is obtained leveraging the Multidimensional Gini index
    (weighted average of the Unidimensional Gini coefficients) and the ZCA whitening process.

    Parameters:
    - y_test: True labels for the test set.
    - whitened_proba_test: whitened predicted probabilities for the test set.
    - y_train: True labels for the train set.
    - whitened_proba_train: whitened predicted probabilities for the train set.
    - gini_weights_test: Gini-based weights for each class. Should sum to 1.
    - gini_weights_train: Gini-based weights for each class. Should sum to 1.
    - n_points: Number of points to interpolate along FPR axis.
    - plot: Whether to show the ROC curve plot.

    Returns:
    - plot object: Matplotlib plot object showing the Multiclass ROC curve.
    """
    n_classes = len(np.unique(y_test))
        
    # Always use the benchmark line: y = x / n_classes
    baseline_slope = 1.0 / n_classes
    baseline_auc = 1 / n_classes
    baseline_label = f'Benchmark (AUC = {baseline_auc:.3f}), y = x / {n_classes}'
    
    common_fpr, agg_tpr_test, agg_thresholds_test, agg_auc_test, std_test = metrics_multiroc(y_test, whitened_proba_test, gini_weights_test, n_points=100)
    common_fpr, agg_tpr_train, agg_thresholds_train, agg_auc_train, std_train =  metrics_multiroc(y_train, whitened_proba_train, gini_weights_train, n_points=100)

    # Plotting
    if plot:
        plt.figure(figsize=(10, 7))
        # CI bands
        plt.fill_between(common_fpr,
                         agg_tpr_test - std_test,
                         agg_tpr_test + std_test,
                         color='blue', alpha=0.1,
                         label='Test ±1 std')
        plt.fill_between(common_fpr,
                         agg_tpr_train - std_train,
                         agg_tpr_train + std_train,
                         color='green', alpha=0.1,
                         label='Train ±1 std')
        plt.plot(common_fpr, agg_tpr_test, color='blue', lw=2,
                 label=f'Gini-Weighted ROC test set (AUC = {agg_auc_test:.3f})')
        plt.plot(common_fpr, agg_tpr_train, color='green', lw=2,
            label=f'Gini-Weighted ROC train set (AUC = {agg_auc_train:.3f})')
        
        # Plot appropriate baseline
        plt.plot([0, 1], [0, baseline_slope], 'k--', label=baseline_label)
        
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('Multidimensional Gini Weighted ROC Curve'+ '-' + classifier_name, fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right', fontsize=12)

        plt.tight_layout()

        return plt.show()


def plot_marginal_and_multi_roc(y_proba_w,y_test, X_test, gini_weights_test, est, clf_name, multilabel=False):
    """
    Plots both marginal ROC curves (one for each class) and an aggregate curve.
    The marginal ROC curves are computed using the predicted probabilities from the classifier.

    Parameters:
    - y_proba_w: Predicted probabilities for each class (after whitening).
    - y_test: True labels for the test set.
    - X_test: Test features.
    - aggregated_fpr: False Positive Rates for the aggregate ROC curve.
    - aggregated_tpr: True Positive Rates for the aggregate ROC curve.
    - aggregate_auc: Aggregate AUC value for the aggregate ROC curve.
    - est: The trained classifier.
    - clf_name: Name of the classifier for labeling in the plot.
    - n_classes: Number of classes in the classification problem.
    - multilabel: Boolean indicating if the problem is multilabel or not.

    Returns:
    - fig: Plotly figure object containing the ROC curves.

    """
    n_classes = len(np.unique(y_test))
        
    # Convert to numpy arrays if needed
    if hasattr(y_test, 'to_numpy'):
        y_test = y_test.to_numpy()
    
    aggregated_fpr, aggregated_tpr, agg_thresholds_test, aggregate_auc, std_test = metrics_multiroc(y_test, y_proba_w, gini_weights_test, n_points=100)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thr = dict()

    for i in range(n_classes):
        # Properly handle different target formats
        if multilabel:
            # For multilabel: directly use the binary indicators
            true_labels = y_test[:, i] if y_test.ndim > 1 else (y_test == i).astype(int)
        else:
            # For multiclass: convert to one-vs-rest binary indicators
            if y_test.ndim > 1:  # one-hot encoded
                true_labels = y_test[:, i]
            else:  # class indices
                true_labels = (y_test == i).astype(int)
        
        # Get probabilities for this class
        class_probs = y_proba_w[:, i]
        
        # Compute ROC curve for this class
        fpr[i], tpr[i], _ = roc_curve(true_labels, class_probs)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Add traces for each class
    colors = cycle(['red', 'green', 'blue', 'purple', 'greenyellow', 'yellow', 
                    'pink', 'aqua', 'lightskyblue', 'orange'])
    
    fig = go.Figure()  # Initialize the figure object
    for i, color in zip(range(n_classes), colors):
        fig.add_trace(go.Scatter(
            x=fpr[i], y=tpr[i],
            mode='lines',
            name=f'Class {i} (AUC={roc_auc[i]:.2f})',
            line=dict(color=color, width=1.5)
        ))
    
    # Add aggregate ROC curve
    fig.add_trace(go.Scatter(
        x=aggregated_fpr,
        y=aggregated_tpr,
        mode='lines',
        name=f'Aggregate ROC (AUC={aggregate_auc:.2f})',
        line=dict(color='black', width=3, dash='dash')
    ))
    
    # Benchmark line: y = x / n_classes
    baseline_slope = 1.0 / n_classes
    baseline_auc = 1 / n_classes
    baseline_label = f'Benchmark (AUC = {baseline_auc:.3f}), y = x / {n_classes}'
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, baseline_slope],
        mode='lines',
        line=dict(color='gray', dash='dot'),
        name=baseline_label
    ))
    
    # Random guessing line: y = x (45-degree line)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='lightgray', dash='dashdot'),
        name='Random Guessing (AUC = 0.5)'
    ))
    
    fig.update_layout(
        title=f"Marginal ROC Curves with multiclass Gini-ROC curve for {clf_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=700,
        xaxis=dict(range=[0, 1]), 
        yaxis=dict(range=[0, 1])  
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig  


def gini_weighted_pr_curves(y_test, proba, gini_weights, n_points=100):
    """
    Compute and plot multiclass Precision-Recall curves across two separate plots:
    - Plot 1: Gini-weighted average, micro-average, and macro-average curves
    - Plot 2: Per-class curves with the Gini-weighted average

    Additionally returns a metrics_dict for the Gini-weighted curve at each recall threshold,
    and marks the point where precision equals recall.

    Parameters
    ----------
    y_test : array-like, shape (n_samples,)
        True class labels (0…n_classes-1).

    proba : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class.

    gini_weights : array-like, shape (n_classes,)
        Nonnegative weights (sum to 1) from your Gini-whitening framework.

    n_points : int, default=100
        Number of points for the common recall grid when averaging.

    Returns
    -------
    metrics_dict : dict
        Nested dict mapping each recall threshold to its precision and recall on the
        Gini-weighted PR curve.
    """
    n_classes = len(np.unique(y_test))

    # 1) Binarize
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    if y_bin.shape[1] != n_classes:
        raise ValueError("y_test must contain all classes 0…n_classes-1")

    # 2) MICRO-AVERAGE
    prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), proba.ravel())
    ap_micro = average_precision_score(y_bin, proba, average="micro")

    # 3) PER-CLASS CURVES + APs
    prec = {}
    rec = {}
    ap = {}
    for i in range(n_classes):
        prec[i], rec[i], _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        ap[i] = average_precision_score(y_bin[:, i], proba[:, i])

    # 4) MACRO-AVERAGE (equal weight)
    common_rec = np.linspace(0, 1, n_points)
    prec_macro_sum = np.zeros_like(common_rec)
    for i in range(n_classes):
        prec_macro_sum += np.interp(common_rec, rec[i][::-1], prec[i][::-1])
    prec_macro = prec_macro_sum / n_classes
    ap_macro = np.mean(list(ap.values()))

    # 5) GINI-WEIGHTED AVERAGE
    prec_gini_sum = np.zeros_like(common_rec)
    for i in range(n_classes):
        prec_gini_sum += gini_weights[i] * np.interp(common_rec, rec[i][::-1], prec[i][::-1])
    prec_gini = prec_gini_sum
    ap_gini = sum(gini_weights[i] * ap[i] for i in range(n_classes))

    # Identify point where precision == recall on Gini curve
    diffs = np.abs(prec_gini - common_rec)
    idx_equal = np.argmin(diffs)
    rec_eq = common_rec[idx_equal]
    prec_eq = prec_gini[idx_equal]

    # Build metrics_dict for Gini-weighted curve
    metrics_dict = {}
    for rec_val, prec_val in zip(common_rec, prec_gini):
        # Calculate F1 score from precision and recall values
        if prec_val + rec_val > 0:  # Avoid division by zero
            f1 = 2 * (prec_val * rec_val) / (prec_val + rec_val)
        else:
            f1 = 0.0
            
        metrics_dict[f"{rec_val:.3f}"] = {
            'recall': float(rec_val),
            'precision': float(prec_val),
            'f1_score': float(f1)
        }

    # 6) PLOT EVERYTHING - SPLIT INTO TWO SEPARATE PLOTS
    
    # PLOT 1: Aggregate curves only (micro, macro, gini)
    plt.figure(figsize=(10, 8), dpi=300)
    
    # micro
    plt.plot(rec_micro, prec_micro,
             label=f"micro-avg PR (AP={ap_micro:.2f})",
             linewidth=3)

    # macro
    plt.plot(common_rec, prec_macro,
             linestyle="--",
             label=f"macro-avg PR (AP={ap_macro:.2f})",
             linewidth=3)

    # gini-weighted
    plt.plot(common_rec, prec_gini,
             linestyle=":",
             label=f"Gini-weighted PR (AP={ap_gini:.2f})",
             linewidth=3)

    # mark where precision == recall on Gini curve
    plt.scatter(rec_eq, prec_eq,
                color='red',
                s=100,
                marker='o',
                zorder=5,
                label='Precision = Recall')

    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title("Multiclass Precision–Recall Curves (Aggregate Methods)", fontsize=15)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # PLOT 2: Per-class curves + Gini-weighted
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Per-class curves
    for i in range(n_classes):
        plt.plot(rec[i], prec[i],
                 alpha=0.7,
                 linewidth=1.5,
                 label=f"class {i} (AP={ap[i]:.2f})")
    
    # Gini-weighted (highlight it with thicker line)
    plt.plot(common_rec, prec_gini,
             linestyle=":",
             color='black',
             linewidth=3,
             label=f"Gini-weighted PR (AP={ap_gini:.2f})")
    
    # mark where precision == recall on Gini curve
    plt.scatter(rec_eq, prec_eq,
                color='red',
                s=100,
                marker='o',
                zorder=5,
                label='Precision = Recall')
    
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title("Per-Class and Gini-weighted Precision–Recall Curves", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    return metrics_dict

# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)