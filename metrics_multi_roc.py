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
# This file contains functions to compute the multiclass ROC analysis usinng the Multidimensional Gini index.

import numpy as np
from gini_whitening import multidim_gini
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d

def aggregate_AUC(whitened_proba_test, whitened_proba_train):
    ''' This function computes the multidimensional Gini index and the theoretical AUC values for the test and train sets.

    Parameters:
    whitened_proba_test : numpy.ndarray
        Whitened test probabilities.
    whitened_proba_train : numpy.ndarray
        Whitened training probabilities.

    Returns:
    AUC_test : float
        Theoretical AUC value for the test set. 
    AUC_train : float
        Theorethicals AUC value for the train set.
    gini_weights_test : numpy.ndarray
        Each class' weight used for the Multidimensional test set Gini.
    gini_weights_train : nuFmpy.ndarray                              
        Each class' weight used for the Multidimensional train set Gini.
    '''
    
    GM_test, gini_weights_test = multidim_gini(whitened_proba_test)
    GM_train, gini_weights_train = multidim_gini(whitened_proba_train)

    # Compute AUC values
    AUC_test = (GM_test + 1) / 2
    AUC_train = (GM_train + 1) / 2

    print(f'Multidimensional Theoretical Gini Test set {GM_test}')
    print(f'Multidimensional Theoretical Gini Train set {GM_train}')
    print(f'Multiclass Theoretical AUC Test set {AUC_test}')
    print(f'Multiclass Theoretical AUC Train set {AUC_train}')

    return AUC_test, AUC_train, gini_weights_test, gini_weights_train



def metrics_multiroc(y_test, whitened_proba_test, gini_weights_test, n_points=100):
    """
    Compute aggregated multiclass ROC metrics using Gini-weighted averaging.
    
    This function computes ROC curves for each class individually, interpolates them 
    to a common FPR grid, and then aggregates them using Gini-based weights to 
    produce a single multiclass ROC curve with associated metrics.
    -
    Parameters:
    -----------
    y_test : array-like, shape (n_samples,) or (n_samples, n_classes)
        True class labels. Can be class indices (1D) or one-hot/multilabel format (2D)
    whitened_proba_test : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class (after whitening process)
    gini_weights_test : array-like, shape (n_classes,)
        Gini-based weights for each class used for weighted averaging
    n_points : int, default=100
        Number of points for interpolation along the common FPR axis
        
    Returns:
    --------
    common_fpr : numpy.ndarray, shape (n_points,)
        Common false positive rate axis for all classes
    agg_tpr_test : numpy.ndarray, shape (n_points,)
        Aggregated true positive rates using Gini weights
    agg_thresholds_test : numpy.ndarray, shape (n_points,)
        Aggregated classification thresholds using Gini weights
    agg_auc_test : float
        Area under the aggregated ROC curve        std_test : numpy.ndarray, shape (n_points,)
            Standard deviation of TPR across classes at each FPR point
        """
    n_classes = len(np.unique(y_test))
        
    if np.sum(gini_weights_test) != 1:
        gini_weights_test_clean = gini_weights_test/np.sum(gini_weights_test)
    else:
        gini_weights_test_clean = gini_weights_test
    
    # Arrays to store interpolated values
    all_tpr_interp_test = np.zeros((n_classes, n_points))
    all_threshold_interp_test = np.zeros((n_classes, n_points))

    # Define common FPR grid
    common_fpr = np.linspace(0, 1, n_points)
    
    for i in range(n_classes):
        # Handle different target formats
        if y_test.ndim == 1:  # Class indices
            true_labels_test = (y_test == i).astype(int)
        else:  # One-hot or multilabel
            true_labels_test = y_test[:, i]
        
        # Compute ROC curve for this class
        fpr_test, tpr_test, thresholds_test = roc_curve(true_labels_test, whitened_proba_test[:, i])
        
        # Clean and interpolate
        thresholds_test = np.nan_to_num(thresholds_test, nan=0.0, posinf=1.0, neginf=0.0)
        thresholds_test = np.clip(thresholds_test, 0, 1)
        tpr_interp_test = interp1d(fpr_test, tpr_test, bounds_error=False, fill_value=(0, 1))(common_fpr)
        threshold_interp_test = interp1d(fpr_test, thresholds_test, bounds_error=False,
                                fill_value=(thresholds_test[0], thresholds_test[-1]))(common_fpr)
        
        all_tpr_interp_test[i] = tpr_interp_test
        all_threshold_interp_test[i] = threshold_interp_test

        agg_tpr_test = np.average(all_tpr_interp_test, axis=0, weights=gini_weights_test_clean)
        agg_thresholds_test = np.average(all_threshold_interp_test, axis=0, weights=gini_weights_test_clean)
        agg_auc_test = auc(common_fpr, agg_tpr_test)

        # Compute ±1 std across classes
        var_test  = np.average((all_tpr_interp_test  - agg_tpr_test )**2,
                            axis=0, weights=gini_weights_test_clean)
        std_test  = np.sqrt(var_test)

    return common_fpr, agg_tpr_test, agg_thresholds_test, agg_auc_test, std_test

# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)