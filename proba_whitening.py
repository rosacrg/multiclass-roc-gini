# Multidimensional Gini-weighted ROC analysis for multiclass classification. 
# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)
# This file contains functions for whitening predicted probabilities using ZCA whitening.

import numpy as np
from gini_whitening import whitening_process, whitening_process_stable

def whitening_predicted_proba_stable(X_test, X_train, est, y_test, multilabel=None):
    '''
    This function whitens predicted probabilities using stable ZCA whitening process.
    The whitening matrix is derived from TRAINING probabilities and applied to test probabilities.

    Parameters:
    -----------
    X_test : array-like
        Test features
    X_train : array-like  
        Training features (needed to compute whitening matrix)
    est : estimator
        Trained classifier
    n_classes : int
        Number of classes
    multilabel : bool
        Whether the problem is multilabel

    Returns:
    --------
    whitened_test_proba : numpy.ndarray
        Whitened test probabilities (with rank-based rescaling)
    whitened_train_proba : numpy.ndarray
        Whitened training probabilities (with rank-based rescaling)
    '''
    
    n_classes = len(np.unique(y_test))

    # Get probabilities for both sets
    y_proba_test = est.predict_proba(X_test)
    y_proba_train = est.predict_proba(X_train)
    
    if multilabel is True:
        # Handle multilabel format
        y_proba_test_array = np.zeros((len(X_test), n_classes))
        y_proba_train_array = np.zeros((len(X_train), n_classes))
        
        for i in range(n_classes):
            y_proba_test_array[:, i] = y_proba_test[i][:, 1]
            y_proba_train_array[:, i] = y_proba_train[i][:, 1]
    else:
        # Handle multiclass format
        y_proba_test_array = y_proba_test
        y_proba_train_array = y_proba_train
    
    # Add small noise to break ties and perfect correlations
    noise_scale = 1e-8
    y_proba_train_array += np.random.normal(0, noise_scale, y_proba_train_array.shape)
    y_proba_test_array += np.random.normal(0, noise_scale, y_proba_test_array.shape)
    
    # Compute whitening matrix from TRAINING probabilities only
    var_train = np.diag(np.cov(y_proba_train_array, rowvar=False))  
    corr_train = np.corrcoef(y_proba_train_array, rowvar=False)
    
    # Create stable whitening matrix from training statistics
    W = whitening_process_stable(corr_train, var_train)
    
    # Apply the SAME whitening matrix to both sets
    whitened_test_proba = (W @ y_proba_test_array.T).T
    whitened_train_proba = (W @ y_proba_train_array.T).T
    
    # Rank-based rescaling to [0,1] for interpretable thresholds
    from scipy.stats import rankdata
    whitened_test_proba = rankdata(whitened_test_proba.ravel()).reshape(whitened_test_proba.shape) / len(whitened_test_proba.ravel())
    whitened_train_proba = rankdata(whitened_train_proba.ravel()).reshape(whitened_train_proba.shape) / len(whitened_train_proba.ravel())
    
    return whitened_test_proba, whitened_train_proba


# Use the following function to whiten predicted probabilities if numerical stability is not a concern.
def whitening_predicted_proba(X_test, X_train, est, n_classes, multilabel):
    '''
    This function whitens predicted probabilities using ZCA whitening process.
    The whitening matrix is derived from TRAINING probabilities and applied to test probabilities.

    Parameters:
    -----------
    X_test : array-like
        Test features
    X_train : array-like  
        Training features (needed to compute whitening matrix)
    est : estimator
        Trained classifier
    n_classes : int
        Number of classes
    multilabel : bool
        Whether the problem is multilabel

    Returns:
    --------
    whitened_test_proba : numpy.ndarray
        Whitened test probabilities
    whitened_train_proba : numpy.ndarray
        Whitened training probabilities
    '''
    
    # Get probabilities for both sets
    y_proba_test = est.predict_proba(X_test)
    y_proba_train = est.predict_proba(X_train)
    
    if multilabel is True:
        # Handle multilabel format
        y_proba_test_array = np.zeros((len(X_test), n_classes))
        y_proba_train_array = np.zeros((len(X_train), n_classes))
        
        for i in range(n_classes):
            y_proba_test_array[:, i] = y_proba_test[i][:, 1]
            y_proba_train_array[:, i] = y_proba_train[i][:, 1]
    else:
        # Handle multiclass format
        y_proba_test_array = y_proba_test
        y_proba_train_array = y_proba_train
    
    # Compute whitening matrix from TRAINING probabilities only
    var_train = np.diag(np.cov(y_proba_train_array, rowvar=False))  
    corr_train = np.corrcoef(y_proba_train_array, rowvar=False)
    
    # Create whitening matrix from training statistics
    W = whitening_process(corr_train, var_train)
    
    # Apply the SAME whitening matrix to both sets
    whitened_test_proba = (W @ y_proba_test_array.T).T
    whitened_train_proba = (W @ y_proba_train_array.T).T
    
    return whitened_test_proba, whitened_train_proba

# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)