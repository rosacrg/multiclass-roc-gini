# Multidimensional Gini-weighted ROC analysis for multiclass classification.
# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)
# This file contains functions for calculating the Gini Mean Difference (GMD), performing ZCA whitening, and computing the multidimensional Gini index.

import numpy as np
from scipy.spatial.distance import pdist

def gmd(x):
    """
    Calculate the Gini Mean Difference (GMD).
    
    Parameters:
    x : array-like (list, numpy array, pandas Series, pandas DataFrame)
        Data values for which GMD should be calculated
        
    Returns:
    float or numpy.ndarray
        GMD value(s) - a single value for a vector input, or 
        an array of values for a DataFrame input (one per column).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n <= 1:
        return np.nan  # Not enough data
    sum_diffs = np.sum(pdist(x[:, np.newaxis], metric='cityblock'))  
    return sum_diffs / (n * (n - 1))


# Whitening Process: ZCA 
def whitening_process(corr, var):
    ''' W = E * D^(-1/2) * E^T * D_var^(-1/2) 
    ZCA whitening process following formula 14 in "Extending the Gini Index to Higher Dimensions via Whitening
    Processes" by Gennaro Auricchio, Paolo Giudici, and Giuseppe Toscani.
    - E is the eigenvector matrix of the correlation matrix,
    - D is the diagonal matrix of eigenvalues of the correlation matric
    - D_var is the diagonal matrix of variances. 

    Parameters:
    corr : numpy.ndarray
    var : numpy.ndarray

    Returns:
    numpy.ndarray
    W: ZCA whitening matrix
    '''
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[eigvals <= 0] = np.finfo(float).eps  # Avoid division by zero
    D = np.diag(eigvals**(-1/2))
    D_var = np.diag(var**(-1/2))
    W = eigvecs @ D @ eigvecs.T @ D_var
    return W


def whitening_process_stable(corr, var, regularization=1e-6):
    """
    Numerically stable ZCA whitening with proper regularization, to avoid probabilities to explode.

     W = E * D^(-1/2) * E^T * D_var^(-1/2) 
    ZCA whitening process following formula 14 in "Extending the Gini Index to Higher Dimensions via Whitening
    Processes" by Gennaro Auricchio, Paolo Giudici, and Giuseppe Toscani.
    - E is the eigenvector matrix of the correlation matrix,
    - D is the diagonal matrix of eigenvalues of the correlation matric
    - D_var is the diagonal matrix of variances. 

    Parameters:
    corr : numpy.ndarray
    var : numpy.ndarray

    Returns:
    numpy.ndarray
    W: ZCA whitening matrix
    """
    # Handle edge cases
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=0.0)
    var = np.nan_to_num(var, nan=1.0, posinf=1.0, neginf=1e-12)
    
    # Strong regularization for stability
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, regularization)
    var_reg = np.maximum(var, regularization)
    
    # Cap maximum scaling to prevent explosion
    max_scaling = 100.0
    eigvals_scaled = np.minimum(eigvals**(-1/2), max_scaling)
    var_scaled = np.minimum(var_reg**(-1/2), max_scaling)
    
    D = np.diag(eigvals_scaled)
    D_var = np.diag(var_scaled)
    W = eigvecs @ D @ eigvecs.T @ D_var
    
    # Final safety check - clip extreme values
    W = np.clip(W, -max_scaling, max_scaling)
    
    return W


# Multidimensional Gini index
def multidim_gini(whitened_data):
    ''' GM = sum((|m_i| / sum(|m_i|)) * G_i_whitenned)
    Multidimensional Gini index following formula 15 in "Extending the Gini Index to Higher Dimensions via Whitening
    Processes" by Gennaro Auricchio, Paolo Giudici, and Giuseppe Toscani.

    It is a weighted avergage of the Gini index of each componenent.

    Parameters:
    whitened_data : numpy.ndarray

    Returns:
    - weights of each unidimensional Gini index: numpy.ndarray
    - multidimensional_gini : float
    '''
    # Compute the whitened mean
    m_star = np.mean(whitened_data, axis=0)
    abs_means = np.abs(m_star)
    
    # Get the number of features
    n_features = whitened_data.shape[1]
    
    # Calculate all Gini components first
    gini_components = np.zeros(n_features)
    for i in range(n_features):
        gini_components[i] = gmd(whitened_data[:, i])
        print(f'Unidimensional Gini for class {i}: {gini_components[i]:.4f}')
    
    # Calculate weights
    weights = abs_means / np.sum(abs_means)
    
    # Compute the multidimensional Gini coefficient
    multidimensional_gini = np.sum(weights * gini_components)
    
    return multidimensional_gini, weights

# Aggregating Multiclass ROC Curves, with Applications to ESG and Credit Risk Management. Rosa Carolina Rosciano, Univerity of Pavia (2025)
