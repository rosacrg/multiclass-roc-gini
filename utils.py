from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_regressor
import torch
from typing import Union
from xgboost import XGBClassifier, XGBRegressor


def manipulate_testdata(xtrain: pd.DataFrame, 
                        xtest: pd.DataFrame, 
                        model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator,
                               torch.nn.Module],
                        variable: str):
    """
    Manipulate the given variable column in test data based on values of that variable in train data.

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    model : Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor.
    variable: str 
            Name of variable.

    Returns
    -------
    pd.DataFrame
            The manipulated data.
    """
    # create xtest_rm
    xtest_rm = xtest.copy()
    if isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
        # specific settings for CatBoost models
        cat_indices = model.get_cat_feature_indices()
        # replace variable with mode or mean based on its type
        if xtrain.columns.get_loc(variable) not in cat_indices:
            mean_value = xtrain[variable].mean()
            xtest_rm[variable] = mean_value
        else:
            mode_value = xtrain[variable].mode()[0]
            xtest_rm[variable] = mode_value
        
    elif isinstance(model, (BaseEstimator, XGBClassifier, XGBRegressor)):
        # specific settings for sklearn and xgboost models
        if isinstance(xtrain[variable].dtype, pd.CategoricalDtype):
            mode_value = xtrain[variable].mode()[0]
            xtest_rm[variable] = mode_value
        else:
            mean_value = xtrain[variable].mean()
            xtest_rm[variable] = mean_value

    elif isinstance(model, torch.nn.Module):
        # specific settings for torch models
        if isinstance(xtrain[variable].dtype, pd.CategoricalDtype):
            mode_value = xtrain[variable].mode()[0]
            xtest_rm[variable] = mode_value
        else:
            mean_value = xtrain[variable].mean()
            xtest_rm[variable] = mean_value
    else:
        raise ValueError("Unsupported model type")
    return xtest_rm


def convert_to_dataframe(*args):
    """
    Convert inputs to DataFrames.

    Parameters
    ----------
    *args
            A variable number of input objects that can be converted into Pandas DataFrames (e.g., lists, dictionaries, numpy arrays).

    Returns
    -------
    list of pd.DataFrame
            A list of Pandas DataFrames created from the input objects.    
    """
    return [pd.DataFrame(arg).reset_index(drop=True) for arg in args]


def validate_variables(variables: Union[list, str], xtrain: pd.DataFrame):
    """
    Check if variables are valid and exist in the train dataset.

    Parameters
    ----------
    variables: list or str
            Variables.
    xtrain : pd.DataFrame
            A dataframe including train data.

    Raises
    -------
    ValueError
            If variables is not a list, not a string or if any variable does not exist in xtrain.
    """
    if isinstance(variables, str):
        variables = [variables]
    elif not isinstance(variables, list):
        raise ValueError("Variables input must be a list")
    for var in variables:
        if var not in xtrain.columns:
            raise ValueError(f"{var} is not in the variables")


def check_nan(*dataframes):
    """
    Check if any of the provided DataFrames contain missing values.

    Parameters
    ----------
    *dataframes : pd.DataFrame
        A variable number of DataFrame objects to check for NaN values.

    Raises
    ------
    ValueError
        If any DataFrame contains missing (NaN) values.
    TypeError
        If any input is not a Pandas DataFrame.
    """

    for i, df in enumerate(dataframes, start=1):
        if isinstance(df, pd.DataFrame):  # Ensure df is a DataFrame
            if df.isna().sum().sum() > 0:  # Check if there are any missing values
                raise ValueError(f"DataFrame {i} contains missing values")
        else:
            raise TypeError(f"Item {i} is not a pandas DataFrame")



def find_yhat(model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator,
              torch.nn.Module],
              xtest: pd.DataFrame):
    """
    Find predicted values for the manipulated data.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    model : Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor.

    Returns
    -------
            The yhat value.
    """
    if is_classifier(model):
        yhat = [x[1] for x in model.predict_proba(xtest)]
    elif is_regressor(model):
        yhat = model.predict(xtest)
    elif isinstance(model, torch.nn.Module):
        xtest_tensor = torch.tensor(xtest.values, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            yhat = model(xtest_tensor)
        if yhat.shape[1] == 2:  # binary classification
            yhat = yhat[:, 1].numpy()
        else:
            yhat = yhat.numpy()
    else:
        raise ValueError("The model type is not recognized for prediction")

    return yhat


def plot_model_curves(x, curves, model_name, prefix="Curve", title="", xlabel='Steps', ylabel='Values'):
    """
    Plot RGA/RGE/RGR curves for a given model using a template.

    Parameters:
    x (np.ndarray or list): X-axis values.
    curves (list of np.ndarray): List of curves to plot (e.g., [rga, rge, rgr])
    model_name (str): Name of the model (e.g., "RF", "XGB") used for labeling
    prefix (str): Prefix for legend labels (e.g., "Curve", "Difference Random")
    title (str): Title of the plot
    xlabel (str): Label for the x-axis
    ylabel (str): Label for the y-axis
    """
    labels_base = ["RGA", "RGE", "RGR"]
    labels = [f"{label} {prefix} {model_name}" for label in labels_base]

    plt.figure(figsize=(6, 4))
    for curve, label in zip(curves, labels):
        plt.plot(x, curve, linestyle='-', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xlim([0, 1])
    plt.grid(True)


def plot_metric_distribution(metric_values, print_label, xlabel, title, bar_label="Model", bins=60):
    """
    Plot a histogram for the given metric values with normalized counts.

    Parameters:
        metric_values (np.ndarray): Computed metric values
        print_label (str): Label to print for the mean volume
        xlabel (str): Label for the x-axis of the plot
        title (str): Title for the histogram plot
        bar_label (str): Label in the legend
        bins (int): Number of bins for the histogram
    """
    # Flatten and compute mean
    flat_vals = metric_values.flatten()
    total_sum = np.sum(flat_vals)
    num_elements = flat_vals.size
    normalized_volume = total_sum / num_elements

    # Print mean volume
    print(f"{print_label}: {normalized_volume:.5f}")

    # Histogram
    counts, bin_edges = np.histogram(flat_vals, bins=bins)
    max_count = counts.max()
    counts_norm = counts / max_count
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts_norm, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, label=bar_label)
    plt.xlabel(xlabel)
    plt.ylabel('Normalized Counts')
    plt.title(title)
    plt.grid(True)
    plt.legend()

def plot_metric_distribution_diff(metric_values, print_label, xlabel, title, bar_label="Model", bins=60):
    """
    Plot a histogram for the given metric differences values with normalized counts.

    Parameters:
        metric_values (np.ndarray): Computed metric values
        print_label (str): Label to print for the mean volume
        xlabel (str): Label for the x-axis of the plot
        title (str): Title for the histogram plot
        bar_label (str): Label in the legend
        bins (int): Number of bins for the histogram
    """
    # Flatten and compute mean
    flat_vals = metric_values.flatten()
    total_sum = np.sum(flat_vals)
    num_elements = flat_vals.size
    normalized_volume = total_sum / num_elements

    # Print mean volume
    print(f"{print_label}: {normalized_volume:.5f}")

    # Histogram
    counts, bin_edges = np.histogram(flat_vals, bins=bins)
    max_count = counts.max()
    counts_norm = counts / max_count
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts_norm, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, label=bar_label, color='green')
    plt.axvline(0, color='red', linestyle='--', label='No Difference')
    plt.xlabel(xlabel)
    plt.ylabel('Normalized Counts')
    plt.title(title)
    plt.grid(True)
    plt.legend()

def plot_mean_histogram(rga, rge, rgr, *,
                        model_name: str,
                        bar_label: str,
                        mean_type: str):
    """
    Compute different means (arithmetic, geometric, quadratic) of (rga, rge, rgr),
    then call plot_metric_distribution_diff with the appropriate labels.

    Parameters
    ----------
    rga, rge, rgr : array‐like
        Three metrics for this model.
    model_name : str
        Name used inside the “print_label” (e.g. "Logistic Regression").
    bar_label : str
        The text to show under each bar in the histogram legend.
    mean_type : str
        Mean formula.
    """
    if mean_type == "arithmetic":
        values = (rga + rge + rgr) / 3
        print_label = f"Mean volume Arithmetic {model_name}"
        xlabel = "Normalized Arithmetic Mean"
        title = f"Histogram of Normalized Arithmetic Mean Values ({model_name})"
    elif mean_type == "geometric":
        values = np.cbrt(rga * rge * rgr)
        print_label = f"Mean volume Geometric {model_name}"
        xlabel = "Normalized Geometric Mean (1/3)"
        title = f"Histogram of Normalized Geometric Mean (1/3) Values ({model_name})"
    elif mean_type == "quadratic":
        values = np.sqrt((rga ** 2 + rge ** 2 + rgr ** 2) / 3)
        print_label = f"Mean volume Quadratic Mean (RMS) {model_name}"
        xlabel = "Normalized Quadratic Mean (RMS)"
        title = f"Histogram of Normalized Quadratic Mean (RMS) Values ({model_name})"
    else:
        raise ValueError("`mean_type` is not added yet")

    plot_metric_distribution(
        metric_values=values,
        print_label=print_label,
        xlabel=xlabel,
        title=title,
        bar_label=bar_label
    )

def plot_diff_mean_histogram(rga, rge, rgr, *,
                        model_name: str,
                        bar_label: str,
                        mean_type: str):
    """
    Compute different means of differences values with base model (arithmetic, geometric, quadratic) of (rga, rge, rgr),
    then call plot_metric_distribution_diff with the appropriate labels.

    Parameters
    ----------
    rga, rge, rgr : array‐like
        Three metrics for this model.
    model_name : str
        Name used inside the “print_label” (e.g. "Logistic Regression").
    bar_label : str
        The text to show under each bar in the histogram legend.
    mean_type : str
        Mean formula.
    """
    if mean_type == "arithmetic":
        values = (rga + rge + rgr) / 3
        print_label = f"Difference Arithmetic {model_name}"
        xlabel = "Normalized Difference Arithmetic Mean"
        title = f"Histogram of Normalized Difference Arithmetic Mean Values  ({model_name})"
    elif mean_type == "geometric":
        values = np.cbrt(rga * rge * rgr)
        print_label = f"Difference Geometric Mean (1/3) {model_name}"
        xlabel = "Normalized Difference Geometric Mean (1/3)"
        title = f"Histogram of Normalized Difference Geometric Mean (1/3) Values ({model_name})"
    elif mean_type == "quadratic":
        values = np.sqrt((rga ** 2 + rge ** 2 + rgr ** 2) / 3)
        print_label = f"Difference Mean volume Quadratic Mean (RMS) {model_name}"
        xlabel = "Normalized Difference Quadratic Mean (RMS)"
        title = f"Histogram of Normalized Difference Quadratic Mean (RMS) Values ({model_name})"
    else:
        raise ValueError("`mean_type` is not added yet")

    plot_metric_distribution_diff(
        metric_values=values,
        print_label=print_label,
        xlabel=xlabel,
        title=title,
        bar_label=bar_label
    )