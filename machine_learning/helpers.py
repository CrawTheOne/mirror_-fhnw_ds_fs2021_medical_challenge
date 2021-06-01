import pandas as pd
import regex as re
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve


def plot_wrapper(plot_object, title, subtitle, xlabel, ylabel, xtick_rotation=None):
    """
    Simple function to set title, labels etc. of a seaborn plot. Helps with readability.
    The function prints the plot.
    
    Arguments
    ---------
    :param plot_object:    a seaborn plot object
    :param title:          str, sets a bold title
    :param subtitle:       str, sets a subtitle
    :param xlabel:         str, replaces xlabel
    :param ylabel:         str, replaces ylabel
    :param xtick_rotation: int, rotates xticks
    
    Returns
    -------
    :return None:          the function prints a plot
    """
    g = plot_object

    # Title, Subtitle and Axis
    g.text(x=0.5,
           y=1.06,
           s=title,
           fontsize=10, weight='bold', ha='center', va='bottom', transform=g.transAxes)
    g.text(x=0.5,
           y=1.01,
           s=subtitle,
           fontsize=10, alpha=0.75, ha='center', va='bottom', transform=g.transAxes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xtick_rotation is not None:
        plt.xticks(rotation=xtick_rotation)
    plt.show()


def plot_validation_curve(estimator, train, test, param_name, param_range, scoring):
    """
    Plot validation curve for param_name in param_range.
    Function has been taken and adapted from "Machine Learning with Python Cookbook", page 202
    """
    train_scores, test_scores = validation_curve(
        # Classifier
        estimator,
        # Feature matrix
        train,
        # Target vector
        test,
        # Hyperparameter to examine
        param_name=param_name,
        # Range of hyperparameters values
        param_range=param_range,
        # Number of folds
        cv=3,
        # Performance metric
        scoring=scoring,
        # Use all computer cores
        n_jobs=-1)

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # Plot mean accuracy scores for training and test sets
    g = plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
    # Plot accuracy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std,
                     test_mean + test_std, color="gainsboro")

    return g

def get_feature_names(pipeline_object):
    """return feature names after transformation for feature name tracking after one_hot_encoding
    
    To be used with a pipeline object, utilizing a preprocessor (sklearn ColumnTransformer) as first step
    Inspiration: https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api/57534118#57534118
    """
    num_feat = pipeline_object[0].transformers_[0][2]
    one_hot_feat = pipeline_object[0].transformers_[1][1]['onehot'].get_feature_names(categorical_features)
    return list(num_feat) + list(one_hot_feat)
