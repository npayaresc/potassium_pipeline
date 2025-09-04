"""
Classification Helper Functions for Magnesium Pipeline

This module provides utilities for converting the regression pipeline to classification,
specifically for binary classification of magnesium levels (above/below threshold).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef
)
import logging

logger = logging.getLogger(__name__)


def create_binary_labels(y_values, threshold=0.3):
    """
    Convert continuous magnesium values to binary labels.
    
    Args:
        y_values: Array or Series of magnesium concentration values
        threshold: Threshold for classification (default 0.3%)
        
    Returns:
        Binary labels: 1 if >= threshold (high), 0 if < threshold (low)
    """
    binary_labels = (y_values >= threshold).astype(int)
    
    # Log class distribution
    n_high = np.sum(binary_labels == 1)
    n_low = np.sum(binary_labels == 0)
    total = len(binary_labels)
    
    logger.info(f"Binary label distribution (threshold={threshold}):")
    logger.info(f"  High (>={threshold}): {n_high}/{total} ({n_high/total*100:.1f}%)")
    logger.info(f"  Low (<{threshold}): {n_low}/{total} ({n_low/total*100:.1f}%)")
    
    # Check for class imbalance
    imbalance_ratio = max(n_high, n_low) / min(n_high, n_low) if min(n_high, n_low) > 0 else np.inf
    if imbalance_ratio > 3:
        logger.warning(f"Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
        logger.warning("Consider using class weights or resampling techniques")
    
    return binary_labels


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities for positive class (optional)
        
    Returns:
        Dictionary of classification metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity and sensitivity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    
    # AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = None
            logger.warning("Could not calculate ROC AUC - possibly only one class in y_true")
    
    # Sample counts
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(np.sum(y_true == 1))
    metrics['n_negative'] = int(np.sum(y_true == 0))
    
    return metrics


def print_classification_report(y_true, y_pred, model_name="Model"):
    """
    Print a formatted classification report.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        model_name: Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} CLASSIFICATION REPORT")
    print('='*60)
    
    # Get metrics
    metrics = calculate_classification_metrics(y_true, y_pred)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Low   High")
    print(f"Actual Low     [{metrics['true_negatives']:5d} {metrics['false_positives']:5d}]")
    print(f"       High    [{metrics['false_negatives']:5d} {metrics['true_positives']:5d}]")
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1 Score:           {metrics['f1']:.4f}")
    print(f"  MCC:                {metrics['mcc']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    
    # Print class distribution
    print(f"\nClass Distribution:")
    print(f"  High (>=0.3):  {metrics['n_positive']}/{metrics['n_samples']} ({metrics['n_positive']/metrics['n_samples']*100:.1f}%)")
    print(f"  Low (<0.3):    {metrics['n_negative']}/{metrics['n_samples']} ({metrics['n_negative']/metrics['n_samples']*100:.1f}%)")
    
    print('='*60)
    
    # Detailed sklearn classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=['Low (<0.3)', 'High (>=0.3)'],
                               digits=4))


def calculate_class_weights(y_train, method='balanced'):
    """
    Calculate class weights for handling imbalanced data.
    
    Args:
        y_train: Training labels
        method: 'balanced' or 'custom'
        
    Returns:
        Dictionary of class weights
    """
    unique_classes = np.unique(y_train)
    
    if method == 'balanced':
        # Sklearn's balanced formula: n_samples / (n_classes * np.bincount(y))
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        class_counts = np.bincount(y_train)
        
        weights = {}
        for cls in unique_classes:
            weights[cls] = n_samples / (n_classes * class_counts[cls])
            
    elif method == 'custom':
        # Custom weights based on magnesium importance
        # Give more weight to minority class and critical misclassifications
        class_counts = np.bincount(y_train)
        
        # Base weight inversely proportional to frequency
        base_weight_0 = 1.0 / class_counts[0]
        base_weight_1 = 1.0 / class_counts[1]
        
        # Normalize so the larger weight is 1.0
        max_weight = max(base_weight_0, base_weight_1)
        weights = {
            0: base_weight_0 / max_weight,
            1: base_weight_1 / max_weight
        }
        
        # Additional adjustment: High magnesium (>=0.3) is agronomically important
        # Slightly increase weight for high class to reduce false negatives
        weights[1] *= 1.2
        
    else:
        # No weighting
        weights = {cls: 1.0 for cls in unique_classes}
    
    logger.info(f"Class weights ({method}): {weights}")
    return weights


def get_stratified_split_indices(X, y, test_size=0.2, random_state=42):
    """
    Get stratified train/test split indices for classification.
    
    Args:
        X: Features
        y: Binary labels
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        train_indices, test_indices
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y))
    
    # Log split statistics
    y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
    y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
    
    logger.info(f"Stratified split created:")
    logger.info(f"  Train: {len(train_idx)} samples")
    logger.info(f"    High: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    logger.info(f"    Low: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    logger.info(f"  Test: {len(test_idx)} samples")
    logger.info(f"    High: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
    logger.info(f"    Low: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
    
    return train_idx, test_idx