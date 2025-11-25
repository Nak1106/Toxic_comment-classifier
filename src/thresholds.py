"""
Threshold optimization utilities for multi-label classification.
"""
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import f1_score


def find_optimal_threshold_per_label(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    search_range: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Find optimal threshold for each label independently.
    
    Args:
        y_true: true labels (N, L)
        y_prob: predicted probabilities (N, L)
        metric: metric to optimize ("f1", "precision", "recall")
        search_range: array of thresholds to try (default: 0.1 to 0.9 in steps of 0.05)
        
    Returns:
        Array of optimal thresholds, one per label
    """
    if search_range is None:
        search_range = np.arange(0.1, 1.0, 0.05)
        
    num_labels = y_true.shape[1]
    optimal_thresholds = []
    
    for label_idx in range(num_labels):
        y_t = y_true[:, label_idx]
        y_p = y_prob[:, label_idx]
        
        best_score = 0.0
        best_thresh = 0.5
        
        for thresh in search_range:
            y_pred = (y_p >= thresh).astype(int)
            
            if metric == "f1":
                score = f1_score(y_t, y_pred, zero_division=0)
            elif metric == "precision":
                from sklearn.metrics import precision_score
                score = precision_score(y_t, y_pred, zero_division=0)
            elif metric == "recall":
                from sklearn.metrics import recall_score
                score = recall_score(y_t, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            if score > best_score:
                best_score = score
                best_thresh = thresh
                
        optimal_thresholds.append(best_thresh)
        
    return np.array(optimal_thresholds)


def apply_thresholds(
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Apply per-label thresholds to probabilities.
    
    Args:
        y_prob: predicted probabilities (N, L)
        thresholds: threshold for each label (L,)
        
    Returns:
        Binary predictions (N, L)
    """
    assert y_prob.shape[1] == len(thresholds), "Mismatch between probabilities and thresholds"
    
    y_pred = np.zeros_like(y_prob, dtype=int)
    for i, thresh in enumerate(thresholds):
        y_pred[:, i] = (y_prob[:, i] >= thresh).astype(int)
        
    return y_pred


def threshold_optimization_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
) -> Dict:
    """
    Generate a report of optimal thresholds per label.
    
    Returns:
        Dictionary with threshold and performance info per label
    """
    optimal_thresholds = find_optimal_threshold_per_label(y_true, y_prob, metric="f1")
    
    report = {}
    for i, label in enumerate(label_names):
        thresh = optimal_thresholds[i]
        y_t = y_true[:, i]
        y_p_default = (y_prob[:, i] >= 0.5).astype(int)
        y_p_optimal = (y_prob[:, i] >= thresh).astype(int)
        
        f1_default = f1_score(y_t, y_p_default, zero_division=0)
        f1_optimal = f1_score(y_t, y_p_optimal, zero_division=0)
        
        report[label] = {
            "optimal_threshold": float(thresh),
            "f1_at_0.5": float(f1_default),
            "f1_at_optimal": float(f1_optimal),
            "improvement": float(f1_optimal - f1_default),
        }
        
    return report

