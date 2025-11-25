"""
Ensemble evaluation utilities.

Combines predictions from multiple models and evaluates ensemble performance.
"""
import numpy as np
from typing import List, Dict, Optional
from .metrics import compute_classification_metrics


def ensemble_average(predictions: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Compute weighted average of model predictions.
    
    Args:
        predictions: list of probability arrays (each N x L)
        weights: optional weights for each model (sum to 1)
        
    Returns:
        Ensemble probabilities (N x L)
    """
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    else:
        assert len(weights) == len(predictions), "Weights must match number of models"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
        
    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += weight * pred
        
    return ensemble_pred


def ensemble_majority_vote(predictions: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Compute majority vote ensemble.
    
    Args:
        predictions: list of probability arrays (each N x L)
        threshold: threshold for converting probabilities to binary
        
    Returns:
        Binary ensemble predictions (N x L)
    """
    binary_preds = [(pred >= threshold).astype(int) for pred in predictions]
    stacked = np.stack(binary_preds, axis=0)  # (num_models, N, L)
    
    # Majority vote
    ensemble_pred = (stacked.sum(axis=0) > len(predictions) / 2).astype(int)
    
    return ensemble_pred


def ensemble_max(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Take maximum prediction across models (more aggressive).
    
    Args:
        predictions: list of probability arrays (each N x L)
        
    Returns:
        Ensemble probabilities (N x L)
    """
    stacked = np.stack(predictions, axis=0)  # (num_models, N, L)
    return stacked.max(axis=0)


def ensemble_min(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Take minimum prediction across models (more conservative).
    
    Args:
        predictions: list of probability arrays (each N x L)
        
    Returns:
        Ensemble probabilities (N x L)
    """
    stacked = np.stack(predictions, axis=0)  # (num_models, N, L)
    return stacked.min(axis=0)


def evaluate_ensemble(
    y_true: np.ndarray,
    predictions: List[np.ndarray],
    model_names: List[str],
    label_names: List[str],
    ensemble_methods: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate multiple ensemble strategies.
    
    Args:
        y_true: true labels (N, L)
        predictions: list of model probabilities
        model_names: names of the models
        label_names: names of the labels
        ensemble_methods: list of methods to try ("average", "max", "min", "vote")
        
    Returns:
        Dictionary of evaluation results
    """
    if ensemble_methods is None:
        ensemble_methods = ["average", "max", "min"]
        
    results = {}
    
    # Individual model performance
    for name, pred in zip(model_names, predictions):
        metrics = compute_classification_metrics(y_true, pred, threshold=0.5, label_names=label_names)
        results[name] = metrics
        
    # Ensemble methods
    for method in ensemble_methods:
        if method == "average":
            ensemble_pred = ensemble_average(predictions)
        elif method == "max":
            ensemble_pred = ensemble_max(predictions)
        elif method == "min":
            ensemble_pred = ensemble_min(predictions)
        elif method == "vote":
            # For vote we need binary, so compute probabilities from vote
            binary_vote = ensemble_majority_vote(predictions)
            ensemble_pred = binary_vote.astype(float)
        else:
            continue
            
        metrics = compute_classification_metrics(y_true, ensemble_pred, threshold=0.5, label_names=label_names)
        results[f"ensemble_{method}"] = metrics
        
    return results


def create_ensemble_comparison_table(
    ensemble_results: Dict,
    label_names: List[str],
) -> Dict:
    """
    Create a summary table comparing all models and ensembles.
    
    Returns:
        Dictionary with model names as keys and metric summaries as values
    """
    summary = {}
    
    for model_name, metrics in ensemble_results.items():
        summary[model_name] = {
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
        }
        
        # Add per-label F1
        for label in label_names:
            if label in metrics["per_label"]:
                summary[model_name][f"{label}_f1"] = metrics["per_label"][label]["f1"]
                
    return summary


def find_optimal_ensemble_weights(
    y_true: np.ndarray,
    predictions: List[np.ndarray],
    label_names: List[str],
    n_trials: int = 100,
) -> np.ndarray:
    """
    Search for optimal ensemble weights using random search.
    
    Args:
        y_true: true labels (N, L)
        predictions: list of model probabilities
        label_names: names of labels
        n_trials: number of random weight combinations to try
        
    Returns:
        Array of optimal weights
    """
    best_f1 = 0.0
    best_weights = None
    
    for _ in range(n_trials):
        # Generate random weights that sum to 1
        weights = np.random.dirichlet(np.ones(len(predictions)))
        
        ensemble_pred = ensemble_average(predictions, weights=weights.tolist())
        metrics = compute_classification_metrics(y_true, ensemble_pred, threshold=0.5, label_names=label_names)
        
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_weights = weights
            
    print(f"Optimal weights: {best_weights}")
    print(f"Macro F1 with optimal weights: {best_f1:.4f}")
    
    return best_weights

