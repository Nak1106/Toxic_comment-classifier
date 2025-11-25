"""
Fairness and bias evaluation utilities.

Evaluates model bias with respect to identity terms and demographic subgroups.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable


# Common identity terms for bias evaluation
IDENTITY_TERMS = {
    "race": ["black", "white", "asian", "hispanic", "latino", "african"],
    "religion": ["muslim", "christian", "jewish", "hindu", "buddhist", "atheist"],
    "gender": ["man", "woman", "male", "female", "gay", "lesbian", "transgender", "queer"],
    "disability": ["disabled", "blind", "deaf", "mentally ill"],
}


def generate_identity_templates(
    identity_terms: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """
    Generate neutral test sentences with identity terms.
    
    Returns:
        List of dicts with keys: "text", "category", "term"
    """
    if identity_terms is None:
        identity_terms = IDENTITY_TERMS
        
    templates = [
        "I am {term}.",
        "As a {term} person, I think this is interesting.",
        "{term} people deserve respect.",
        "My friend is {term}.",
        "I identify as {term}.",
    ]
    
    examples = []
    for category, terms in identity_terms.items():
        for term in terms:
            for template in templates:
                text = template.format(term=term)
                examples.append({
                    "text": text,
                    "category": category,
                    "term": term,
                })
                
    return examples


def evaluate_identity_bias(
    predict_fn: Callable,
    identity_examples: List[Dict],
    label_names: List[str],
) -> pd.DataFrame:
    """
    Evaluate model predictions on identity templates.
    
    Args:
        predict_fn: function that takes text and returns probabilities (1, L)
        identity_examples: list of identity template dicts
        label_names: names of the labels
        
    Returns:
        DataFrame with predictions for each identity term
    """
    results = []
    
    for example in identity_examples:
        text = example["text"]
        probs = predict_fn(text)
        
        row = {
            "text": text,
            "category": example["category"],
            "term": example["term"],
        }
        
        for i, label in enumerate(label_names):
            row[f"prob_{label}"] = float(probs[0, i])
            
        results.append(row)
        
    return pd.DataFrame(results)


def compute_bias_metrics(
    identity_results: pd.DataFrame,
    label_names: List[str],
) -> Dict:
    """
    Compute bias metrics from identity evaluation results.
    
    Returns:
        Dictionary of bias metrics per identity category and label
    """
    metrics = {}
    
    for category in identity_results["category"].unique():
        metrics[category] = {}
        subset = identity_results[identity_results["category"] == category]
        
        for label in label_names:
            prob_col = f"prob_{label}"
            
            # Mean and std of predicted toxicity for this identity category
            mean_prob = subset[prob_col].mean()
            std_prob = subset[prob_col].std()
            max_prob = subset[prob_col].max()
            
            # Per-term statistics
            term_stats = subset.groupby("term")[prob_col].mean().to_dict()
            
            metrics[category][label] = {
                "mean_prob": float(mean_prob),
                "std_prob": float(std_prob),
                "max_prob": float(max_prob),
                "term_stats": term_stats,
            }
            
    return metrics


def compare_subgroup_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_mask: np.ndarray,
    label_names: List[str],
) -> Dict:
    """
    Compare model performance on a subgroup vs the rest.
    
    Args:
        y_true: true labels (N, L)
        y_prob: predicted probabilities (N, L)
        subgroup_mask: boolean mask indicating subgroup membership (N,)
        label_names: names of labels
        
    Returns:
        Dictionary with AUC and F1 comparison
    """
    from sklearn.metrics import roc_auc_score, f1_score
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    comparison = {}
    
    for i, label in enumerate(label_names):
        # Subgroup performance
        subgroup_y_true = y_true[subgroup_mask, i]
        subgroup_y_prob = y_prob[subgroup_mask, i]
        subgroup_y_pred = y_pred[subgroup_mask, i]
        
        # Rest performance
        rest_y_true = y_true[~subgroup_mask, i]
        rest_y_prob = y_prob[~subgroup_mask, i]
        rest_y_pred = y_pred[~subgroup_mask, i]
        
        try:
            subgroup_auc = roc_auc_score(subgroup_y_true, subgroup_y_prob)
        except:
            subgroup_auc = float("nan")
            
        try:
            rest_auc = roc_auc_score(rest_y_true, rest_y_prob)
        except:
            rest_auc = float("nan")
            
        subgroup_f1 = f1_score(subgroup_y_true, subgroup_y_pred, zero_division=0)
        rest_f1 = f1_score(rest_y_true, rest_y_pred, zero_division=0)
        
        comparison[label] = {
            "subgroup_auc": float(subgroup_auc),
            "rest_auc": float(rest_auc),
            "subgroup_f1": float(subgroup_f1),
            "rest_f1": float(rest_f1),
            "auc_gap": float(subgroup_auc - rest_auc) if not np.isnan(subgroup_auc) else float("nan"),
            "f1_gap": float(subgroup_f1 - rest_f1),
        }
        
    return comparison

