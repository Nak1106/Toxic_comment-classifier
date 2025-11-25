"""
Calibration utilities for probabilistic predictions.

Includes temperature scaling, ECE, Brier score, and reliability diagrams.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: true labels (N, L) for multi-label
        y_prob: predicted probabilities (N, L)
        n_bins: number of bins for calibration
        
    Returns:
        ECE score (lower is better)
    """
    # Flatten for multi-label case
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob_flat > bin_lower) & (y_prob_flat <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true_flat[in_bin].mean()
            avg_confidence_in_bin = y_prob_flat[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilistic predictions).
    
    Args:
        y_true: true labels (N, L)
        y_prob: predicted probabilities (N, L)
        
    Returns:
        Brier score (lower is better)
    """
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram.
    
    Returns:
        bin_centers: center of each confidence bin
        accuracies: actual accuracy in each bin
        counts: number of samples in each bin
    """
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    accuracies = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob_flat > bin_lower) & (y_prob_flat <= bin_upper)
        count = in_bin.sum()
        
        if count > 0:
            accuracy = y_true_flat[in_bin].mean()
        else:
            accuracy = 0.0
            
        accuracies.append(accuracy)
        counts.append(count)
        
    return bin_centers, np.array(accuracies), np.array(counts)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.
    
    Applies a single learned temperature parameter to logits before sigmoid.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits):
        """Scale logits by temperature."""
        return logits / self.temperature
        
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Fit temperature parameter using validation set.
        
        Args:
            logits: raw model outputs (N, L)
            labels: true labels (N, L)
            lr: learning rate
            max_iter: max optimization iterations
        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        print(f"Learned temperature: {self.temperature.item():.4f}")
        
        return self


def temperature_scale_predictions(
    logits_train: np.ndarray,
    labels_train: np.ndarray,
    logits_test: np.ndarray,
) -> np.ndarray:
    """
    Apply temperature scaling calibration.
    
    Args:
        logits_train: training/validation logits for fitting temperature
        labels_train: training/validation labels
        logits_test: test logits to calibrate
        
    Returns:
        Calibrated probabilities for test set
    """
    device = torch.device("cpu")
    temp_scaler = TemperatureScaling().to(device)
    
    logits_train_t = torch.tensor(logits_train, dtype=torch.float32, device=device)
    labels_train_t = torch.tensor(labels_train, dtype=torch.float32, device=device)
    
    temp_scaler.fit(logits_train_t, labels_train_t)
    
    with torch.no_grad():
        logits_test_t = torch.tensor(logits_test, dtype=torch.float32, device=device)
        scaled_logits = temp_scaler(logits_test_t)
        calibrated_probs = torch.sigmoid(scaled_logits).cpu().numpy()
        
    return calibrated_probs

