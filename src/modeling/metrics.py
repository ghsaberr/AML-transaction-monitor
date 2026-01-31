import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score


def evaluate_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    return {
        "alert_rate": y_pred.mean(),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_pr_auc(y_true, y_score):
    return average_precision_score(y_true, y_score)
