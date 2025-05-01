from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics including accuracy, confusion matrix and F1 scores.
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_by_class': f1_score(y_true, y_pred, average=None)
    }
    return metrics