import numpy as np
from scipy.stats import spearmanr, pearsonr

def safe_srcc(x, y):
    """Calculate Spearman Rank Correlation Coefficient."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return 0.0
    v = spearmanr(x, y)[0]
    return 0.0 if np.isnan(v) else float(v)

def safe_plcc(x, y):
    """Calculate Pearson Linear Correlation Coefficient."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return 0.0
    v = pearsonr(x, y)[0]
    return 0.0 if np.isnan(v) else float(v)