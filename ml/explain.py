"""SHAP helpers shared by training (global plots) and the backend (per-transaction)."""
import numpy as np


def build_explainer(classifier, kind, background=None):
    import shap  # imported lazily so the web app can start without it
    if kind == "tree":
        return shap.TreeExplainer(classifier)
    return shap.LinearExplainer(classifier, background)


def positive_class_shap(explainer, X):
    """Return a 2-D array (n_samples, n_features) of SHAP values for the FRAUD class."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv
