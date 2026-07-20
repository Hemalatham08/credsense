"""
CardioSense - SHAP Explanation Formatting
===========================================
train_model.py builds:
    shap.KernelExplainer(ensemble.predict_proba, background)

KernelExplainer.shap_values() on a 2-class predict_proba function returns
EITHER:
  - a list of 2 arrays, each shape (n_samples, n_features) — one per class
    (older SHAP versions), or
  - a single array of shape (n_samples, n_features, n_classes)
    (newer SHAP versions, ~0.42+)

This module normalizes both into one flat dict per feature so the API
response and the `shap_values` JSONB column always look the same
regardless of which SHAP version is installed.

Run this against your installed environment once and check which branch
actually fires (print type(raw) and raw.shape/len(raw)) — that's the
"confirm SHAP output shape" checklist item.
"""

import numpy as np


def explain_prediction(explainer, X_scaled: np.ndarray, feature_names: list) -> dict:
    raw = explainer.shap_values(X_scaled)

    if isinstance(raw, list):
        # Old-style: list of per-class arrays, each (n_samples, n_features)
        class1_values = raw[1][0]
    else:
        raw = np.array(raw)
        if raw.ndim == 3:
            # Shape (n_samples, n_features, n_classes)
            class1_values = raw[0, :, 1]
        else:
            # Shape (n_samples, n_features) — single-output explainer
            class1_values = raw[0]

    contributions = {
        name: float(value) for name, value in zip(feature_names, class1_values)
    }

    # Sorted by absolute contribution, largest first — drives the
    # recommendation engine and whatever SHAP chart the frontend renders
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "values": contributions,
        "ranked": [{"feature": k, "contribution": v} for k, v in ranked],
    }