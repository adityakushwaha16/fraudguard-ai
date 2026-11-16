"""Loads the trained artifacts ONCE and serves predictions + SHAP explanations.

Never returns a made-up score: if artifacts are missing (or were produced from synthetic
smoke-test data) it raises ModelNotAvailable and the API answers 503.
"""
import json
import threading
from pathlib import Path
import joblib
import pandas as pd

from ml.features import (BINARY_FEATURES, FEATURE_LABELS, FEATURES, INPUT_COLUMNS,
                         engineer_features)


class ModelNotAvailable(Exception):
    pass


def risk_level(prob: float, threshold: float) -> str:
    """High = at/above the tuned decision threshold; Medium = at least half of it; else Low."""
    if prob >= threshold:
        return "High"
    if prob >= threshold * 0.5:
        return "Medium"
    return "Low"


class Predictor:
    def __init__(self, model_dir, allow_synthetic=False):
        self.model_dir = Path(model_dir)
        self.allow_synthetic = allow_synthetic
        self._lock = threading.Lock()
        self._loaded = False
        self.error = None

    # ---------- loading ----------
    def _load(self):
        with self._lock:
            if self._loaded:
                return
            try:
                cfg_path = self.model_dir / "model_config.json"
                pipe_path = self.model_dir / "fraud_pipeline.joblib"
                if not cfg_path.exists() or not pipe_path.exists():
                    raise ModelNotAvailable(
                        "No trained model found. Download PaySim into datasets/ and run "
                        "`python -m ml.train --data datasets/<file>.csv`.")
                self.config = json.loads(cfg_path.read_text())
                if self.config.get("synthetic") and not self.allow_synthetic:
                    raise ModelNotAvailable("Model was trained on synthetic smoke-test data and is disabled.")
                # Training and prediction must use EXACTLY the same feature columns.
                trained_with = (self.config.get("features") or {}).get("all")
                if trained_with != FEATURES:
                    raise ModelNotAvailable(
                        "Model feature mismatch: this model was trained with a different feature set "
                        f"({trained_with}) than the current code expects ({FEATURES}). "
                        "Retrain with `python -m ml.train`.")
                self.pipeline = joblib.load(pipe_path)
                fitted_cols = getattr(self.pipeline, "feature_names_in_", None)
                if fitted_cols is not None and list(fitted_cols) != FEATURES:
                    raise ModelNotAvailable("Model feature mismatch inside the saved pipeline. Retrain.")
                self.prep, self.clf = self.pipeline[:-1], self.pipeline[-1]
                self.threshold = float(self.config["threshold"])
                self.feature_names = self.config["transformed_feature_names"]
                self.explainer = None
                self.explain_error = None
                try:
                    from ml.explain import build_explainer
                    bg_path = self.model_dir / "shap_background.joblib"
                    bg = joblib.load(bg_path) if bg_path.exists() else None
                    self.explainer = build_explainer(self.clf, self.config["explainer"], bg)
                except Exception as exc:  # shap missing or explainer failed
                    self.explain_error = f"SHAP unavailable: {exc}"
                self._loaded = True
            except ModelNotAvailable as exc:
                self.error = str(exc)
                raise
            except Exception as exc:
                self.error = f"Failed to load model: {exc}"
                raise ModelNotAvailable(self.error)

    def status(self):
        try:
            self._load()
        except ModelNotAvailable as exc:
            return {"available": False, "reason": str(exc)}
        metrics_path = self.model_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        sel = metrics.get("selected")
        return {"available": True, "model_name": self.config["model_name"],
                "model_version": self.config["model_version"], "threshold": self.threshold,
                "dataset": self.config.get("dataset"), "explanations": self.explainer is not None,
                "test_metrics": metrics.get("models", {}).get(sel, {}).get("test"),
                "comparison": {k: v["test"] for k, v in metrics.get("models", {}).items()},
                "global_importance": (metrics.get("shap") or {}).get("global_importance", [])[:10]}

    # ---------- inference ----------
    def predict(self, tx: dict) -> dict:
        self._load()
        row = pd.DataFrame([{c: tx[c] for c in INPUT_COLUMNS}])
        feats = engineer_features(row)
        prob = float(self.pipeline.predict_proba(feats)[0, 1])
        result = {"fraud_probability": prob,
                  "label": "Fraudulent" if prob >= self.threshold else "Legitimate",
                  "risk_level": risk_level(prob, self.threshold),
                  "threshold": self.threshold, "model_version": self.config["model_version"],
                  "explanation": None, "explanation_error": self.explain_error}
        if self.explainer is not None:
            try:
                result["explanation"] = self._explain(feats, prob, result["risk_level"])
            except Exception as exc:
                result["explanation_error"] = f"SHAP failed: {exc}"
        return result

    def _explain(self, feats, prob, risk):
        from ml.explain import positive_class_shap
        Xt = self.prep.transform(feats)
        sv = positive_class_shap(self.explainer, Xt)[0]
        raw_vals = self._display_values(feats.iloc[0])
        contribs = []
        for f, s in zip(self.feature_names, sv):
            c = {"feature": f, "label": FEATURE_LABELS.get(f, f), "value": raw_vals.get(f),
                 "kind": "binary" if f in BINARY_FEATURES else "number",
                 "shap": float(s), "direction": "increases" if s > 0 else "decreases"}
            c["display"] = self._fmt(c)
            contribs.append(c)
        contribs.sort(key=lambda d: -abs(d["shap"]))
        return {"contributions": contribs, "summary": self._summarise(contribs, prob, risk)}

    @staticmethod
    def _display_values(row):
        vals = {k: float(row[k]) for k in row.index if k != "type"}
        vals["type_TRANSFER"] = 1.0 if row["type"] == "TRANSFER" else 0.0
        vals["type_CASH_OUT"] = 1.0 if row["type"] == "CASH_OUT" else 0.0
        return vals

    @staticmethod
    def _fmt(c):
        v, f = c["value"], c["feature"]
        if f in BINARY_FEATURES:
            return "yes" if v else "no"
        if f == "amount_to_orig_balance":
            return f"{v * 100:.0f}% of balance" if v < 10 else ">= 1000% of balance"
        if f == "hour_of_day":
            return f"{int(v):02d}:00"
        return f"{v:,.2f}"

    def _summarise(self, contribs, prob, risk):
        up = [c for c in contribs if c["shap"] > 0][:3]
        down = [c for c in contribs if c["shap"] < 0][:2]
        text = f"This transaction was scored as {risk} risk (estimated fraud probability {prob*100:.1f}%). "
        if up:
            text += "The factors that pushed the risk up most were: " + "; ".join(
                f"{c['label']} ({self._fmt(c)})" for c in up) + ". "
        if down:
            text += "Factors that pulled the risk down: " + "; ".join(
                f"{c['label']} ({self._fmt(c)})" for c in down) + ". "
        text += "This is a model estimate based on learned patterns, not confirmed fraud."
        return text
