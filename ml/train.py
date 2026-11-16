"""Chronological PaySim training, validation, final testing and production export.

Usage (recommended first chronological run):
    python -m ml.train --data datasets/paysim.csv --sample 300000 --n-iter 2 --tune-rows 100000

Full-data run (slower and more memory-intensive):
    python -m ml.train --data datasets/paysim.csv --n-iter 6 --tune-rows 200000

The data is split by PaySim's `step` column BEFORE any optional sampling:
    earliest 60% of time -> training
    next 20%             -> validation/model selection/threshold tuning
    latest 20%           -> final holdout test

If --sample is supplied, rows are sampled separately inside each already-defined
chronological partition. No future-period rows can enter the training partition.
The final production pipeline is refitted on train+validation after the holdout
metrics are computed. The test set is never used for fitting or threshold selection.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    ALLOWED_TYPES, CATEGORICAL_FEATURES, EXCLUDED_COLUMNS, FEATURES,
    FEATURE_LABELS, INPUT_COLUMNS, NUMERIC_FEATURES, PAYSIM_REQUIRED_COLUMNS,
    TARGET, engineer_features, from_paysim,
)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(categories=[ALLOWED_TYPES], handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def clean_names(names):
    return [n.split("__", 1)[1] for n in names]


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_val, proba_val, beta=1.0):
    p, r, thresholds = precision_recall_curve(y_val, proba_val)
    p, r = p[:-1], r[:-1]
    if len(thresholds) == 0:
        return 0.5
    f_beta = (1 + beta**2) * p * r / np.maximum(beta**2 * p + r, 1e-12)
    return float(thresholds[int(np.argmax(f_beta))])


def candidates(pos_weight, demo=False):
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [8, 12, None],
             "clf__min_samples_leaf": [1, 5, 20]},
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (
            XGBClassifier(scale_pos_weight=pos_weight, eval_metric="aucpr", n_jobs=-1,
                          random_state=SEED, tree_method="hist"),
            {"clf__n_estimators": [150, 300], "clf__max_depth": [4, 6, 8],
             "clf__learning_rate": [0.05, 0.1, 0.2], "clf__subsample": [0.7, 1.0]},
        )
    except ImportError:
        if not demo:
            sys.exit("xgboost is not installed. Run: pip install -r requirements.txt")
        print("WARNING (demo only): xgboost missing, skipping XGBoost")
    return models


def chronological_partition(df):
    """Split by whole PaySim steps; never put the same step in multiple partitions."""
    counts = df.groupby("step", sort=True).size()
    if len(counts) < 3:
        raise ValueError("Not enough distinct PaySim step values for chronological train/val/test.")
    cumulative = counts.to_numpy().cumsum()
    n = int(cumulative[-1])
    cut1 = int(np.searchsorted(cumulative, n * 0.60, side="left") + 1)
    cut2 = int(np.searchsorted(cumulative, n * 0.80, side="left") + 1)
    cut1 = min(max(cut1, 1), len(counts) - 2)
    cut2 = min(max(cut2, cut1 + 1), len(counts) - 1)
    steps = counts.index.to_numpy()
    ranges = [steps[:cut1], steps[cut1:cut2], steps[cut2:]]
    parts = [df[df["step"].isin(s)].copy() for s in ranges]
    return parts


def sample_partition(part, n):
    """Randomly sample only within a single time partition, stratifying where possible."""
    if n >= len(part):
        return part
    if n < 2:
        raise ValueError("The requested --sample is too small for three chronological partitions.")
    counts = part[TARGET].value_counts()
    stratify = part[TARGET] if len(counts) == 2 and counts.min() >= 2 else None
    sampled, _ = train_test_split(part, train_size=n, random_state=SEED, stratify=stratify)
    return sampled


def maybe_sample_chronologically(parts, total_sample):
    if not total_sample:
        return parts
    total_rows = sum(len(p) for p in parts)
    targets = [int(round(total_sample * len(p) / total_rows)) for p in parts]
    targets[-1] = total_sample - sum(targets[:-1])
    sampled = [sample_partition(p, min(target, len(p))) for p, target in zip(parts, targets)]
    return sampled


def step_range(part):
    return {"min": int(part["step"].min()), "max": int(part["step"].max())}


def main():
    parser = argparse.ArgumentParser(description="Chronological PaySim fraud model training")
    parser.add_argument("--data")
    parser.add_argument("--sample", type=int, default=0,
                        help="optional total row sample, drawn separately within each chronological split")
    parser.add_argument("--n-iter", type=int, default=6, help="random-search iterations per model")
    parser.add_argument("--tune-rows", type=int, default=200000,
                        help="maximum training-partition rows used for hyperparameter search")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="F-beta used to choose validation threshold; beta=2 favours recall")
    parser.add_argument("--demo-synthetic", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    demo = args.demo_synthetic
    output_dir = Path(args.out) if args.out else ROOT / ("models_smoke_test" if demo else "models")
    image_dir = (ROOT / "static" / "img" / "ml") if not demo else output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if demo:
        from .synthetic import make_synthetic
        print("!!! SYNTHETIC DATA: pipeline smoke test only; artifacts are NOT production-usable. !!!")
        frame = make_synthetic()
        data_name = "SYNTHETIC (smoke test)"
    else:
        if not args.data or not Path(args.data).exists():
            sys.exit("Dataset not found. Pass --data datasets/paysim.csv")
        frame = pd.read_csv(args.data)
        data_name = Path(args.data).name

    missing = [col for col in PAYSIM_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        sys.exit(f"Dataset is missing expected columns: {missing}")

    raw_count = len(frame)
    frame = frame[frame["type"].isin(ALLOWED_TYPES)].drop_duplicates()
    frame = frame.dropna(subset=PAYSIM_REQUIRED_COLUMNS).copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=["step", TARGET])
    frame[TARGET] = frame[TARGET].astype(int)
    frame = frame.sort_values("step", kind="mergesort").reset_index(drop=True)

    # Split the complete filtered timeline first. Optional sampling happens only inside each period.
    partitions = chronological_partition(frame)
    if args.sample:
        if args.sample < 30:
            sys.exit("For a meaningful three-way run, --sample must be at least 30; use 300000 as planned.")
        partitions = maybe_sample_chronologically(partitions, args.sample)
    train_df, val_df, test_df = partitions

    for name, part in [("training", train_df), ("validation", val_df), ("final test", test_df)]:
        frauds = int(part[TARGET].sum())
        if frauds == 0 or frauds == len(part):
            sys.exit(f"The chronological {name} partition contains only one target class. Cannot evaluate ROC-AUC/PR-AUC reliably.")

    print(f"Rows: {raw_count} raw -> {sum(map(len, partitions))} filtered/sampled rows")
    print("Chronological split (whole PaySim step ranges):")
    for label, part in [("TRAIN", train_df), ("VALIDATION", val_df), ("FINAL TEST", test_df)]:
        print(f"  {label}: {len(part):,} rows, fraud={int(part[TARGET].sum()):,}, "
              f"fraud_rate={part[TARGET].mean()*100:.4f}%, steps={part['step'].min()}..{part['step'].max()}")
    if not (train_df["step"].max() < val_df["step"].min() and val_df["step"].max() < test_df["step"].min()):
        raise AssertionError("Chronological split overlap detected")

    def xy(part):
        features = engineer_features(from_paysim(part))
        target = part[TARGET].astype(int).reset_index(drop=True)
        return features.reset_index(drop=True), target

    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)
    assert list(X_train.columns) == FEATURES
    assert not set(X_train.columns) & set(EXCLUDED_COLUMNS)
    print(f"Model features ({len(FEATURES)}): {FEATURES}")
    print(f"Excluded columns: {sorted(EXCLUDED_COLUMNS)}")

    if int(y_train.sum()) < 3:
        sys.exit("Training period has fewer than 3 fraud examples; more data is needed for cross-validation.")
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    if len(X_train) > args.tune_rows:
        tune_idx, _ = train_test_split(
            np.arange(len(X_train)), train_size=args.tune_rows, random_state=SEED,
            stratify=y_train,
        )
        X_tune, y_tune = X_train.iloc[tune_idx], y_train.iloc[tune_idx]
    else:
        X_tune, y_tune = X_train, y_train
    min_class = int(y_tune.value_counts().min())
    if min_class < 3:
        sys.exit("Tuning subset has too few examples of one class for 3-fold stratified CV; increase --tune-rows.")

    results, validation_fitted, test_probabilities = {}, {}, {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for model_name, (estimator, search_space) in candidates(pos_weight, demo).items():
        started = time.time()
        print(f"\n=== {model_name}: tuning on TRAIN only ===")
        pipeline = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        iterations = min(args.n_iter, int(np.prod([len(v) for v in search_space.values()])))
        search = RandomizedSearchCV(
            pipeline, search_space, n_iter=iterations, scoring="average_precision",
            cv=cv, random_state=SEED, n_jobs=1, refit=False,
        )
        search.fit(X_tune, y_tune)
        pipeline.set_params(**search.best_params_)
        pipeline.fit(X_train, y_train)

        validation_probability = pipeline.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, validation_probability, args.beta)
        test_probability = pipeline.predict_proba(X_test)[:, 1]
        results[model_name] = {
            "best_params": {key.replace("clf__", ""): value for key, value in search.best_params_.items()},
            "validation": metrics_at(y_val, validation_probability, threshold),
            "test": metrics_at(y_test, test_probability, threshold),
            "train_seconds": round(time.time() - started, 1),
        }
        validation_fitted[model_name] = (pipeline, threshold)
        test_probabilities[model_name] = test_probability
        val_metrics = results[model_name]["validation"]
        print(f"{model_name}: validation PR-AUC={val_metrics['pr_auc']:.4f}, "
              f"recall={val_metrics['recall']:.4f}, precision={val_metrics['precision']:.4f}, "
              f"threshold={threshold:.4f}")

    selected_name = max(results, key=lambda key: results[key]["validation"]["pr_auc"])
    selected_candidate, selected_threshold = validation_fitted[selected_name]
    print(f"\nSELECTED MODEL (validation PR-AUC): {selected_name}; threshold={selected_threshold:.6f}")
    print("Final test is evaluated once using the selected threshold; it is not used for tuning.")

    # Test plots are generated from candidate models fitted on TRAIN only.
    plt.figure(figsize=(6, 5))
    for name, probabilities in test_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        plt.plot(false_positive_rate, true_positive_rate,
                 label=f"{name} (AUC={results[name]['test']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(); plt.title("ROC curves (chronological final test)")
    plt.savefig(image_dir / "roc_curves.png", dpi=120, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6, 5))
    for name, probabilities in test_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        plt.plot(recall, precision, label=f"{name} (AP={results[name]['test']['pr_auc']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.title("Precision-recall curves (chronological final test)")
    plt.savefig(image_dir / "pr_curves.png", dpi=120, bbox_inches="tight"); plt.close()

    selected_test_metrics = results[selected_name]["test"]
    cm = selected_test_metrics["confusion_matrix"]
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    plt.figure(figsize=(4.5, 4)); plt.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        plt.text(col, row, f"{value:,}", ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Legit", "Fraud"]); plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix - {selected_name} (chronological test)")
    plt.savefig(image_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close()

    # Refit the chosen model on TRAIN + VALIDATION for production. FINAL TEST remains untouched.
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    production_pipeline = Pipeline([
        ("prep", make_preprocessor()),
        ("clf", candidates(pos_weight, demo)[selected_name][0]),
    ])
    production_pipeline.set_params(**{
        f"clf__{key}": value for key, value in results[selected_name]["best_params"].items()
    })
    production_pipeline.fit(X_train_val, y_train_val)

    # SHAP explanations are generated for the production pipeline using held-out test rows.
    preprocessor, classifier = production_pipeline[:-1], production_pipeline[-1]
    transformed_names = clean_names(preprocessor.get_feature_names_out())
    explainer_kind = "linear" if selected_name == "Logistic Regression" else "tree"
    shap_info = {"available": False}
    try:
        import shap
        from .explain import build_explainer, positive_class_shap
        shap_rows = X_test.sample(min(2000, len(X_test)), random_state=SEED)
        transformed_test = preprocessor.transform(shap_rows)
        background = (preprocessor.transform(X_train_val.sample(min(200, len(X_train_val)), random_state=SEED))
                      if explainer_kind == "linear" else None)
        explainer = build_explainer(classifier, explainer_kind, background)
        shap_values = positive_class_shap(explainer, transformed_test)
        importance = np.abs(shap_values).mean(axis=0)
        shap_info = {
            "available": True, "explainer": explainer_kind,
            "global_importance": sorted([
                {"feature": feature, "label": FEATURE_LABELS.get(feature, feature),
                 "mean_abs_shap": float(value)}
                for feature, value in zip(transformed_names, importance)
            ], key=lambda item: -item["mean_abs_shap"]),
        }
        plt.figure(); shap.summary_plot(shap_values, transformed_test,
                                         feature_names=transformed_names, show=False)
        plt.savefig(image_dir / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
        plt.figure(); shap.summary_plot(shap_values, transformed_test,
                                         feature_names=transformed_names, plot_type="bar", show=False)
        plt.savefig(image_dir / "shap_importance.png", dpi=120, bbox_inches="tight"); plt.close()
        if background is not None:
            joblib.dump(background, output_dir / "shap_background.joblib")
    except ImportError:
        if not demo:
            sys.exit("shap is not installed. Run: pip install -r requirements.txt")
        print("WARNING (demo only): SHAP missing; explanation plots skipped")

    # Save the production model, not the train-only evaluation candidate.
    joblib.dump(production_pipeline, output_dir / "fraud_pipeline.joblib")
    model_version = time.strftime("%Y%m%d-%H%M%S")
    config = {
        "model_name": selected_name, "model_version": model_version,
        "synthetic": demo, "dataset": data_name, "threshold": selected_threshold,
        "beta": args.beta,
        "features": {"all": FEATURES, "categorical": CATEGORICAL_FEATURES,
                     "numeric": NUMERIC_FEATURES, "input_columns": INPUT_COLUMNS,
                     "allowed_types": ALLOWED_TYPES, "excluded_columns": EXCLUDED_COLUMNS},
        "leakage_policy": "pre-transaction information only",
        "evaluation_protocol": "chronological_by_paysim_step",
        "transformed_feature_names": transformed_names, "explainer": explainer_kind,
        "class_balance": {"train_fraud": int(y_train.sum()), "train_total": int(len(y_train)),
                           "production_fit_fraud": int(y_train_val.sum()),
                           "production_fit_total": int(len(y_train_val))},
        "split": {
            "method": "chronological_by_whole_step_values",
            "train": len(X_train), "validation": len(X_val), "test": len(X_test),
            "train_steps": step_range(train_df), "validation_steps": step_range(val_df),
            "test_steps": step_range(test_df), "production_fit_rows": len(X_train_val),
        },
    }
    (output_dir / "model_config.json").write_text(json.dumps(config, indent=2))
    metrics_payload = {
        "selected": selected_name, "models": results, "shap": shap_info, "synthetic": demo,
        "evaluation_protocol": "chronological_by_paysim_step",
        "production_model_refit_on_train_plus_validation": True,
        "test_set_used_for_fitting_or_threshold_selection": False,
        "split": config["split"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=float))

    print(f"\nProduction model refitted on TRAIN + VALIDATION rows: {len(X_train_val):,}")
    print(f"Artifacts saved in {output_dir}")
    print(f"CHRONOLOGICAL FINAL TEST ({selected_name}): "
          f"precision={selected_test_metrics['precision']:.4f} "
          f"recall={selected_test_metrics['recall']:.4f} "
          f"f1={selected_test_metrics['f1']:.4f} "
          f"roc_auc={selected_test_metrics['roc_auc']:.4f} "
          f"pr_auc={selected_test_metrics['pr_auc']:.4f} "
          f"accuracy={selected_test_metrics['accuracy']:.4f}")
    print("Note: reported final-test metrics belong to the model fitted on TRAIN only; "
          "the saved production model is refitted on TRAIN + VALIDATION.")


if __name__ == "__main__":
    main()
