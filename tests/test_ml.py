import json
import numpy as np
import pandas as pd
import pytest
from ml.features import (BINARY_FEATURES, EXCLUDED_COLUMNS, FEATURE_LABELS, FEATURES, INPUT_COLUMNS,
                         NUMERIC_FEATURES, engineer_features, from_paysim)
from backend.services.predictor import ModelNotAvailable, Predictor, risk_level

PAYSIM_ROW = dict(step=26, type="TRANSFER", amount=100.0, nameOrig="C1", oldbalanceOrg=100.0,
                  newbalanceOrig=0.0, nameDest="C2", oldbalanceDest=0.0, newbalanceDest=100.0,
                  isFraud=1, isFlaggedFraud=0)


def _inputs(**kw):
    base = dict(type="TRANSFER", amount=100.0, oldbalanceOrg=100.0, oldbalanceDest=50.0, hour_of_day=2)
    return pd.DataFrame([{**base, **kw}])


# ---------- leakage guards ----------
def test_no_excluded_or_post_transaction_column_is_a_feature():
    assert not set(FEATURES) & set(EXCLUDED_COLUMNS)
    assert not set(INPUT_COLUMNS) & set(EXCLUDED_COLUMNS)
    for bad in ("newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "orig_emptied"):
        assert bad not in FEATURES


def test_features_depend_only_on_input_columns():
    """Adding post-transaction columns (or changing them) must not change the features."""
    base = _inputs()
    noisy = base.assign(newbalanceOrig=0.0, newbalanceDest=12345.0, isFraud=1)
    changed = base.assign(newbalanceOrig=99999.0, newbalanceDest=-5.0, isFraud=0)
    pd.testing.assert_frame_equal(engineer_features(base), engineer_features(noisy))
    pd.testing.assert_frame_equal(engineer_features(noisy), engineer_features(changed))
    engineer_features(base[INPUT_COLUMNS])          # works with the input columns alone


def test_training_and_inference_use_identical_feature_columns():
    train_side = engineer_features(from_paysim(pd.DataFrame([PAYSIM_ROW])))
    serve_side = engineer_features(_inputs())
    assert list(train_side.columns) == list(serve_side.columns) == FEATURES
    # the same transaction through both routes gives identical feature values
    same = _inputs(amount=100.0, oldbalanceOrg=100.0, oldbalanceDest=0.0, hour_of_day=2)
    pd.testing.assert_frame_equal(train_side.reset_index(drop=True), engineer_features(same))


def test_from_paysim_drops_post_columns_and_converts_step():
    out = from_paysim(pd.DataFrame([PAYSIM_ROW]))
    assert list(out.columns) == INPUT_COLUMNS and out.loc[0, "hour_of_day"] == 2   # 26 % 24


def test_every_feature_has_a_label_and_binary_features_exist():
    onehot = {f"type_{t}" for t in ("TRANSFER", "CASH_OUT")}
    assert set(FEATURE_LABELS) == set(NUMERIC_FEATURES) | onehot
    assert BINARY_FEATURES <= set(FEATURE_LABELS)


# ---------- feature values ----------
def test_drain_and_zero_balance_flags():
    f = engineer_features(_inputs(amount=100.0, oldbalanceOrg=100.0, oldbalanceDest=0.0))
    assert f.loc[0, "drains_origin"] == 1 and f.loc[0, "dest_balance_zero"] == 1
    assert f.loc[0, "amount_to_orig_balance"] == pytest.approx(1.0) and f.loc[0, "hour_of_day"] == 2
    g = engineer_features(_inputs(amount=10.0, oldbalanceOrg=1000.0, oldbalanceDest=5.0))
    assert g.loc[0, "drains_origin"] == 0 and g.loc[0, "dest_balance_zero"] == 0
    assert g.loc[0, "amount_to_orig_balance"] == pytest.approx(0.01)


def test_zero_sender_balance_has_no_division_error():
    f = engineer_features(_inputs(oldbalanceOrg=0.0))
    assert f.loc[0, "orig_balance_zero"] == 1 and f.loc[0, "drains_origin"] == 0
    assert f.loc[0, "amount_to_orig_balance"] == 0.0
    assert np.isfinite(f.drop(columns="type").to_numpy(dtype=float)).all()


def test_ratio_is_capped():
    assert engineer_features(_inputs(amount=1e9, oldbalanceOrg=1.0)).loc[0, "amount_to_orig_balance"] == 10.0


def test_risk_levels():
    assert risk_level(0.9, 0.8) == "High" and risk_level(0.5, 0.8) == "Medium" and risk_level(0.1, 0.8) == "Low"


# ---------- predictor safety ----------
def test_synthetic_model_refused(tmp_path):
    (tmp_path / "model_config.json").write_text(json.dumps({"synthetic": True}))
    (tmp_path / "fraud_pipeline.joblib").write_bytes(b"x")
    with pytest.raises(ModelNotAvailable):
        Predictor(tmp_path).predict({})


def test_model_trained_with_old_features_is_rejected(tmp_path):
    old = ["step", "type", "amount", "newbalanceOrig", "errorBalanceOrig", "orig_emptied"]
    (tmp_path / "model_config.json").write_text(json.dumps({"synthetic": False, "features": {"all": old}}))
    (tmp_path / "fraud_pipeline.joblib").write_bytes(b"x")
    with pytest.raises(ModelNotAvailable, match="feature mismatch"):
        Predictor(tmp_path).predict({})


def test_training_pipeline_smoke(tmp_path):
    """Runs the full training script on SYNTHETIC data to prove the code path works."""
    import subprocess, sys
    out = tmp_path / "m"
    subprocess.run([sys.executable, "-m", "ml.train", "--demo-synthetic", "--n-iter", "1", "--out", str(out)], check=True)
    cfg = json.loads((out / "model_config.json").read_text())
    assert cfg["synthetic"] is True and 0 < cfg["threshold"] < 1
    assert cfg["features"]["all"] == FEATURES
    assert not set(cfg["transformed_feature_names"]) & set(EXCLUDED_COLUMNS)
    assert set(cfg["features"]["excluded_columns"]) == set(EXCLUDED_COLUMNS)
    # a smoke-test model must still be refused by the web app, and must load when explicitly allowed
    with pytest.raises(ModelNotAvailable):
        Predictor(out).predict({})
    res = Predictor(out, allow_synthetic=True).predict(
        dict(type="TRANSFER", amount=500.0, oldbalanceOrg=500.0, oldbalanceDest=0.0, hour_of_day=3))
    assert 0.0 <= res["fraud_probability"] <= 1.0
