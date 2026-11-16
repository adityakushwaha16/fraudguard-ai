"""(Post-transaction balance columns are generated on purpose to mimic PaySim; the pipeline must ignore them.)
SYNTHETIC data generator -- ONLY for smoke-testing that the pipeline code runs.
Models trained on this are meaningless and the backend refuses to load them."""
import numpy as np
import pandas as pd


def make_synthetic(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n, p=[.3, .4, .3])
    amount = rng.lognormal(9, 1.5, n)
    old_o = amount * rng.uniform(0.8, 5, n)
    fraud = ((t != "PAYMENT") & (rng.random(n) < 0.03)).astype(int)
    old_o = np.where(fraud == 1, amount, old_o)
    new_o = np.where(fraud == 1, 0.0, np.maximum(old_o - amount, 0))
    old_d = rng.lognormal(8, 2, n) * (rng.random(n) < .6)
    new_d = np.where(fraud == 1, old_d, old_d + amount)
    return pd.DataFrame({"step": rng.integers(1, 744, n), "type": t, "amount": amount,
                         "nameOrig": "C1", "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
                         "nameDest": "C2", "oldbalanceDest": old_d, "newbalanceDest": new_d,
                         "isFraud": fraud, "isFlaggedFraud": 0})
