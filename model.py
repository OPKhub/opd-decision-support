"""Predictive modelling (Chapter 3, Section 3.7).

Task A  - patient-level regression: predict the consultation wait of a patient joining the queue.
Task B  - short-horizon forecasting: predict the hourly average consultation wait 1, 2 and 3 hours ahead.
Task C  - congestion classification: GREEN / AMBER / RED, derived from the predicted wait.

Models : baseline (hourly average), Linear Regression (baseline), Random Forest, XGBoost, LightGBM, LSTM (needs PyTorch).
Design : chronological split by day (70% train / 15% validation / 15% test), time-series cross-validation for tuning,
         Diebold-Mariano and Wilcoxon tests with Holm correction (tests H0_1), SHAP for explainability.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (confusion_matrix, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, r2_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

FEATURES = ["hour", "dow", "arrivals_last60", "consult_q", "triage_q"]
TARGET = "consult_wait"
# Congestion thresholds in minutes. CONFIRM THESE WITH HOSPITAL MANAGEMENT (Section 3.7 of the thesis).
AMBER_MIN, RED_MIN = 15.0, 30.0
CLASS_NAMES = ["GREEN", "AMBER", "RED"]
BASELINE, LINEAR = "Baseline (hourly average)", "Linear Regression"


def rag(wait_min):
    return "RED" if wait_min >= RED_MIN else "AMBER" if wait_min >= AMBER_MIN else "GREEN"


def classify(w):
    w = np.asarray(w, dtype=float)
    return np.select([w >= RED_MIN, w >= AMBER_MIN], [2, 1], 0)


# ------------------------------------------------------------------------------------ features
def build_features(df):
    d = df.reset_index(drop=True).copy()
    d["hour"] = (d["consult_arr"] // 60).astype(int) + 7
    arr_last = np.zeros(len(d))
    for _, idx in d.groupby("day").indices.items():
        a = np.sort(d["arrival"].values[idx])
        t = d["consult_arr"].values[idx]
        arr_last[idx] = np.searchsorted(a, t, side="right") - np.searchsorted(a, t - 60, side="left")
    d["arrivals_last60"] = arr_last
    return d.sort_values(["day", "consult_arr"]).reset_index(drop=True)


def split_by_day(d, train=0.70, val=0.15):
    days = np.sort(d["day"].unique())
    a, b = days[int(len(days) * train) - 1], days[int(len(days) * (train + val)) - 1]
    return d[d.day <= a].reset_index(drop=True), d[(d.day > a) & (d.day <= b)].reset_index(drop=True), \
        d[d.day > b].reset_index(drop=True)


def metrics(y, p):
    p = np.clip(p, 0, None)
    return {"MAE (min)": mean_absolute_error(y, p), "RMSE (min)": mean_squared_error(y, p) ** 0.5, "R2": r2_score(y, p)}


# ------------------------------------------------------------------------------------ LSTM (optional)
def make_sequences(X, days, seq_len):
    """For each visit, the features of the last `seq_len` visits of the same day (left-padded with the first visit)."""
    out = np.zeros((len(X), seq_len, X.shape[1]), dtype=np.float32)
    for _, idx in pd.Series(np.arange(len(X))).groupby(days).indices.items():
        pos = np.arange(len(X))[days == days[idx[0]]]
        xd = X[pos]
        pad = np.vstack([np.repeat(xd[:1], seq_len - 1, axis=0), xd])
        for i in range(len(pos)):
            out[pos[i]] = pad[i:i + seq_len]
    return out


def lstm_fit_predict(tr, va, te, seq_len=8, epochs=15, seed=42):
    """Returns (pred_val, pred_test) or None if PyTorch is not installed.
    NOTE: written for the thesis pipeline; install PyTorch (pip install torch) to enable it."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return None
    torch.manual_seed(seed)
    mu, sd = tr[FEATURES].mean().values, tr[FEATURES].std().replace(0, 1).values
    ym, ys = tr[TARGET].mean(), tr[TARGET].std() or 1.0
    seq = lambda d: make_sequences(((d[FEATURES].values - mu) / sd).astype(np.float32), d["day"].values, seq_len)
    Xtr, Xva, Xte = seq(tr), seq(va), seq(te)
    ytr = ((tr[TARGET].values - ym) / ys).astype(np.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(FEATURES), 32, batch_first=True)
            self.out = nn.Linear(32, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.out(h[:, -1, :]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=256, shuffle=True)
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb), yb)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pv = net(torch.tensor(Xva)).numpy() * ys + ym
        pt = net(torch.tensor(Xte)).numpy() * ys + ym
    return pv, pt


# ------------------------------------------------------------------------------------ statistics
def diebold_mariano(e_base, e_model, lags=10):
    """DM test on squared-error loss with a Newey-West variance (lags). d > 0 means the model is better."""
    d = np.asarray(e_base) ** 2 - np.asarray(e_model) ** 2
    n, dc = len(d), d - d.mean()
    lrv = np.mean(dc * dc)
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.mean(dc[k:] * dc[:-k])
    dm = d.mean() / np.sqrt(max(lrv, 1e-12) / n)
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def test_against_baselines(y, preds):
    rows = []
    for base in (BASELINE, LINEAR):
        for name, p in preds.items():
            if name in (BASELINE, LINEAR):
                continue
            e_b, e_m = y - np.clip(preds[base], 0, None), y - np.clip(p, 0, None)
            dm, p_dm = diebold_mariano(e_b, e_m)
            try:
                w = stats.wilcoxon(np.abs(e_b), np.abs(e_m))
                p_w = w.pvalue
            except ValueError:
                p_w = 1.0
            r = abs(stats.norm.ppf(max(p_w, 1e-300) / 2)) / np.sqrt(len(y))
            rows.append({"Model": name, "Compared with": base,
                         "MAE improvement (%)": 100 * (np.abs(e_b).mean() - np.abs(e_m).mean()) / np.abs(e_b).mean(),
                         "DM statistic": dm, "DM p": p_dm, "Wilcoxon p": p_w, "Effect size r": r})
    t = pd.DataFrame(rows)
    t["Holm-adjusted p (DM)"] = holm(t["DM p"])
    t["Holm-adjusted p (Wilcoxon)"] = holm(t["Wilcoxon p"])
    sig = (t["Holm-adjusted p (DM)"] < 0.05) & (t["Holm-adjusted p (Wilcoxon)"] < 0.05)
    t["Decision on H0_1"] = np.where(sig & (t["MAE improvement (%)"] > 0), "Reject H0 (model better)",
                             np.where(sig, "Reject H0 (baseline better)", "Fail to reject H0"))
    return t


# ------------------------------------------------------------------------------------ main experiment
def _tune(est, grid, X, y, n_iter, seed):
    s = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=TimeSeriesSplit(n_splits=3),
                           scoring="neg_mean_absolute_error", random_state=seed, n_jobs=-1)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_


def forecast_horizons(d, seed=42):
    """Task B: hourly mean consultation wait 1, 2 and 3 hours ahead (LightGBM vs persistence)."""
    g = d.groupby(["day", "hour"]).agg(arrivals=("arrival", "size"), wait=(TARGET, "mean"), q=("consult_q", "mean")).reset_index()
    full = pd.MultiIndex.from_product([sorted(g["day"].unique()), range(7, 16)], names=["day", "hour"])
    g = g.set_index(["day", "hour"]).reindex(full, fill_value=0).reset_index()
    g["dow"] = g["day"].map(d.drop_duplicates("day").set_index("day")["dow"])
    g["arrivals_lag1"] = g.groupby("day")["arrivals"].shift(1).fillna(0)
    g["wait_lag1"] = g.groupby("day")["wait"].shift(1).fillna(0)
    feats = ["hour", "dow", "arrivals", "wait", "q", "arrivals_lag1", "wait_lag1"]
    tr, _, te = split_by_day(g)
    rows = []
    for h in (1, 2, 3):
        for part in (tr, te):
            part[f"y{h}"] = part.groupby("day")["wait"].shift(-h)
        a, b = tr.dropna(subset=[f"y{h}"]), te.dropna(subset=[f"y{h}"])
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1, n_jobs=1)
        m.fit(a[feats], a[f"y{h}"])
        p, naive = m.predict(b[feats]), b["wait"].values
        rows.append({"Horizon": f"{h} hour(s) ahead", "Persistence MAE (min)": mean_absolute_error(b[f"y{h}"], naive),
                     "LightGBM MAE (min)": mean_absolute_error(b[f"y{h}"], p),
                     "Persistence RMSE (min)": mean_squared_error(b[f"y{h}"], naive) ** 0.5,
                     "LightGBM RMSE (min)": mean_squared_error(b[f"y{h}"], p) ** 0.5})
    return pd.DataFrame(rows).round(3)


def run_experiments(df, seed=42, n_iter=6, use_lstm=True, shap_rows=1000):
    import shap
    d = build_features(df).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    tr, va, te = split_by_day(d)
    Xtr, ytr = tr[FEATURES], tr[TARGET].values

    base_map = tr.groupby("hour")[TARGET].mean()
    pred_va = {BASELINE: va["hour"].map(base_map).fillna(ytr.mean()).values}
    pred_te = {BASELINE: te["hour"].map(base_map).fillna(ytr.mean()).values}

    models, params = {}, {}
    models[LINEAR] = LinearRegression().fit(Xtr, ytr)
    models["Random Forest"], params["Random Forest"] = _tune(
        RandomForestRegressor(random_state=seed, n_jobs=1),
        {"n_estimators": [100, 200], "max_depth": [8, 12, None], "min_samples_leaf": [2, 5, 10]}, Xtr, ytr, n_iter, seed)
    models["XGBoost"], params["XGBoost"] = _tune(
        xgb.XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist"),
        {"n_estimators": [150, 300], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.07, 0.15], "subsample": [0.8, 1.0]},
        Xtr, ytr, n_iter, seed)
    models["LightGBM"], params["LightGBM"] = _tune(
        lgb.LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1),
        {"n_estimators": [150, 300], "num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.07, 0.15], "min_child_samples": [20, 50]},
        Xtr, ytr, n_iter, seed)
    for name, m in models.items():
        pred_va[name], pred_te[name] = m.predict(va[FEATURES]), m.predict(te[FEATURES])

    lstm_ok = False
    if use_lstm:
        out = lstm_fit_predict(tr, va, te, seed=seed)
        if out is not None:
            pred_va["LSTM"], pred_te["LSTM"] = out
            lstm_ok = True

    yva, yte = va[TARGET].values, te[TARGET].values
    tab = lambda y, P: pd.DataFrame([{"Model": n, **metrics(y, p)} for n, p in P.items()]).round(3)
    table_test, table_val = tab(yte, pred_te), tab(yva, pred_va)

    cands = [n for n in models]                       # best interactive model chosen on VALIDATION MAE
    best = min(cands, key=lambda n: mean_absolute_error(yva, np.clip(pred_va[n], 0, None)))
    trees = [n for n in ("Random Forest", "XGBoost", "LightGBM")]
    best_tree = min(trees, key=lambda n: mean_absolute_error(yva, np.clip(pred_va[n], 0, None)))

    # Task C: classification
    crow, true_c = [], classify(yte)
    for n, p in pred_te.items():
        pc = classify(np.clip(p, 0, None))
        pr, rc, f1, _ = precision_recall_fscore_support(true_c, pc, average="macro", zero_division=0)
        red = true_c == 2
        crow.append({"Model": n, "Precision": pr, "Recall": rc, "F1-score": f1,
                     "Red-state episodes missed (%)": 100 * float((pc[red] != 2).mean()) if red.any() else np.nan})
    cls_table = pd.DataFrame(crow).round(3)
    cm = confusion_matrix(true_c, classify(np.clip(pred_te[best_tree], 0, None)), labels=[0, 1, 2])

    tests = test_against_baselines(yte, pred_te).round(4)

    sample = te[FEATURES].sample(min(shap_rows, len(te)), random_state=seed)
    sv = shap.TreeExplainer(models[best_tree]).shap_values(sample)
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    return {"best_model": models[best], "best_name": best, "best_tree_name": best_tree, "models": models,
            "params": params, "table_test": table_test, "table_val": table_val, "classification": cls_table,
            "confusion": pd.DataFrame(cm, index=[f"True {c}" for c in CLASS_NAMES], columns=[f"Pred {c}" for c in CLASS_NAMES]),
            "tests": tests, "horizons": forecast_horizons(d, seed), "shap_importance": shap_imp, "shap_values": sv,
            "shap_sample": sample, "n_train": len(tr), "n_val": len(va), "n_test": len(te), "lstm_ran": lstm_ok,
            "test_frame": te.assign(**{f"pred_{best_tree}": np.clip(pred_te[best_tree], 0, None)})}
