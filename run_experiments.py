"""Runs the full modelling experiment and saves the tables and figures for Chapter 4.
Usage:  python run_experiments.py            (uses data/visits.csv)
Outputs in the `results` folder: table_4_3, table_4_4, hypothesis tests, forecast horizons, SHAP and plots."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from model import run_experiments

path = sys.argv[1] if len(sys.argv) > 1 else "data/visits.csv"
if not os.path.exists(path):
    sys.exit(f"{path} not found. Run preprocess.py first.")
src = open("data/visits_source.txt").read() if os.path.exists("data/visits_source.txt") else ""
if "synthetic" in src.lower():
    print("WARNING: SYNTHETIC data. These outputs are for testing only and are NOT thesis results.\n")
R = run_experiments(pd.read_csv(path))
os.makedirs("results", exist_ok=True)
R["table_test"].to_csv("results/table_4_3_test_performance.csv", index=False)
R["table_val"].to_csv("results/validation_performance.csv", index=False)
R["classification"].to_csv("results/table_4_4_classification.csv", index=False)
R["confusion"].to_csv("results/confusion_matrix.csv")
R["tests"].to_csv("results/hypothesis_1_tests.csv", index=False)
R["horizons"].to_csv("results/forecast_horizons.csv", index=False)
R["shap_importance"].rename("mean_abs_shap").to_csv("results/shap_importance.csv")
pd.Series({k: str(v) for k, v in R["params"].items()}).to_csv("results/tuned_hyperparameters.csv")

shap.summary_plot(R["shap_values"], R["shap_sample"], show=False)
plt.savefig("results/fig_shap_summary.png", dpi=200, bbox_inches="tight"); plt.close("all")
te = R["test_frame"]; col = f"pred_{R['best_tree_name']}"
wk = te[te["day"].isin(sorted(te["day"].unique())[:5])].reset_index(drop=True)
plt.figure(figsize=(9, 3.5)); plt.plot(wk["consult_wait"], lw=0.7, label="Observed"); plt.plot(wk[col], lw=0.7, label="Predicted")
plt.xlabel("Visits in time order (first 5 test days)"); plt.ylabel("Consultation wait (min)"); plt.legend()
plt.savefig("results/fig_predicted_vs_observed.png", dpi=200, bbox_inches="tight"); plt.close("all")

print(R["table_test"].to_string(index=False)); print()
print(R["classification"].to_string(index=False)); print()
print(R["tests"][["Model", "Compared with", "MAE improvement (%)", "Holm-adjusted p (DM)", "Holm-adjusted p (Wilcoxon)", "Decision on H0_1"]].to_string(index=False))
print("\nLSTM included:", R["lstm_ran"], "| files saved in the 'results' folder")
