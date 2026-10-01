"""Analyses the usability questionnaire (Chapter 3, Section 3.10) and produces Tables 4.8 and 4.9.
Usage:  python analyse_survey.py survey_responses.csv
Enter one row per participant, using the layout of survey_template.csv:
  participant_id, role, years_experience, SUS1..SUS10 (1-5), USE_current_1..6 (1-5), USE_system_1..6 (1-5)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

SUS = [f"SUS{i}" for i in range(1, 11)]
CUR = [f"USE_current_{i}" for i in range(1, 7)]
SYS = [f"USE_system_{i}" for i in range(1, 7)]


def cronbach_alpha(items):
    items = items.dropna()
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))


def main(path):
    d = pd.read_csv(path)
    need = SUS + CUR + SYS
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"Missing columns: {miss}")
    if d[need].isna().any().any() or not d[need].isin([1, 2, 3, 4, 5]).all().all():
        print("WARNING: some ratings are blank or outside 1-5; please check the data entry.\n")

    # SUS scoring: odd items = score-1, even items = 5-score, sum x 2.5
    contrib = pd.DataFrame({c: (d[c] - 1) if int(c[3:]) % 2 == 1 else (5 - d[c]) for c in SUS})
    d["SUS_score"] = contrib.sum(axis=1) * 2.5
    n = len(d)
    os.makedirs("results", exist_ok=True)

    # ---- Table 4.8 participants
    t48 = d.groupby("role").agg(Number=("participant_id", "count"), Mean_years_experience=("years_experience", "mean")).round(1)
    t48.loc["Total"] = [n, round(d["years_experience"].mean(), 1)]
    t48.to_csv("results/table_4_8_participants.csv")

    # ---- Table 4.9 usability and usefulness
    cur_m, sys_m = d[CUR].mean(axis=1), d[SYS].mean(axis=1)
    diff = sys_m - cur_m
    try:
        w = stats.wilcoxon(sys_m, cur_m)
        z = abs(stats.norm.ppf(w.pvalue / 2))
        p_w, r = w.pvalue, z / np.sqrt(n)
    except ValueError:
        p_w, r = np.nan, np.nan
    t_sus = stats.ttest_1samp(d["SUS_score"], 68) if n > 1 else None
    rows = [
        ["Participants (n)", n, ""],
        ["Mean SUS score (0-100)", f"{d['SUS_score'].mean():.1f} (SD {d['SUS_score'].std(ddof=1):.1f})", "Benchmark 68 (Bangor et al., 2008)"],
        ["Median SUS score", f"{d['SUS_score'].median():.1f}", ""],
        ["SUS vs benchmark 68 (one-sample t-test)", f"t = {t_sus.statistic:.2f}, p = {t_sus.pvalue:.4f}" if t_sus else "", "Above 68 = above average"],
        ["Cronbach's alpha (SUS)", f"{cronbach_alpha(contrib):.2f}", ">= 0.70 acceptable"],
        ["Cronbach's alpha (usefulness, current approach)", f"{cronbach_alpha(d[CUR]):.2f}", ">= 0.70 acceptable"],
        ["Cronbach's alpha (usefulness, proposed system)", f"{cronbach_alpha(d[SYS]):.2f}", ">= 0.70 acceptable"],
        ["Mean usefulness, current approach (1-5)", f"{cur_m.mean():.2f}", ""],
        ["Mean usefulness, proposed system (1-5)", f"{sys_m.mean():.2f}", ""],
        ["Mean difference (system - current)", f"{diff.mean():.2f}", ""],
        ["Wilcoxon signed-rank test (H0_2)", f"p = {p_w:.4f}, effect size r = {r:.2f}", "Reject H0_2 if p < 0.05 and system rated higher"],
    ]
    t49 = pd.DataFrame(rows, columns=["Measure", "Result", "Note"])
    t49.to_csv("results/table_4_9_usability.csv", index=False)
    pd.DataFrame({"Current approach": d[CUR].mean(), "Proposed system": d[SYS].mean().values}).round(2) \
        .to_csv("results/usefulness_item_means.csv")
    d[["participant_id", "SUS_score"]].to_csv("results/sus_scores_by_participant.csv", index=False)

    print(t48.to_string(), "\n")
    print(t49.to_string(index=False))
    if p_w < 0.05 and diff.mean() > 0:
        print("\nDecision: reject H0_2 (system rated significantly more useful).")
    else:
        print("\nDecision: fail to reject H0_2.")
    print("\nFiles saved in the 'results' folder.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python analyse_survey.py survey_responses.csv")
    main(sys.argv[1])
