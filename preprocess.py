"""ETL / preprocessing pipeline (Chapter 3, Sections 3.5 and 3.6).

Usage:
    python preprocess.py path/to/raw_export.csv

What it does, in order:
  1. Reads the raw export and renames columns using COLUMN_MAP (edit this for your hospital's field names).
  2. Drops personal identifiers (PII) and replaces the visit ID with an irreversible hash.
  3. Removes duplicate records.
  4. Excludes records with missing critical timestamps.
  5. Excludes visits outside opening hours and visits with implausible durations.
  6. Excludes records whose timestamps are out of logical order.
  7. Imputes missing non-critical timestamps with MICE-style chained equations (fitted on the TRAINING days only).
  8. Builds the stage arrival times, waiting times and queue lengths the app needs.
  9. Writes data/visits.csv (for the app) plus the tables for your thesis:
       data/table_4_1_data_quality.csv   and   data/table_4_2_descriptive_stats.csv
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ------------------------------------------------------------------ SETTINGS (edit for your hospital)
COLUMN_MAP = {  # raw column name in your export  ->  name used here
    "visit_id": "visit_id",
    "arrival_time": "arrival_time",
    "registration_start": "registration_start", "registration_end": "registration_end",
    "triage_start": "triage_start", "triage_end": "triage_end",
    "consult_start": "consult_start", "consult_end": "consult_end",
    "lab_start": "lab_start", "lab_end": "lab_end",
    "radiology_start": "radiology_start", "radiology_end": "radiology_end",
    "pharmacy_start": "pharmacy_start", "pharmacy_end": "pharmacy_end",
    "departure_time": "departure_time",
}
PII_COLUMNS = ["patient_name", "phone_number", "insurance_id", "national_id", "address", "email"]
OPEN_HOUR, CLOSE_HOUR = 7, 16        # arrivals outside [07:00, 16:00) are excluded
MAX_TOTAL_MIN = 600                  # visits longer than 10 hours are treated as implausible
MAX_MISSING_TIMESTAMPS = 2           # more missing than this -> record excluded
TRAIN_FRACTION = 0.70                # imputer is fitted on the first 70% of days only (no leakage)

CHAIN = ["registration_start", "registration_end", "triage_start", "triage_end",
         "consult_start", "consult_end", "pharmacy_start", "pharmacy_end", "departure_time"]
CRITICAL = ["arrival_time", "consult_start"]
ALL_TS = ["arrival_time"] + CHAIN[:6] + ["lab_start", "lab_end", "radiology_start", "radiology_end"] + CHAIN[6:]


def main(path):
    if "synthetic" in os.path.basename(path).lower():
        print("NOTE: input is SYNTHETIC. Results from it are for testing the software only.")
    raw = pd.read_csv(path)
    report = []
    def log(step, n_affected, action):
        report.append({"Check": step, "Records affected": int(n_affected),
                       "Percent of raw records": round(100 * n_affected / n_raw, 2), "Action taken": action})

    n_raw = len(raw)
    df = raw.rename(columns=COLUMN_MAP)

    # 1-2. anonymise ------------------------------------------------------------------------
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
    df["visit_id"] = df["visit_id"].astype(str).map(lambda v: hashlib.sha256(v.encode()).hexdigest()[:12])
    for c in ALL_TS:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # 3. duplicates --------------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "visit_id"] + [], keep="first")
    df = df.drop_duplicates(subset="visit_id", keep="first")
    log("Duplicate records", before - len(df), "Removed")

    # 4. critical missing ------------------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=CRITICAL)
    log("Missing critical timestamp (arrival or consultation start)", before - len(df), "Excluded")

    # 5a. opening hours ---------------------------------------------------------------------
    before = len(df)
    df = df[(df["arrival_time"].dt.hour >= OPEN_HOUR) & (df["arrival_time"].dt.hour < CLOSE_HOUR)]
    log(f"Arrival outside {OPEN_HOUR:02d}:00-{CLOSE_HOUR:02d}:00", before - len(df), "Excluded")

    # 5b. implausible durations ------------------------------------------------------------
    before = len(df)
    end = df[CHAIN].max(axis=1)
    total = (end - df["arrival_time"]).dt.total_seconds() / 60
    same_day = end.dt.date == df["arrival_time"].dt.date
    df = df[(total <= MAX_TOTAL_MIN) & same_day]
    log(f"Total time above {MAX_TOTAL_MIN} min or ending on another day", before - len(df), "Excluded")

    # 6. logical order -------------------------------------------------------------------
    before = len(df)
    order = ["arrival_time"] + CHAIN
    bad = pd.Series(False, index=df.index)
    for a, b in zip(order[:-1], order[1:]):
        ok = df[a].isna() | df[b].isna()
        bad |= ~ok & (df[b] < df[a])
    lab_bad = df["lab_start"].notna() & ((df["lab_start"] < df["consult_end"]) | df["lab_end"].isna() | (df["lab_end"] < df["lab_start"]))
    before_rad = df["lab_end"].fillna(df["consult_end"])        # radiology follows the lab (if any), else the consultation
    rad_bad = df["radiology_start"].notna() & ((df["radiology_start"] < before_rad) | df["radiology_end"].isna() | (df["radiology_end"] < df["radiology_start"]))
    df = df[~(bad | lab_bad | rad_bad)]
    log("Timestamps out of logical order", before - len(df), "Excluded")

    # 6b. too many missing ------------------------------------------------------------------
    before = len(df)
    nmiss = df[CHAIN].isna().sum(axis=1)
    df = df[nmiss <= MAX_MISSING_TIMESTAMPS].copy()
    log(f"More than {MAX_MISSING_TIMESTAMPS} missing timestamps", before - len(df), "Excluded")

    # day index / time in minutes since opening ----------------------------------------------
    df["date"] = df["arrival_time"].dt.normalize()
    days = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
    df["day"] = df["date"].map(days)
    df["dow"] = df["date"].dt.dayofweek
    opening = df["date"] + pd.Timedelta(hours=7)
    mins = lambda col: (df[col] - opening).dt.total_seconds() / 60

    # 7. impute missing non-critical timestamps (MICE-style; fitted on training days only) ------
    n_imputed_rows = int(df[CHAIN].isna().any(axis=1).sum())
    rel = pd.DataFrame({c: (df[c] - df["arrival_time"]).dt.total_seconds() / 60 for c in CHAIN}, index=df.index)
    if rel.isna().any().any():
        cut = df["day"].max() * TRAIN_FRACTION
        imp = IterativeImputer(max_iter=15, random_state=0, min_value=0)
        imp.fit(rel[df["day"] <= cut])
        filled = pd.DataFrame(imp.transform(rel), columns=CHAIN, index=df.index)
        was_na = rel.isna().values
        imputed = filled.values
        lab_s = ((df["lab_start"] - df["arrival_time"]).dt.total_seconds() / 60).values
        lab_e = ((df["lab_end"] - df["arrival_time"]).dt.total_seconds() / 60).values
        rad_s = ((df["radiology_start"] - df["arrival_time"]).dt.total_seconds() / 60).values
        rad_e = ((df["radiology_end"] - df["arrival_time"]).dt.total_seconds() / 60).values
        k = {c: i for i, c in enumerate(CHAIN)}
        upper = np.full(imputed.shape, np.inf)                       # next REAL timestamp to the right
        for i in range(len(CHAIN) - 2, -1, -1):
            upper[:, i] = np.where(~was_na[:, i + 1], rel.values[:, i + 1], upper[:, i + 1])
        first_diag = np.fmin(np.where(np.isnan(lab_s), np.inf, lab_s), np.where(np.isnan(rad_s), np.inf, rad_s))
        upper[:, k["consult_end"]] = np.fmin(upper[:, k["consult_end"]], first_diag)
        vals = rel.values.copy()
        prev = np.zeros(len(vals))
        for i, c in enumerate(CHAIN):                                # keep every imputed time between its neighbours
            lo = prev if c != "pharmacy_start" else np.fmax(prev, np.fmax(np.where(np.isnan(lab_e), 0, lab_e), np.where(np.isnan(rad_e), 0, rad_e)))
            fill_v = np.minimum(np.maximum(imputed[:, i], lo), upper[:, i])
            vals[:, i] = np.where(was_na[:, i], fill_v, vals[:, i])
            prev = vals[:, i]
        filled = pd.DataFrame(vals, columns=CHAIN, index=df.index)
        for c in CHAIN:
            df[c] = df[c].fillna(df["arrival_time"] + pd.to_timedelta(filled[c], unit="m"))
    log("Missing non-critical timestamps", n_imputed_rows, "Imputed (MICE-style, fitted on training days)")

    # 8. build app columns ------------------------------------------------------------------
    out = pd.DataFrame({"visit_id": df["visit_id"], "day": df["day"], "dow": df["dow"]})
    out["arrival"] = mins("arrival_time")
    out["hour"] = (out["arrival"] // 60).astype(int) + 7
    has_lab, has_rad = df["lab_start"].notna(), df["radiology_start"].notna()
    after_lab = mins("lab_end").where(has_lab, mins("consult_end"))           # when the patient is ready for radiology
    after_diag = mins("radiology_end").where(has_rad, after_lab)              # when the patient is ready for pharmacy
    arr = {"reg": out["arrival"], "triage": mins("registration_end"), "consult": mins("triage_end"),
           "lab": mins("consult_end"), "rad": after_lab, "pharm": after_diag}
    arr["lab"], arr["rad"] = arr["lab"].where(has_lab), arr["rad"].where(has_rad)   # only users of these services
    start = {"reg": mins("registration_start"), "triage": mins("triage_start"), "consult": mins("consult_start"),
             "lab": mins("lab_start"), "rad": mins("radiology_start"), "pharm": mins("pharmacy_start")}
    endt = {"reg": mins("registration_end"), "triage": mins("triage_end"), "consult": mins("consult_end"),
            "lab": mins("lab_end"), "rad": mins("radiology_end"), "pharm": mins("pharmacy_end")}
    for s in arr:
        out[s + "_arr"], out[s + "_start"], out[s + "_end"] = arr[s], start[s], endt[s]
        out[s + "_wait"] = (start[s] - arr[s]).clip(lower=0)
    out["depart"] = mins("departure_time")
    out["total_time"] = out["depart"] - out["arrival"]
    out["visit_type"] = df.get("visit_type")
    out["triage_category"] = df.get("triage_category")
    out["age"] = df.get("age")

    # queue length when a patient joins a stage = people who arrived earlier and have not yet started
    for s in arr:
        q = np.full(len(out), np.nan)
        for _, idx in out.groupby("day").groups.items():
            ix = out.index.get_indexer(idx)
            a_i, st = arr[s].values[ix], start[s].values[ix]
            a_all, s_all = np.sort(a_i[~np.isnan(a_i)]), np.sort(st[~np.isnan(st)])
            qi = np.clip(np.searchsorted(a_all, a_i, "left") - np.searchsorted(s_all, a_i, "right"), 0, None)
            q[ix] = np.where(np.isnan(a_i), np.nan, qi)
        out[s + "_q"] = q
    out = out.sort_values(["day", "arrival"]).reset_index(drop=True)

    # 9. outputs ---------------------------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/visits.csv", index=False)
    open("data/visits_source.txt", "w").write(os.path.basename(path))
    t41 = pd.DataFrame(report)
    t41.loc[len(t41)] = ["Records retained for analysis", len(out), round(100 * len(out) / n_raw, 2), "Used in modelling"]
    t41.to_csv("data/table_4_1_data_quality.csv", index=False)

    rows = []
    for s, label in [("reg", "Registration"), ("triage", "Triage"), ("consult", "Consultation"),
                     ("lab", "Laboratory"), ("rad", "Radiology"), ("pharm", "Pharmacy")]:
        w = out[s + "_wait"][out[s + "_start"].notna()]
        svc = (out[s + "_end"] - out[s + "_start"]).dropna()
        rows.append({"Stage": label, "Mean wait": w.mean(), "Median wait": w.median(),
                     "90th percentile": w.quantile(0.9), "Mean service time": svc.mean()})
    rows.append({"Stage": "Total OPD turnaround", "Mean wait": out["total_time"].mean(),
                 "Median wait": out["total_time"].median(), "90th percentile": out["total_time"].quantile(0.9),
                 "Mean service time": np.nan})
    pd.DataFrame(rows).round(1).to_csv("data/table_4_2_descriptive_stats.csv", index=False)

    print("\nDATA QUALITY REPORT (paste into Table 4.1)")
    print(t41.to_string(index=False))
    print("\nDESCRIPTIVE STATISTICS, minutes (paste into Table 4.2)")
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print(f"\nWrote {len(out):,} clean visits to data/visits.csv. Start the app to use them.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python preprocess.py path/to/raw_export.csv")
    main(sys.argv[1])
