"""Creates a SYNTHETIC, deliberately messy file that looks like a raw hospital (LHIMS-style) export.
Purpose: develop and test preprocess.py before the real extract is available.
THIS IS NOT REAL DATA. Never use it for thesis results."""
import numpy as np
import pandas as pd
from sim import generate_dataset

rng = np.random.default_rng(7)
DAYS = 120
sim = generate_dataset(DAYS, seed0=500)

# map simulated day numbers onto real working days (Mon-Fri)
dates = pd.bdate_range("2025-03-03", periods=DAYS)
base = pd.to_datetime(sim["day"].map(dict(enumerate(dates)))) + pd.Timedelta(hours=7)
ts = lambda col: base + pd.to_timedelta(sim[col], unit="m")

raw = pd.DataFrame({
    "visit_id": [f"V{100000+i}" for i in range(len(sim))],
    "patient_name": [f"Test Patient {i:05d}" for i in range(len(sim))],       # fake PII
    "phone_number": [f"000-000-{i%10000:04d}" for i in range(len(sim))],       # fake PII
    "insurance_id": [f"TEST-{i:07d}" for i in range(len(sim))],                # fake PII
    "age": rng.integers(1, 85, len(sim)),
    "sex": rng.choice(["F", "M"], len(sim), p=[0.58, 0.42]),
    "visit_type": rng.choice(["New", "Review"], len(sim), p=[0.45, 0.55]),
    "icd10_chapter": rng.choice(list("ABCDEGIJKMNR"), len(sim)),
    "triage_category": rng.choice(["Green", "Yellow", "Orange"], len(sim), p=[0.7, 0.25, 0.05]),
    "arrival_time": ts("arrival"),
    "registration_start": ts("reg_start"), "registration_end": ts("reg_end"),
    "triage_start": ts("triage_start"), "triage_end": ts("triage_end"),
    "consult_start": ts("consult_start"), "consult_end": ts("consult_end"),
    "lab_start": ts("lab_start"), "lab_end": ts("lab_end"),
    "radiology_start": ts("rad_start"), "radiology_end": ts("rad_end"),
    "pharmacy_start": ts("pharm_start"), "pharmacy_end": ts("pharm_end"),
    "departure_time": ts("depart"),
})
n = len(raw)
time_cols = [c for c in raw.columns if c.endswith(("_start", "_end", "_time"))]
noncritical = ["registration_start", "registration_end", "triage_start", "triage_end",
               "consult_end", "pharmacy_start", "pharmacy_end", "departure_time"]

# ---- introduce realistic data problems --------------------------------------------------
pick = lambda frac: rng.choice(n, int(n * frac), replace=False)
for i in pick(0.03):                                   # missing non-critical timestamps
    for c in rng.choice(noncritical, rng.integers(1, 3), replace=False):
        raw.loc[i, c] = pd.NaT
for i in pick(0.008):                                  # missing critical arrival time
    raw.loc[i, "arrival_time"] = pd.NaT
for i in pick(0.01):                                   # out-of-order timestamps
    raw.loc[i, "consult_start"] = raw.loc[i, "triage_end"] - pd.Timedelta(minutes=int(rng.integers(5, 30)))
for i in pick(0.004):                                  # implausible departure (next day)
    raw.loc[i, "departure_time"] = raw.loc[i, "departure_time"] + pd.Timedelta(days=1)
for i in pick(0.005):                                  # visits logged before opening time
    raw.loc[i, time_cols] = raw.loc[i, time_cols] - pd.Timedelta(minutes=110)
dups = raw.sample(frac=0.02, random_state=1)           # duplicate rows
raw = pd.concat([raw, dups]).sample(frac=1, random_state=2).reset_index(drop=True)

for c in time_cols:
    raw[c] = raw[c].dt.strftime("%Y-%m-%d %H:%M:%S")
raw.to_csv("sample_data/SYNTHETIC_sample_lhims_export.csv", index=False)
print("Wrote sample_data/SYNTHETIC_sample_lhims_export.csv with", len(raw), "rows (SYNTHETIC)")
