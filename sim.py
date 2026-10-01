"""Discrete-event simulation of the OPD pathway (Chapter 3, Section 3.8).

Stages: registration -> triage -> consultation -> (laboratory, probability p_lab) -> (radiology, probability p_rad) -> pharmacy.
Time is in minutes from 07:00. Arrivals follow a Poisson process whose hourly rate varies through the day.
Replace the DEFAULT parameters with values fitted from the hospital's own data.
"""
import random
import numpy as np
import pandas as pd
import simpy

DAY_START_HOUR = 7
DEFAULT = dict(
    clerks=3, nurses=3, doctors=5, lab_staff=3, radiographers=2, pharmacists=3,
    mean_reg=3.0, mean_triage=5.0, mean_consult=12.0, mean_lab=15.0, mean_rad=15.0, mean_pharm=4.0,
    p_lab=0.35, p_rad=0.25,
    rates=[18, 28, 30, 25, 20, 18, 14, 10, 8],   # patients/hour for 07:00-16:00
    demand_mult=1.0,
    extra_doctors=0, extra_triage=0, extra_lab=0, extra_rad=0,
    extra_start=2, extra_end=5,                  # hours after 07:00 (2 -> 09:00, 5 -> 12:00)
)


def _windowed_resource(env, base, extra, start_h, end_h):
    """Resource with `extra` units that are only available between start_h and end_h."""
    res = simpy.PriorityResource(env, capacity=base + extra)
    if extra > 0:
        env.process(_blocker(env, res, extra, start_h * 60, end_h * 60))
    return res


def _blocker(env, res, extra, s, e):
    held = [res.request(priority=-1) for _ in range(extra)]
    yield env.timeout(s)
    for r in held:
        res.release(r)
    yield env.timeout(e - s)
    _ = [res.request(priority=-1) for _ in range(extra)]
    yield env.timeout(10_000)


def _qlen(res):
    return sum(1 for r in res.queue if getattr(r, "priority", 0) == 0)


def _patient(env, pid, res, p, rng, rec_list):
    rec = {"pid": pid, "arrival": env.now}

    def stage(name, resource, mean):
        rec[name + "_arr"] = env.now
        rec[name + "_q"] = _qlen(resource)
        with resource.request(priority=0) as rq:
            yield rq
            rec[name + "_start"] = env.now
            yield env.timeout(rng.gammavariate(2.0, mean / 2.0))
        rec[name + "_end"] = env.now

    yield from stage("reg", res["reg"], p["mean_reg"])
    yield from stage("triage", res["triage"], p["mean_triage"])
    yield from stage("consult", res["consult"], p["mean_consult"])
    if rng.random() < p["p_lab"]:
        yield from stage("lab", res["lab"], p["mean_lab"])
    if rng.random() < p["p_rad"]:
        yield from stage("rad", res["rad"], p["mean_rad"])
    yield from stage("pharm", res["pharm"], p["mean_pharm"])
    rec["depart"] = env.now
    rec_list.append(rec)


def run_day(params=None, seed=0):
    p = {**DEFAULT, **(params or {})}
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    env = simpy.Environment()
    res = {
        "reg": simpy.PriorityResource(env, capacity=p["clerks"]),
        "triage": _windowed_resource(env, p["nurses"], p["extra_triage"], p["extra_start"], p["extra_end"]),
        "consult": _windowed_resource(env, p["doctors"], p["extra_doctors"], p["extra_start"], p["extra_end"]),
        "lab": _windowed_resource(env, p["lab_staff"], p["extra_lab"], p["extra_start"], p["extra_end"]),
        "rad": _windowed_resource(env, p["radiographers"], p["extra_rad"], p["extra_start"], p["extra_end"]),
        "pharm": simpy.PriorityResource(env, capacity=p["pharmacists"]),
    }
    times = []
    for h, rate in enumerate(p["rates"]):
        n = nprng.poisson(rate * p["demand_mult"])
        times += list(h * 60 + nprng.uniform(0, 60, n))
    times.sort()
    recs = []

    def arrivals():
        for i, t in enumerate(times):
            yield env.timeout(max(t - env.now, 0))
            env.process(_patient(env, i, res, p, rng, recs))

    env.process(arrivals())
    env.run(until=len(p["rates"]) * 60 + 600)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["total_time"] = df["depart"] - df["arrival"]
    for s in ["reg", "triage", "consult", "lab", "rad", "pharm"]:
        if s + "_start" in df:
            df[s + "_wait"] = df[s + "_start"] - df[s + "_arr"]
    df["hour"] = (df["arrival"] // 60).astype(int) + DAY_START_HOUR
    return df


def run_replications(params=None, n=20, seed0=1000):
    p = {**DEFAULT, **(params or {})}
    rows = []
    for i in range(n):
        d = run_day(p, seed0 + i)
        busy = (d["consult_end"] - d["consult_start"]).sum()
        cap = (p["doctors"] + p["extra_doctors"]) * (len(p["rates"]) * 60)
        rows.append({
            "Reception wait (min)": d["reg_wait"].mean(),
            "Triage wait (min)": d["triage_wait"].mean(),
            "Consultation wait (min)": d["consult_wait"].mean(),
            "Laboratory wait (min)": d["lab_wait"].mean() if "lab_wait" in d else 0.0,
            "Radiology wait (min)": d["rad_wait"].mean() if "rad_wait" in d else 0.0,
            "Pharmacy wait (min)": d["pharm_wait"].mean(),
            "Total time in system (min)": d["total_time"].mean(),
            "90th pct total time (min)": d["total_time"].quantile(0.9),
            "Patients served": len(d),
            "Doctor utilisation (%)": 100 * busy / cap,
        })
    r = pd.DataFrame(rows)
    out = pd.DataFrame({"mean": r.mean(), "ci95": 1.96 * r.std(ddof=1) / np.sqrt(n)})
    return out


def generate_dataset(days=90, params=None, seed0=1):
    frames = []
    for d in range(days):
        x = run_day(params, seed0 + d)
        x["day"] = d
        x["dow"] = d % 7
        frames.append(x)
    return pd.concat(frames, ignore_index=True)
