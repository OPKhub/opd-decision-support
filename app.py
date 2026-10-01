"""Web-based predictive analytics and scenario-testing system for OPD operations (starter prototype)."""
import os
import numpy as np
import pandas as pd
import streamlit as st

import auth
from sim import generate_dataset, run_replications
from model import AMBER_MIN, FEATURES, RED_MIN, rag, run_experiments

st.set_page_config(page_title="OPD Decision Support", page_icon="🏥", layout="wide")
DATA_FILE = "data/visits.csv"
STAGES = {"reg": "Registration", "triage": "Triage", "consult": "Consultation", "lab": "Laboratory",
          "rad": "Radiology", "pharm": "Pharmacy"}
COLOURS = {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴"}


@st.cache_data(show_spinner="Loading data...")
def load_data():
    """Uses the hospital's de-identified extract if data/visits.csv exists, otherwise synthetic demo data."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE), False
    return generate_dataset(90), True


@st.cache_resource(show_spinner="Training and comparing models (about a minute)...")
def get_results(df):
    return run_experiments(df)


# ---------------------------------------------------------------- login
def login_screen():
    st.title("🏥 OPD Predictive Analytics and Scenario Testing")
    st.caption("Secure access for authorised hospital staff only")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        role = auth.verify(u, p)
        if role:
            st.session_state.update(user=u, role=role)
            auth.audit(u, "login")
            st.rerun()
        else:
            auth.audit(u or "unknown", "failed login")
            st.error("Invalid username or password.")
    if auth.using_demo_passwords():
        with st.expander("Demo accounts (change before real use)"):
            st.write("admin / ChangeMe-Admin1 · manager / ChangeMe-Manager1 · supervisor / ChangeMe-Super1")


if "user" not in st.session_state:
    login_screen()
    st.stop()

user, role = st.session_state["user"], st.session_state["role"]
df, is_demo = load_data()

with st.sidebar:
    st.markdown(f"**{user}** ({role})")
    pages = {"dashboard": "1. Live queue dashboard", "alerts": "2. Predictive bottleneck alerts",
             "simulator": "3. What-if simulator", "audit": "Audit log"}
    allowed = auth.ROLES[role]
    choice = st.radio("Module", [pages[k] for k in allowed])
    if st.button("Sign out"):
        auth.audit(user, "logout")
        st.session_state.clear()
        st.rerun()
    with st.expander("Security reminders"):
        st.markdown("- Never share your password.\n- Sign out on shared computers.\n"
                    "- Do not export or screenshot dashboards outside approved channels.\n"
                    "- Report suspicious messages asking for your login details.")

src = open("data/visits_source.txt").read() if os.path.exists("data/visits_source.txt") else ""
if not is_demo and "synthetic" in src.lower():
    st.error("SYNTHETIC DATA: the loaded file comes from '" + src + "'. It is for testing the software only "
             "and must NOT be reported as real hospital data or thesis results.")
if is_demo:
    st.warning("DEMO MODE: the data shown are synthetic. Place the hospital's de-identified extract at "
               "`data/visits.csv` to use real data.")

# ---------------------------------------------------------------- module 1
if choice == pages["dashboard"]:
    auth.audit(user, "viewed dashboard")
    st.header("Live queue telemetry")
    day = st.selectbox("Day to display (replace with 'today' when connected to live LHIMS data)",
                       sorted(df["day"].unique())[::-1][:14])
    hour = st.slider("Hour of day", 7, 15, 10)
    d = df[(df["day"] == day)]
    cols = st.columns(len(STAGES))
    for c, (k, label) in zip(cols, STAGES.items()):
        w = d[d["hour"] == hour][k + "_wait"].mean()
        w = 0.0 if np.isnan(w) else w
        c.metric(f"{COLOURS[rag(w)]} {label}", f"{w:.1f} min")
    st.subheader("Average waiting time by hour")
    by_h = d.groupby("hour")[[k + "_wait" for k in STAGES]].mean()
    by_h.columns = list(STAGES.values())
    st.line_chart(by_h)
    st.subheader("Arrivals per hour")
    st.bar_chart(d.groupby("hour").size().rename("Arrivals"))

# ---------------------------------------------------------------- module 2
elif choice == pages["alerts"]:
    auth.audit(user, "viewed alerts")
    R = get_results(df)
    st.header("Predictive bottleneck alerts")
    st.write(f"Chronological split by day: **{R['n_train']:,}** training, **{R['n_val']:,}** validation and "
             f"**{R['n_test']:,}** test visits. Best model on validation data: **{R['best_name']}**.")
    if not R["lstm_ran"]:
        st.info("The LSTM was skipped because PyTorch is not installed (`pip install torch`).")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Predict now", "Model comparison", "Congestion classes",
                                      "Statistical tests", "Explainability (SHAP)", "Forecast horizon"])
    with t1:
        st.subheader("Predict the consultation wait for a patient joining the queue now")
        c = st.columns(4)
        hour = c[0].slider("Hour", 7, 15, 9)
        arr = c[1].number_input("Arrivals in last 60 min", 0, 100, 28)
        cq = c[2].number_input("People waiting for consultation", 0, 100, 8)
        tq = c[3].number_input("People waiting at triage", 0, 100, 3)
        x = pd.DataFrame([{"hour": hour, "dow": 1, "arrivals_last60": arr, "consult_q": cq, "triage_q": tq}])[FEATURES]
        pred = max(float(R["best_model"].predict(x)[0]), 0.0)
        st.metric(f"{COLOURS[rag(pred)]} Predicted consultation wait", f"{pred:.1f} min",
                  help=f"GREEN < {AMBER_MIN:.0f}, AMBER {AMBER_MIN:.0f}-{RED_MIN:.0f}, RED >= {RED_MIN:.0f} minutes (confirm thresholds with the hospital)")
        if rag(pred) != "GREEN":
            st.info("Suggested action: consider redeploying staff to consultation and test options in the What-if simulator.")
    with t2:
        st.subheader("Held-out test set (Table 4.3)")
        st.dataframe(R["table_test"], hide_index=True, width="stretch")
        st.subheader("Validation set (used to choose the best model)")
        st.dataframe(R["table_val"], hide_index=True, width="stretch")
        st.caption("Tuned hyperparameters: " + "; ".join(f"{k}: {v}" for k, v in R["params"].items()))
    with t3:
        st.subheader("GREEN / AMBER / RED classification (Table 4.4)")
        st.dataframe(R["classification"], hide_index=True, width="stretch")
        st.write("Confusion matrix for the best tree model")
        st.dataframe(R["confusion"], width="stretch")
    with t4:
        st.subheader("Test of H0_1: are the ML models different from the baselines?")
        st.dataframe(R["tests"], hide_index=True, width="stretch")
        st.caption("Diebold-Mariano (squared error, Newey-West variance) and Wilcoxon signed-rank (absolute errors), "
                   "Holm-corrected. A model is declared different only if BOTH corrected p-values are below 0.05.")
    with t5:
        import matplotlib.pyplot as plt
        import shap
        st.subheader(f"What drives predicted waiting time? ({R['best_tree_name']})")
        st.bar_chart(R["shap_importance"].rename("Mean |SHAP| (minutes)"))
        shap.summary_plot(R["shap_values"], R["shap_sample"], show=False)
        st.pyplot(plt.gcf())
        plt.close("all")
    with t6:
        st.subheader("Hourly average consultation wait 1 to 3 hours ahead")
        st.dataframe(R["horizons"], hide_index=True, width="stretch")

# ---------------------------------------------------------------- module 3
elif choice == pages["simulator"]:
    auth.audit(user, "opened simulator")
    st.header("What-if scenario simulator")
    st.caption("Discrete-event simulation of the OPD pathway (SimPy). Results are averages over repeated simulated days.")
    c = st.columns(3)
    extra_doc = c[0].slider("Extra doctors", 0, 4, 2)
    extra_tri = c[1].slider("Extra triage nurses", 0, 3, 0)
    win = c[2].slider("Extra staff hours (after 07:00)", 0, 9, (2, 5))
    c2 = st.columns(3)
    extra_lab = c2[0].slider("Extra laboratory staff", 0, 3, 0)
    extra_rad = c2[1].slider("Extra radiographers", 0, 2, 0)
    mult = c2[2].slider("Demand change (%)", -20, 50, 0)
    reps = st.select_slider("Replications", [10, 20, 30, 50], 20)
    if st.button("Run simulation", type="primary"):
        auth.audit(user, f"ran scenario doc+{extra_doc} tri+{extra_tri} lab+{extra_lab} rad+{extra_rad} demand{mult}%")
        with st.spinner("Simulating..."):
            base = run_replications({"demand_mult": 1 + mult / 100}, n=reps)
            scen = run_replications({"demand_mult": 1 + mult / 100, "extra_doctors": extra_doc,
                                     "extra_triage": extra_tri, "extra_lab": extra_lab, "extra_rad": extra_rad,
                                     "extra_start": win[0], "extra_end": win[1]}, n=reps)
        out = pd.DataFrame({"Baseline": base["mean"], "Scenario": scen["mean"]})
        out["Change (%)"] = 100 * (out["Scenario"] - out["Baseline"]) / out["Baseline"].replace(0, np.nan)
        st.dataframe(out.round(1), width="stretch")
        key = ["Triage wait (min)", "Consultation wait (min)", "Laboratory wait (min)", "Radiology wait (min)",
               "Total time in system (min)"]
        st.bar_chart(out.loc[key, ["Baseline", "Scenario"]])
        st.caption("Validate the baseline against your hospital's observed waiting times before relying on these results.")

# ---------------------------------------------------------------- audit
elif choice == pages["audit"]:
    st.header("Audit log")
    if os.path.exists(auth.AUDIT_FILE):
        st.dataframe(pd.read_csv(auth.AUDIT_FILE).iloc[::-1], width="stretch")
    else:
        st.write("No entries yet.")
