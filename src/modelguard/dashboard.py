"""Streamlit dashboard. Thin on purpose: every number comes from dashboard_data."""
import os

import altair as alt
import streamlit as st

from modelguard import dashboard_data as dd
from modelguard import store

DRIFT_RED, THRESHOLD_AMBER = "#E11D48", "#F59E0B"
MONITOR_NAMES = {"psi": "PSI", "ks": "KS test", "chi2": "Chi-square", "rolling_accuracy": "Accuracy", "rolling_f1": "F1"}
MONITOR_HELP = {
    "psi": "Population Stability Index: how far the binned distribution has moved from the reference. Higher = more drift; alerts above the amber line.",
    "ks": "Kolmogorov-Smirnov test p-value (numeric features). Lower = more evidence of drift; alerts below the amber line.",
    "chi2": "Chi-square test p-value (categorical features). Lower = more evidence of drift; alerts below the amber line.",
}
SCENARIO_HELP = {
    "none": "A clean stream with no injected drift. Any alert here is a false alarm.",
    "sudden_shift": "A feature jumps to a new distribution at one batch and stays there.",
    "gradual_shift": "A feature drifts slowly over several batches, which is harder to catch early.",
    "label_shift": "The mix of target labels changes, so features drift only indirectly.",
    "missing_feature": "A feature starts arriving empty, as if an upstream pipeline broke.",
}


def vline(start):
    return alt.Chart({"values": [{"x": start}]}).mark_rule(color=DRIFT_RED, strokeDash=[4, 3], strokeWidth=2).encode(x="x:Q")


def drift_band(start, n_batches):
    """Shaded region from the true drift start to the last batch, so 'after drift' reads at a glance."""
    band = alt.Chart({"values": [{"x": start, "x2": n_batches - 1}]}).mark_rect(color=DRIFT_RED, opacity=0.08).encode(x="x:Q", x2="x2:Q")
    return [band, vline(start)]


def monitor_chart(g, monitor, marker):
    base = alt.Chart(g).encode(x=alt.X("batch_index:Q", title="Batch"))
    # Log axis over [P_FLOOR, 1]: clean p-values stay readable, drifted ones pin to the bottom edge below the alpha line
    y = alt.Y("value:Q", title="PSI score") if monitor == "psi" else alt.Y("value:Q", title="p-value (log)", scale=alt.Scale(type="log", domain=[dd.P_FLOOR, 1]), axis=alt.Axis(values=[dd.P_FLOOR, 1e-4, 1e-2, 1], format="~g"))
    tip = [alt.Tooltip("batch_index:Q", title="Batch"), alt.Tooltip("value:Q", title="Value", format=".3g"), alt.Tooltip("flagged:N", title="Breach")]
    line = base.mark_line(strokeWidth=2.5).encode(y=y)
    dots = base.mark_circle(size=55, opacity=1).encode(
        y=y, tooltip=tip,
        color=alt.Color("flagged:N", scale=alt.Scale(domain=[False, True], range=["#7C3AED", DRIFT_RED]), legend=alt.Legend(title="Breach", orient="top")))
    thr = alt.Chart({"values": [{"t": float(g["threshold"].iloc[0])}]}).mark_rule(color=THRESHOLD_AMBER, strokeWidth=2).encode(y="t:Q")
    return alt.layer(*marker, line, dots, thr).properties(height=240)


def header(scenario, s):
    st.title(":material/shield_with_heart: ModelGuard")
    st.caption("Watches a deployed model's input data and predictions batch by batch, and raises an alert when something drifts. "
               "Each scenario below replays a simulated data stream where the true drift point is known, so you can see how fast the monitors catch it.")
    with st.container(horizontal=True):
        st.badge(scenario.replace("_", " "), icon=":material/stream:", color="violet")
        if s["drift_feature"]:
            st.badge(f"drifted feature: {s['drift_feature']}", icon=":material/target:", color="blue")
        if s["drift_start"] is None:
            st.badge("clean stream" if not s["n_alerts"] else "false alarms", icon=":material/check_circle:" if not s["n_alerts"] else ":material/error:",
                     color="green" if not s["n_alerts"] else "orange")
        elif s["first_alert"] is not None:
            st.badge(f"drift caught {s['delay']} batch(es) after it started", icon=":material/check_circle:", color="green")
        else:
            st.badge("drift missed", icon=":material/error:", color="red")

    with st.container(horizontal=True):
        st.metric("Batches monitored", s["n_batches"], border=True, icon=":material/view_timeline:",
                  help="Each batch is one chunk of live traffic scored by the model and checked by every monitor.")
        st.metric("Drift injected at", "none" if s["drift_start"] is None else f"batch {s['drift_start']}", border=True, icon=":material/bolt:",
                  help="Ground truth from the simulator: the batch where the data really started to change.")
        st.metric("First alert", "none" if s["first_alert"] is None else f"batch {s['first_alert']}", border=True, icon=":material/notifications_active:",
                  help="First alert at or after the true drift start.")
        st.metric("Detection delay", "n/a" if s["delay"] is None else f"{s['delay']} batches", border=True, icon=":material/timer:",
                  help="Batches between the drift starting and the first alert. Lower is better; alerts need several breaches in a row, so it is never 0.")
        st.metric("False alarms", s["false_alarms"], border=True, icon=":material/report:",
                  help="Alerts raised before any drift existed. Lower is better.")


def alerts_tab(con, scenario):
    st.caption("An alert fires when one monitor breaches its threshold for several batches in a row, which filters out one-off noise.")
    feed = dd.alert_feed(con, scenario)
    if feed.empty:
        st.success("Every monitor stayed within its threshold for the whole stream.", title="No alerts raised", icon=":material/verified:")
        return
    st.warning(f"{len(feed)} alert(s) raised. Open the other tabs to see the signal behind each one.", icon=":material/notifications_active:")
    counts = feed.assign(monitor=feed["monitor"].map(MONITOR_NAMES).fillna(feed["monitor"])).groupby("monitor").size().reset_index(name="alerts")
    with st.container(border=True):
        st.subheader("Alerts by monitor")
        bar = alt.Chart(counts).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("alerts:Q", title="Alerts", axis=alt.Axis(tickMinStep=1)), y=alt.Y("monitor:N", title=None, sort="-x"),
            color=alt.Color("monitor:N", legend=None), tooltip=["monitor", "alerts"])
        st.altair_chart(bar.properties(height=alt.Step(44)), width="stretch", alt="Number of alerts per monitor")
    st.dataframe(feed, hide_index=True, width="stretch", alt="Alert log for the selected scenario", column_config={
        "batch_index": st.column_config.NumberColumn("Batch", width="small"),
        "batch_id": st.column_config.TextColumn("Batch ID"),
        "monitor": st.column_config.TextColumn("Monitor", width="small"),
        "feature": st.column_config.TextColumn("Feature", width="small"),
        "reason": st.column_config.TextColumn("Why it fired", width="large"),
        "timestamp": st.column_config.TextColumn("Raised at"),
    })


def drift_tab(con, scenario, s, marker):
    st.caption("Compares each feature's live distribution with the reference data the model was trained on. "
               "Red dots are batches that breached the amber threshold; the shaded area is where drift was really happening.")
    feats = dd.features(con, scenario)
    default = s["drift_feature"] if s["drift_feature"] in feats else feats[0]
    feature = st.pills("Feature", feats, default=default, required=True, key=f"feature_{scenario}")
    series = dd.drift_series(con, scenario, feature)
    groups = list(series.groupby("monitor"))
    for col, (monitor, g) in zip(st.columns(len(groups)), groups):
        with col, st.container(border=True):
            st.subheader(f"{MONITOR_NAMES.get(monitor, monitor)} on `{feature}`")
            st.caption(MONITOR_HELP.get(monitor, ""))
            st.altair_chart(monitor_chart(g, monitor, marker), width="stretch",
                            alt=f"{monitor} for {feature} per batch with its alert threshold and the true drift start")


def missing_tab(con, scenario):
    st.caption("Share of empty values per feature per batch. A feature that suddenly lights up usually means a broken upstream pipeline, not a change in the world.")
    m = dd.missing_matrix(con, scenario)
    heat = alt.Chart(m).mark_rect(cornerRadius=2).encode(
        x=alt.X("batch_index:O", title="Batch"), y=alt.Y("feature:N", title=None),
        color=alt.Color("rate:Q", title="Missing", scale=alt.Scale(domain=[0, 1], scheme="purplered"), legend=alt.Legend(format="%")),
        tooltip=["feature", alt.Tooltip("batch_index:O", title="Batch"), alt.Tooltip("rate:Q", title="Missing", format=".1%")])
    with st.container(border=True):
        st.altair_chart(heat.properties(height=380), width="stretch", alt="Heatmap of missing-value rate per feature per batch")


def performance_tab(con, scenario, marker):
    st.caption("Accuracy and F1 over a rolling window of recent labeled rows. Labels arrive a few batches late, so this view always lags behind the drift monitors. "
               "Dashed lines are the alert floors.")
    perf = dd.performance_series(con, scenario)
    if perf.empty:
        st.info("No labels have arrived yet.", icon=":material/hourglass_empty:")
        return
    perf = perf.assign(metric=perf["monitor"].map(MONITOR_NAMES).fillna(perf["monitor"]))
    color = alt.Color("metric:N", title=None, legend=alt.Legend(orient="top"))
    lines = alt.Chart(perf).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("batch_index:Q", title="Batch"), y=alt.Y("statistic:Q", title="Score", scale=alt.Scale(zero=False)), color=color,
        tooltip=["metric", alt.Tooltip("batch_index:Q", title="Batch"), alt.Tooltip("statistic:Q", title="Score", format=".3f")])
    floors = alt.Chart(perf).mark_rule(strokeDash=[6, 4], strokeWidth=1.5).encode(y="threshold:Q", color=color)
    with st.container(border=True):
        st.altair_chart(alt.layer(*marker, lines, floors).properties(height=320), width="stretch", alt="Rolling accuracy and F1 per batch with their alert floors")


def main() -> None:
    st.set_page_config(page_title="ModelGuard", page_icon=":material/shield_with_heart:", layout="wide")
    con = store.connect(os.environ.get("MODELGUARD_DB", "data/demo.db"))
    scenarios = dd.list_scenarios(con)
    if not scenarios:
        st.error("No runs in this database. Run `make demo` or `make simulate`.")
        return

    with st.sidebar:
        st.header(":material/tune: Scenario")
        scenario = st.selectbox("Scenario", scenarios, key="scenario", bind="query-params", format_func=lambda x: x.replace("_", " ").capitalize(), label_visibility="collapsed")
        s = dd.summary(con, scenario)
        st.info(SCENARIO_HELP.get(s["drift_type"], s["drift_type"]), icon=":material/info:")
        st.markdown("**How to read the charts**\n\n"
                    ":red[**Red shading / dashed line**] — true drift start\n\n"
                    ":orange[**Amber line**] — alert threshold\n\n"
                    ":violet[**Purple dots**] — batch within threshold\n\n"
                    ":red[**Red dots**] — threshold breached")

    if not s["n_batches"]:
        st.title(":material/shield_with_heart: ModelGuard")
        st.info("This scenario was simulated but its monitor results are missing (an older database?). Re-run `make simulate`.", title="No monitor results", icon=":material/hourglass_empty:")
        return
    marker = drift_band(s["drift_start"], s["n_batches"]) if s["drift_start"] is not None else []
    header(scenario, s)

    alerts, drift, missing, perf = st.tabs([":material/notifications: Alerts", ":material/ssid_chart: Feature drift",
                                            ":material/grid_view: Missing values", ":material/monitoring: Model performance"])
    with alerts:
        alerts_tab(con, scenario)
    with drift:
        drift_tab(con, scenario, s, marker)
    with missing:
        missing_tab(con, scenario)
    with perf:
        performance_tab(con, scenario, marker)


if __name__ == "__main__":
    main()
