"""Streamlit dashboard. Thin on purpose: every number comes from dashboard_data."""
import os

import altair as alt
import streamlit as st

from modelguard import dashboard_data as dd
from modelguard import store


def vline(start):
    return alt.Chart({"values": [{"x": start}]}).mark_rule(color="red", strokeDash=[2, 2]).encode(x="x:Q")


def main() -> None:
    st.set_page_config(page_title="ModelGuard", layout="wide")
    st.title("ModelGuard")
    con = store.connect(os.environ.get("MODELGUARD_DB", "data/demo.db"))
    scenarios = dd.list_scenarios(con)
    if not scenarios:
        st.error("No runs in this database. Run `make demo` or `make simulate`.")
        return
    scenario = st.sidebar.selectbox("Scenario", scenarios, key="scenario")
    start = dd.drift_start(con, scenario)
    st.sidebar.write(f"True drift starts at batch: {start if start is not None else 'none (clean stream)'}")
    marker = [vline(start)] if start is not None else []

    st.header("Alerts")
    feed = dd.alert_feed(con, scenario)
    if feed.empty:
        st.success("No alerts raised.")
    else:
        st.warning(f"{len(feed)} alert(s) raised.")
    st.dataframe(feed, hide_index=True, width="stretch")

    st.header("Drift per feature")
    feature = st.selectbox("Feature", dd.features(con, scenario), key="feature")
    series = dd.drift_series(con, scenario, feature)
    for monitor, g in series.groupby("monitor"):
        base = alt.Chart(g).encode(x=alt.X("batch_index:Q", title="batch"))
        # Log axis over [P_FLOOR, 1]: clean p-values stay readable, drifted ones pin to the bottom edge below the alpha line
        y = alt.Y("value:Q", title="PSI score") if monitor == "psi" else alt.Y("value:Q", title=f"{monitor} p-value (log, floored at {dd.P_FLOOR:g})", scale=alt.Scale(type="log", domain=[dd.P_FLOOR, 1]))
        line = base.mark_line(point=True).encode(y=y)
        thr = alt.Chart({"values": [{"t": float(g["threshold"].iloc[0])}]}).mark_rule(color="orange").encode(y="t:Q")
        st.altair_chart(alt.layer(line, thr, *marker).properties(title=f"{monitor} on {feature} (orange = threshold)", height=200), width="stretch",
                        alt=f"{monitor} for {feature} per batch with its alert threshold and the true drift start")

    st.header("Missing rate")
    m = dd.missing_matrix(con, scenario)
    heat = alt.Chart(m).mark_rect().encode(x="batch_index:O", y="feature:N", color=alt.Color("rate:Q", scale=alt.Scale(domain=[0, 1])), tooltip=["feature", "batch_index", "rate"])
    st.altair_chart(heat.properties(height=350), width="stretch", alt="Heatmap of missing-value rate per feature per batch")

    st.header("Rolling performance (labels arrive late)")
    perf = dd.performance_series(con, scenario)
    if perf.empty:
        st.info("No labels have arrived yet.")
    else:
        lines = alt.Chart(perf).mark_line(point=True).encode(x="batch_index:Q", y=alt.Y("statistic:Q", scale=alt.Scale(zero=False)), color="monitor:N")
        floors = alt.Chart(perf).mark_rule(strokeDash=[4, 4]).encode(y="threshold:Q", color="monitor:N")
        st.altair_chart(alt.layer(lines, floors, *marker).properties(height=250), width="stretch", alt="Rolling accuracy and F1 per batch with their alert floors")


if __name__ == "__main__":
    main()
