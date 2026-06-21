"""
Enforcement dashboard: hotspot heatmap, ranked priority table, and an
hour-of-day breakdown for the top zones.

Run generate_demo_data.py + scoring.py first (or detect_violations.py
+ scoring.py for real footage) so data/violations_scored.csv and
data/hotspot_summary.csv exist.

Usage:
    cd src
    streamlit run app_dashboard.py
"""
import os

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

DATA_DIR = "../data"

st.set_page_config(page_title="Parking-induced congestion intelligence", layout="wide")
st.title("Parking-induced congestion intelligence")
st.caption("AI-driven illegal-parking hotspot detection and enforcement prioritization")


@st.cache_data
def load_data():
    scored = pd.read_csv(os.path.join(DATA_DIR, "violations_scored.csv"))
    hotspots = pd.read_csv(os.path.join(DATA_DIR, "hotspot_summary.csv"))
    return scored, hotspots


try:
    scored, hotspots = load_data()
except FileNotFoundError:
    st.error(
        "No scored data found yet. Run `python generate_demo_data.py` then "
        "`python scoring.py` first (or `detect_violations.py` + `scoring.py` "
        "once you have real footage)."
    )
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Violations logged", len(scored))
col2.metric("Locations covered", scored["location_id"].nunique())
col3.metric("Highest priority score", f"{hotspots['priority_score'].max():.1f}")

st.subheader("Hotspot heatmap")
center_lat = hotspots["lat"].mean()
center_lon = hotspots["lon"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

by_location = (
    hotspots.groupby(["location_id", "lat", "lon"])["priority_score"]
    .sum()
    .reset_index()
)

HeatMap(
    by_location[["lat", "lon", "priority_score"]].values.tolist(),
    radius=25, blur=15,
).add_to(m)

for _, row in by_location.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=7,
        popup=f"{row['location_id']} (priority {row['priority_score']:.1f})",
        color="#D85A30",
        fill=True,
        fill_opacity=0.8,
    ).add_to(m)

st_folium(m, width=None, height=480)

st.subheader("Zones ranked by enforcement priority")
display_cols = ["location_id", "hour", "violation_count", "total_blocked_minutes",
                 "avg_impact_score", "priority_score"]
st.dataframe(
    hotspots[display_cols].rename(columns={
        "location_id": "Location", "hour": "Hour of day",
        "violation_count": "Violations", "total_blocked_minutes": "Blocked (min)",
        "avg_impact_score": "Avg impact (0-100)", "priority_score": "Priority score",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Violations by hour of day -- top 5 zones")
top_locations = hotspots.groupby("location_id")["priority_score"].sum().nlargest(5).index
chart_data = (
    scored[scored["location_id"].isin(top_locations)]
    .groupby(["location_id", "hour"])
    .size()
    .unstack(fill_value=0)
    .T
)
st.bar_chart(chart_data)

st.subheader("Recommended enforcement plan")
rec_path = os.path.join(DATA_DIR, "recommendations.txt")
if os.path.exists(rec_path):
    with open(rec_path) as f:
        for line in f:
            if line.strip():
                st.markdown(f"- {line.strip()}")
else:
    st.info("Run `python scoring.py` to generate recommendations.")
