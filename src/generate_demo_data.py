"""
Generates a synthetic camera_config.csv, traffic_profile.csv, and
violations.csv across a handful of illustrative Bengaluru-style
locations, so the dashboard renders end-to-end immediately -- even
before (or independent of) the real video pipeline being wired up.

This is your safety net: run this first to sanity-check the dashboard
and scoring math, then layer your real detected violations on top
(or alongside) for the actual demo.

Usage:
    cd src
    python generate_demo_data.py
"""
import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd

random.seed(7)

LOCATIONS = [
    {"location_id": "majestic_metro_gate2", "lat": 12.9767, "lon": 77.5713, "lanes": 2, "road_width_m": 7.0},
    {"location_id": "koramangala_5th_block", "lat": 12.9352, "lon": 77.6245, "lanes": 2, "road_width_m": 6.5},
    {"location_id": "indiranagar_100ft", "lat": 12.9716, "lon": 77.6412, "lanes": 3, "road_width_m": 10.0},
    {"location_id": "silk_board_jn", "lat": 12.9172, "lon": 77.6228, "lanes": 4, "road_width_m": 14.0},
    {"location_id": "whitefield_itpl", "lat": 12.9858, "lon": 77.7325, "lanes": 2, "road_width_m": 7.0},
]

# Relative how-bad-is-this-spot weighting, just for a believable demo
LOCATION_WEIGHT = {
    "silk_board_jn": 1.8,
    "majestic_metro_gate2": 1.5,
    "koramangala_5th_block": 1.2,
}


def build_camera_config(out_path):
    rows = [{
        **loc,
        "video_path": f"../sample_videos/{loc['location_id']}.mp4",
        "simulated_start_time": "2026-06-20T08:00:00",
    } for loc in LOCATIONS]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_traffic_profile(out_path):
    rows = []
    for loc in LOCATIONS:
        for hour in range(24):
            # two daily peaks, ~9am and ~7pm
            base = (
                200
                + 600 * max(0, 1 - abs(hour - 9) / 4)
                + 800 * max(0, 1 - abs(hour - 19) / 4)
            )
            rows.append({
                "location_id": loc["location_id"],
                "hour": hour,
                "avg_vehicles_per_hour": round(base + random.uniform(-30, 30), 1),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_violations(out_path, n_per_location=40):
    rows = []
    start_date = datetime(2026, 6, 13)
    peak_weights = [
        1 + 5 * max(0, 1 - abs(h - 9) / 4) + 6 * max(0, 1 - abs(h - 19) / 4)
        for h in range(24)
    ]
    for loc in LOCATIONS:
        n = int(n_per_location * LOCATION_WEIGHT.get(loc["location_id"], 1.0))
        for _ in range(n):
            day_offset = random.randint(0, 6)
            hour = random.choices(range(24), weights=peak_weights)[0]
            minute = random.randint(0, 59)
            ts = start_date + timedelta(days=day_offset, hours=hour, minutes=minute)
            duration = max(30, random.gauss(180, 90))
            rows.append({
                "location_id": loc["location_id"],
                "timestamp": ts.isoformat(),
                "duration_sec": round(duration, 1),
                "lanes": loc["lanes"],
                "road_width_m": loc["road_width_m"],
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../data")
    parser.add_argument("--config-dir", default="../config")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)

    build_camera_config(os.path.join(args.config_dir, "camera_config.csv"))
    build_traffic_profile(os.path.join(args.config_dir, "traffic_profile.csv"))
    build_violations(os.path.join(args.out_dir, "violations.csv"))

    print("Generated synthetic camera_config.csv, traffic_profile.csv, violations.csv")
    print("Next: python scoring.py   then   streamlit run app_dashboard.py")


if __name__ == "__main__":
    main()
