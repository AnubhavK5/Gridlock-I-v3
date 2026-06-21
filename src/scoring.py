"""
Turns the raw violations log into:
  1. A per-violation congestion impact score (0-100)
  2. A hotspot-level summary (per location x hour-of-day), ranked by
     enforcement priority
  3. A plain-English enforcement recommendation for the top zones

Impact score formula (deliberately simple and explainable):

    raw_impact = lane_fraction_blocked x duration_minutes x traffic_volume_index

  - lane_fraction_blocked: assumes an illegally parked / encroaching
    vehicle removes one lane's worth of carriageway width
    (1 / number_of_lanes). A blocked lane on a 2-lane road hurts more
    than the same block on a 6-lane road.
  - duration_minutes: how long the vehicle dwelt in the zone.
  - traffic_volume_index: that location's average vehicle volume for
    that hour of day, normalized 0-1 against the busiest hour in the
    dataset. Blocking a lane at 6pm near a metro gate matters more
    than at 2am.

  raw_impact is then min-max scaled to 0-100 across the whole dataset
  so it reads cleanly on a dashboard. This is intentionally a
  transparent, tunable heuristic -- not a calibrated traffic
  simulation -- which is appropriate for a prototype: it's
  defensible, fast to compute, and easy to explain to a judge.

Usage:
    cd src
    python scoring.py
    python scoring.py --violations ../data/violations.csv \
        --traffic-profile ../config/traffic_profile.csv \
        --camera-config ../config/camera_config.csv \
        --out-dir ../data --top-n 5
"""
import argparse
import os

import pandas as pd


def lane_fraction_blocked(lanes):
    lanes = max(int(lanes), 1)
    return 1.0 / lanes


def compute_impact_scores(violations, traffic_profile):
    df = violations.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    df = df.merge(traffic_profile, on=["location_id", "hour"], how="left")
    df["avg_vehicles_per_hour"] = df["avg_vehicles_per_hour"].fillna(
        traffic_profile["avg_vehicles_per_hour"].median()
    )

    df["lane_fraction"] = df["lanes"].apply(lane_fraction_blocked)
    df["duration_min"] = df["duration_sec"] / 60.0

    max_volume = traffic_profile["avg_vehicles_per_hour"].max()
    df["volume_index"] = df["avg_vehicles_per_hour"] / max_volume

    df["raw_impact"] = df["lane_fraction"] * df["duration_min"] * df["volume_index"]

    max_raw = df["raw_impact"].max() or 1.0
    df["impact_score"] = (df["raw_impact"] / max_raw) * 100.0
    return df


def build_hotspot_summary(scored, camera_config):
    grouped = (
        scored.groupby(["location_id", "hour"])
        .agg(
            violation_count=("impact_score", "count"),
            total_blocked_minutes=("duration_min", "sum"),
            avg_impact_score=("impact_score", "mean"),
            max_impact_score=("impact_score", "max"),
        )
        .reset_index()
    )
    # Priority blends "how bad each violation is" with "how often it happens"
    grouped["priority_score"] = grouped["avg_impact_score"] * grouped["violation_count"]

    grouped = grouped.merge(
        camera_config[["location_id", "lat", "lon", "road_width_m", "lanes"]].drop_duplicates(),
        on="location_id", how="left",
    )
    return grouped.sort_values("priority_score", ascending=False).reset_index(drop=True)


def build_recommendations(hotspot_summary, top_n=5):
    recs = []
    for _, row in hotspot_summary.head(top_n).iterrows():
        hour = int(row["hour"])
        recs.append(
            f"{row['location_id']}: deploy enforcement around "
            f"{hour:02d}:00-{(hour + 1) % 24:02d}:00 "
            f"(priority {row['priority_score']:.1f}, "
            f"{int(row['violation_count'])} violation(s) observed, "
            f"avg impact {row['avg_impact_score']:.1f}/100)"
        )
    return recs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--violations", default="../data/violations.csv")
    parser.add_argument("--traffic-profile", default="../config/traffic_profile.csv")
    parser.add_argument("--camera-config", default="../config/camera_config.csv")
    parser.add_argument("--out-dir", default="../data")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    violations = pd.read_csv(args.violations)
    if violations.empty:
        print("No violations found -- run detect_violations.py (or generate_demo_data.py) first.")
        return

    traffic_profile = pd.read_csv(args.traffic_profile)
    camera_config = pd.read_csv(args.camera_config)

    scored = compute_impact_scores(violations, traffic_profile)
    hotspot_summary = build_hotspot_summary(scored, camera_config)
    recs = build_recommendations(hotspot_summary, args.top_n)

    os.makedirs(args.out_dir, exist_ok=True)
    scored.to_csv(os.path.join(args.out_dir, "violations_scored.csv"), index=False)
    hotspot_summary.to_csv(os.path.join(args.out_dir, "hotspot_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "recommendations.txt"), "w") as f:
        f.write("\n".join(recs))

    print(f"Scored {len(scored)} violations across {scored['location_id'].nunique()} location(s).")
    print("\nTop enforcement recommendations:")
    for r in recs:
        print(" -", r)


if __name__ == "__main__":
    main()
