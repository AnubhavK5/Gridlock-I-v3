"""
Detects vehicles in one or more videos, checks whether they sit inside
a no-parking zone polygon, and logs a violation once a vehicle has
dwelt in the zone for longer than a threshold.

Run define_zone.py first to create the zone polygon(s) for your
camera(s), and fill in config/camera_config.csv with one row per
video you have (see config/camera_config.example.csv for the format).

Usage:
    cd src
    python detect_violations.py
    python detect_violations.py --config ../config/camera_config.csv \
        --zones ../config/zones.json --out ../data/violations.csv

Notes / things kept deliberately simple for prototype scope:
  - Vehicle detector: pretrained YOLOv8n (COCO classes), no fine-tuning.
    First run downloads ~6MB of weights -- needs internet access once.
  - "Inside the zone" is tested using the bottom-center point of each
    box (closer to where the vehicle actually touches the road than
    the box centroid).
  - Tracking is a simple centroid tracker (see tracker.py), not a
    full re-identification model. Good enough at modest frame-skip
    rates; will occasionally lose a track in heavy occlusion.
  - --frame-skip controls how many frames we skip between detector
    calls. Higher = faster but coarser duration measurement.
"""
import argparse
import os
from datetime import datetime, timedelta

import cv2
import pandas as pd

from tracker import CentroidTracker
from zone_utils import load_zones, point_in_zone

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def process_video(video_path, location_id, zone_polygon, sim_start_dt,
                   frame_skip, dwell_threshold_sec, model, road_width_m, lanes):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    seconds_per_processed_frame = frame_skip / fps

    tracker = CentroidTracker(
        max_disappeared=max(int(2 * fps / frame_skip), 3),
        max_distance=80,
    )

    # track_id -> running state while it's inside the zone
    track_state = {}
    violations = []

    frame_idx = 0
    video_time_sec = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        results = model.predict(frame, verbose=False)[0]
        centroids = []
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name not in VEHICLE_CLASS_NAMES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            centroids.append(bottom_center((x1, y1, x2, y2)))

        objects, deregistered_ids = tracker.update(centroids)

        for tid, centroid in objects.items():
            inside = point_in_zone(centroid, zone_polygon)
            state = track_state.setdefault(
                tid, {"in_zone_total": 0.0, "in_zone": False, "first_in_zone_time": None}
            )
            if inside:
                if not state["in_zone"]:
                    state["first_in_zone_time"] = video_time_sec
                state["in_zone"] = True
                state["in_zone_total"] += seconds_per_processed_frame
            else:
                state["in_zone"] = False

        for tid in deregistered_ids:
            state = track_state.pop(tid, None)
            if state and state["in_zone_total"] >= dwell_threshold_sec:
                event_time = sim_start_dt + timedelta(seconds=state["first_in_zone_time"] or 0)
                violations.append({
                    "location_id": location_id,
                    "timestamp": event_time.isoformat(),
                    "duration_sec": round(state["in_zone_total"], 1),
                    "lanes": lanes,
                    "road_width_m": road_width_m,
                })

        frame_idx += 1
        video_time_sec += seconds_per_processed_frame

    # Flush tracks still active (in zone) when the video ends
    for state in track_state.values():
        if state["in_zone_total"] >= dwell_threshold_sec:
            event_time = sim_start_dt + timedelta(seconds=state["first_in_zone_time"] or 0)
            violations.append({
                "location_id": location_id,
                "timestamp": event_time.isoformat(),
                "duration_sec": round(state["in_zone_total"], 1),
                "lanes": lanes,
                "road_width_m": road_width_m,
            })

    cap.release()
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/camera_config.csv")
    parser.add_argument("--zones", default="../config/zones.json")
    parser.add_argument("--out", default="../data/violations.csv")
    parser.add_argument("--frame-skip", type=int, default=5,
                         help="process every Nth frame (default 5)")
    parser.add_argument("--dwell-threshold-sec", type=float, default=90.0,
                         help="min seconds inside the zone to count as a violation")
    args = parser.parse_args()

    cameras = pd.read_csv(args.config)
    zones = load_zones(args.zones) if os.path.exists(args.zones) else {}
    model = load_model()

    all_violations = []
    for _, row in cameras.iterrows():
        location_id = row["location_id"]
        if location_id not in zones:
            print(f"[skip] no zone polygon for '{location_id}' -- run define_zone.py first")
            continue
        sim_start_dt = datetime.fromisoformat(str(row["simulated_start_time"]))
        print(f"[run] {location_id}: {row['video_path']}")
        v = process_video(
            video_path=row["video_path"],
            location_id=location_id,
            zone_polygon=zones[location_id],
            sim_start_dt=sim_start_dt,
            frame_skip=args.frame_skip,
            dwell_threshold_sec=args.dwell_threshold_sec,
            model=model,
            road_width_m=row["road_width_m"],
            lanes=row["lanes"],
        )
        print(f"       -> {len(v)} violation(s) logged")
        all_violations.extend(v)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(
        all_violations,
        columns=["location_id", "timestamp", "duration_sec", "lanes", "road_width_m"],
    ).to_csv(args.out, index=False)
    print(f"Saved {len(all_violations)} violations to {args.out}")


if __name__ == "__main__":
    main()
