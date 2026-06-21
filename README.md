# Parking-Induced Congestion Intelligence -- Prototype

Detects illegal/spillover parking from camera footage, scores how much each
incident actually hurts traffic flow, and ranks zones so enforcement can be
targeted instead of patrol-based and reactive.

Built for: Gridlock Hackathon 2.0, Round 2 -- "Poor Visibility on
Parking-Induced Congestion."

## Pipeline

```
camera feeds -----> detection & tracking -----> violation log
                                                       |
traffic baseline data --------------------------------+
                                                       v
                                          congestion impact scoring
                                                       |
                                                       v
                                          hotspot heatmap & ranking
                                                       |
                                                       v
                                          enforcement dashboard
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`ultralytics` will download YOLOv8n weights (~6MB) on first use -- needs
internet once.

## Run order

### 1. Sanity-check everything with synthetic data first

```bash
cd src
python generate_demo_data.py
python scoring.py
streamlit run app_dashboard.py
```

This populates `config/camera_config.csv`, `config/traffic_profile.csv`,
and `data/violations.csv` with believable synthetic data across five
illustrative locations, scores it, and opens the full dashboard. Do this
first -- it confirms the scoring math and dashboard work before you touch
real video, and it's your fallback if the CV part isn't ready in time for
the demo.

### 2. Plug in your real footage

1. Drop your video file(s) into `sample_videos/`.
2. For each video, draw the no-parking zone once:
   ```bash
   python define_zone.py --video ../sample_videos/yourclip.mp4 --location-id cam1
   ```
   Click 3+ points around the kerb-side area where spillover parking
   happens, press `s` to save. (No display available? Use
   `--dump-frame frame.jpg` to export the first frame, eyeball pixel
   coordinates in any image viewer, and hand-edit `config/zones.json`.)
3. Copy `config/camera_config.example.csv` to `config/camera_config.csv`
   and fill in one row per video: `location_id` (must match what you used
   in step 2), `video_path`, `lat`/`lon` (approximate real-world
   coordinates for the dashboard map), `lanes`, `road_width_m`, and
   `simulated_start_time` (an ISO datetime -- since a short clip can't
   span a full day, you assign it a "this is what time of day this footage
   represents" so the impact score and dashboard reflect realistic
   conditions, e.g. set one clip to `...T08:00:00` for a morning shot and
   another to `...T19:00:00` for evening).
4. Copy `config/traffic_profile.example.csv` to `config/traffic_profile.csv`
   and fill in approximate vehicles/hour by location and hour. If you
   don't have real counts, it's fine to reuse the two-peak shape from
   `generate_demo_data.py` -- the relative day-shape matters more than the
   absolute numbers for this prototype.
5. Run detection, then scoring, then the dashboard:
   ```bash
   python detect_violations.py
   python scoring.py
   streamlit run app_dashboard.py
   ```

You can mix real and synthetic locations in the same `camera_config.csv` /
`violations.csv` if you want a denser-looking demo -- just don't overwrite
your real `violations.csv` by re-running `generate_demo_data.py` afterward.

## How the impact score works

```
raw_impact   = lane_fraction_blocked x duration_minutes x traffic_volume_index
impact_score = raw_impact, min-max scaled to 0-100 across the dataset
```

- **lane_fraction_blocked** = `1 / number_of_lanes`. A car blocking one
  lane on a 2-lane road matters more than the same car on a 6-lane road.
- **duration_minutes** = how long the vehicle dwelt inside the no-parking
  zone (from the tracker).
- **traffic_volume_index** = that location's typical vehicle volume for
  that hour of day, normalized against the busiest hour in the dataset --
  blocking a lane during evening rush hour near a metro gate matters far
  more than the same block at 2am.

Zones are then ranked by `priority_score = avg_impact_score x violation_count`
per (location, hour) -- so a spot with frequent *and* high-impact violations
rises to the top, and the dashboard's "recommended enforcement plan"
surfaces the top N as suggested patrol windows.

This is a transparent, tunable heuristic, not a calibrated traffic
simulation -- which is the right level of rigor for a prototype: it's fast,
explainable in a judging Q&A, and easy to defend ("here's exactly why this
zone scored higher than that one").

## What's simplified, on purpose, for prototype scope

- **Detection**: pretrained YOLOv8n (COCO classes), no fine-tuning on
  Bengaluru-specific footage. Good enough to demonstrate the concept;
  accuracy would improve with fine-tuning on local data in a real
  deployment.
- **Tracking**: a lightweight centroid-distance tracker (`tracker.py`),
  not a full re-identification model. It can occasionally lose a track
  under heavy occlusion -- acceptable at the frame-skip rates used here.
- **Lane-blocking estimate**: assumes one violating vehicle removes one
  lane's worth of width, rather than measuring exact pixel-to-meter
  encroachment from the bounding box. A real deployment would calibrate
  this per camera.
- **Traffic volume**: a configurable baseline profile, not live sensor
  data. Swap in real loop-detector or GPS-probe data if available.

## Files

```
src/
  tracker.py             centroid tracker (dwell-time tracking)
  zone_utils.py           polygon load + point-in-zone test
  define_zone.py          draw a no-parking zone on a video frame
  detect_violations.py    YOLO + tracker + zone -> violations.csv
  scoring.py               violations.csv -> impact scores + hotspot ranking
  generate_demo_data.py   synthetic fallback data for the whole pipeline
  app_dashboard.py        Streamlit dashboard
config/
  camera_config.example.csv     schema for real cameras
  traffic_profile.example.csv   schema for baseline traffic volume
  zones.json              (created by define_zone.py)
data/                      (created by the scripts -- violations, scored, hotspot summary)
sample_videos/             put your footage here
```
