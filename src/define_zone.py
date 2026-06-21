"""
Interactive helper to outline a no-parking zone on the first frame of
a video. Click points around the area (e.g. the kerb-side stretch
where spillover parking happens), then:

    s = save this polygon for the given --location-id into zones.json
    r = reset the current points
    q = quit without saving

Usage:
    cd src
    python define_zone.py --video ../sample_videos/cam1.mp4 --location-id cam1

Note: this needs a display (it opens an OpenCV window), so run it on
your own machine, not in a headless environment. If you're on a
machine without a display, use the --dump-frame option instead to
export the first frame as an image, eyeball pixel coordinates in any
image viewer, and hand-write the polygon into config/zones.json
yourself (it's just a JSON dict of location_id -> list of [x, y]
points).
"""
import argparse
import json
import os

import cv2
import numpy as np

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--location-id", required=True)
    parser.add_argument("--zones", default="../config/zones.json")
    parser.add_argument("--dump-frame", default=None,
                         help="instead of opening a window, just save the first frame "
                              "to this path so you can eyeball coordinates manually")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {args.video}")

    if args.dump_frame:
        cv2.imwrite(args.dump_frame, frame)
        h, w = frame.shape[:2]
        print(f"Saved first frame ({w}x{h}) to {args.dump_frame}.")
        print("Open it in any image viewer, note pixel coordinates of the zone "
              "corners, then add to zones.json manually, e.g.:")
        print(f'  {{"{args.location_id}": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}}')
        return

    window = "Click zone corners | s=save  r=reset  q=quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, tuple(p), 4, (0, 0, 255), -1)
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
        elif key == ord("q"):
            break
        elif key == ord("s"):
            if len(points) < 3:
                print("Need at least 3 points to form a polygon.")
                continue
            zones = {}
            if os.path.exists(args.zones):
                with open(args.zones) as f:
                    zones = json.load(f)
            zones[args.location_id] = points
            os.makedirs(os.path.dirname(args.zones) or ".", exist_ok=True)
            with open(args.zones, "w") as f:
                json.dump(zones, f, indent=2)
            print(f"Saved zone for '{args.location_id}' -> {args.zones}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
