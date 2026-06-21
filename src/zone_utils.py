"""
Helpers for loading no-parking zone polygons and testing whether a
point (vehicle position) falls inside one.
"""
import json

import cv2
import numpy as np


def load_zones(path):
    """zones.json maps location_id -> list of [x, y] pixel points."""
    with open(path) as f:
        return json.load(f)


def point_in_zone(point, polygon):
    """polygon: list of [x, y] pixel points (>=3). point: (x, y)."""
    contour = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0
