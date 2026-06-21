"""
A minimal centroid-distance tracker.

We don't need a full deep-learning tracker (DeepSORT etc.) for this
prototype -- we just need to know "is this the same vehicle as last
frame" so we can measure how long it dwells inside a parking zone.
Matching nearest centroids frame-to-frame is good enough at the
frame-skip rates this pipeline uses.
"""
from collections import OrderedDict

import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=80):
        """
        max_disappeared: how many consecutive missed frames before we
            consider a track gone (handles brief detection dropouts).
        max_distance: max pixel distance to consider two centroids the
            same vehicle between frames.
        """
        self.next_id = 0
        self.objects = OrderedDict()       # id -> (x, y)
        self.disappeared = OrderedDict()   # id -> consecutive missed frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        """
        input_centroids: list of (x, y) tuples detected this frame.
        Returns (current_objects, deregistered_ids) where current_objects
        is {id: (x, y)} for everything currently visible, and
        deregistered_ids is the list of track ids that JUST disappeared
        on this call (so the caller can flush their stats).
        """
        deregistered_ids = []

        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    deregistered_ids.append(object_id)
                    self.deregister(object_id)
            return dict(self.objects), deregistered_ids

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return dict(self.objects), deregistered_ids

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()), dtype=float)
        input_arr = np.array(input_centroids, dtype=float)

        # pairwise distance matrix: existing objects (rows) vs new detections (cols)
        D = np.linalg.norm(object_centroids[:, None, :] - input_arr[None, :, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                deregistered_ids.append(object_id)
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return dict(self.objects), deregistered_ids
