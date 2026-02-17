# src/phase5/ped_track.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

Box = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class Track:
    track_id: int
    centroid: np.ndarray  # shape (2,), float32
    box: Box
    age: int = 0          # frames since created
    hits: int = 1         # number of successful matches
    misses: int = 0       # consecutive frames not matched

    def as_tuple(self) -> Tuple[int, float, float]:
        return (self.track_id, float(self.centroid[0]), float(self.centroid[1]))


class NearestNeighborTracker:
    """
    Very simple multi-object tracker:
      - greedy nearest-neighbor association (by centroid distance)
      - keep ID stable across frames
      - delete tracks after max_missed consecutive misses
      - optional EMA smoothing of centroid updates

    This is intentionally lightweight (no Kalman filter, no Hungarian),
    and is perfect for "even rough" tracking.
    """

    def __init__(
        self,
        max_dist: float = 45.0,      # pixels
        max_missed: int = 4,         # frames
        min_hits: int = 2,           # hits before "confirmed"
        ema_alpha: float = 0.6,      # 0..1, higher = more smoothing
    ):
        self.max_dist = float(max_dist)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.ema_alpha = float(ema_alpha)

        self._next_id = 1
        self._tracks: List[Track] = []

        # stats / convenience
        self.total_created = 0

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.total_created = 0

    def update(self, boxes: List[Box], centroids: np.ndarray) -> List[Track]:
        """
        Update tracker with this frame's detections.

        Args:
          boxes: list of (x,y,w,h)
          centroids: Nx2 float array aligned with boxes

        Returns:
          list of current tracks (including unconfirmed ones).
        """
        # Normalize inputs
        if centroids is None or len(boxes) == 0:
            centroids = np.zeros((0, 2), dtype=np.float32)
            boxes = []
        else:
            centroids = np.asarray(centroids, dtype=np.float32)
            if centroids.ndim != 2 or centroids.shape[1] != 2:
                raise ValueError(f"centroids must be Nx2, got {centroids.shape}")

        # Age all tracks; mark misses by default (will be reset for matched ones)
        for tr in self._tracks:
            tr.age += 1
            tr.misses += 1

        # If no existing tracks: create new ones from all detections
        if len(self._tracks) == 0:
            for i in range(len(boxes)):
                self._create_track(boxes[i], centroids[i])
            return self.tracks

        # If no detections: just prune missed tracks
        if len(boxes) == 0:
            self._prune_tracks()
            return self.tracks

        # Build list of all (track_idx, det_idx, dist) pairs
        pairs = []
        track_centroids = np.stack([t.centroid for t in self._tracks], axis=0)  # (M,2)

        # Compute distances (M x N) without scipy
        # dist^2 = (dx^2 + dy^2)
        diffs = track_centroids[:, None, :] - centroids[None, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=2))  # (M,N)

        M, N = dists.shape
        for ti in range(M):
            for di in range(N):
                pairs.append((float(dists[ti, di]), ti, di))
        pairs.sort(key=lambda x: x[0])  # smallest dist first

        assigned_tracks = set()
        assigned_dets = set()

        # Greedy assignment with gating
        for dist, ti, di in pairs:
            if dist > self.max_dist:
                break
            if ti in assigned_tracks or di in assigned_dets:
                continue
            assigned_tracks.add(ti)
            assigned_dets.add(di)
            self._apply_match(self._tracks[ti], boxes[di], centroids[di])

        # Create new tracks for unassigned detections
        for di in range(N):
            if di not in assigned_dets:
                self._create_track(boxes[di], centroids[di])

        # Prune old tracks
        self._prune_tracks()
        return self.tracks

    def confirmed_tracks(self) -> List[Track]:
        """Return tracks with hits >= min_hits (less jittery for downstream repulsion)."""
        return [t for t in self._tracks if t.hits >= self.min_hits]

    def _apply_match(self, tr: Track, box: Box, det_centroid: np.ndarray) -> None:
        tr.box = box
        tr.hits += 1
        tr.misses = 0

        # EMA smoothing to reduce jitter
        a = self.ema_alpha
        tr.centroid = (a * tr.centroid + (1.0 - a) * det_centroid).astype(np.float32)

    def _create_track(self, box: Box, centroid: np.ndarray) -> None:
        tr = Track(
            track_id=self._next_id,
            centroid=np.asarray(centroid, dtype=np.float32).copy(),
            box=tuple(map(int, box)),
            age=0,
            hits=1,
            misses=0,
        )
        self._tracks.append(tr)
        self._next_id += 1
        self.total_created += 1

    def _prune_tracks(self) -> None:
        self._tracks = [t for t in self._tracks if t.misses <= self.max_missed]


def tracks_to_points(tracks: List[Track], confirmed_only: bool = True, min_hits: int = 2) -> np.ndarray:
    """
    Convert tracks into Nx2 repulsion points.
    """
    if confirmed_only:
        pts = [t.centroid for t in tracks if t.hits >= min_hits]
    else:
        pts = [t.centroid for t in tracks]
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(pts, axis=0).astype(np.float32)


__all__ = ["Track", "NearestNeighborTracker", "tracks_to_points"]
