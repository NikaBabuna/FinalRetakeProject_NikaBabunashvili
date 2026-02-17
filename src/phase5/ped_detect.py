# src/phase5/ped_detect.py

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

Box = Tuple[int, int, int, int]  # (x, y, w, h)


def boxes_to_centroids(boxes: List[Box]) -> np.ndarray:
    """Convert list of boxes to Nx2 centroids array float32: [[cx, cy], ...]"""
    if not boxes:
        return np.zeros((0, 2), dtype=np.float32)
    cents = [(x + 0.5 * w, y + 0.5 * h) for (x, y, w, h) in boxes]
    return np.asarray(cents, dtype=np.float32)


class PedestrianDetectorMOG2:
    """
    Pedestrian-ish blob detector using background subtraction (MOG2).

    Returns bounding boxes (x,y,w,h) of foreground blobs after threshold + morphology.
    Robustness improvement:
      - Large merged blobs are *split* into smaller boxes instead of being discarded.
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 25.0,
        detect_shadows: bool = False,
        mask_thresh: int = 200,         # binary threshold on MOG2 output
        min_area: int = 80,
        morph_k: int = 5,
        open_iters: int = 1,
        close_iters: int = 1,

        # geometric filters
        min_w: int = 6,
        min_h: int = 10,
        aspect_min: float = 0.15,
        aspect_max: float = 6.0,
        ignore_border: bool = False,
        border_px: int = 2,

        # "large blob" handling (trigger splitting, NOT auto-reject)
        large_area_ratio: float = 0.03,   # fraction of frame area that counts as "too big"
        large_w_frac: float = 0.40,       # fraction of frame width
        large_h_frac: float = 0.40,       # fraction of frame height
        split_large_blobs: bool = True,

        # how to split large blobs
        split_erode_k: int = 3,           # small erosion kernel inside large blob ROI
        split_erode_iters: int = 1,       # 1 is usually enough
        split_open_iters: int = 1,        # helps separate pedestrians in clumps
        split_min_area: int | None = None # if None, uses min_area
    ):
        self.min_area = int(min_area)
        self.morph_k = int(max(1, morph_k))
        self.open_iters = int(max(0, open_iters))
        self.close_iters = int(max(0, close_iters))
        self.mask_thresh = int(mask_thresh)

        self.min_w = int(min_w)
        self.min_h = int(min_h)
        self.aspect_min = float(aspect_min)
        self.aspect_max = float(aspect_max)

        self.ignore_border = bool(ignore_border)
        self.border_px = int(max(0, border_px))

        self.large_area_ratio = float(large_area_ratio)
        self.large_w_frac = float(large_w_frac)
        self.large_h_frac = float(large_h_frac)
        self.split_large_blobs = bool(split_large_blobs)

        self.split_erode_k = int(max(1, split_erode_k))
        self.split_erode_iters = int(max(0, split_erode_iters))
        self.split_open_iters = int(max(0, split_open_iters))
        self.split_min_area = int(split_min_area) if split_min_area is not None else None

        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=int(history),
            varThreshold=float(var_threshold),
            detectShadows=bool(detect_shadows),
        )

        # Morphology kernels (global)
        k = self.morph_k
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kc = max(1, k + 2)
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))

        # Kernels used to split large blobs (local ROI ops)
        sk = self.split_erode_k
        self.kernel_split = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sk, sk))

    def _passes_geom_filters(self, x: int, y: int, w: int, h: int, W: int, H: int) -> bool:
        if w <= 0 or h <= 0:
            return False
        if w < self.min_w or h < self.min_h:
            return False
        aspect = w / float(h)
        if aspect < self.aspect_min or aspect > self.aspect_max:
            return False
        if self.ignore_border and self.border_px > 0:
            bx = self.border_px
            if x <= bx or y <= bx or (x + w) >= (W - bx) or (y + h) >= (H - bx):
                return False
        return True

    def _is_large_blob(self, w: int, h: int, frame_area: float, W: int, H: int) -> bool:
        box_area = float(w * h)
        if box_area > self.large_area_ratio * frame_area:
            return True
        if w > self.large_w_frac * W or h > self.large_h_frac * H:
            return True
        return False

    def _split_blob_into_boxes(self, fg_mask: np.ndarray, contour: np.ndarray, min_area_local: int) -> List[Box]:
        """
        Try to split a big merged contour into multiple smaller boxes by:
          - drawing it to a mask
          - cropping ROI
          - eroding + opening
          - finding sub-contours
        """
        H, W = fg_mask.shape[:2]

        # mask just this contour
        blob_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=-1)

        x0, y0, w0, h0 = cv2.boundingRect(contour)
        if w0 <= 0 or h0 <= 0:
            return []

        roi = blob_mask[y0:y0 + h0, x0:x0 + w0].copy()

        # If ROI is tiny, don't bother splitting
        if roi.size == 0 or w0 < (self.min_w * 2) or h0 < (self.min_h * 2):
            return []

        # Erode to break thin bridges between pedestrians
        if self.split_erode_iters > 0:
            roi = cv2.erode(roi, self.kernel_split, iterations=self.split_erode_iters)

        # Optional open to separate clumps further
        if self.split_open_iters > 0:
            roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, self.kernel_split, iterations=self.split_open_iters)

        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Box] = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area_local:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # translate ROI coords back to full image coords
            gx, gy = x0 + x, y0 + y
            boxes.append((int(gx), int(gy), int(w), int(h)))

        return boxes

    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        H, W = frame_bgr.shape[:2]
        frame_area = float(H * W)

        fg = self.sub.apply(frame_bgr)

        # If shadows are enabled, they appear as 127. Threshold removes them.
        _, fg = cv2.threshold(fg, int(self.mask_thresh), 255, cv2.THRESH_BINARY)

        # Global cleanup
        if self.open_iters > 0:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel_open, iterations=self.open_iters)
        if self.close_iters > 0:
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close, iterations=self.close_iters)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Box] = []
        min_area_local = int(self.split_min_area) if self.split_min_area is not None else int(self.min_area)

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # If it's a large merged blob, attempt splitting first (instead of rejecting it)
            if self.split_large_blobs and self._is_large_blob(w, h, frame_area, W, H):
                sub_boxes = self._split_blob_into_boxes(fg, c, min_area_local=min_area_local)

                # Keep only good sub-boxes
                kept = []
                for (sx, sy, sw, sh) in sub_boxes:
                    # Also filter out absurdly large sub-boxes (still merged)
                    if self._is_large_blob(sw, sh, frame_area, W, H):
                        continue
                    if not self._passes_geom_filters(sx, sy, sw, sh, W, H):
                        continue
                    kept.append((sx, sy, sw, sh))

                if kept:
                    boxes.extend(kept)
                    continue  # don't keep original large box

                # If splitting failed (rare), fall through and consider original box

            # Normal case: apply filters to the original box
            if not self._passes_geom_filters(x, y, w, h, W, H):
                continue

            # Soft guard: if STILL huge and splitting is off/fails, skip to avoid giant squares
            if self._is_large_blob(w, h, frame_area, W, H):
                continue

            boxes.append((int(x), int(y), int(w), int(h)))

        return boxes


__all__ = ["PedestrianDetectorMOG2", "boxes_to_centroids"]
