import numpy as np
import cv2
from heapq import heappush, heappop

from src.path.path_extraction import extract_path_mask
from src.map_tools.map_click_ab import ensure_AB_points


def compute_cost_field(mask: np.ndarray):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist = dist + 1e-3
    cost = 1.0 / dist
    return cost


def dijkstra(cost, start, goal):
    h, w = cost.shape
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    pq = []
    heappush(pq, (0.0, (sx, sy)))

    dist_map = {(sx, sy): 0.0}
    parent = {(sx, sy): None}

    while pq:
        cur_cost, (x, y) = heappop(pq)

        if (x, y) == (gx, gy):
            break

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy

                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if cost[ny, nx] <= 0:
                    continue

                new_cost = cur_cost + cost[ny, nx]

                if (nx, ny) not in dist_map or new_cost < dist_map[(nx, ny)]:
                    dist_map[(nx, ny)] = new_cost
                    parent[(nx, ny)] = (x, y)
                    heappush(pq, (new_cost, (nx, ny)))

    if (gx, gy) not in parent:
        raise RuntimeError("Pathfinding failed from A to B")

    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()
    return path


def extract_centerline_points(mask=None, A=None, B=None):
    """
    FINAL API:
    Returns Nx2 numpy array ONLY
    """

    if mask is None:
        mask = extract_path_mask()

    if A is None or B is None:
        A, B = ensure_AB_points()

    cost = compute_cost_field(mask)
    path = dijkstra(cost, A, B)

    pts = np.array(path, dtype=float)
    return pts
