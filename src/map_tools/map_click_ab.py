import cv2
from src.map_tools.map_loader import load_map_image
from src import config


def _click_points_ui():
    """
    Opens OpenCV window and lets user click A and B.
    Returns:
        A (x,y), B (x,y)
    """

    print("======================================")
    print("Select A and B on map")
    print("Click start point A, then end point B")
    print("Press ESC to cancel")
    print("======================================")

    img = load_map_image()
    display = img.copy()

    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal display

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Clicked: ({x}, {y})")

            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Select A and B", display)

    cv2.imshow("Select A and B", display)
    cv2.setMouseCallback("Select A and B", mouse_callback)

    while True:
        cv2.imshow("Select A and B", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if len(points) >= 2:
            break

    cv2.destroyAllWindows()

    if len(points) < 2:
        raise RuntimeError("A/B selection cancelled or incomplete.")

    return points[0], points[1]


def ensure_AB_points():
    """
    Ensures config.A_PIX and config.B_PIX exist.
    If not, launches click UI to obtain them.

    Returns:
        A, B
    """

    if config.A_PIX is not None and config.B_PIX is not None:
        print("[INFO] Using existing A/B from config")
        return config.A_PIX, config.B_PIX

    print("[INFO] A/B not set. Launching selection UI...")
    A, B = _click_points_ui()

    config.A_PIX = A
    config.B_PIX = B

    print(f"[INFO] Stored A = {A}")
    print(f"[INFO] Stored B = {B}")

    return A, B
