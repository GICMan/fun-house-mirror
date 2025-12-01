import numpy as np
import cv2
import time


def make_warp_field(h, w, amplitude=80, freq=0.01):
    """
    Precompute a fun-house warp field:
    dx = A * sin(y * freq)
    dy = 0
    """
    # Create grid of coordinates
    ys, xs = np.indices((h, w), dtype=np.float32)

    # Distort horizontally
    dx = amplitude * np.sin(ys * freq)
    dy = np.zeros_like(dx)

    map_x = xs + dx
    map_y = ys + dy

    return map_x, map_y


cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

ret, frame = cam.read()

h, w = frame.shape[:2]
map_x, map_y = make_warp_field(h, w)

previous_time = 0
frame_count = 0
while True:
    ret, frame = cam.read()

    diff_time = time.time() - previous_time
    previous_time = time.time()
    if diff_time > 0:
        fps = 1 / diff_time
    else:
        fps = 0  # FPS is 0 initially or if time delta is too small

    warped = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    fps_text = f"FPS: {int(fps)}"
    cv2.putText(warped, fps_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Camera', warped)
    frame_count += 1

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
