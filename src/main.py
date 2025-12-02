import cv2
import time
import tps
import pose_detect
import numpy as np
from filters import init_filters

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1240)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cam.read()


frame_height, frame_width = frame.shape[:2]
filters = init_filters((frame_width, frame_height))

warper = tps.TPS((frame_width, frame_height), (50, 50), verbose=True)

ctrl_nx, ctrl_ny = 2, 2
xs = np.linspace(0, frame_width - 1, ctrl_nx)
ys = np.linspace(0, frame_height - 1, ctrl_ny)
grid_cx, grid_cy = np.meshgrid(xs, ys)
src = np.vstack([grid_cx.ravel(), grid_cy.ravel()]).T.astype(np.float64)
np.append(src, [0, 0])
dst = src.copy()


def update_control_points(landmarks):
    global src
    global dst
    src, dst = filters[1].filter(landmarks)


landmarker = pose_detect.pose_detector(
    (frame_width, frame_height), update_control_points)


previous_time = 0
frame_count = 0
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    landmarker.get_pose(frame)

    diff_time = time.time() - previous_time
    previous_time = time.time()
    if diff_time > 0:
        fps = 1 / diff_time
    else:
        fps = 0  # FPS is 0 initially or if time delta is too small

    warper.update_src_points(src)
    map_x, map_y = warper.compute_map(dst)

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

    cv2.circle(warped, src[-1].astype(int), 8, (255, 255, 255), -1)

    cv2.imshow('Fun House Mirror', warped)
    frame_count += 1

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
