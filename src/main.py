import cv2
import time
import tps
import pose_detect
import numpy as np

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cam.read()

frame_height, frame_width = frame.shape[:2]

previous_time = 0
frame_count = 0

warper = tps.TPS((frame_width, frame_height), (50, 50), verbose=True)

ctrl_nx, ctrl_ny = 2, 2
xs = np.linspace(0, frame_width - 1, ctrl_nx)
ys = np.linspace(0, frame_height - 1, ctrl_ny)
grid_cx, grid_cy = np.meshgrid(xs, ys)
src = np.vstack([grid_cx.ravel(), grid_cy.ravel()]).T.astype(np.float64)
np.append(src, [0, 0])
dst = src.copy()
# dst[7] = dst[7] + [0, -100]
# dst[12] = dst[12] + [0, 100]


def effect_1(landmarks):
    head_norm = landmarks[0]
    head_x = head_norm[0] * frame_width
    head_y = head_norm[1] * frame_height

    box_size = 100
    src = np.array([
        [0, 0],
        [0, frame_height - 1],
        [frame_width - 1, 0],
        [frame_width - 1, frame_height - 1],
        [head_x, head_y + box_size],
        [head_x + box_size, head_y],
        [head_x, head_y - box_size],
        [head_x + box_size, head_y],
        [head_x, head_y],
    ])

    dst = src.copy()
    dst[-1] = [head_x, head_y - 50]

    return src, dst


def effect_2(landmarks):
    left_shoulder = landmarks[12]
    right_shoulder = landmarks[11]

    left_shoulder_x = left_shoulder[0] * frame_width
    left_shoulder_y = left_shoulder[1] * frame_height

    right_shoulder_x = right_shoulder[0] * frame_width
    right_shoulder_y = right_shoulder[1] * frame_height

    # box_size = 100
    src = np.array([
        [0, 0],
        [0, frame_height - 1],
        [frame_width - 1, 0],
        [frame_width - 1, frame_height - 1],
        [left_shoulder_x, left_shoulder_y + 50],
        [right_shoulder_x, right_shoulder_y + 50],
        [left_shoulder_x, left_shoulder_y - 100],
        [right_shoulder_x, right_shoulder_y - 100],
        [left_shoulder_x, left_shoulder_y],
        [right_shoulder_x, right_shoulder_y],
    ])

    dst = src.copy()
    dst[-2] = [left_shoulder_x - 150, left_shoulder_y]
    dst[-1] = [right_shoulder_x + 150, right_shoulder_y]

    return src, dst


def update_control_points(landmarks):
    global src
    global dst
    src, dst = effect_2(landmarks)


landmarker = pose_detect.pose_detector(
    (frame_width, frame_height), update_control_points)


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
    # cv2.imshow('Camera', frame)
    frame_count += 1

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
