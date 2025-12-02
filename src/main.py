import time
import argparse
import cv2
import numpy as np
import tps
import pose_detect
from filters import init_filters


MESH_SIZE = 50


def main():
    parser = argparse.ArgumentParser(
        description='A digital fun-house mirror.')
    parser.add_argument(
        '--cam', '-c', help='Specify a camera index.',
        default=0, type=int)
    parser.add_argument(
        '--rotate', '-r', help='If true, frames will be rotated 90 degrees',
        default=False, type=bool)
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Verbose output',
        default=False)
    parser.add_argument(
        '--filter', '-f', help='Filter number',
        default=0)

    args = parser.parse_args()
    verbose = args.verbose

    if verbose:
        print(f"Using camera: {args.cam}")

    cam = cv2.VideoCapture(args.cam)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cam.read()
    frame_height, frame_width = frame.shape[:2]

    if verbose:
        print(f"Frame size: {frame_width} x {frame_height}")

    filters = init_filters((frame_width, frame_height))

    if verbose:
        print("Initializing filters:")
        for filter in filters:
            print(f"-> {filter.name}")

    warper = tps.TPS((frame_width, frame_height),
                     (MESH_SIZE, MESH_SIZE))

    if verbose:
        print(f"Warper created with grid: {MESH_SIZE} x {MESH_SIZE}")

    src = np.array([
        [0, 0],
        [0, frame_height - 1],
        [frame_width - 1, 0],
        [frame_width - 1, frame_height - 1]
    ])
    dst = src.copy()
    mask = np.full((frame_height, frame_width), 255, dtype=np.uint8)

    def update_control_points(landmarks, mask0):
        nonlocal src
        nonlocal dst
        nonlocal mask
        src, dst = filters[args.filter].filter(landmarks)
        # mask = (mask0 * 255).astype(np.uint8)

    landmarker = pose_detect.pose_detector(
        (frame_width, frame_height), update_control_points)

    if verbose:
        print("Landmarker created")

    previous_time = 0
    while True:
        times = [time.time()]

        diff_time = time.time() - previous_time
        previous_time = time.time()
        if diff_time > 0:
            fps = 1 / diff_time
        else:
            fps = 0

        ret, frame = cam.read()

        frame = cv2.flip(frame, 1)

        landmarker.get_pose(frame)

        blured_mask = cv2.blur(mask, (30, 30))
        frame = cv2.bitwise_and(frame, frame, mask=blured_mask)

        times.append(time.time())

        warper.update_src_points(src)
        times.append(time.time())

        map_x, map_y = warper.compute_map(dst)
        times.append(time.time())

        warped = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        times.append(time.time())

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(warped, fps_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.circle(warped, src[-1].astype(int), 8, (255, 255, 255), -1)

        cv2.imshow('Fun House Mirror', warped)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
