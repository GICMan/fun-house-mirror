import time
import threading
import queue
import argparse
import cv2
import numpy as np
import tps
import pose_detect
from filters import init_filters


MESH_SIZE = 50
FPS_ALPHA = 0.1

WIN_NAME = "Mirror"


class Camera:
    def __init__(self, src=0, hd=False):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 if hd else 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 if hd else 720)
        self.queue = queue.Queue(maxsize=1)
        self.running = True

        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # keep only the most recent frame
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(
        description='A digital fun-house mirror.')
    parser.add_argument(
        '--cam', '-c', help='Specify a camera index.',
        default=0, type=int)
    parser.add_argument(
        '--rotate', '-r', action='store_true',
        help='If true, frames will be rotated 90 degrees',
        default=False)
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Verbose output',
        default=False)
    parser.add_argument(
        '--filter', '-f', help='Filter number',
        default=0, type=int)
    parser.add_argument(
        '--high-def', '-hd', help='High definition resolution',
        action='store_true',
        default=False)

    args = parser.parse_args()
    verbose = args.verbose

    cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if verbose:
        print(f"Using camera: {args.cam}")

    cam = Camera(args.cam, args.high_def)

    frame = cam.read()
    if args.rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

    def update_control_points(landmarks):
        nonlocal src
        nonlocal dst
        src, dst = filters[args.filter].filter(landmarks)

    landmarker = pose_detect.pose_detector(
        (frame_width, frame_height), update_control_points)

    if verbose:
        print("Landmarker created")

    previous_time = 0
    prev_fps = 0
    while True:
        diff_time = time.time() - previous_time
        previous_time = time.time()
        if diff_time > 0:
            fps = FPS_ALPHA * (1 / diff_time) + (1 - FPS_ALPHA) * prev_fps
            prev_fps = fps
        else:
            fps = 0

        frame = cam.read()
        if args.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)

        landmarker.get_pose(frame)

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

        cv2.imshow(WIN_NAME, warped)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
