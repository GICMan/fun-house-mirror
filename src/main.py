import time
import argparse
import cv2
import numpy as np
import tps
import pose_detect
from filters import init_filters
from camera import Camera


MESH_SIZE = 50
FPS_ALPHA = 0.1

WIN_NAME = "Mirror"


def generate_grid_image_transparent(x_num, y_num, width, height,
                                    line_color=(0, 0, 0, 255),
                                    line_thickness=1):
    # Create a transparent image
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Compute spacing
    x_spacing = width / x_num
    y_spacing = height / y_num

    # Draw vertical lines
    for i in range(1, x_num):
        x = int(i * x_spacing)
        cv2.line(img, (x, 0), (x, height),
                 line_color, thickness=line_thickness)

    # Draw horizontal lines
    for j in range(1, y_num):
        y = int(j * y_spacing)
        cv2.line(img, (0, y), (width, y), line_color, thickness=line_thickness)

    return img


def draw_control(img, src, dst):
    for i in range(len(src)):
        if np.array_equal(src[i], dst[i]):
            cv2.circle(img, (int(src[i][0]), int(
                src[i][1])), 4, (255, 255, 255), -1)
        else:
            cv2.circle(img, (int(src[i][0]), int(
                src[i][1])), 4, (255, 255, 255), -1)
            cv2.circle(img, (int(dst[i][0]), int(
                dst[i][1])), 4, (0, 255, 255), -1)
            cv2.line(img, (int(src[i][0]), int(src[i][1])),
                     (int(dst[i][0]), int(dst[i][1])), (0, 255, 255), 2, -1)


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

    grid_img = generate_grid_image_transparent(
        MESH_SIZE, MESH_SIZE, frame_width, frame_height)

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
    show_grid = False
    show_ctl = False
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

        if show_grid:
            warped_grid = cv2.remap(
                grid_img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            alpha = warped_grid[:, :, 3:] / 255.0
            overlay_bgr = warped_grid[:, :, :3]
            warped = (overlay_bgr * alpha + warped *
                      (1 - alpha)).astype(np.uint8)

        if show_ctl:
            draw_control(warped, src, dst)

        cv2.imshow(WIN_NAME, warped)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            show_ctl = not show_ctl
        elif key == ord('g'):
            show_grid = not show_grid

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
