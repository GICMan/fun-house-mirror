import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

cam = cv2.VideoCapture(1)

previous_time = 0
frame_count = 0

sized_mask = np.full((1080, 1920), 255, dtype=np.uint8)
while True:
    ret, frame = cam.read()

    # results = model(frame, device='mps', stream=True)
    # result = next(results)
    # if result and result.masks is not None:
    #     mask_data = result.masks.data[0]
    #     binary_mask = (mask_data.cpu().numpy() * 255).astype("uint8")
    #     sized_mask = cv2.resize(binary_mask, (1920, 1080))

    masked = cv2.bitwise_and(frame, frame, mask=sized_mask)

    diff_time = time.time() - previous_time
    previous_time = time.time()
    if diff_time > 0:
        fps = 1 / diff_time
    else:
        fps = 0  # FPS is 0 initially or if time delta is too small

    fps_text = f"FPS: {int(fps)}"
    cv2.putText(masked, fps_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Camera', masked)
    frame_count += 1

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
