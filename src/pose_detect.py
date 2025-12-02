import time
import mediapipe as mp
import numpy as np
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2


class pose_detector:
    def __init__(self, frame_size, callback):
        self.callback = callback
        self.frame_width, self.frame_height = frame_size
        model_path = './src/models/pose_landmarker_full.task'

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        # PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            output_segmentation_masks=True,
            result_callback=self.process_result)

        self.landmarker = PoseLandmarker.create_from_options(options)

        self.alpha = 0.2
        self.prev = None

    def get_pose(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=frame)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def process_result(self, result, output_image, timestamp_ms):
        if len(result.pose_landmarks) == 0:
            return

        landmarks = result.pose_landmarks[0]
        mask = result.segmentation_masks[0]
        landmark_arr = np.zeros((len(landmarks), 3))
        for i, lmk in enumerate(landmarks):
            landmark_arr[i, 0] = lmk.x
            landmark_arr[i, 1] = lmk.y
            landmark_arr[i, 2] = lmk.visibility

        if self.prev is None:
            self.prev = landmark_arr.copy()
            self.callback(landmark_arr, mask.numpy_view())

        smoothed = self.alpha * landmark_arr + (1 - self.alpha) * self.prev
        self.prev = smoothed
        self.callback(smoothed, mask.numpy_view())
