import time
import mediapipe as mp
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
            result_callback=self.process_result)

        self.landmarker = PoseLandmarker.create_from_options(options)

    def get_pose(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=frame)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def process_result(self, result, output_image, timestamp_ms):
        if len(result.pose_landmarks) == 0:
            return

        landmarks = result.pose_landmarks[0]
        self.callback(landmarks)

        # def print_result(result, output_image: mp.Image, timestamp_ms: int):
        #     global pose_result
        #     pose_result = result
        #
        #
        # def draw_landmarks_on_image(rgb_image, detection_result):
        #     pose_landmarks_list = detection_result.pose_landmarks
        #     annotated_image = np.copy(rgb_image)
        #
        #     # Loop through the detected poses to visualize.
        #     for idx in range(len(pose_landmarks_list)):
        #         pose_landmarks = pose_landmarks_list[idx]
        #
        #         # Draw the pose landmarks.
        #         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #         pose_landmarks_proto.landmark.extend([
        #             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        #         ])
        #         solutions.drawing_utils.draw_landmarks(
        #             annotated_image,
        #             pose_landmarks_proto,
        #             solutions.pose.POSE_CONNECTIONS,
        #             solutions.drawing_styles.get_default_pose_landmarks_style())
        #     return annotated_image
        #
        #
        # cam = cv2.VideoCapture(1)
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
        #
        # model_path = './src/models/pose_landmarker_full.task'
        #
        # BaseOptions = mp.tasks.BaseOptions
        # PoseLandmarker = mp.tasks.vision.PoseLandmarker
        # PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        # PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        # VisionRunningMode = mp.tasks.vision.RunningMode
        #
        # options = PoseLandmarkerOptions(
        #     base_options=BaseOptions(model_asset_path=model_path),
        #     running_mode=VisionRunningMode.LIVE_STREAM,
        #     result_callback=print_result)
        #
        # with PoseLandmarker.create_from_options(options) as landmarker:
        #     previous_time = 0
        #     frame_count = 0
        #     pose_result = None
        #     while True:
        #         ret, frame = cam.read()
        #         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
        #                             data=frame)
        #         landmarker.detect_async(mp_image, int(time.time() * 1000))
        #
        #         if pose_result:
        #             posed = draw_landmarks_on_image(frame, pose_result)
        #         else:
        #             posed = frame
        #
        #         diff_time = time.time() - previous_time
        #         previous_time = time.time()
        #         if diff_time > 0:
        #             fps = 1 / diff_time
        #         else:
        #             fps = 0  # FPS is 0 initially or if time delta is too small
        #
        #         fps_text = f"FPS: {int(fps)}"
        #         cv2.putText(posed, fps_text, (20, 70),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        #
        #         cv2.imshow('Camera', posed)
        #         frame_count += 1
        #
        #         if cv2.waitKey(1) == ord('q'):
        #             break
        #
        # cam.release()
        # cv2.destroyAllWindows()
