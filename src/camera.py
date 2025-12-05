import threading
import queue
import cv2


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
