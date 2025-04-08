import cv2
import argparse
import numpy as np
import time
from datetime import datetime
from threading import Thread
from queue import Queue
from imutils.video import FPS
import requests
import uuid
import json

def timestamp_to_str(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

class GrafanaMeteoriteAlertSender:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_alert(self, event_time, probability, photo_url, state="alerting"):
        # Форматирование времени для уведомления
        alert_time = timestamp_to_str(event_time)
        formatted_time = alert_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Заголовок и сообщение уведомления
        title = f"Обнаружена угроза метеоритного удара"
        message = (
            f"Метеорит был обнаружен в {formatted_time}.\n"
            f"Вероятность удара: {probability}%\n"
            f"Смотрите фото для получения дополнительной информации."
        )

        # Формирование полезной нагрузки для отправки в вебхук
        payload = {
            "title": title,
            "message": message,
            "ruleName": "Правило обнаружения метеоритов",
            "state": state,
            "tags": {
                "event": "meteorite",
                "probability": str(probability),
                "time": formatted_time
            },
            "evalMatches": [
                {"metric": "probability", "value": probability}
            ],
            "photo_url": photo_url
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.webhook_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            print("🚨 Уведомление о метеоритном ударе отправлено успешно!")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Не удалось отправить уведомление о метеоритном ударе: {e}")
            return None
    def get_send_alert_handler(self):
        def handler(timestamp):
            self.send_alert(timestamp, 1, "")
        return handler

class AbstractClaryfier:
    def __init__(self):
        self.event_handler = self.default_handler

    def send(self, roi, time):
        pass

    @staticmethod
    def default_handler(timestamp):
        print("Alert: Meteorite detected! Time: ", timestamp_to_str(timestamp))

class DummyClaryfier(AbstractClaryfier):
    def __init__(self):
        super().__init__()
    def send(self, roi, time):
        self.event_handler(time)

class CloudClaryfier(AbstractClaryfier):
    def __init__(self, oauth_token, oauth_provider):
        super().__init__()
        self.key = ''
        self.thread = None
        self.last_meteorite_time = 0
        self.req_freq_sec = 5
        self.q = Queue()
        self.oauth_token = oauth_token
        self.oauth_provider = oauth_provider
    
    def start(self):
        if self.thread:
            raise RuntimeError("Already started")
        self.thread = Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
    def send(self, roi, time):
        self.q.put((roi, time))
    
    def stop(self):
        if not self.thread: return
        self.q.put(None)
        self.thread.join()

    def _update(self):
        prev_req_time = 0
        while True:
            message = self.q.get()
            if not message:
                break
            t = time.time()
            if t - prev_req_time < self.req_freq_sec:
                time.sleep(self.req_freq_sec - (t - prev_req_time))
            frame, timestamp = message
            print("starting cloud recognizion...")
            obj, prob = self._neuro_recognize(frame)
            if obj == "meteorite":
                self.event_handler(timestamp)
            prev_req_time = time.time()
    def _neuro_recognize(self, frame):
        success, buf = cv2.imencode('.jpg', frame)
        if not success:
            raise Exception("Failed to encode frame to JPEG")
        
        img_buf = buf.tobytes()

        url = "https://smarty.mail.ru/api/v1/objects/detect"
        params = {
            "oauth_token": self.oauth_token,
            "oauth_provider": self.oauth_provider
        }
        headers = {
            "accept": "application/json"
        }
        meta = {
            "mode": ["object"],
            "images": [{"name": "file"}]
        }
        files = {
            "file": ("frame.jpg", img_buf, "image/jpeg"),
            "meta": (None, json.dumps(meta), "application/json")
        }
        
        try:
            response, body = None, None
            while True:
                response = requests.post(url, params=params, headers=headers, files=files)
                body = response.json()
                if "error" in body["body"]["object_labels"][0] and body["body"]["object_labels"][0]["error"] == "internal error: image crc mismatch":
                    print("cloud internal trouble, sleeping and trying again...")
                    time.sleep(30)
                    continue
                break
            data = body["body"]["object_labels"][0]["labels"]
            data.sort(key=lambda x: x["prob"], reverse=True)
            meteorite_prob = 0
            aviation_prob = 0
            bird_prob = 0
            for elem in data:
                if elem["eng"] == "Astronomical Object":
                    meteorite_prob += elem["prob"]
                elif elem["eng"] == "Vehicle" or elem["eng"] == "Airplane" or elem["eng"] == "Aviation":
                    aviation_prob += elem["prob"]
                elif elem["eng"] == "Bird":
                    bird_prob += elem["prob"]
            if meteorite_prob > aviation_prob and meteorite_prob > bird_prob:
                return "meteorite", meteorite_prob
            elif aviation_prob > meteorite_prob and aviation_prob > bird_prob:
                return "airplane", aviation_prob
            elif bird_prob > meteorite_prob and bird_prob > aviation_prob:
                return "birds", bird_prob
            else:
                return None, 0
        except Exception as e:
            print("_neuro_recognize exception: ", e)
            return None, 0


class AsyncVideoReader:
    def __init__(self, stream, queueSize=4):
        self.stream = stream
        self.started = False
        self.q = Queue(maxsize=queueSize)
    def start(self):
        if self.started:
            raise RuntimeError("Already started")
        self.started = True
        Thread(target=self._update, args=(), daemon=True).start()
        return self
    def _update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                return
            self.q.put(frame)
    def read(self):
        if not self.isOpened():
            return False, None
        return True, self.q.get()
    def stop(self):
        self.started = False
    def isOpened(self):
        return self.started or not self.q.empty()
    def release(self):
        self.stop()
        self.stream.release()
    
class AsyncVideoWriter:
    def __init__(self, stream, queueSize=4):
        self.stream = stream
        self.started = False
        self.q = Queue(maxsize=queueSize)
    def start(self):
        if self.started:
            raise RuntimeError("Already started")
        self.started = True
        Thread(target=self._update, args=(), daemon=True).start()
        return self
    def _update(self):
        while self.started:
            frame = self.q.get()
            self.stream.write(frame)
    def write(self, frame):
        if not self.isOpened():
            return False
        self.q.put(frame)
        return True
    def stop(self):
        self.started = False
    def isOpened(self):
        return self.started
    def release(self):
        self.stop()
        self.stream.release()

def red_intensity_from_hue(hue):
    # OpenCV Hue range: 0–179
    # Normalize redness: 0.0 (not red) to 1.0 (very red)

    # Red can be near 0 or near 180 (wraps around)
    distance_from_red = min(abs(hue - 0), abs(hue - 179))
    intensity = max(0, 1 - (distance_from_red / 20))  # 20 is a loose threshold
    return round(intensity, 2)

class TrackedMeteorite:
    def __init__(self, brightness_threshold = 50, red_coef = 10):
        self.time = 0
        self.meteorite_likelihood = 0.0
        self.meteorite_likelihood_acc = 0.0
        self.brightness_threshold = brightness_threshold
        self.red_coef = red_coef
        self.detected = False
    def update(self, frame, hsv, red_mask, bbox):
        self.time += 1
        x,y,w,h = bbox
        if w*h <= 0:
            self.meteorite_likelihood = 0
            return 0
        mean_color = cv2.mean(hsv[y:y+h, x:x+w])[:3]
        trail_area = red_mask[y:y+h, x:x+w]
        br_score = 0
        if mean_color[2] > self.brightness_threshold:
            br_score = (mean_color[2] - self.brightness_threshold) / (255 - self.brightness_threshold)
        red_intensity = red_intensity_from_hue(mean_color[0])
        red_density = cv2.countNonZero(trail_area) / (w*h)
        likelihood = (red_intensity + red_density * self.red_coef) * br_score
        self.meteorite_likelihood_acc += likelihood
        self.meteorite_likelihood = likelihood
        return self.meteorite_likelihood

class MultiObjectTracker:
    def __init__(self, tracker_creator, min_contour_area=1000, max_distance=50):
        """
        Custom MultiTracker for managing multiple objects dynamically.

        :param tracker_creator: A function that returns a new OpenCV tracker instance.
        :param min_contour_area: Minimum area of contours to be considered objects.
        :param max_distance: Max distance to consider the same object across frames.
        """
        self.objects = {}
        self.tracker_creator = tracker_creator  # Function to create a tracker
        self.min_contour_area = min_contour_area
        self.max_distance = max_distance  # Max distance to consider the same object

    def add(self, frame, bbox, tag):
        try:
            tracker = cv2.TrackerCSRT.create()  # Use function to create tracker
            tracker.init(frame, bbox)
            self.objects[uuid.uuid4()] = (tracker, bbox, tag)
        except Exception as e:
            print(f"Error while adding tracker: {e}")

    def update(self, frame):
        """Update all trackers, remove lost ones, and return active bounding boxes."""
        for obj_id in list(self.objects.keys()):
            tracker, bbox, tag = self.objects[obj_id]
            try:
                success, bbox = tracker.update(frame)
                if not success:
                    del self.objects[obj_id]
                self.objects[obj_id] = (tracker, bbox, tag)
            except Exception as e:
                del self.objects[obj_id]
                print(f"Error while updating tracker {obj_id}: {e}")

        return self.objects

    def detect_and_add(self, frame, bbox, tag):
        """Detect a single moving object and add it to tracking if not already tracked."""

        x, y, w, h = bbox
        new_center = (x + w // 2, y + h // 2)  # Compute the center of the object

        # Check if this object is already tracked using center distance
        is_new = True
        for _, tbox, _ in self.objects.values():
            tx, ty, tw, th = tbox
            tracked_center = (tx + tw // 2, ty + th // 2)
            distance = np.linalg.norm(np.array(new_center) - np.array(tracked_center))
            if distance < self.max_distance:  # Object is close enough -> already tracked
                is_new = False
                break
        if is_new:
            self.add(frame, (x, y, w, h), tag)  # Add new tracker

def detect_meteorite(cap, claryfier, mask=None, crop_factor = 4, gray_blur_r = 5, time_threshold = 5):
    global last_meteorite_time
    def resize_frame(frame):
        frame = cv2.resize(frame, (frame.shape[1] // crop_factor, frame.shape[0] // crop_factor))
        return frame
    def prepare_gray(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (gray_blur_r, gray_blur_r), 0)
        if mask is not None:
            frame = cv2.bitwise_and(frame, mask)
        return frame

    # Expanded red color range (covering more shades)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([40, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    fps = FPS().start()
    multi_tracker = MultiObjectTracker(tracker_creator=cv2.TrackerCSRT)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgbg = cv2.createBackgroundSubtractorMOG2()  # Background subtractor
    ret, first_frame = cap.read()
    first_frame = resize_frame(first_frame)
    if mask is not None:
        mask = cv2.resize(mask, (first_frame.shape[1], first_frame.shape[0]))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_frame(frame)
        gray = prepare_gray(frame)
        fgmask = fgbg.apply(gray)  # Background subtraction
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_ellipse)  # Remove small noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_ellipse) # Close small gaps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            bbox = cv2.boundingRect(cnt)
            if area < 10 or area > 10_000 or bbox[2] * bbox[3] <= 0:
                continue
            meteorite = TrackedMeteorite()
            meteorite.update(frame, hsv, red_mask, bbox)
            if meteorite.meteorite_likelihood > 0.5:
                multi_tracker.detect_and_add(frame, bbox, meteorite)
        
        multi_tracker.update(frame)

        # Draw tracked objects
        for obj_id in list(multi_tracker.objects.keys()):
            _, bbox, meteorite = multi_tracker.objects[obj_id]
            meteorite.update(frame, hsv, red_mask, bbox)
            if meteorite.meteorite_likelihood < 0.5:
                del multi_tracker.objects[obj_id]
                continue
            if not meteorite.detected and meteorite.time > time_threshold and meteorite.meteorite_likelihood_acc / meteorite.time > 0.5:
                meteorite.detected = True
                # get extended ROI of the meteorite
                margin = 200
                x1 = max(x - margin, 0)
                y1 = max(y - margin, 0)
                x2 = min(x + w + margin, frame.shape[1])
                y2 = min(y + h + margin, frame.shape[0])

                # Extract extended ROI
                roi = frame[y1:y2, x1:x2].copy()
                cv2.imshow("roi", roi)
                claryfier.send(roi, time.time())

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(obj_id), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow('Meteorite Detection', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        fps.update()
        #if detected:
        #    while True: continue
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the detection
    video_path = None  # Change this to your video file path
    # parse args --video
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="path to video file")
    parser.add_argument("--mask", help="path to mask file")
    parser.add_argument("--mailru_oauth_token", help="mail.ru cloud oauth token")
    parser.add_argument("--grafana-webhook-url", help="grafana webhook url")
    args = parser.parse_args()
    cap = None
    mask = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_COLOR)
        if mask is None:
            print("Cannot open mask")
            exit(1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    if cap is None or not cap.isOpened():
        print("Cannot open video")
        exit(1)

    cap = AsyncVideoReader(cap)
    cap.start()

    claryfier = DummyClaryfier()
    if args.mailru_oauth_token:
        claryfier = CloudClaryfier(args.mailru_oauth_token, "mcs")
        claryfier.start()
    else:
        print("WARNING: no cloud object detection system enabled")

    if args.grafana_webhook_url:
        claryfier.event_handler = GrafanaMeteoriteAlertSender(args.grafana_webhook_url).get_send_alert_handler()

    detect_meteorite(cap, claryfier, mask=mask)