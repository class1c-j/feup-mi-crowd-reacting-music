import cv2
from ultralytics import YOLO


class PeopleRecognition:
    def __init__(self, model_path="yolov8n-face.pt"):
        self.model = YOLO(model_path)

    def crop_people(self, frame):
        cropped_images = []
        results = self.model(frame)

        for result in results:
            for rect in result.boxes.xyxy:
                x1, y1, x2, y2 = rect
                x1, y1, x2, y2 = (
                    int(x1.item()),
                    int(y1.item()),
                    int(x2.item()),
                    int(y2.item()),
                )
                cropped_image = frame[y1:y2, x1:x2]
                cropped_images.append(cropped_image)

        return cropped_images

    def detect_people(self, frame):
        results = self.model(frame)
        return results


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    recognizer = PeopleRecognition()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            cropped_images = recognizer.crop_people(frame)

            for i, image in enumerate(cropped_images):
                cv2.imshow("cropped_" + str(i), image)

            cv2.namedWindow("original", cv2.WINDOW_NORMAL)
            cv2.imshow("original", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./data/people_footage/party1.mp4"
    process_video(video_path)
