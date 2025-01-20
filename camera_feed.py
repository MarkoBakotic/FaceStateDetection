import cv2

def main():
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    video_capture = cv2.VideoCapture(0)

    color_filter = None

    def detect_bounding_box(vid, original):
        gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return vid

    while True:
        result, video_frame = video_capture.read()
        if not result:
            break

        processed_frame = video_frame.copy()

        if color_filter == "red":
            processed_frame[:, :, 0] = 0
            processed_frame[:, :, 1] = 0
        elif color_filter == "green":
            processed_frame[:, :, 0] = 0
            processed_frame[:, :, 2] = 0
        elif color_filter == "blue":
            processed_frame[:, :, 1] = 0
            processed_frame[:, :, 2] = 0

        processed_frame = detect_bounding_box(processed_frame, video_frame)

        cv2.imshow("My Face Detection Project", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            color_filter = "red"
        elif key == ord("g"):
            color_filter = "green"
        elif key == ord("b"):
            color_filter = "blue"
        elif key == ord("n"):
            color_filter = None

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
