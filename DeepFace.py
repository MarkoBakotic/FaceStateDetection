import cv2
from deepface import DeepFace

AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

AGE_RANGES = ['(0-2)', '(3-6)', '(7-12)', '(13-20)', '(21-32)', '(33-43)', '(44-53)', '(54-100)']
GENDERS = ['Male', 'Female']

age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

def predict_age_gender(face_image):
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDERS[gender_preds[0].argmax()]
    
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_RANGES[age_preds[0].argmax()]
    
    return gender, age

def main():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
    color_filter = None
    frame_skip = 5
    frame_count = 0

    def detect_bounding_box(vid, original):
        gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            face = original[y:y+h, x:x+w]

            gender, age = predict_age_gender(face)

            try:
                analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
                emotion = analysis[0]["dominant_emotion"]
            except Exception as e:
                emotion = "Error" 

            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(vid, f"{gender}, {age}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(vid, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vid

    while True:
        result, video_frame = video_capture.read()
        if not result:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue 
        video_frame = cv2.resize(video_frame, (640, 480))
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
        cv2.imshow("Face Detection", processed_frame)

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
