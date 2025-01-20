import os
from deepface import DeepFace
import cv2

def verify_emotion(image_path):
    print(f"Analyzing image: {image_path}")
    try:
        # Analyze the image for emotions using DeepFace
        result = DeepFace.analyze(image_path, actions=['emotion'])
        print("Analysis result:", result)  # Print the result for debugging

        if not result:
            print("No faces detected or analysis failed.")
            return

        # Get the dominant emotion from the result
        dominant_emotion = result[0]['dominant_emotion']
        print(f"Dominant Emotion: {dominant_emotion}")

    except Exception as e:
        print(f"Error during analysis: {e}")

def main():
    image_path = r"C:\Users\Davor\Desktop\1\OpenCV_KI_V1\test_image.jpeg"  # Replace with your image path

    # Check if the image exists and is accessible
    if not os.path.exists(image_path):
        print("Error: Image file does not exist or the path is incorrect.")
        return

    # Read the image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image file cannot be opened. Check the file format or path.")
        return

    print("Image loaded successfully.")
    verify_emotion(image_path)

if __name__ == "__main__":
    main()
