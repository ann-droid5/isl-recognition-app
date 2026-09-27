import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
import json
import os
import mediapipe as mp

# 1. We placed your model loading inside a function so Streamlit can cache it
def load_model_and_classes(model_path="isl_mobilenet_model_final.keras", classes_path="class_names_final.json"):
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, "r") as f:
        class_names = json.load(f)
    return model, class_names


# 2. MediaPipe Hand Landmarker setup
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


def predict_frame(frame, model, class_names):

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe: hand presence check only
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = detector.detect(mp_image)

    # Skip if no hand
    if len(result.hand_landmarks) == 0:
        return None, None

    # Fixed crop used in the successful experiment
    height, width, _ = frame_rgb.shape

    y1 = int(height * 0.15)
    y2 = int(height * 0.75)

    x1 = 0
    x2 = width

    cropped = frame_rgb[
        y1:y2,
        x1:x2
    ]

    # Resize
    cropped = cv2.resize(
        cropped,
        (224, 224)
    )

    # Prepare input
    cropped = cropped.astype(
        np.float32
    )

    cropped = np.expand_dims(
        cropped,
        axis=0
    )

    # MobileNetV2 prediction
    prediction = model.predict(
        cropped,
        verbose=0
    )

    class_index = np.argmax(
        prediction[0]
    )

    confidence = prediction[0][
        class_index
    ]

    label = (
        class_names[str(class_index)]
        if isinstance(class_names, dict)
        else class_names[class_index]
    )

    return label, confidence


def predict_video(video_path, model, class_names):

    cap = cv2.VideoCapture(
        video_path
    )

    predictions = []
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 5 != 0:
            continue

        label, confidence = predict_frame(
            frame,
            model,
            class_names
        )

        # Skip frames where MediaPipe
        # did not detect a hand
        if label is None:
            continue

        if confidence >= 0.60:
            predictions.append(label)

    cap.release()

    return predictions


def get_final_prediction(predictions):

    if len(predictions) == 0:
        return "No confident prediction"

    counter = Counter(
        predictions
    )

    final_prediction = counter.most_common(
        1
    )[0][0]

    return final_prediction


# 3. Wrapped your testing code so it doesn't execute
# when Streamlit imports this file
if __name__ == "__main__":

    model, class_names = load_model_and_classes()

    video_path = "test_video.mp4"

    if os.path.exists(video_path):

        predictions = predict_video(
            video_path,
            model,
            class_names
        )

        print("Frame predictions:")
        print(predictions)

        final_prediction = get_final_prediction(
            predictions
        )

        print(
            "Final prediction:",
            final_prediction
        )

    else:

        print(
            f"Please place a '{video_path}' "
            "file in this folder to test "
            "this script directly."
        )