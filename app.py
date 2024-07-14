import base64
from flask import Flask, request, jsonify, render_template
import cv2 as cv
import numpy as np
import io
from io import BytesIO
from PIL import Image
import copy
import itertools
from collections import Counter, deque
import csv
import mediapipe as mp
from flask_cors import CORS
from keras.models import load_model
from utils import CvFpsCalc
from modelLetters.keypoint_classifier.keypoint_classifier import KeyPointClassifier as KeyPointClassifierLetters
from modelNumbers.keypoint_classifier.keypoint_classifier import KeyPointClassifier as KeyPointClassifierNumbers

from modelLetters.point_history_classifier.point_history_classifier import PointHistoryClassifier as PointHistoryClassifierLetters
from modelNumbers.point_history_classifier.point_history_classifier import PointHistoryClassifier as PointHistoryClassifierNumbers

app = Flask(__name__)
CORS(app)

# Configuration
USE_STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
HISTORY_LENGTH = 16

# Initialize MediaPipe Hands once
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=USE_STATIC_IMAGE_MODE,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# Load models and labels once
model_verbs = load_model("modelVerbs/keras_model.h5", compile=False)
class_names_verbs = open("modelVerbs/labels.txt", "r").readlines()

keypoint_classifier_numbers = KeyPointClassifierNumbers()
point_history_classifier_numbers = PointHistoryClassifierNumbers()

keypoint_classifier_letters = KeyPointClassifierLetters()
point_history_classifier_letters = PointHistoryClassifierLetters()

with open('modelLetters/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels_letters = [row[0] for row in csv.reader(f)]

with open('modelNumbers/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels_numbers = [row[0] for row in csv.reader(f)]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictLetters', methods=['POST'])
def predict_letters():
    try:
        data = request.get_json()
        image_data = data['image'].split(",")[1]  # Remove the data URL prefix
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)

        # Convert RGB to BGR as OpenCV expects images in BGR format
        image_np = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)

        # Process the image and get predictions
        predictions = process_image_letters(image_np)

        if not predictions:
            return jsonify({"error": "No hand sign detected"}), 400

        return jsonify(predictions)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        image.close()  # Ensure the PIL image resource is released


def process_image_letters(image):
    point_history = deque(maxlen=HISTORY_LENGTH)
    finger_gesture_history = deque(maxlen=HISTORY_LENGTH)

    # Flip the image horizontally for a later selfie-view display
    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    # Convert the BGR image to RGB before processing
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    predictions = {}
    accuracy = 0.0  # Initialize accuracy

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

            # Hand sign classification
            hand_sign_id = keypoint_classifier_letters(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Point gesture
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            # Finger gesture classification
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == (HISTORY_LENGTH * 2):
                finger_gesture_id = point_history_classifier_letters(pre_processed_point_history_list)

            # Calculates the gesture IDs in the latest detection
            finger_gesture_history.append(finger_gesture_id)

            predicted_hand_sign = keypoint_classifier_labels_letters[hand_sign_id]

            # Calculate accuracy based on confidence levels
            accuracy = handedness.classification[0].score * 100

            predictions = {
                "hand_sign": predicted_hand_sign,
                "accuracy": accuracy
            }
            print(predictions)
    else:
        point_history.append([0, 0])

    # Return None if no hand landmarks are detected
    if not predictions:
        return None

    return predictions


@app.route('/predictNumbers', methods=['POST'])
def predict_numbers():
    try:
        data = request.get_json()
        image_data = data['image'].split(",")[1]  # Remove the data URL prefix
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)

        # Convert RGB to BGR as OpenCV expects images in BGR format
        image_np = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)

        # Process the image and get predictions
        predictions = process_image_numbers(image_np)

        if not predictions:
            return jsonify({"error": "No hand sign detected"}), 400

        return jsonify(predictions)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        image.close()


def process_image_numbers(image):
    point_history = deque(maxlen=HISTORY_LENGTH)
    finger_gesture_history = deque(maxlen=HISTORY_LENGTH)

    # Flip the image horizontally for a later selfie-view display
    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    # Convert the BGR image to RGB before processing
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    predictions = {}
    accuracy = 0.0  # Initialize accuracy

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

            # Hand sign classification
            hand_sign_id = keypoint_classifier_numbers(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Point gesture
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            # Finger gesture classification
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == (HISTORY_LENGTH * 2):
                finger_gesture_id = point_history_classifier_numbers(pre_processed_point_history_list)

            # Calculates the gesture IDs in the latest detection
            finger_gesture_history.append(finger_gesture_id)

            predicted_hand_sign = keypoint_classifier_labels_numbers[hand_sign_id]

            # Calculate accuracy based on confidence levels
            accuracy = handedness.classification[0].score * 100

            predictions = {
                "hand_sign": predicted_hand_sign,
                "accuracy": accuracy
            }
            print(predictions)
    else:
        point_history.append([0, 0])

    # Return None if no hand landmarks are detected
    if not predictions:
        return None

    return predictions

def get_predictions_from_image_verbs(image):
    # Resize the image to the required size (224x224)
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)

    # Convert the image to a numpy array and normalize it
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model_verbs.predict(image)
    index = np.argmax(prediction)
    class_name = class_names_verbs[index].strip()
    confidence_score = prediction[0][index]

    # Return prediction and confidence score
    return {
        "hand_sign": class_name,
        "accuracy": str(np.round(confidence_score * 100))[:-2] + "%"
    }

@app.route('/predictVerbs', methods=['POST'])
def predict_verbs():
    try:
        data = request.get_json()
        image_data = data['image'].split(",")[1]  # Remove the data URL prefix
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)

        # Convert RGB to BGR as OpenCV expects images in BGR format
        image_np = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)

        predictions = get_predictions_from_image_verbs(image_np)

        if not predictions:
            return jsonify({"error": "No hand sign detected"}), 400

        return jsonify(predictions)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([(int(landmark.x * image_width), int(landmark.y * image_height))
                               for landmark in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(landmark.x * image_width), image_width - 1),
             min(int(landmark.y * image_height), image_height - 1)]
            for landmark in landmarks.landmark]


def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    temp_landmark_list = [[x - base_x, y - base_y] for x, y in landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list), default=1)
    return [n / max_value for n in temp_landmark_list]


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    base_x, base_y = point_history[0] if point_history else (0, 0)
    temp_point_history = [[(x - base_x) / image_width, (y - base_y) / image_height] for x, y in point_history]
    return list(itertools.chain.from_iterable(temp_point_history))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
