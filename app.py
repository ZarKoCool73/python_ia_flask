import os
import cv2
import numpy as np
import math
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import base64
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
classifier = None
offset = 20
imgSize = 250
detector = HandDetector(maxHands=1)
signs = {
    '0': {'index': 0}, '1': {'index': 2}, '2': {'index': 2},
    '3': {'index': 2}, '4': {'index': 2}, '5': {'index': 2},
    '6': {'index': 2}, '7': {'index': 2}, '8': {'index': 2},
    '9': {'index': 2}, '10': {'index': 2}
}
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
sign_selected = None


def load_model(sign_type):
    model_path = os.path.join(current_dir, f"Model/keras_model_{sign_type}.h5")
    labels_path = os.path.join(current_dir, f"Model/labels_{sign_type}.txt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"No se encontró el archivo de etiquetas en {labels_path}")
    return Classifier(model_path, labels_path)


def preprocess_image(img):
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            return imgWhite, img
    return None, img


@app.route('/process_image', methods=['POST'])
def process_image():
    global classifier, sign_selected, signs
    data = request.get_json()
    image_data = data.get('imageData')
    expressions = data.get('Expressions')
    sign_selected = expressions
    classifier = load_model(sign_selected)
    if not image_data:
        return jsonify({
            'sign': None,
            'accuracy': 0,
            'error': 'No image data provided'
        }), 400

    try:
        # Decodificar la imagen desde base64
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({
            'sign': '-',
            'accuracy': 0,
            'error': 'Seña no encontrada'
        }), 400

    img_preprocessed, _ = preprocess_image(img)
    if img_preprocessed is not None:
        prediction, index = classifier.getPrediction(img_preprocessed)
        sign = '-'
        accuracy = float(np.max(prediction)) * 100

        if index == signs.get(str(sign_selected), {}).get('index') and accuracy > 95:
            sign = sign_selected
        else:
            accuracy = 0

        return jsonify({
            'sign': sign,
            'accuracy': accuracy
        })
    else:
        return jsonify({
            'sign': '-',
            'accuracy': 0,
            'error': 'Seña no encontrada'
        }), 400


@app.route('/camera')
def video_feed():
    return render_template('index.html')


@app.route('/expressions')
def index():
    global classifier, sign_selected
    camera_id = request.args.get('camera_id', default=0, type=str)
    sign_selected = camera_id
    classifier = load_model(camera_id)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
