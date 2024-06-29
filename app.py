import os
import cv2
import numpy as np
import math
import base64
from flask import Flask, request, jsonify, Response, render_template
from PIL import Image
import io
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)
video_config_camera = 0

# Obtén la ruta absoluta del directorio actual
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
sign_selected = None
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
          "10"]  # Asegúrate de que las etiquetas en labels_8.txt correspondan a estas etiquetas


def load_model(sign_type):
    # Construye las rutas absolutas para los archivos
    model_path = os.path.join(current_dir, f"Model/keras_model_{sign_type}.h5")
    labels_path = os.path.join(current_dir, f"Model/labels_{sign_type}.txt")

    # Verificar la existencia de los archivos
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


def generate_frames():
    cap = cv2.VideoCapture(video_config_camera)  # Iniciar la captura de video al inicio de la función
    while True:
        success, img = cap.read()
        if not success:
            break

        img_preprocessed, img_with_bbox = preprocess_image(img)
        if img_preprocessed is not None or classifier is None:
            prediction, index = classifier.getPrediction(img_preprocessed)
            global sign_selected, signs
            sign = str(sign_selected)
            accuracy = float(np.max(prediction)) * 100

            # Dibujar predicción y rectángulo en la imagen original
            hands, img = detector.findHands(img, draw=False)
            print(signs[str(sign_selected)])
            if hands and index == signs[str(sign_selected)]['index'] and accuracy > 97:
                print(prediction)
                hand = hands[0]
                x, y, w, h = hand['bbox']
                cv2.putText(img, sign, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Liberar la captura de video al salir de la función
    cap.release()


# Vista sin diseño
@app.route('/camera')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Vista con diseño
@app.route('/expressions')
def index():
    global classifier, sign_selected
    camera_id = request.args.get('camera_id', default=0, type=str)
    sign_selected = camera_id
    classifier = load_model(camera_id)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
