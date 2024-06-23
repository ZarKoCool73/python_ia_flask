import cv2
import numpy as np
import math
from flask import Flask, Response
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)

model_path="Model/keras_model.h5"
labels_path="Model/labels.txt"

offset = 20
imgSize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)
labels = ["A", "B", "C"]  # Asegúrate de que las etiquetas en labels.txt correspondan a estas etiquetas


def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break

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

                prediction, index = classifier.getPrediction(imgWhite)
                print(prediction, index)

                cv2.putText(img, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/expressions')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
