import os

# Deshabilitar la GPU para TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import base64
import io

from PIL import Image

# Cargar el modelo preentrenado
model = load_model('lenguaje_detector_1.model')
model1 = load_model('modelo_verbos_3.h5')

# Diccionario de etiquetas para las letras del alfabeto en lenguaje de señas
labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
    12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
    18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N',
    24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}
labels_dict_verbos = {
    0: 'ABRAZAR', 1: 'CALLAR', 2: 'DORMIR', 3: 'CERRAR', 4: 'CURIOSEAR', 5: 'PERDONAR',
}

# Inicializar Flask
app = Flask(__name__)
CORS(app)

# Inicializar el detector de manos de Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)


# Función para preprocesar la imagen de entrada para la mano
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float') / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Función para preprocesar la imagen de entrada para verbos
def preprocess_image_verbos(image):
    image = cv2.resize(image, (228, 241))
    image = image.astype('float') / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Ruta de la página principal
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/expressions')
def expressions():
    return render_template("index.html")


@app.route('/comprehension')
def comprehension():
    return render_template("index.html")


@app.route('/process_image', methods=['POST'])
def process_image():
    img_data = request.get_json()['imageData']
    expressions = request.get_json()['Expressions']

    # Decodificar imagen
    img_bytes = base64.b64decode(img_data.split(',')[1])
    img = np.array(Image.open(io.BytesIO(img_bytes)))

    if expressions:
        # Detección de expresiones

        prep_img = preprocess_image(img)
        pred = model.predict(prep_img)
        sign = labels_dict[np.argmax(pred)]
        accuracy = float(np.max(pred)) * 100

    else:
        # Detección de compresión

        prep_img = preprocess_image_verbos(img)
        pred = model1.predict(prep_img)
        sign = labels_dict_verbos[np.argmax(pred)]
        accuracy = float(np.max(pred)) * 100

    if sign == "" or accuracy < 80:
        return jsonify({
            'message': 'No se detectó seña con suficiente precisión'
        })

    else:
        # encode image
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': f'data:image/jpeg;base64,{img_base64}',
            'sign': sign,
            'accuracy': accuracy
        })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

# Liberar los recursos
cv2.destroyAllWindows()
