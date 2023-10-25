from tensorflow import keras
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from flask import Flask, Response, jsonify
from flask_cors import CORS

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


# Crear una función para preprocesar la imagen de entrada
def preprocess_image(image):
    # Redimensionar la imagen a un tamaño compatible con el modelo (por ejemplo, 224x224)
    image = cv2.resize(image, (224, 224))
    # Normalizar los valores de píxel en el rango [0, 1]
    image = image.astype('float') / 255.0
    # Agregar una dimensión adicional para representar el lote de imágenes (batch)
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image_verbos(image):
    # Redimensionar la imagen a un tamaño compatible con el modelo (por ejemplo, 200x200)
    image = cv2.resize(image, (228, 241))
    # Normalizar los valores de píxel en el rango [0, 1]
    image = image.astype('float') / 255.0
    # Agregar una dimensión adicional para representar el lote de imágenes (batch)
    image = np.expand_dims(image, axis=0)
    return image


# Inicializar la cámara con la resolución deseada
width, height = 1280, 720  # Cambia la resolución a la que desees
cap = cv2.VideoCapture(0)
cap.set(3, width)  # Establecer el ancho del fotograma
cap.set(4, height)  # Establecer la altura del fotograma

# Inicializar el detector de manos de Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# Fuente y tamaño del texto
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1  # Reducir el tamaño de la letra
font_thickness = 2  # Reducir el grosor de la letra

# Inicializar Flask
app = Flask(__name__)
CORS(app)
transmitiendo_video = True
transmitiendo_verbos = True


# Ruta de la página principal
@app.route('/')
def index():
    return "message"


# Función para obtener los frames de la cámara
def get_frame():
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)
    global transmitiendo_video
    while True:
        # Leer el fotograma actual de la cámara
        ret, frame = cap.read()
        # Verificar si el fotograma es válido
        if not ret or frame is None:
            continue  # Salta este fotograma y sigue con el siguiente
        # Convertir el fotograma a escala de grises
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detectar las manos en el fotograma
        results = hands.process(frame_rgb)
        # Verificar si se detectaron manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtener la posición del dedo pulgar
                thumb_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                x_center = int(np.mean([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])

                # Determinar si la mano está a la izquierda o derecha basándose en la posición del dedo pulgar
                if thumb_x < x_center:
                    hand_label = "Izquierda"
                    text_x = 10  # Ubicación en la esquina izquierda
                else:
                    hand_label = "Derecha"
                    text_x = frame.shape[1] - 200  # Ubicación en la esquina derecha
                # Dibujar los puntos de referencia de la mano en el fotograma
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                # Preprocesar la imagen de entrada para la mano
                preprocessed_frame = preprocess_image(frame)
                # Realizar la predicción utilizando el modelo
                predictions = model.predict(preprocessed_frame)
                predicted_class = np.argmax(predictions[0])
                # Obtener la etiqueta correspondiente a la clase predicha
                predicted_label = labels_dict[predicted_class]
                if predicted_class < 9:
                    message = "Numero"
                else:
                    message = "Letra"
                # Mostrar la etiqueta en la parte superior izquierda o derecha de la pantalla según la mano detectada
                # cv2.putText(frame, f'Mano: {hand_label}', (text_x, 50), font, font_scale, (0, 255, 0), font_thickness)
                cv2.putText(frame, f'{message}: {predicted_label}', (text_x, 100), font, font_scale, (0, 255, 0),
                            font_thickness)

        # Mostrar el fotograma en una ventana
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_frame_verbos():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Preprocesar
        img = preprocess_image_verbos(frame)

        # Predicción
        pred = model1.predict(img)

        # Obtener etiqueta
        label = labels_dict_verbos[pred.argmax()]

        # Mostrar resultado
        cv2.putText(frame, label, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Ruta para el streaming de video
@app.route('/api/video')
def video_feed():
    global transmitiendo_video
    transmitiendo_video = True
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/videoVerbos')
def video_verbos_feed():
    global transmitiendo_verbos
    transmitiendo_verbos = True
    return Response(get_frame_verbos(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Ruta para detener el servicio de video
@app.route('/api/stop_video')
def stop_video():
    global transmitiendo_video
    global transmitiendo_verbos
    transmitiendo_video = False
    transmitiendo_verbos = False
    return jsonify(message='Servicio de video detenido.')


if __name__ == '__main__':
    # Iniciar la aplicación Flask y hacer que escuche en todas las interfaces de red
    app.run(debug=False, host='0.0.0.0')

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
