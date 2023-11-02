import os

# Deshabilitar la GPU para TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from flask import Flask, Response, jsonify, render_template, request, make_response
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
from aiortc import RTCPeerConnection, RTCSessionDescription
import mediapipe as mp
import asyncio
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


# Función para obtener los frames de la cámara para letras
def get_frame():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No funciona la cámara")
        return

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                    x_center = int(np.mean([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])

                    if thumb_x < x_center:
                        hand_label = "Izquierda"
                        text_x = 10
                    else:
                        hand_label = "Derecha"
                        text_x = frame.shape[1] - 200

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    preprocessed_frame = preprocess_image(frame)
                    predictions = model.predict(preprocessed_frame)
                    predicted_class = np.argmax(predictions[0])
                    predicted_label = labels_dict[predicted_class]
                    if predicted_class < 9:
                        message = "Numero"
                    else:
                        message = "Letra"

                    cv2.putText(frame, f'{message}: {predicted_label}', (text_x, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


def preprocess_image1(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float') / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/process_image', methods=['POST'])
def process_image():
    img_data = request.get_json()['imageData']

    # Decodificar imagen
    img_bytes = base64.b64decode(img_data.split(',')[1])
    img = np.array(Image.open(io.BytesIO(img_bytes)))

    # Preprocesar
    prep_img = preprocess_image1(img)

    # Predicción
    pred = model.predict(prep_img)
    sign = labels_dict[np.argmax(pred)]

    # Convertir imagen a base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}',
        'sign': sign
    })


# Función para obtener los frames de la cámara para verbos
def get_frame_verbos():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No funciona camara1", cv2.Error)
        print("No funciona camara2", cv2.getBuildInformation())
    try:
        while True:
            ret, frame = cap.read()
            print("Captura de fotograma exitosa o no:", ret)

            if not ret:
                break

            img = preprocess_image_verbos(frame)
            pred = model1.predict(img)
            label = labels_dict_verbos[pred.argmax()]

            cv2.putText(frame, label, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


def create_answer(offer):
    # Configurar la conexión peer
    configuration = {'iceServers': [{'urls': 'stun:stun.l.google.com:19302'}]}
    peer_connection = RTCPeerConnection(configuration)

    # Establecer la descripción de la oferta remota
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def set_remote_description():
        await peer_connection.setRemoteDescription(offer)

    loop.run_until_complete(set_remote_description())

    # Crear una respuesta a la oferta
    answer = loop.run_until_complete(peer_connection.createAnswer())

    # Establecer la descripción local y obtener la respuesta
    async def set_local_description():
        await peer_connection.setLocalDescription(answer)

    loop.run_until_complete(set_local_description())

    return peer_connection.localDescription


# Ruta para el streaming de video para letras
@app.route('/alpha', methods=['GET'])
def video_feed():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No funciona la cámara")
        return

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                    x_center = int(np.mean([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])

                    if thumb_x < x_center:
                        hand_label = "Izquierda"
                        text_x = 10
                    else:
                        hand_label = "Derecha"
                        text_x = frame.shape[1] - 200

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    preprocessed_frame = preprocess_image(frame)
                    predictions = model.predict(preprocessed_frame)
                    predicted_class = np.argmax(predictions[0])
                    predicted_label = labels_dict[predicted_class]
                    if predicted_class < 9:
                        message = "Numero"
                    else:
                        message = "Letra"

                    cv2.putText(frame, f'{message}: {predicted_label}', (text_x, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


# Ruta para el streaming de video para verbos
@app.route('/betta', methods=['GET'])
def video_verbos_feed():
    return Response(get_frame_verbos(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Ruta para detener el servicio de video
@app.route('/api/stop_video', methods=['GET'])
def stop_video():
    return jsonify(message='Servicio de video detenido.')


@app.route('/offer')
def offer():
    offer = RTCSessionDescription(sdp=request.json['offer']['sdp'], type=...)
    pc = RTCPeerConnection()
    pc.setRemoteDescription(offer)
    answer = pc.createAnswer()
    pc.setLocalDescription(answer)
    return jsonify({'answer': {'sdp': answer.sdp, 'type': answer.type}})


    # No es necesario ejecutar app.run() en un entorno de producción


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

# Liberar los recursos
cv2.destroyAllWindows()
