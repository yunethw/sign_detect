import cv2 as cv
import data
import record
from tensorflow import keras
import numpy as np

TEXT_TOP_POS = (10, 30)
FONT = cv.FONT_HERSHEY_SIMPLEX
COLOR = (255, 255, 255)

cap = cv.VideoCapture(0)
features = []
model = keras.models.load_model('alphabet_model.h5')
print(model.summary())

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    frame, results = record.mediapipe_detect(frame, record.hands)
    if results.multi_hand_landmarks:
        record.draw_landmarks(frame, results.multi_hand_landmarks)
        features = data.get_features(results)
        features = np.array(features).reshape(1, 168)
        prediction = model.predict(features)
        letter = chr(np.argmax(prediction) + 65)
        probability = prediction[0][np.argmax(prediction)]
        if probability > 0.5:
            frame = cv.putText(frame, '{0}: {1:.4}'.format(letter, probability), TEXT_TOP_POS, FONT, 1, COLOR)

    cv.imshow('OpenCV Feed', frame)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break




