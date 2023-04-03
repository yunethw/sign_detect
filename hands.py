import cv2
from mediapipe import solutions

mp_hands = solutions.hands
mp_drawing = solutions.drawing_utils


def mediapipe_detect(image, mp_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image,
    #                           results.face_landmarks,
    #                           mp_holistic.FACEMESH_TESSELATION,
    #                           landmark_drawing_spec=mp_drawing.DrawingSpec((255, 0, 0), 1, 1),
    #                           connection_drawing_spec=mp_drawing.DrawingSpec((255, 255, 255), 1, 1))
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    for hand_landmarks in results:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detect(frame, hands)

        print(len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 'No hand')
        # if results.multi_hand_landmarks:
        #     index_right = results.multi_hand_landmarks[0].landmark[8]
        #     draw_landmarks(image, results.multi_hand_landmarks)
        #     print(index_right.z)

        # if results.multi_hand_landmarks:
        #     draw_landmarks(image, results.multi_hand_landmarks)
        # cv2.flip()
        cv2.imshow('OpenCV Feed', cv2.flip(image, flipCode=1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

