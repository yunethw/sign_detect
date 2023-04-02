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
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            wrist_left = results.multi_hand_landmarks[0].landmark[0]
            wrist_right = results.multi_hand_landmarks[1].landmark[0]
            index_left = results.multi_hand_landmarks[0].landmark[8]
            index_right = results.multi_hand_landmarks[1].landmark[8]
            rel_x = abs((index_right.x - index_left.x)/(wrist_right.x - wrist_left.x))
            rel_y = abs((index_right.y - index_left.y)/(wrist_right.y - wrist_left.y))
            print(rel_x, rel_y)
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

