import cv2
from mediapipe import solutions

mp_hands = solutions.hands
mp_drawing = solutions.drawing_utils


def mediapipe_detect(image, mp_model):
    flip_image = cv2.flip(image, flipCode=1)
    image = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)
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
        # cv2.flip(frame, 1)
        image, results = mediapipe_detect(frame, hands)

        # print(results.multi_handedness[0].classification[0].label)
        # print(results.multi_handedness[1].classification[0].label)
        if results.multi_hand_landmarks:
            for i, result in enumerate(results.multi_hand_landmarks):
                print(i, result.landmark[8].x, end='\t')
            print()
            for result in results.multi_hand_world_landmarks:
                print(result.landmark[8].x, end='\t')
            print()
            for result in results.multi_handedness:
                print(result.classification[0].label, end='\t')
            print('\n')

            # index_right = results.multi_hand_landmarks[0].landmark[8]
            # draw_landmarks(image, results.multi_hand_landmarks)
            # print(index_right.z)

        if results.multi_hand_landmarks:
            draw_landmarks(image, results.multi_hand_landmarks)

        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

