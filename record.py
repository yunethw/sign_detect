import cv2 as cv
import mediapipe as mp
import extract_landmarks_copy as extract

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detect(image, mp_model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    for hand_landmarks in results:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def start():
    start_tick = 0
    printed = False
    extracted = False
    letter = '0'
    cap = cv.VideoCapture(0)
    freq = cv.getTickFrequency()
    text_top_position = (5, 30)
    text_below_position = (5, 60)
    font = cv.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    datafile_path = 'data.xlsx'

    results_array = []
    frame_num = 0

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame, results = mediapipe_detect(frame, hands)

        if results.multi_hand_landmarks:
            draw_landmarks(frame, results.multi_hand_landmarks)
        key = cv.waitKey(10)
        if key > 0:
            if key == ord('Q'):
                break
            elif ord('a') <= key <= ord('z'):
                letter = chr(key).upper()
                printed = False
                extracted = False
                start_tick = cv.getTickCount()

        time = int((cv.getTickCount() - start_tick) / freq)
        if time < 13:  # start recording procedure
            frame = cv.putText(frame, letter, text_top_position, font, 1, color)
            if time <= 3:  # count down
                frame = cv.putText(frame, 'Recording in {}'.format(3 - time), text_below_position, font, 1, color)
            elif 3 < time <= 8:  # recording 5 seconds
                frame = cv.putText(frame, '{}'.format(time - 3), text_below_position, font, 1, color)
                results_array.append(results)
                frame_num += 1
            else:  # recording complete
                if not printed:
                    print('Letter {}: {} frames recorded'.format(letter, frame_num))
                    printed = True
                    frame = cv.putText(frame, 'Processing...', text_below_position, font, 1, color)
                elif printed and not extracted:
                    print('Extracting landmarks...')
                    extract.extract_landmarks(results_array[0:frame_num], datafile_path, letter)
                    extracted = True
                    print('Done.')
        else:
            frame = cv.putText(frame, 'Press letter key to record', text_top_position, font, 1, color)

        cv.imshow('Feed', frame)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    start()
