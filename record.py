import cv2 as cv
import mediapipe as mp
import data

TEXT_TOP_POS = (5, 30)
TEXT_BELOW_POS = (5, 60)
FONT = cv.FONT_HERSHEY_SIMPLEX
COLOR = (255, 255, 255)
DATAFILE_PATH = 'data.xlsx'

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
    freq = cv.getTickFrequency()

    printed = False
    extracted = False

    letter = '0'
    results_array = []
    dataframe = data.SignData()

    cap = cv.VideoCapture(0)

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
                results_array.clear()
                start_tick = cv.getTickCount()

        time = int((cv.getTickCount() - start_tick) / freq)
        if time < 13:  # start recording procedure
            frame = cv.putText(frame, letter, TEXT_TOP_POS, FONT, 1, COLOR)
            if time <= 3:  # count down
                frame = cv.putText(frame, 'Recording in {}'.format(3 - time), TEXT_BELOW_POS, FONT, 1, COLOR)
            elif 3 < time <= 8:  # recording 5 seconds
                frame = cv.putText(frame, '{}'.format(time - 3), TEXT_BELOW_POS, FONT, 1, COLOR)
                results_array.append(results)
            else:  # recording complete
                if not printed:
                    print('Letter {}: {} frames recorded'.format(letter, len(results_array)))
                    printed = True
                    frame = cv.putText(frame, 'Processing...', TEXT_BELOW_POS, FONT, 1, COLOR)
                elif printed and not extracted:
                    print('Extracting landmarks...')
                    dataframe.add_data(results_array, letter)
                    extracted = True
                    print('Done.')
        else:
            frame = cv.putText(frame, 'Press letter key to record', TEXT_TOP_POS, FONT, 1, COLOR)

        cv.imshow('Feed', frame)

    dataframe.save_data(DATAFILE_PATH)
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    start()
