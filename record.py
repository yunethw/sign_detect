import cv2 as cv

start_tick = 0
letter = '0'


def start():
    global start_tick, letter
    cap = cv.VideoCapture(0)
    freq = cv.getTickFrequency()
    text_top_position = (5, 30)
    text_below_position = (5, 60)
    font = cv.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        key = cv.waitKey(10)
        if key > 0:
            if key == ord('Q'):
                break
            elif ord('a') <= key <= ord('z'):
                letter = chr(key).upper()
                start_tick = cv.getTickCount()

        time = int((cv.getTickCount() - start_tick) / freq)
        if time < 15:  # start recording procedure
            frame = cv.putText(frame, letter, text_top_position, font, 1, color)
            if time <= 3:  # count down
                frame = cv.putText(frame, 'Recording in {}'.format(3 - time), text_below_position, font, 1, color)
            elif 3 < time <= 8:  # recording 5 seconds
                frame = cv.putText(frame, '{}'.format(time - 3), text_below_position, font, 1, color)
            else:  # recording complete
                frame = cv.putText(frame, 'Recording Complete', text_below_position, font, 1, color)
        else:
            frame = cv.putText(frame, 'Press letter key to record', text_top_position, font, 1, color)

        cv.imshow('Feed', frame)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    start()
