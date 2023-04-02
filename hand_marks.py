import cv2
import numpy as np
import random
import mediapipe as mp
import pandas as pd

lx_cols = [f'lx{i}' for i in range(21)]
ly_cols = [f'ly{i}' for i in range(21)]
rx_cols = [f'rx{i}' for i in range(21)]
ry_cols = [f'ry{i}' for i in range(21)]

dynm_cols = lx_cols + ly_cols + rx_cols + ry_cols

df = pd.DataFrame(columns=dynm_cols)


def extract_landmarks(frames):
    hands = mp.solutions.hands.Hands()

    landmark_data = []

    # landmarks_array=[]

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_world_landmarks and len(results.multi_hand_world_landmarks) == 2:
            landmarks_ar_lx = [lmk.x for lmk in results.multi_hand_world_landmarks[0].landmark]
            landmarks_ar_ly = [lmk.y for lmk in results.multi_hand_world_landmarks[0].landmark]
            landmarks_ar_rx = [lmk.x for lmk in results.multi_hand_world_landmarks[1].landmark]
            landmarks_ar_ry = [lmk.y for lmk in results.multi_hand_world_landmarks[1].landmark]
            landmarks_ar_lz = [[lmk.z] for lmk in results.multi_hand_landmarks[0].landmark]
            landmarks_ar_rz = [[lmk.z] for lmk in results.multi_hand_landmarks[1].landmark]

            # landmarks_array = landmarks_ar_lx_ly.append(landmarks_ar_rx_ry)
            # landmarks_array = np.concatenate((landmarks_ar_lx_ly, landmarks_ar_rx_ry), axis=1)
            landmarks_array = landmarks_ar_lx + landmarks_ar_ly + landmarks_ar_rx + landmarks_ar_ry
            # print(len(landmarks_array))
            # print(landmarks_ar_rx_ry)

            # print(landmarks_array)

            df.loc[i] = landmarks_array

        # if results.multi_hand_world_landmarks and len(results.multi_hand_world_landmarks) == 2:
        #     for h in range(2):
        #         for i in range(21):
        #             print("2 hands")
        #
        #             landmarks_array.append(results.multi_hand_world_landmarks[h].landmark[i].x)
        #             landmarks_array.append(results.multi_hand_world_landmarks[h].landmark[i].y)
        #
        #             # landmarks_array = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[h].landmark])
        #
        #             # landmark_data.append(lan/dmarks_array.reshape(-1))
        #
        #             # print(f"X coordinate of hand {h} landmark {i}",
        #             #       results.multi_hand_world_landmarks[h].landmark[i].x)
        #             # print(f"Y coordinate of hand {h} landmark {i} ",
        #             #       results.multi_hand_world_landmarks[h].landmark[i].y)
        #             # if h == 0:
        #             #     print("---------------------------------------------------------")
        #             #     print(f"Relative values of landmark {i} ",
        #             #           abs(results.multi_hand_landmarks[h].landmark[i].x -
        #             #               results.multi_hand_landmarks[h + 1].landmark[i].x))
        #             # print(f"Z coordinate of hand {h} landmark {i} ",
        #             #       results.multi_hand_landmarks[h].landmark[i].z)
        #             # print("------------------------")
        #
        #
        # elif results.multi_hand_world_landmarks and len(results.multi_hand_world_landmarks) == 1:
        # print("Hand :", len(results.multi_hand_landmarks), results.multi_hand_landmarks)
        # print("1 hand")

    # print(len(landmarks_array))
    # df = pd.DataFrame(landmark_data, columns=dynamic_columns)
    # excel_path = 'C:\\Users\\User\\Downloads\\hand_landmarks.xlsx'
    # with pd.ExcelWriter(excel_path) as writer:
    #     df.to_excel(writer, sheet_name='Landmarks', index=False)


def frame_capture(path):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_array = np.empty((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames_array[i] = frame
    selected_frames = random.choices(frames_array, k=5)
    return selected_frames


def initiate_hand_marks(path):
    selected_frames = frame_capture(path)
    extract_landmarks(selected_frames)
    excel_path = '/home/thalal/PycharmProjects/sign_detect/hand_marks.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='Landmarks', index=False)


