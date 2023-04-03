import cv2
import numpy as np
import random
import mediapipe as mp
import pandas as pd

landmarks_ar_lx_rx_ly_ry_lz_rz = []
landmarks_ar_wlx_wly = []
landmarks_ar_wrx_wry = []


def extract_landmarks(frames, path):
    dynm_cols = ['LETTER']

    for idx_d_col_1 in range(21):
        dynm_cols = dynm_cols + [f'L{idx_d_col_1}X'] + [f'L{idx_d_col_1}Y']

    for idx_d_col_2 in range(21):
        dynm_cols = dynm_cols + [f'R{idx_d_col_2}X'] + [f'R{idx_d_col_2}Y']

    for idx_d_col_3 in range(21):
        dynm_cols = dynm_cols + [f'{idx_d_col_3}X'] + [f'{idx_d_col_3}Y'] + [f'L{idx_d_col_3}Z'] + [f'R{idx_d_col_3}Z']

    df = pd.DataFrame(columns=dynm_cols)

    hands = mp.solutions.hands.Hands()

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_world_landmarks and len(results.multi_hand_world_landmarks) == 2:
            landmarks_ar_wlx = [lmk.x for lmk in results.multi_hand_world_landmarks[0].landmark]
            landmarks_ar_wly = [lmk.y for lmk in results.multi_hand_world_landmarks[0].landmark]
            landmarks_ar_wrx = [lmk.x for lmk in results.multi_hand_world_landmarks[1].landmark]
            landmarks_ar_wry = [lmk.y for lmk in results.multi_hand_world_landmarks[1].landmark]

            landmarks_ar_lx = [lmk.x for lmk in results.multi_hand_landmarks[0].landmark]
            landmarks_ar_rx = [lmk.x for lmk in results.multi_hand_landmarks[1].landmark]
            landmarks_ar_ly = [lmk.y for lmk in results.multi_hand_landmarks[0].landmark]
            landmarks_ar_ry = [lmk.y for lmk in results.multi_hand_landmarks[1].landmark]

            landmarks_ar_lz = [lmk.z for lmk in results.multi_hand_landmarks[0].landmark]
            landmarks_ar_rz = [lmk.z for lmk in results.multi_hand_landmarks[1].landmark]

        else:
            if not results.multi_hand_world_landmarks[0]:
                landmarks_ar_wlx = [0] * 21
                landmarks_ar_wly = [0] * 21  # replaced all with [0] * 21 for clarity
                landmarks_ar_lx = [0] * 21
                landmarks_ar_ly = [0] * 21
                landmarks_ar_lz = [0] * 21

                landmarks_ar_wrx = [lmk.x for lmk in results.multi_hand_world_landmarks[1].landmark]
                landmarks_ar_wry = [lmk.y for lmk in results.multi_hand_world_landmarks[1].landmark]
                landmarks_ar_rx = [lmk.x for lmk in results.multi_hand_landmarks[1].landmark]
                landmarks_ar_ry = [lmk.y for lmk in results.multi_hand_landmarks[1].landmark]

                landmarks_ar_rz = [lmk.z for lmk in results.multi_hand_landmarks[1].landmark]
            else:
                landmarks_ar_wrx = [0] * 21
                landmarks_ar_wry = landmarks_ar_wrx
                landmarks_ar_rx = landmarks_ar_wrx
                landmarks_ar_ry = landmarks_ar_wrx
                landmarks_ar_rz = landmarks_ar_wrx

                landmarks_ar_wlx = [lmk.x for lmk in results.multi_hand_world_landmarks[0].landmark]
                landmarks_ar_wly = [lmk.y for lmk in results.multi_hand_world_landmarks[0].landmark]

                landmarks_ar_lx = [lmk.x for lmk in results.multi_hand_landmarks[0].landmark]
                landmarks_ar_ly = [lmk.y for lmk in results.multi_hand_landmarks[0].landmark]

                landmarks_ar_lz = [lmk.z for lmk in results.multi_hand_landmarks[0].landmark]

        for idx_l in range(21):
            landmarks_ar_wlx_wly.append(landmarks_ar_wlx[idx_l])
            landmarks_ar_wlx_wly.append(landmarks_ar_wly[idx_l])

        for idx_r in range(21):
            landmarks_ar_wrx_wry.append(landmarks_ar_wrx[idx_r])
            landmarks_ar_wrx_wry.append(landmarks_ar_wry[idx_r])

        for idx in range(21):
            landmarks_ar_lx_rx_ly_ry_lz_rz.append(abs(landmarks_ar_lx[idx] - landmarks_ar_rx[idx]))
            landmarks_ar_lx_rx_ly_ry_lz_rz.append(abs(landmarks_ar_ly[idx] - landmarks_ar_ry[idx]))
            landmarks_ar_lx_rx_ly_ry_lz_rz.append(landmarks_ar_lz[idx])
            landmarks_ar_lx_rx_ly_ry_lz_rz.append(landmarks_ar_rz[idx])

        landmarks_array = [path[0]] + landmarks_ar_wlx_wly + landmarks_ar_wrx_wry + landmarks_ar_lx_rx_ly_ry_lz_rz

        landmarks_ar_wlx_wly.clear()
        landmarks_ar_wrx_wry.clear()
        landmarks_ar_lx_rx_ly_ry_lz_rz.clear()

        df.loc[i] = landmarks_array

    excel_path = '/home/thalal/PycharmProjects/sign_detect/hand_marks.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='Landmarks', index=False)


def frame_capture(path):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_array = np.empty((50, frame_height, frame_width, 3), dtype=np.uint8)
    rand_int = np.random.choice(num_frames, 50, False)
    j = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret and i in rand_int:
            frames_array[j] = frame
            j += 1

    return frames_array



def initiate_hand_marks(path):
    selected_frames = frame_capture(path)
    extract_landmarks(selected_frames, path)


if __name__ == '__main__':
    initiate_hand_marks('a.mp4')

