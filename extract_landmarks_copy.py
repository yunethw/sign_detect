import cv2
import numpy as np
import pandas as pd

landmarks_relative = []
landmarks_world_left = []
landmarks_world_right = []


def extract_landmarks(results, datafile, letter):
    def select_results(all_results):

        rand_int = np.random.choice(len(all_results), 30, replace=False)
        results_array = []
        j = 0
        for i in range(len(all_results)):
            if i in rand_int:
                results_array.append(all_results[i])
                j += 1
        return results_array

    selected_results = select_results(results)
    dynm_cols = ['LETTER']
    for idx_d_col_1 in range(21):
        dynm_cols = dynm_cols + [f'L{idx_d_col_1}X'] + [f'L{idx_d_col_1}Y']
    for idx_d_col_2 in range(21):
        dynm_cols = dynm_cols + [f'R{idx_d_col_2}X'] + [f'R{idx_d_col_2}Y']
    for idx_d_col_3 in range(21):
        dynm_cols = dynm_cols + [f'{idx_d_col_3}X'] + [f'{idx_d_col_3}Y'] + [f'L{idx_d_col_3}Z'] + [f'R{idx_d_col_3}Z']

    df = pd.DataFrame(columns=dynm_cols)

    landmarks_ar_wlx = [0] * 21
    landmarks_ar_wly = [0] * 21
    landmarks_ar_wrx = [0] * 21
    landmarks_ar_wry = [0] * 21
    landmarks_ar_lx = [0] * 21
    landmarks_ar_rx = [0] * 21
    landmarks_ar_ly = [0] * 21
    landmarks_ar_ry = [0] * 21
    landmarks_ar_lz = [0] * 21
    landmarks_ar_rz = [0] * 21


    for i, result in enumerate(selected_results):
        if result.multi_hand_landmarks:
            if len(result.multi_hand_world_landmarks) == 2:
                landmarks_ar_wlx = [lmk.x for lmk in result.multi_hand_world_landmarks[0].landmark]
                landmarks_ar_wly = [lmk.y for lmk in result.multi_hand_world_landmarks[0].landmark]
                landmarks_ar_wrx = [lmk.x for lmk in result.multi_hand_world_landmarks[1].landmark]
                landmarks_ar_wry = [lmk.y for lmk in result.multi_hand_world_landmarks[1].landmark]

                landmarks_ar_lx = [lmk.x for lmk in result.multi_hand_landmarks[0].landmark]
                landmarks_ar_rx = [lmk.x for lmk in result.multi_hand_landmarks[1].landmark]
                landmarks_ar_ly = [lmk.y for lmk in result.multi_hand_landmarks[0].landmark]
                landmarks_ar_ry = [lmk.y for lmk in result.multi_hand_landmarks[1].landmark]

                landmarks_ar_lz = [lmk.z for lmk in result.multi_hand_landmarks[0].landmark]
                landmarks_ar_rz = [lmk.z for lmk in result.multi_hand_landmarks[1].landmark]
            elif len(results.multi_hand_world_landmarks) == 1:
                landmarks_ar_wrx = [lmk.x for lmk in result.multi_hand_world_landmarks[0].landmark]
                landmarks_ar_wry = [lmk.y for lmk in result.multi_hand_world_landmarks[0].landmark]
                landmarks_ar_rx = [lmk.x for lmk in result.multi_hand_landmarks[0].landmark]
                landmarks_ar_ry = [lmk.y for lmk in result.multi_hand_landmarks[0].landmark]

                landmarks_ar_rz = [lmk.z for lmk in result.multi_hand_landmarks[0].landmark]

        for idx in range(21):
            landmarks_world_left.append(landmarks_ar_wlx[idx])
            landmarks_world_left.append(landmarks_ar_wly[idx])
            landmarks_world_right.append(landmarks_ar_wrx[idx])
            landmarks_world_right.append(landmarks_ar_wry[idx])

        for idx in range(21):
            landmarks_relative.append(abs(landmarks_ar_lx[idx] - landmarks_ar_rx[idx]))
            landmarks_relative.append(abs(landmarks_ar_ly[idx] - landmarks_ar_ry[idx]))
            landmarks_relative.append(landmarks_ar_lz[idx])
            landmarks_relative.append(landmarks_ar_rz[idx])

        landmarks_array = [ord(letter) - 64] + landmarks_world_left + landmarks_world_right + landmarks_relative

        landmarks_world_left.clear()
        landmarks_world_right.clear()
        landmarks_relative.clear()

        df.loc[i] = landmarks_array

    excel_path = datafile
    with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name='Landmarks', index=False)


def frame_capture(path):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_array = np.empty((50, frame_height, frame_width, 3), dtype=np.uint8)
    rand_int = np.random.choice(num_frames, 50, replace=False)
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