import pandas as pd
import random

ROWS_PER_LETTER = 65


def get_features(result):
    landmarks_relative = []
    landmarks_world_left = []
    landmarks_world_right = []
    global wlx, wly, wrx, wry, lx, ly, lz, rx, ry, rz

    if result.multi_hand_landmarks:
        if len(result.multi_hand_landmarks) == 2:
            match result.multi_handedness[0].classification[0].label:
                case 'Right':
                    right, left = 0, 1
                case 'Left':
                    left, right = 0, 1

            wlx = [lmk.x for lmk in result.multi_hand_world_landmarks[left].landmark]
            wly = [lmk.y for lmk in result.multi_hand_world_landmarks[left].landmark]
            wrx = [lmk.x for lmk in result.multi_hand_world_landmarks[right].landmark]
            wry = [lmk.y for lmk in result.multi_hand_world_landmarks[right].landmark]

            lx = [lmk.x for lmk in result.multi_hand_landmarks[left].landmark]
            ly = [lmk.y for lmk in result.multi_hand_landmarks[left].landmark]
            lz = [lmk.z for lmk in result.multi_hand_landmarks[left].landmark]

            rx = [lmk.x for lmk in result.multi_hand_landmarks[right].landmark]
            ry = [lmk.y for lmk in result.multi_hand_landmarks[right].landmark]
            rz = [lmk.z for lmk in result.multi_hand_landmarks[right].landmark]
        elif len(result.multi_hand_landmarks) == 1:
            wlx = [0] * 21
            wly = [0] * 21
            lx = [0] * 21
            ly = [0] * 21
            lz = [0] * 21
            wrx = [lmk.x for lmk in result.multi_hand_world_landmarks[0].landmark]
            wry = [lmk.y for lmk in result.multi_hand_world_landmarks[0].landmark]
            rx = [lmk.x for lmk in result.multi_hand_landmarks[0].landmark]
            ry = [lmk.y for lmk in result.multi_hand_landmarks[0].landmark]
            rz = [lmk.z for lmk in result.multi_hand_landmarks[0].landmark]
    else:
        wlx = [0] * 21
        wly = [0] * 21
        wrx = [0] * 21
        wry = [0] * 21
        lx = [0] * 21
        ly = [0] * 21
        lz = [0] * 21
        rx = [0] * 21
        ry = [0] * 21
        rz = [0] * 21

    for idx in range(21):
        landmarks_world_left.append(wlx[idx])
        landmarks_world_left.append(wly[idx])
        landmarks_world_right.append(wrx[idx])
        landmarks_world_right.append(wry[idx])

    for idx in range(21):
        landmarks_relative.append(abs(lx[idx] - rx[idx]))
        landmarks_relative.append(abs(ly[idx] - ry[idx]))
        landmarks_relative.append(lz[idx])
        landmarks_relative.append(rz[idx])

    data = landmarks_world_left + landmarks_world_right + landmarks_relative
    return data


class SignData:
    def __init__(self):
        # creating array of column names
        cols = ['LETTER']
        for idx in range(21):
            cols = cols + [f'L{idx}X'] + [f'L{idx}Y']
        for idx in range(21):
            cols = cols + [f'R{idx}X'] + [f'R{idx}Y']
        for idx in range(21):
            cols = cols + [f'{idx}X'] + [f'{idx}Y'] + [f'L{idx}Z'] + [f'R{idx}Z']

        self.df = pd.DataFrame(columns=cols)

    def add_data(self, results: list, letter: str):
        # w - world, l/r - left/right, x/y/z - coordinates
        landmarks_relative = []
        landmarks_world_left = []
        landmarks_world_right = []

        selected_results = random.sample(results, k=ROWS_PER_LETTER) if len(results) > ROWS_PER_LETTER else results
        for result in selected_results:
            if result.multi_hand_landmarks:
                if len(result.multi_hand_landmarks) == 2:
                    match result.multi_handedness[0].classification[0].label:
                        case 'Right':
                            right, left = 0, 1
                        case 'Left':
                            left, right = 0, 1

                    wlx = [lmk.x for lmk in result.multi_hand_world_landmarks[left].landmark]
                    wly = [lmk.y for lmk in result.multi_hand_world_landmarks[left].landmark]
                    wrx = [lmk.x for lmk in result.multi_hand_world_landmarks[right].landmark]
                    wry = [lmk.y for lmk in result.multi_hand_world_landmarks[right].landmark]

                    lx = [lmk.x for lmk in result.multi_hand_landmarks[left].landmark]
                    ly = [lmk.y for lmk in result.multi_hand_landmarks[left].landmark]
                    lz = [lmk.z for lmk in result.multi_hand_landmarks[left].landmark]

                    rx = [lmk.x for lmk in result.multi_hand_landmarks[right].landmark]
                    ry = [lmk.y for lmk in result.multi_hand_landmarks[right].landmark]
                    rz = [lmk.z for lmk in result.multi_hand_landmarks[right].landmark]
                elif len(result.multi_hand_landmarks) == 1:
                    wlx = [0] * 21
                    wly = [0] * 21
                    lx = [0] * 21
                    ly = [0] * 21
                    lz = [0] * 21
                    wrx = [lmk.x for lmk in result.multi_hand_world_landmarks[0].landmark]
                    wry = [lmk.y for lmk in result.multi_hand_world_landmarks[0].landmark]
                    rx = [lmk.x for lmk in result.multi_hand_landmarks[0].landmark]
                    ry = [lmk.y for lmk in result.multi_hand_landmarks[0].landmark]
                    rz = [lmk.z for lmk in result.multi_hand_landmarks[0].landmark]
            else:
                continue

            for idx in range(21):
                landmarks_world_left.append(wlx[idx])
                landmarks_world_left.append(wly[idx])
                landmarks_world_right.append(wrx[idx])
                landmarks_world_right.append(wry[idx])

            for idx in range(21):
                landmarks_relative.append(abs(lx[idx] - rx[idx]))
                landmarks_relative.append(abs(ly[idx] - ry[idx]))
                landmarks_relative.append(lz[idx])
                landmarks_relative.append(rz[idx])

            # complete row of data
            landmarks_array = [ord(letter) - 65] + landmarks_world_left + landmarks_world_right + landmarks_relative
            next_row_index = len(self.df.index)
            self.df.loc[next_row_index] = landmarks_array

            # reset row data
            landmarks_world_left.clear()
            landmarks_world_right.clear()
            landmarks_relative.clear()

    def save_data(self, excel_datafile_path):
        excel_path = excel_datafile_path
        with pd.ExcelWriter(excel_path) as writer:
            self.df.to_excel(writer, sheet_name='Landmarks', index=False)
        print(f'Data saved at {excel_path}')
        self.df.astype({'LETTER': 'int32'})
        self.df.to_csv('data.csv', index=False, header=False)
        print(f'CSV Data saved at data.csv')
