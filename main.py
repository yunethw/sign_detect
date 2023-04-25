from pathlib import Path

from tkinter import *
from tkinter import Tk, Canvas, Button, PhotoImage
from PIL import Image, ImageTk
import cv2 as cv
import data
import record
from tensorflow import keras
import numpy as np


word = ''
prev_letter = ''
start_tick = 0
freq = cv.getTickFrequency()

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")

TEXT_TOP_POS = (10, 100)
FONT = cv.FONT_HERSHEY_SIMPLEX
COLOR = (255, 255, 255)

model = keras.models.load_model('alphabet_model.h5')
print(model.summary())


def center_screen(screen):
    global screen_height, screen_width, x_cordinate, y_cordinate
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (1080 / 2))
    y_cordinate = int((screen_height / 2) - (720 / 2))
    screen.geometry("{}x{}+{}+{}".format(1080, 720, x_cordinate, y_cordinate))


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def talk_now():
    global word
    word = ''
    cap = cv.VideoCapture(0)
    # New tkinter window
    window.withdraw()
    new_window = Toplevel()
    new_window.title("Handy")
    new_window.configure(bg="black")
    center_screen(new_window)
    ico = Image.open(relative_to_assets("hand.png"))
    photo = ImageTk.PhotoImage(ico)
    new_window.title("Handy")
    new_window.wm_iconphoto(False, photo)

    # Label to display the text output
    text_label = Label(new_window, text="", font=("Raleway SemiBold", 24), fg="white", bg="black")
    text_label.pack(side="bottom", pady=10)

    # Label to display video stream
    video_label = Label(new_window)
    video_label.pack()

    # Back button to release the webcam and close all windows
    back_button_image = PhotoImage(file=relative_to_assets("back_yellow.png"))
    back_button = Button(new_window, image=back_button_image, borderwidth=0, highlightthickness=0,
                         command=lambda: [cap.release(), new_window.destroy(), window.deiconify()], relief="flat",
                         compound='center')
    back_button.image = back_button_image
    back_button.place(x=25.0,
                      y=25.0)

    # Function to update the video stream
    def update_video():
        global prev_letter, word, start_tick, freq
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame, results = record.mediapipe_detect(frame, record.hands)
        if results.multi_hand_landmarks:
            record.draw_landmarks(frame, results.multi_hand_landmarks)
            features = data.get_features(results)
            features = np.array(features).reshape(1, 176)
            prediction = model.predict(features)
            letter = chr(np.argmax(prediction) + 65)
            probability = prediction[0][np.argmax(prediction)]
            time = int((cv.getTickCount() - start_tick) / freq)
            frame = cv.putText(frame, '{0}: {1:.4}'.format(letter, probability), TEXT_TOP_POS, FONT, 1, COLOR,
                               thickness=2)
            if probability > 0.85 and time > 1:
                if letter != prev_letter:
                    word = word + letter
                    start_tick = cv.getTickCount()
                    prev_letter = letter
                text_label.config(text=f"{word}")
        #     else:
        #         text_label.config(text="")
        # else:
        #     text_label.config(text="")

        # Convert the frame to PIL Image
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((1080, 720))

        photo = ImageTk.PhotoImage(image)
        video_label.config(image=photo)
        video_label.image = photo

        # Update funcion call
        new_window.after(5, update_video)

    update_video()


slide = 1


def next_image():
    global slide
    if slide == 4:
        slide = 0
    image_image_2.configure(file=relative_to_assets(f"slide{slide + 1}.png"))
    dots.configure(file=relative_to_assets(f"dots{slide + 1}.png"))
    slide += 1


def previous_image():
    global slide
    if slide == 1:
        slide = 5
    image_image_2.configure(file=relative_to_assets(f"slide{slide - 1}.png"))
    dots.configure(file=relative_to_assets(f"dots{slide - 1}.png"))
    slide -= 1


window = Tk()
ico = Image.open(relative_to_assets("hand.png"))
photo = ImageTk.PhotoImage(ico)

window.title("Handy")
window.wm_iconphoto(False, photo)

window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=720,
    width=1080,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
canvas.create_rectangle(
    0.0,
    2.842170943040401e-14,
    1080.0,
    720.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    677.0,
    2.842170943040401e-14,
    1080.0,
    720.0,
    fill="#E14E4E",
    outline="")

canvas.create_text(
    708.0,
    431.0,
    anchor="nw",
    text="Hands Can\nTalk Too.",
    fill="#FFFFFF",
    font=("Raleway Black", 64 * -1)
)

canvas.create_text(
    151.0,
    108.0,
    anchor="nw",
    text="Welcome to Handy.",
    fill="#E14E4E",
    font=("Raleway SemiBold", 40 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    932.0,
    258.0,
    image=image_image_1
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: talk_now(),
    relief="flat"
)
button_1.place(
    x=869.0,
    y=638.0,
    width=178.0,
    height=45.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: previous_image(),
    relief="flat"
)
button_2.place(
    x=35.0,
    y=389.0,
    width=55.0,
    height=55.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: next_image(),
    relief="flat"
)
button_3.place(
    x=588.0,
    y=389.0,
    width=55.0,
    height=55.0
)

image_image_2 = PhotoImage(
    file=relative_to_assets("slide1.png"))
dots = PhotoImage(file=relative_to_assets("dots1.png"))
image_2 = canvas.create_image(
    338.0,
    415.0,
    image=image_image_2
)
dots1 = canvas.create_image(
    338.0,
    610.0,
    image=dots
)

center_screen(window)

window.resizable(False, False)
window.mainloop()
