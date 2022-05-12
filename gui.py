import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import *

import cv2
import numpy as np
from PIL import ImageTk, Image
import threading

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# import trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# adding voice to output
import pyttsx3

engine = pyttsx3.init()


def say(sign):
    engine.say(sign)
    engine.runAndWait()


def getClassName(classNo):
    # print(classNo)
    classes = {1: 'Speed limit (20km/h)',
               2: 'Speed limit (30km/h)',
               3: 'Speed limit (50km/h)',
               4: 'Speed limit (60km/h)',
               5: 'Speed limit (70km/h)',
               6: 'Speed limit (80km/h)',
               7: 'End of speed limit (80km/h)',
               8: 'Speed limit (100km/h)',
               9: 'Speed limit (120km/h)',
               10: 'No passing',
               11: 'No passing veh over 3.5 tons',
               12: 'Right-of-way at intersection',
               13: 'Priority road',
               14: 'Yield',
               15: 'Stop',
               16: 'No vehicles',
               17: 'Veh > 3.5 tons prohibited',
               18: 'No entry',
               19: 'General caution',
               20: 'Dangerous curve left',
               21: 'Dangerous curve right',
               22: 'Double curve',
               23: 'Bumpy road',
               24: 'Slippery road',
               25: 'Road narrows on the right',
               26: 'Road work',
               27: 'Traffic signals',
               28: 'Pedestrians',
               29: 'Children crossing',
               30: 'Bicycles crossing',
               31: 'Beware of ice/snow',
               32: 'Wild animals crossing',
               33: 'End speed + passing limits',
               34: 'Turn right ahead',
               35: 'Turn left ahead',
               36: 'Ahead only',
               37: 'Go straight or right',
               38: 'Go straight or left',
               39: 'Keep right',
               40: 'Keep left',
               41: 'Roundabout mandatory',
               42: 'End of no passing',
               43: 'End no passing veh > 3.5 tons'}
    return classes[int(classNo) + 1]


# image preprocessing
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    imf = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def real_time():
    began_count = False
    frameWidth = 640
    frameHeight = 480
    brightness = 180
    threshold = 0.90
    font = cv2.FONT_HERSHEY_SIMPLEX

    # video camera
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)

    # import trained model
    pickle_in = open("model_trained.p", "rb")
    model = pickle.load(pickle_in)



    while True:
        # read image
        success, imgOriginal = cap.read()

        # process image
        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))

        img = preprocessing(img)
        cv2.imshow("processed img", img)
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgOriginal, "CLASS ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, "PROBABILITY ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        # predict image
        predicitions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predicitions)

        if probabilityValue > threshold:
            # print(getClassName(classIndex))
            cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2,
                        cv2.LINE_AA)

        cv2.imshow("Result", imgOriginal)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break




def classify(file_path):
    img = Image.open(file_path)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    print(img.shape)
    img = preprocessing(img)
    cv2.imshow("processed img", img)
    img = img.reshape(1, 32, 32, 1)

    predict_x = model.predict([img])[0]
    pred = np.argmax(predict_x, axis=0)
    sign = getClassName(pred)
    print(sign)

    label.configure(foreground='#011638', text=sign)
    t = threading.Thread(target=say, args=(sign,))
    t.start()


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="UPLOAD AN IMAGE", command=upload_image, padx=20, pady=10)
upload.configure(background='blue', foreground='red', font=('frontierswoman', 10, 'bold'))

# upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
# upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)

rt = Button(top, text="realtime", command=real_time, padx=10, pady=5)
rt.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
rt.pack(side=BOTTOM, padx=10, pady=50)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
