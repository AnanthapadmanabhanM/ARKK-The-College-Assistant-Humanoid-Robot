import requests
import json
import cv2
import pyttsx3
import speech_recognition as sr 
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

model = Sequential()
classifier = load_model('emotion_model_1.h5')

class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

engine=pyttsx3.init()
engine.setProperty('rate',205)
t=[]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_detector_image(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) # Convert the image into GrayScale image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
    return rects, allfaces, img

def face_detector_video(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y + h, x:x + w]

    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    return (x, w, y, h), roi_gray, img

def emotionVideo(frame):

        rect, face, image = face_detector_video(frame)
        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classifier.predict(roi,verbose=0)[0]
            label = class_labels[preds.argmax()]
            return label

        else:
            return 0


def get_bot_response(userText):
    response={}
    data = json.dumps({"sender" : "Rasa","message" : userText})
    headers = {'Content-type' : 'application/json', 'Accept' : 'text/plain'}
    response = requests.post('http://localhost:5005/webhooks/rest/webhook', data = data, headers = headers)
    response = response.json()
    print("*************************")
    print(response)
    print(len(response))
    if len(response)==0:
        return 0
    return response[0]['text']

def check_face(img):
    id=1000
    while 1:
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)))
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                if (confidence < 70):
                    #id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                    
                elif confidence >= 70:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                return id
                break
        except:
            return "unknown"

    
def speechRec():
    r = sr.Recognizer() 
    for _ in range(3):
        with sr.Microphone() as source:
            print("Please wait. Calibrating microphone...")   
            r.adjust_for_ambient_noise(source, duration=2)  
            #while 1:
            print("Say something!") 
            audio = r.listen(source) 
            try:  
                audio=r.recognize_google(audio)
                print("you said '" +audio + "'")  
                audio=audio.lower()
                return audio
            except sr.UnknownValueError:  
                print("could not understand audio")  
            except sr.RequestError as e:  
                print("error; {0}".format(e))
    return 0

while 1:
    face_flag=0
    cam = cv2.VideoCapture(0)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    ret, img =cam.read()
    if not ret:
        print("Camera not connected")
    name=check_face(img)
    face_flag=1
    print("name:",name)
    emot=emotionVideo(img)
    print(emot)
    if emot=="Angry":
        text="Hello, it looks like you're not your usual self. Would you like to talk about what's going on? How can I help?"
    elif emot=="Fear":
        text="Hi, I can sense some unease in your tone. Want to talk about what's scaring you? How can I assist you?"
    elif emot=="Happy":
        text="Hi there! You're looking cheerful today. What's on your mind? What can I do for you?"
    elif emot=="Neutral":
        text="Hi there, you seem pretty calm. What's on your mind? How can I assist you?"
    elif emot=="Sad":
        text="Hey, it looks like something's on your mind. I'm here to listen if you want to share. What can I do for you?"
    elif emot=="Surprise":
        text="Hi there. I can tell  you are very surprised. How can I assist today?"
    else:
        text="Hey, it looks like something's grossing you out. Want to talk about what's bothering you? How can I help?"
    if name=="unknown":
        engine.say(text)
        engine.runAndWait()
    else:
        #text="Hello "+str(name)+" how can i help you?"
        engine.say(text)
        engine.runAndWait()
    while face_flag:
        t=speechRec()
        if t!=0:
            res=get_bot_response(t)
            if res !=0:
                engine.say(res)
                engine.runAndWait()
                

        else:
            engine.say("bye bye...")
            engine.runAndWait()
    