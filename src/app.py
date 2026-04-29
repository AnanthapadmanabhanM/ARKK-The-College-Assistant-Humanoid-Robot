import requests
import textwrap
import json
import cv2
import pyttsx3
#import google.generativeai as genai
import speech_recognition as sr 
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from IPython.display import display
from IPython.display import Markdown
import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
from tkinter import Toplevel, Label, Entry, Button
import time

d={'MG03':'i.jpg','MG01':'i.jpg','Ladies Amenities Centre':'i.jpg','MG09':'i.jpg',
   'IT Staff room':'i.jpg','UG Dean Office':'i.jpg','Reprographic Center':'i.jpg',
   'MG07':'i.jpg','Library':'i.jpg','MGO2':'i.jpg','Placement Cell':'h.jpg','Unnati':'h.jpg',
   'M501':'h.jpg','M502':'h.jpg', 'Zoom IT Hall':'g.jpg','M403':'g.jpg','TPLC':'g.jpg',
   'M404':'g.jpg','AP Electrical and Electronics Engineering':'g.jpg','AP IT':'g.jpg',
   'AP Civil':'f.jpg','M309':'f.jpg','M307':'f.jpg','M305':'f.jpg','M303':'f.jpg','M311':'f.jpg',
   'M313':'f.jpg','Electrical and Electronics Engineering Staffroom':'f.jpg',
   'Electronics and Communication Engineering Staffroom':'f.jpg','CCE':'f.jpg','CERD':'f.jpg',
   'M300C':'f.jpg','M300A':'f.jpg','HOD Electromnics and Communication Engineering':'f.jpg',
   'M301':'f.jpg','HOD IT':'f.jpg','M304':'f.jpg','M306':'f.jpg','M312':'f.jpg','M208':'f.jpg',
   'M310':'f.jpg','Electronics and Communication Engineering Staffroom':'f.jpg','HOD Physical Education':'f.jpg',
   'AP Mechanical':'e.jpg','M205':'e.jpg','M203A':'e.jpg','M203B':'e.jpg','M203C':'e.jpg',
   'AP Electronics and Communiation Engineering':'e.jpg','M207':'e.jpg','M209':'e.ipg',
   'HOD Electrical and Electronics Engineering':'e.jpg','HOD Civil':'e.jpg','HOD Mechanical':'e.jpg',
   'PTA Office':'e.jpg','M201':'e.jpg','M200B':'e.jpg','M200A':'e.jpg','M204':'e.jpg','M208':'e.jpg',
   'M206':'e.jpg','M212':'e.jpg','M210':'e.jpg','Science Staffroom':'e.jpg','HOD GS':'e.jpg','M103':'d.jpg',
   'M101':'d.jpg','Toilet Men':'d.jpg','Toilet Women':'d.jpg','Administrative Office':'M103',
   'Examination Office':'d.jpg','M100':'d.jpg','Principal Room':'d.jpg','M102':'d.jpg','Conference Hall':'d.jpg',
   'TEQUIP Office':'d.jpg','M104':'d.jpg','M106':'d.jpg','AP Mechanical':'d.jpg','Mechanical Staffroom':'d.jpg',
   'M106':'d.jpg','M112':'d.jpg','Bus Office':'d.jpg','KTU Office':'d.jpg','Staff Room IT':'c.jpg','Amenity':'c.jpg',
   'UPS Room':'c.jpg','Web Application Lab':'c.jpg','ITG08':'c.jpg','Programming Lab':'c.jpg','ITG09':'c.jpg',
   'Electronics Circuit Lab':'c.jpg','Signal Processing Lab':'c.jpg','Fibre Optics Lab':'c.jpg','Maintenance Cell':'c.jpg',
   'Microcontroller Lab':'c.jpg','Project Lab IT':'c.jpg','Software Testing Lab':'c.jpg','Network Security Lab':'c.jpg',
   'IT208':'c.jpg','IT Staffroom':'c.jpg','IT Department Library':'c.jpg','Internal combustion engine lab':'b.jpg',
   'Project Lab Mechanical':'b.jpg','Staffroom Mechanical':'b.jpg','Thermal Engineering Lab':'b.jpg',
   'PG computational lab':'a.jpg','Computer Aided Design Lab':'a.jpg','CAD Lab':'a.jpg','Staff room Mechanical':'a.jpg',
   'Non Destructive Testing Facility':'a.jpg','ADAM':'a.jpg','Metrology and advanced manufacturing lab':'j.jpg',
   'Metrology and advanced manufacturing lab':'j.jpg','Geotechnical lab':'j.jpg','PG03':'j.jpg','Transportation Engineering Lab':'j.jpg',
   'Strength of Materials Lab':'j.jpg','Electronics and Communication Engineering Seminar Hall':'k.jpg','Nano Electronics and Research lab':'k.jpg',
   'Communication Engineering Lab':'k.jpg','Systems and Control Lab':'k.jpg','Power Electronics Lab':'k.jpg','P204':'l.jpg',
   'P202':'l.jpg','Digital Electronics Lab':'l.jpg','P203':'l.jpg','Environmental Engineering Lab':'l.jpg','P201':'l.jpg',
   'Metrology  and advanced manufacturing lab':'m.jpg','Geotechnical lab':'m.jpg',
   'Transportation Engineering Lab':'m.jpg','Strength of materials lab':'m.jpg','Electronics and communication engineering seminar hall':'n.jpg',
   'Easy IT seminar hall':'n.jpg','EC IT seminar hall':'n.jpg','ASAP':'n.jpg','Nano electronics and research lab':'n.jpg',
   'Communication Engineering Lab':'n.jpg','Systems and Control lab':'n.jpg','Power Electronics Lab':'n.jpg','P204':'o.jpg',
   'Digital Electronics Lab':'o.jpg','P202':'o.jpg','Environmental Engineering Lab':'o.jpg','Machine Shop':'p.jpg','Machines Lab':'p.jpg',
   'Metrology Lab':'p.jpg','MB07':'q.jpg','MB01':'q.jpg'}

#gemini integration
#GOOGLE_API_KEY = 'AIzaSyAdgIHqcD6ATooLvzXaxu-tlWStO_glk1o'
#genai.configure(api_key=GOOGLE_API_KEY)
#model1 = genai.GenerativeModel('gemini-pro')
#generation.config = GenerationConfi(temprature=0.9,top_p=1.0,top_k=32,candidate_count)


model = Sequential()
classifier = load_model('emotion_model_1.h5')

class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

engine=pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')  
engine.setProperty('volume',1.0)
engine.setProperty('voice', voices[1].id)
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

#def to_markdown(text):
    #text = text.replace('•', '  *')
    #text = text.replace('*', ' ')
    #return textwrap.indent(text, '> ', predicate=lambda _: True)
    #print(text)
    #return text


class Application:
    def __init__(self):

        self.root=tk.Tk()
        self.root.title('ARKK')
        #self.root.attributes('-fullscreen', True)
        #self.root.state('zoomed')
        self.root.configure(bg="#205591")
        self.root.protocol('WM_DELETE_WINDOW',self.destructor)
        self.root.geometry("1500x950")
       
        
        self.setup_gui(self,image_path="bg.png")
        self.status_label = tk.Label(self.root, text=" ", font=("Helvetica", 32), bg="#205591",fg="#ffc45d")
        screen_height=self.root.winfo_screenheight()
        self.status_label.place(x=20, y=screen_height-25)
        self.name=1000
        self.cam=cv2.VideoCapture(0)
        self.mloop()

    def setup_gui(self,image_path="bg.png"):
        image= Image.open(image_path)
        screen_width=self.root.winfo_screenwidth()
        screen_height=self.root.winfo_screenheight()
        image=image.resize((screen_width,screen_height-50))
        print(screen_width,screen_height)

        self.photo_image=ImageTk.PhotoImage(image)
        self.image_label=tk.Label(self.root,image=self.photo_image)
        self.image_label.place(x=0,y=20,relwidth=1, relheight=1)

        # self.status_label = Label(self, text="Initializing...", bg='lightgrey', fg='black')
        # self.status_label.pack(side='bottom', fill='x')

        # self.image_label = Label(self, bg='black')
        # self.image_label.pack(padx=10, pady=10, expand=True, fill='both')
        # self.status_label = Label(self, text="Initializing...", bg='lightgrey', fg='black')
        # self.status_label.pack(side='bottom', fill='x')
    
    def speechRec(self):
        r = sr.Recognizer() 
        for ye in range(3):

            with sr.Microphone() as source:
                print("Please wait. Calibrating microphone...") 
                self.status_label.config(text="Please wait. Calibrating microphone...",font=("Helvetica", 40, "bold"))  
                r.adjust_for_ambient_noise(source, duration=2)  
                print(ye)
                print("Say something!") 
                self.status_label.config(text="Say something!",font=("Helvetica", 40, "bold")) 
                audio = r.listen(source,timeout=5,phrase_time_limit=10) 
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
    
    def check_face(self,img):
        id=1000
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(self.minW), int(self.minH)))
        #print(len(faces))
        for(x,y,w,h) in faces:
            print("detected")
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if (confidence < 70):
                #id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                
            elif confidence >= 70:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        return id
            
    
    def mloop(self):
        face_flag=0
        
        self.minW = 0.1*self.cam.get(3)
        self.minH = 0.1*self.cam.get(4)
        ret, img =self.cam.read()
    
        if not ret:
            print("Camera not connected")
            self.status_label.config(text="Camera not connected",font=("Helvetica", 40, "bold")) 
        
        if self.name==1000:
            self.name=self.check_face(img)
        else:
            print(self.name)
            emot=emotionVideo(img)
            
            text="Hello how can i help you?"
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
            elif emot=="Disgust":
                text="Hey, it looks like something's grossing you out. Want to talk about what's bothering you? How can I help?"

            if self.name=="unknown":
                engine.say(text)
                engine.runAndWait()
            else:
                #text="Hello "+name+" how can i help you?"
                engine.say(text)
                engine.runAndWait()
            face_flag=1
            while face_flag:
                t=self.speechRec()
                if t!=0:
                    res=get_bot_response(t)
                    #print(res)
                    if "loc_place" in res:
                        place=res.split(",")[1]
                        for i in d:
                            j=i.lower()
                            j=j.replace(" ","")
                            place=place.lower()
                            place=place.replace(" ","")
                            if j == place:
                                self.setup_gui(image_path="image_folder/"+d[i])
                                time.sleep(10)
                                self.setup_gui()
                    elif "book_app" in res:
                        print("appointment_booked")
                        self.book_appointment()
                    else:
                        if res !=0:
                            engine.say(res)
                            engine.runAndWait()
                        else:
                            #response = model1.generate_content(t,)
                            tt="Sorry i am unable to help you with your query"
                            #print(response.text)
                            engine.say(tt)
                            engine.runAndWait()
                else:
                    if emot=="Angry":
                        text="Okay, see you later! Channel that anger into positive action."
                    elif emot=="Fear":
                        text="Farewell! Embrace the unknown with bravery and resilience."
                    elif emot=="Happy":
                        text="Alright, catch you later! Keep spreading those good vibes!"
                    elif emot=="Neutral":
                        text="Okay, see you later! Channel that anger into positive action."
                    elif emot=="Sad":
                        text="Alright, take care! I hope brighter days are ahead for you."
                    elif emot=="Surprise":
                        text="Alright, take care! I'm glad to leave you surprised."
                    elif emot=="Disgust":
                        text="Alright, take care! If you need to vent further, I'm here to listen."
                    engine.say(text)
                    engine.runAndWait()
                    face_flag=0
                    self.name=1000
        self.root.after(30,self.mloop)
    def book_appointment(self):
        print("Booking appointment...")
        self.new_window = Toplevel(self)
        self.new_window.title("Book Appointment")
        
        Label(self.new_window, text="Enter Details").pack()
        self.info_entry = Entry(self.new_window)
        self.info_entry.pack()
        Button(self.new_window, text="Submit", command=self.submit_appointment).pack()

    def submit_appointment(self):
        info = self.info_entry.get()
        print(f"Information submitted: {info}")
        self.new_window.destroy()

    def destructor(self):
        print("Closing Application...")
        self.vs.release()
        #self.ser.close()
        self.root.destroy()

app=Application()
app.root.mainloop()
    
