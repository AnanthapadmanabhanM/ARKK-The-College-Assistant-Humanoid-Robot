import pyttsx3
engine=pyttsx3.init()

engine.setProperty('rate',115)
engine.setProperty('volume',1.0)
voices = engine.getProperty('voices')
engine.setProperty('voices',voices[1].id)

print("listen")
engine.say("hello")
engine.runAndWait()

