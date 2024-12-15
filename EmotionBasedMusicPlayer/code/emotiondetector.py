import cv2
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser

model = load_model(r'code\TrainedModels\emotion_recognition_model.h5')

face_cascade = cv2.CascadeClassifier(r'code\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
detected_emotion=[]

while True:
  ret, frame = cap.read()
  
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

  for (x, y, w, h) in faces:
    face_roi = gray_frame[y:y+h, x:x+w]
    
    def preprocess_image(img, target_size=(48, 48)):
      img = cv2.resize(img, target_size)
      img = img.astype('float32') / 255.0
      img = img.reshape(target_size[0], target_size[1], 1)
      return img

    preprocessed_face = preprocess_image(face_roi)
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

    prediction = model.predict(preprocessed_face)
    
    predicted_emotion = emotion_labels[np.argmax(prediction[0])]

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    detected_emotion.append(predicted_emotion)
    print(predicted_emotion)

  cv2.imshow('Emotion Recognition', frame)

  if len(detected_emotion) >= 5:
    break

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

def most_repeated(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num

print(f"The Most Repeated Emotion in the person: {most_repeated(detected_emotion)}")

cap.release()
cv2.destroyAllWindows()

def open_playlist(playlist_url):
   webbrowser.open(playlist_url)

playlist_urls = {'Angry': '' , 
                 'Disgust': '' , 
                 'Fear': '' , 
                 'Happy': 'https://open.spotify.com/playlist/37i9dQZF1EpuqQZrqvAenD?si=005749521a874c4c' , 
                 'Neutral': 'https://open.spotify.com/playlist/7hBPTBteuoyasBiJRKIaSR?si=645b653e1fe24088&pt=aa7706046a69434a81f631e231835960' , 
                 'Sad': '' , 
                 'Surprise': '' }

open_playlist(playlist_urls[f'{most_repeated(detected_emotion)}'])