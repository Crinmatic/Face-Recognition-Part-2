
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
model = load_model('model.h5')
#loading cascade
face_cascade = cv2.CascadeClassifier('\frontalFace10\haarcascade_frontalface_alt.xml')
 
def face_extractor(img ):
    #detects face and returns cropped face
    faces = face_cascade.detectMultiScale(img, 1.3,5,0)
    if faces is  ():
        return None
    #crop all faces round
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face
#with webcam
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize (face, (224,224))
        im = Image.fromarray(face, 'RGB')
        #resizing into 128*128 because we trained with this image size
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred =model.predict(img_array)
        print(pred)
        
        name = 'None matching'
        
    
        if pred[0][1] > 0.5:
            name = 'person1' 
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
     
        else:
           if pred.all() < 1.0:
              names = 'unknown face' 
              cv2.putText(frame, names, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
          
       
    else:
       cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()

