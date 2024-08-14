"""
import cv2
facemodel=cv2.CascadeClassifier("face.xml")
vid=cv2.VideoCapture("face.png")
i=1
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        faces=facemodel.detectMultiScale(frame)
        for (x,y,l,w) in faces:
            face_img=frame[y:y+w,x:x+l]
            path="data/"+str(i)+".jpg"
            i=i+1
            cv2.imwrite(path,face_img)
            cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,0),3)
        cv2.namedWindow("my window",cv2.WINDOW_NORMAL)
        cv2.imshow("my window",frame)
        key=cv2.waitKey(20)
        if(key==ord('x')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()   

"""
import streamlit as st
import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image
image_path = 'mask1.jpg'
image = cv2.imread(image_path)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2, minSize=(30, 30))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
#cv2.namedWindow("Face Detection Result",cv2.WINDOW_NORMAL)
#cv2.imshow('Face Detection Result', image)
st.image(image,channels='BGR',width=400)
cv2.waitKey(0)
cv2.destroyAllWindows()
