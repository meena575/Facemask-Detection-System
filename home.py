import streamlit as st
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np
import tempfile
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5')
def app():
    st.markdown("<h1 style='text-align: center;'>Face Mask Detection System</h1>", unsafe_allow_html=True)

         # Load the image
    image="https://news.cgtn.com/news/77497a4e7a457a4e3241444d34636a4e3359444f31457a6333566d54/img/d87b2bb0ca8e47dcbff030e6d644f7de/d87b2bb0ca8e47dcbff030e6d644f7de.jpg"
        # Get the width of the image
    image_width = 500
        
        # Calculate the left margin to center the image
        #left_margin = (st.sidebar.columns[0].width - image_width) / 2
        
        # Center the image using Streamlit layout options
    st.markdown(
            f'<div style="text-align:center;">'
            f'<img src="{image}" style="width:400px">'
            '</div>',
            unsafe_allow_html=True
        )
        #st.image("https://global-uploads.webflow.com/625d567276661e857102867d/63cd55af57b94e9886e36427_A%20Beginners%20Guide%20to%20Employee%20Management%20System.png",width=400)
    st.markdown("### About")
    st.markdown("<p>A face mask detection system is a technology designed to automatically identify whether individuals in images, videos, or real-time streams are wearing face masks. It plays a crucial role in enforcing mask-wearing policies, especially during public health crises such as the COVID-19 pandemic.</h1>", unsafe_allow_html=True)
def image_app():
    st.markdown("<h1 style='text-align: center;'>Image Face Mask Detection</h1>", unsafe_allow_html=True)
    image="https://images.unsplash.com/photo-1584744982491-665216d95f8b?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    st.markdown(
        f'<div style="text-align:center;">'
        f'<img src="{image}" style="width:400px">'
        '</div>',
        unsafe_allow_html=True
    )
    file=st.file_uploader("Upload Image")
    if file:
        b=file.getvalue()
        d=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(d,cv2.IMREAD_COLOR)
        faces=facemodel.detectMultiScale(img)
        labels_dict={0:'MASK',1:'NO MASK'}
        color_dict={0:(0,255,0),1:(0,0,255)}
        for (x,y,w,h) in faces:
            crop_face1=img[y:y+h,x:x+w]
            cv2.imwrite('temp.jpg',crop_face1)
            crop_face=load_img('temp.jpg',target_size=(150,150,3))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            pred=maskmodel.predict(crop_face)
            pred=maskmodel.predict(crop_face)[0][0]
            if pred==1:
                cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[1],2)
                cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[1],-1)
                cv2.putText(img,labels_dict[1],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h+1),color_dict[0],2)
                cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[0],-1)
                cv2.putText(img,labels_dict[0],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),4)
        st.image(img,channels='BGR',width=300)
   
def video_app():
    st.markdown("<h1 style='text-align: center;'>Video Face Mask Detection</h1>", unsafe_allow_html=True)
    image="https://media.gettyimages.com/id/1349349174/photo/doctor-wearing-surgical-mask-examining.jpg?s=612x612&w=gi&k=20&c=2570X8fWtmBnlQYa3K3ao61j0NuMKdilT8SxnHeGf8A="
    st.markdown(
        f'<div style="text-align:center;">'
        f'<img src="{image}" style="width:400px">'
        '</div>',
        unsafe_allow_html=True
    )
    file=st.file_uploader("Upload Video")
    window=st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
        i=1
        labels_dict={0:'MASK',1:'NO MASK'}
        color_dict={0:(0,255,0),1:(0,0,255)}
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for (x,y,w,h) in faces:
                    crop_face1=frame[y:y+h,x:x+w]
                    cv2.imwrite('temp.jpg',crop_face1)
                    crop_face=load_img('temp.jpg',target_size=(150,150,3))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    if pred==1:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[1],2)
                        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[1],-1)
                        cv2.putText(frame,labels_dict[1],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                        path="data_nomask/"+str(i)+".jpg"
                        cv2.imwrite(path,crop_face1)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+w,y+h+1),color_dict[0],2)
                        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[0],-1)
                        cv2.putText(frame,labels_dict[0],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                window.image(frame,channels='BGR')
def cemara_app():
    st.markdown("<h1 style='text-align: center;'>WebCamera Face Mask Detection</h1>", unsafe_allow_html=True)
    image="https://media.fs.com/images/community/upload/kindEditor/202109/01/working-from-home-with-webcam-1630470747-GrdlR4Cm52.png"
    st.markdown(
        f'<div style="text-align:center;">'
        f'<img src="{image}" style="width:400px">'
        '</div>',
        unsafe_allow_html=True
    )
    labels_dict={0:'MASK',1:'NO MASK'}
    color_dict={0:(0,255,0),1:(0,0,255)}
    window=st.empty()
    #k=st.text_input("Enter 0 to open webcam or write url for opening ip camera")
    start_button = st.button('Start Camera')
    stop_button = st.button('Stop Camera')

    if stop_button:
        st.experimental_rerun()
    if start_button:
        vid=cv2.VideoCapture(0)
        i=1
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for (x,y,w,h) in faces:
                    crop_face1=frame[y:y+h,x:x+w]
                    cv2.imwrite('temp1.jpg',crop_face1)
                    crop_face=load_img('temp1.jpg',target_size=(150,150,3))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    if pred==1:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[1],2)
                        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[1],-1)
                        cv2.putText(frame,labels_dict[1],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                        path="data_nomask/"+str(i)+".jpg"
                        cv2.imwrite(path,crop_face1)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+w,y+h+1),color_dict[0],2)
                        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[0],-1)
                        cv2.putText(frame,labels_dict[0],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                window.image(frame,channels='BGR')
        
def ipcemara_app():
    st.markdown("<h1 style='text-align: center;'>IP Camera Face Mask Detection</h1>", unsafe_allow_html=True)
    image="https://www.advantex.uk.com/wp-content/uploads/2017/09/Facial-Recogonition.png"
    st.markdown(
        f'<div style="text-align:center;">'
        f'<img src="{image}" style="width:400px">'
        '</div>',
        unsafe_allow_html=True
    )
    labels_dict={0:'MASK',1:'NO MASK'}
    color_dict={0:(0,255,0),1:(0,0,255)}
    k=st.text_input("Write URL for opening IP Camera")
    btn=st.button("Start Camera")
    window=st.empty()
    btn2=st.button("stop camera")
    if btn2:
        st.experimental_rerun()
    if btn:
        vid=cv2.VideoCapture(k)
        i=1
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for (x,y,w,h) in faces:
                    crop_face1=frame[y:y+h,x:x+w]
                    cv2.imwrite('temp1.jpg',crop_face1)
                    crop_face=load_img('temp1.jpg',target_size=(150,150,3))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    if pred==1:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[1],2)
                        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[1],-1)
                        cv2.putText(frame,labels_dict[1],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                        path="data_nomask/"+str(i)+".jpg"
                        cv2.imwrite(path,crop_face1)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+w,y+h+1),color_dict[0],2)
                        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[0],-1)
                        cv2.putText(frame,labels_dict[0],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                window.image(frame,channels='BGR')
       

   
