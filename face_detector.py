import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('smile.xml')

#img = cv2.imread('come1.jpg')
webcam = cv2.VideoCapture(0)

while True:
    successful, frame = webcam.read()
    

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x,y,w,h) in face_coordinates:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,250,0),2)

      the_face = frame[y:y+h, x:x+w]

      face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

      smiles = smile_detector.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20)

      for (x_,y_,w_,h_) in smiles:
       cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(50,50,200),2)

       if len(smiles) > 0:
         cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
         fontFace=cv2.FONT_HERSHEY_PLAIN, color=(250, 250, 25))



    
    cv2.imshow('face detector',frame)
    key = cv2.waitKey(1) 
    if key==81 or key==113:
        break 
webcam.release()
print("success")