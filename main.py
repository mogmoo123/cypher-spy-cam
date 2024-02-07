import cv2

##Read Video

filename = None

if filename == None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(filename)
img = cv2.imread("img/camera.png",cv2.IMREAD_UNCHANGED)

##Harr Cascade Classifier

bd = cv2.CascadeClassifier('ai files\haarcascade_fullbody.xml')
fc = cv2.CascadeClassifier('ai files\haarcascade_frontalface_default.xml')

## VIdeo
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    size = (int(width),int(height))
    a = 1

    while True:
        ret, frame = cap.read()
        scaleFactor = 1.6
        minNeighbors = 3
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        bodys = bd.detectMultiScale(gray,scaleFactor,minNeighbors)
        faces = fc.detectMultiScale(gray,scaleFactor,minNeighbors)

        number_body = 0
        for (x,y,w,h) in bodys:
            number_body +=1
            cv2.rectangle(frame, (x,y),(x+w,y+h), color=(240,150,79), thickness=2)
            cv2.putText(frame,f"body {number_body}",(x+w,y),cv2.FONT_HERSHEY_PLAIN,fontScale=1.5,color=(240,150,79),thickness= 1)

        face_number = 0
        for (fx,fy,fw,fh) in faces:
            face_number+=1
            cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), color = (255,255,255),thickness=2)
            cv2.putText(frame,f"face {face_number}",(fx+fw,fy),cv2.FONT_HERSHEY_PLAIN,fontScale=1.5,color=(255,255,255),thickness= 1)

        if ret:
            frame = cv2.resize(frame,(int(1920/a),int(1080/a)))
            img = cv2.resize(img,(int(1920/a),int(1080/a)))
            frame = cv2.bitwise_or(frame,img)
            cv2.imshow('camera',frame)

            if cv2.waitKey(int(120/fps)) != -1:
                break

        else:
            print("no frame")
            break
    
else:
    print("can't open camera!")

cap.release()
cv2.destroyAllWindows()
