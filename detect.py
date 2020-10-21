import numpy as np
import argparse
import cv2
import winsound

cap = cv2.VideoCapture(1)

fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧速率
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频尺寸
videoWrite = cv2.VideoWriter(r"out.avi",
                             cv2.VideoWriter_fourcc("I", "4", "2", "0"), fps, size)

#接下来，初始化类的标签和包围框的颜色：
CLASSES = ["background", "knife","people"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")

num=0
# 加载mobilenet-SSD目标检测模型
cvNet = cv2.dnn.readNetFromTensorflow(
	'frozen_inference_graph_3000.pb', 
	'graph_peo_kni_30000.pbtxt')
# 加载人脸检测模型
detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while(True):
    ret,img = cap.read()
    flag_d = False
    flag_knife = False
    flag_people = False
    # print(ret)
    num+=1
    print('第',num,'帧')
    # if num == 150:
    #     break
    if ret is False:
        break
    # img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(400, 400), swapRB=True, crop=False))
    detections = cvNet.forward()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2] #置信度
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.85:
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
            idx = int(detections[0, 0, i, 1]) #取出标签下表
            print('idx',idx)
            if idx==1:
                flag_knife = True
            if idx==2:
                flag_people = True
            box = detections[0, 0, i, 3:7] * np.array([cols, rows, cols, rows])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            # 显示目标检测框
            cv2.rectangle(img, (startX, startY), (endX, endY),COLORS[idx], 2)
            # 显示label信息
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            ##报警
            duration = 500  # millisecond
            freq = 1040  # Hz
            print('Beep')
            winsound.Beep(freq, duration)

    #检测人脸
    face_flag = False
    rects = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2, minSize=(10,10), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in rects:
        # 画矩形框
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
        face_flag = True


    flag_d = False
    data_len = len(rects)
    #print('people',flag_people)
    #print('knife',flag_knife)
    #print('face',data_len>1)
    if flag_knife==True and face_flag==True:
        flag_d = True
        ##报警
        duration = 1000  # millisecond
        freq = 3040  # Hz
        print('Beep')
        winsound.Beep(freq, duration)

        print('已捕捉嫌疑犯！！！')
        cv2.putText(img, 'capturing suspect...', (350, 20),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
    # 显示画面
    cv2.resizeWindow("resized", 300, 960)
    cv2.imshow('Detect System ',img)
    if flag_d==True and num%10==1:
        image_path = './tmp_images/image{}.jpg'.format(num)
        # print(image_path)
        cv2.imwrite(image_path,img)
    
    videoWrite.write(img)
    info = cap.get(2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# show the output image
cap.release()
videoWrite.release()
cv2.destroyAllWindows()
