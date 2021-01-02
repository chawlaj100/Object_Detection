import os
import cv2
import numpy as np
from PIL import Image

label_path = "./coco.names"
LABELS = open(label_path).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
cfg_path = "yolov4-tiny.cfg"

yolo_net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Uncomment the below 2 lines if you are having OpenCV source build with GPU
# yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = yolo_net.getLayerNames()
ln = [ln[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

ret, fr = cap.read()
while True:
    if ret:
        ret, fr = cap.read()
        frame = cv2.resize(fr, (960, 720))
        h, w = frame.shape[0], frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        yolo_net.setInput(blob)
        outputs = yolo_net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        if len(idxs) > 0:
            idf = idxs.flatten()

            for i in idf:
                
                x, y = boxes[i][0], boxes[i][1]
                width, height = boxes[i][2], boxes[i][3]
                cv2.rectangle(frame, (x,y), (x+width, y+height), (0, 120, 255), 2)

                confid = confidences[i]
                name_class = LABELS[classIDs[i]]

                cv2.putText(frame, name_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        #cv2.putText(frame, "Press ESC key to Quit.", (x + height + 5, y + (width//2) - (width//3 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Output Frame", frame)
        
        key = cv2.waitKey(20)
        if key == 27:
            break
    else:
        pass

cap.release()
cv2.destroyAllWindows()

