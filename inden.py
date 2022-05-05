import cv2
import numpy as np
# Installing Yolo Yu


cap = cv2.VideoCapture(0)

while (True):
    frame = cap.read()
    img = frame[1]

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    outputlayers = net.getUnconnectedOutLayersNames()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # Calling Our Picture

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    img = cv2.resize(img, None, fx=2, fy=2)
    height, width, channels = img.shape
    # Detecting objects

    blob = cv2.dnn.blobFromImage(img, 0.00392, (800, 800), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    # We Display Information
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    cv2.imshow('Video', img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

