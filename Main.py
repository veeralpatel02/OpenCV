import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, HTMLResponse
import time
import numpy as np
import tempfile
import os
import base64
#from RpiMotorLib import RpiMotorLib
#import RPi.GPIO as GPIO

app = FastAPI()
camera = cv2.VideoCapture(1)
templates = Jinja2Templates(directory="templates")
GpioPins = [18, 23, 24, 25]

# Declare a named instance of class pass a name and motor type
#mymotortest = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")
# min time between motor steps (ie max speed)
step_time = .002

# PID GainValues (these are just starter values)
Kp = 0.003
Kd = 0.0001
Ki = 0.0001

latest_qr_url = "https://example.com/qr_code"

def gen_frames_red():
     while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_red_objects(frame)
            cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Show the resulting frame with green boxes around the red objects
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def gen_frames():
     while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Show the resulting frame with green boxes around the red objects
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def gen_frames_lBlue():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_light_blue_objects(frame)
            cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Show the resulting frame with green boxes around the red objects
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            k = cv2.waitKey(5)
            if k == 27:
                break

import cv2

def gen_frames_qrCode():
    qr_detector = cv2.QRCodeDetector()
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Detect and draw QR code
        qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame)
        if qr_bbox is not None:
            # Draw the bounding box with red color
            qr_bbox = qr_bbox.astype(int)
            cv2.polylines(frame, [qr_bbox], isClosed=True, color=(0, 0, 255), thickness=3)
            if qr_data:
                cv2.putText(frame, "QR Detected", (qr_bbox[0][0][0], qr_bbox[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)
                latest_qr_url = qr_data
                print(latest_qr_url)

        # Normalize frame and encode it to JPEG
        cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame as JPEG bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

def gen_frames_circle():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_circular_objects(frame)
            cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Show the resulting frame with green boxes around the red objects
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            k = cv2.waitKey(5)
            if k == 27:
                break
def gen_frames_face():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_faces(frame)
            cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Show the resulting frame with green boxes around the red objects
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            k = cv2.waitKey(5)
            if k == 27:
                break
def detect_red_objects(frame):
    # convert to hsv deals better with lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red is on the upper and lower end of the HSV scale, requiring 2 ranges
    lower1 = np.array([0, 200, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 200, 100])
    upper2 = np.array([179, 255, 255])

    # masks input image with upper and lower red ranges
    red_only1 = cv2.inRange(hsv, lower1, upper1)
    red_only2 = cv2.inRange(hsv, lower2, upper2)
    red_only = red_only1 + red_only2

    # run an opening to get rid of any noise
    mask = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(red_only, cv2.MORPH_OPEN, mask)

    # run connected components algo to return all objects it sees.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 4, cv2.CV_32S)
    b = np.matrix(labels)

    # error values
    d_error = 0
    last_error = 0
    sum_error = 0
    for i in range(1, num_labels):  # Exclude background label (0)
        if num_labels > 1:
            start = time.time()
            # extracts the label of the largest none background component
            # and displays distance from center and image.
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)], key = lambda x: x[1])
            Obj = b == max_label
            Obj = np.uint8(Obj)
            Obj[Obj > 0] = 255

            # calculate error from center column of masked image
            error = -1 * (320 - centroids[max_label][0])
            # speed gain calculated from PID gain values
            speed = Kp * error + Ki * sum_error + Kd * d_error

            #if negative speed change direction
            if speed < 0:
                direction = False
            else:
                direction = True

            # inverse speed set for multiplying step time
            # (lower step time = faster speed)
            speed_inv = abs(1/(speed))

            # get delta time between loops
            delta_t = time.time() - start
            # calculate derivative error
            d_error = (error - last_error)/delta_t
            # integrated error
            sum_error += (error * delta_t)
            last_error = error

            # # buffer of 20 only runs within 20
            # if abs(error) > 20:
            #     #mymotortest.motor_run(GpioPins, speed_inv * step_time, 1, direction, False, "full", .05)
            # else:
            #     #run 0 steps if within an error of 20
            #     #mymotortest.motor_run(GpioPins, step_time, 0, direction, False, "full", .05)
            
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(frame, 'Red Object', (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        else:
            print('no object in view')

    return frame
def detect_light_blue_objects(frame):
    # convert to hsv deals better with lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # light blue is in the middle of the HSV scale
    lower1 = np.array([90, 50, 50])
    upper1 = np.array([130, 255, 255])

    # masks input image with upper and lower red ranges
    lBlue_only = cv2.inRange(hsv, lower1, upper1)

    # run an opening to get rid of any noise
    mask = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(lBlue_only, cv2.MORPH_OPEN, mask)

    # run connected components algo to return all objects it sees.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 4, cv2.CV_32S)
    b = np.matrix(labels)

    # error values
    d_error = 0
    last_error = 0
    sum_error = 0

    for i in range(1, num_labels):  # Exclude background label (0)
        if num_labels > 1:
            start = time.time()
            # extracts the label of the largest none background component
            # and displays distance from center and image.
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)], key = lambda x: x[1])
            Obj = b == max_label
            Obj = np.uint8(Obj)
            Obj[Obj > 0] = 255

            # calculate error from center column of masked image
            error = -1 * (320 - centroids[max_label][0])
            # speed gain calculated from PID gain values
            speed = Kp * error + Ki * sum_error + Kd * d_error

            #if negative speed change direction
            if speed < 0:
                direction = False
            else:
                direction = True

            # inverse speed set for multiplying step time
            # (lower step time = faster speed)
            speed_inv = abs(1/(speed))

            # get delta time between loops
            delta_t = time.time() - start
            # calculate derivative error
            d_error = (error - last_error)/delta_t
            # integrated error
            sum_error += (error * delta_t)
            last_error = error

            # buffer of 20 only runs within 20
            # if abs(error) > 20:
            #     #mymotortest.motor_run(GpioPins, speed_inv * step_time, 1, direction, False, "full", .05)
            # else:
            #     #run 0 steps if within an error of 20
            #     mymotortest.motor_run(GpioPins, step_time, 0, direction, False, "full", .05)
        else:
            print('no object in view')
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(frame, 'light-blue object', (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    return frame
def detect_circular_objects(frame):
    # greyscale for Hough Circles
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=50)

    # error values
    d_error = 0
    last_error = 0
    sum_error = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Draw green rectangles around each detected circle
        for (x, y, r) in circles:

            start = time.time()

            error = -1 * (320 - x)
            # speed gain calculated from PID gain values
            speed = Kp * error + Ki * sum_error + Kd * d_error

            #if negative speed change direction
            if speed < 0:
                direction = False
            else:
                direction = True

            # inverse speed set for multiplying step time
            # (lower step time = faster speed)
            speed_inv = abs(1/(speed))

            # get delta time between loops
            delta_t = time.time() - start
            # calculate derivative error
            d_error = (error - last_error)/delta_t
            # integrated error
            sum_error += (error * delta_t)
            last_error = error

            # buffer of 20 only runs within 20
            # if abs(error) > 20:
            #     mymotortest.motor_run(GpioPins, speed_inv * step_time, 1, direction, False, "full", .05)
            # else:
            #     #run 0 steps if within an error of 20
            #     mymotortest.motor_run(GpioPins, step_time, 0, direction, False, "full", .05)
        
            cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
            cv2.putText(frame, 'Circle', (x - r - 10, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return frame

def detect_faces(frame):
    # greyscale for Faces
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # for more contrast
    _, thresh = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(grey, 1.1, 4)

    # error values
    d_error = 0
    last_error = 0
    sum_error = 0

    if faces is not None:
        # Draw green rectangles around each detected circle
        for (x, y, w, h) in faces:
            start = time.time()
            # calculate error from center column of masked image
            error = -1 * (320 - x)
            # speed gain calculated from PID gain values
            speed = Kp * error + Ki * sum_error + Kd * d_error

            #if negative speed change direction
            if speed < 0:
                direction = False
            else:
                direction = True

            # inverse speed set for multiplying step time
            # (lower step time = faster speed)
            speed_inv = abs(1/(speed))

            # get delta time between loops
            delta_t = time.time() - start
            # calculate derivative error
            d_error = (error - last_error)/delta_t
            # integrated error
            sum_error += (error * delta_t)
            last_error = error

            # buffer of 20 only runs within 20
            # if abs(error) > 20:
            #     mymotortest.motor_run(GpioPins, speed_inv * step_time, 1, direction, False, "full", .05)
            # else:
            #     #run 0 steps if within an error of 20
            #     mymotortest.motor_run(GpioPins, step_time, 0, direction, False, "full", .05)
           
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

    return frame
def qrcode(frame):
    qr_detect = cv2.QRCodeDetector()
    data, bbox, _ = qr_detect.detectAndDecode(frame)
    
    if bbox is not None:
        bbox = bbox.astype(int)
        cv2.polylines(frame, [bbox], isClosed=True, color=(0, 0, 255), thickness=3)
        if data:
            cv2.putText(frame, "QR Detected", (bbox[0][0][0], bbox[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)
    return frame, data


# Load YOLO model and classes
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def detect_and_save():
    img = cv2.imread('snapshot.jpg')

    # Load names of classes and get random colors
    classes = open('coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    return img


def detect_objects():
    # Read frame from camera
    success, frame = camera.read()
    if not success:
        return None

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    classes = open('coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Draw bounding boxes
    conf_threshold = 0.5
    nms_threshold = 0.4
    h, w = frame.shape[:2]
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_threshold:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    colors = np.random.randint(0, 255, size=(len(classIDs), 3), dtype='uint8')

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)

    return frame


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/video_feed1')
def video_feed1():
    return StreamingResponse(gen_frames_red(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/video_feed_lBlue')
def video_feed_lBlue():
    return StreamingResponse(gen_frames_lBlue(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/video_feed_circle')
def video_feed_circle():
    return StreamingResponse(gen_frames_circle(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/video_feed_qrcode')
def video_feed_qrcode():
        return StreamingResponse(gen_frames_qrCode(), media_type='multipart/x-mixed-replace; boundary=frame')



@app.get('/video_feed_face')
def video_feed_face():
    return StreamingResponse(gen_frames_face(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/detect", response_class=HTMLResponse)
async def detect(request: Request):
    detected_frame = detect_objects()

    _, img_encoded = cv2.imencode('.jpg', detected_frame)
    img_base64 = base64.b64encode(img_encoded).decode()

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "img_base64": img_base64}
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
