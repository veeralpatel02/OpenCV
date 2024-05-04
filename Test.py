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
from RpiMotorLib import RpiMotorLib
import RPi.GPIO as GPIO


app = FastAPI()
camera = cv2.VideoCapture(0)
templates = Jinja2Templates(directory="templates")


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

       # cap = cv2.VideoCapture(0)
    # Stepper Motor Setup
    GpioPins = [18, 23, 24, 25]

    # Declare a named instance of class pass a name and motor type
    mymotortest = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")
    # min time between motor steps (ie max speed)
    step_time = .002

    # PID GainValues (these are just starter values)
    Kp = 0.003
    Kd = 0.0001
    Ki = 0.0001

    # error values
    d_error = 0
    last_error = 0
    sum_error = 0

    # run an opening to get rid of any noise
    mask = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(red_only, cv2.MORPH_OPEN, mask)

    # run connected components algo to return all objects it sees.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 4, cv2.CV_32S)
    b = np.matrix(labels)

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
            if abs(error) > 20:
                mymotortest.motor_run(GpioPins, speed_inv * step_time, 1, direction, False, "full", .05)
            else:
                #run 0 steps if within an error of 20
                mymotortest.motor_run(GpioPins, step_time, 0, direction, False, "full", .05)
            
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

@app.get('/')
def index(request: Request):
   return templates.TemplateResponse("indexC1.html", {"request": request})



@app.get('/video_feed1')
def video_feed1():
   return StreamingResponse(gen_frames_red(),
                            media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=8000)



