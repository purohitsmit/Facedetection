import cv2
import sys
import copy
import logging as log
import datetime as dt
from time import sleep

cascPath = ["haarcascade_eye.xml","haarcascade_frontalcatface.xml"]
eyeCascade = cv2.CascadeClassifier(cascPath[0])
faceCascade = cv2.CascadeClassifier(cascPath[1])
log.basicConfig(filename='webcam.log',level=log.INFO)


video_capture = cv2.VideoCapture(0)

'''https://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/'''
# OS: Windows 10 Home 64 bits
# PL: Python 2.7.12
# Lib: OpenCV 3.1
# Player: VLC Player
# MJPG&avi is very well combination.
# I had tried XVID&avi / MJPG/mpg all get some trouble.
# and M$ media player with Win7codecs can’t play the video.
# So I turn to VLC Player.

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # cv2.VideoWriter_fourcc() does not exist
video_writer = cv2.VideoWriter("output.mp4", fourcc , 20.0 , (680, 480))
anterior = 0
a=0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    temp = copy.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eyeCascade.detectMultiScale( 
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(str(a)+'.jpg', temp)
        a+=1

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)
    # if ret:
    #     video_writer.write(temp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()