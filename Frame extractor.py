import numpy as np
import cv2 as cv
from AverageDelay import init_data
import os
from myLib import IoU
# Your 2D list array

playback_start = 1


video = "Videos/V_Dark.mp4"
cap = cv.VideoCapture(video)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

frame_no = playback_start
cap.set(cv.CAP_PROP_POS_FRAMES, playback_start)


instances=[]
play=True
while True:

    ret, frame = cap.read()
    cv.imshow("hmm",frame)
    if not ret:
        break

    while True:
        if play==False:
            key = cv.waitKey(0)
        else:
            key =cv.waitKey(1)

        if key == ord('w'):
            play = not play

        if key == ord('d'):
            break

        if key == ord('s'):

            #cv.destroyAllWindows()  # Close the video window before user input
            print(f"\n frame number:{frame_no }")
            cv.imwrite(f"Dataset/Additional Data/50 - Video/{frame_no}.jpg",frame)
            break
        if play==True:
            break

    # if cv.waitKey(30) == 27:
    #     cv.destroyAllWindows()
    frame_no+=1






cap.release()
cv.destroyAllWindows()



