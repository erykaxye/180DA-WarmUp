'''
converting colorspaces: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces 
bounding box: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features 
'''

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame1 = cap.read()
    _, frame2 = cap.read() 

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # define range of black color in HSV
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([0, 0, 50])

    # define range of black color in RGB
    lower_rgb = np.array([0, 0, 0])
    upper_rgb = np.array([50, 50, 50])

    # Threshold the HSV image to get only black colors
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Threshold the RGB image to get only black colors
    mask_rgb = cv2.inRange(rgb, lower_rgb, upper_rgb)

    area = 0
    count = 0

    # Find contours with RGB and display them 
    contours, hierarchy = cv2.findContours(mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Only display rectangles that are bigger than the average rectangle size 
    for cnt in contours: 
        x,y,w,h = cv2.boundingRect(cnt)
        area = area + w*h 
        count = count + 1
    if count > 0: 
        avg_area = area/count 
    else:
        avg_area = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h 
        if area > avg_area: 
            cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)

    # Find contours with HSV and display them 
    contours, hierarchy = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h 
        if area > avg_area: 
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('hsv_thres', frame1)
    cv2.imshow('rgb_thres', frame2)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()