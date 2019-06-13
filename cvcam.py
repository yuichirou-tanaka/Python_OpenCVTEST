#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
argv = sys.argv
if len(argv) > 2:
    filepath = argv[2]
else:
    filepath = "sample.jpg"


img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

#cv2.imshow('result', img)
#cv2.imwrite('img.jpg', img)
gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#cv2.imshow('result', gray_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

canny_img = cv2.Canny(gray_img, 200, 400)
#cv2.imshow("result", canny_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT,dp=2, minDist=20, param1=10,param2=40, minRadius=20, maxRadius=40)

print(circles)
import matplotlib.pyplot as plt
#plt.imshow(canny_img)
#plt.show()

cups_circles = np.copy(img)
if circles is not None and len(circles) > 0:
    circles = circles[0]
    for(x,y,r) in circles:
        x,y,r = int(x), int(y), int(r)
        cv2.circle(cups_circles, (x,y),r,(255,255,0),4)
    plt.imshow(cv2.cvtColor(cups_circles, cv2.COLOR_BGR2RGB))
    plt.show()



print('number of circles detected: %d' % len(circles[0]))
