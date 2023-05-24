import numpy as np
import cv2
import sys
import os


# Getting the args from the command line
args=sys.argv
path_input_img=args[1]
path_output_img=args[2]



image = cv2.imread(path_input_img)
image = cv2.resize(image,(600,800))

orginal=image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


max_cont=contours[0]
for i in contours :
  if cv2.contourArea(max_cont)<cv2.contourArea(i):
     max_cont=i

cv2.drawContours(image, max_cont, -1,  (0, 255, 0), 3) #sharpens the frame in green
peri = cv2.arcLength(max_cont, True)
approx = cv2.approxPolyDP(max_cont, 0.05 * peri, True)

rows, cols = image.shape[:2]

pts1 = np.float32(approx)
pts2 = np.float32([[0, 0], [0, rows], [cols, rows],[cols,0]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(orginal, M, (cols, rows))

dst = cv2.resize(dst,(rows,cols))
cv2.imshow('image',dst)
cv2.waitKey(0)
cv2.imwrite(path_output_img,dst)
