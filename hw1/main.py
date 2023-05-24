import os
import sys
import cv2 as cv
import numpy as np


def RemoveGreen(img,bkimg):
    img1 = cv.imread(img)# reading the image from path that received from the function arguments
    img2 = cv.imread(bkimg) # reading the background image from path that received from the function arguments
    if img2.shape != img1.shape: #if the of background and image not same shape then maked it same
        img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    hvs_img = cv.cvtColor(img1, cv.COLOR_BGR2HSV)  # changing the color of the image from BGR model to HSV model
    lower_green = np.array([36, 50, 70]) # getting the lower level of green color in HSV model
    upper_green =  np.array([89, 255, 255])  # getting the highest level of green color in HSV model
    mask = cv.inRange(hvs_img, lower_green, upper_green)  # creating the mask of the green color from the image
    img1_bg = cv.bitwise_and(img1, img1, mask=mask) # using the mask that created and bitwise
    final = img1 - img1_bg
    final = np.where(final == 0 , img2, final) # merging the two images ( f and HSV img2 ) to get the final result

    return final


if __name__ == "__main__":
    inputIMPath = sys.argv[1] # getting the input path from the command line
    inputBKPath = sys.argv[2]# getting the input of bk path from the command line
    outputPath = sys.argv[3] # getting the output path from the command line


    if os.path.exists(inputIMPath) and os.path.exists(inputBKPath) : # checking if the input string is an exist path
        fixed_image = RemoveGreen(inputIMPath,inputBKPath)  # receiving the image after removing the green color and replace to background image
        cv.imwrite(outputPath, fixed_image)
    else:
        print("input files is not valid ")
