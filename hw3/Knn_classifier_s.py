import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
#hhd_dataset//hhd_dataset
thisdir = ''
imagesList = []
labels = []
dirs = []
listofletter=[]

# processing the image and to convert it to data
def pre_processing(thisdir):
    for dir in range(27):
        letter=[]
        for img in os.listdir(thisdir + '/' + str(dir)):
            pathImage = thisdir + '/' + str(dir) + '/' + img
            image = cv2.imread(pathImage)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            (height, width) = grayscale.shape
            if height > width:
                padding_size = (height - width) // 2
                padded_image = cv2.copyMakeBorder(grayscale, top=0, bottom=0, left=padding_size, right=padding_size,
                                                  borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            elif width > height:
                padding_size = (width - height) // 2
                padded_image = cv2.copyMakeBorder(grayscale, top=padding_size, bottom=padding_size, left=0, right=0,
                                                  borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            else:
                padded_image = grayscale
            # print(height, '------', width)

            image = cv2.resize(padded_image, (32, 32))
            # th, image = cv2.threshold(image, 150, 200, cv2.THRESH_OTSU)
            image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                         transform_sqrt=False, block_norm="L2")
            imagesList.append(image)
            letter.append(image)

            labels.append(int(dir))
        listofletter.append(letter)
        dirs.append(int(dir))

#to get put the model on test and print the Accuracy of all letters
def testing(maxK,model):
    y_test = []
    pred_y = []

    with open("results.txt", "w") as file1:
        file1.write("Max Accuracy take from model k = {0} \n".format(str(maxK)))
        file1.write("Class Accuracy\n")

        for folder_name in range(27):
            for img_name in listofletter[folder_name]:
                y_test.append(int(folder_name))


            y_true = [folder_name]*len(listofletter[folder_name])

            pred_per_class = model.predict(np.asarray(listofletter[folder_name]))
            acc = accuracy_score(np.asarray(y_true), pred_per_class)
            pred_y.append(pred_per_class)

            file1.write("{0}     {1:.2f}%\n".format(folder_name, acc * 100))
    return np.asarray(y_test), np.asarray(pred_y, dtype=object)




if __name__ == "__main__":
    # check if the input is correct , if not print suitable sentence
    if len(sys.argv) < 2:
        print("Error, messing the images directory!")
    else:
        thisdir = sys.argv[1]
        pre_processing(thisdir)
        x = np.asarray(imagesList)
        x = x.reshape(x.shape[0], -1)
        #to divide to training/validation/testing
        X_train, X, y_train, Y = train_test_split(x, labels, test_size=0.2)
        X_val, X_test, y_val, y_test = train_test_split(X, Y, test_size=0.5)

        listAcc = []
        indexacc = []
        for i in range(1, 16, 2):
            neigh = KNeighborsClassifier(n_neighbors=i, n_jobs=-1).fit(X_train, y_train)
            acc = neigh.score(X_val, y_val)
            listAcc.append(acc * 100)
            indexacc.append(i)
            print("Accuracy of model at K=", i, "is", 100 * acc)
        maxAcc = max(listAcc)
        maxK = listAcc.index(maxAcc)
        maxK = indexacc[maxK]
        print("Max Accuracy is", maxAcc, 'in k = ', maxK)
        neigh = KNeighborsClassifier(n_neighbors=maxK, n_jobs=-1).fit(x, labels)

        y_true, pred_y = testing(maxK, neigh)
        print(" Writing confusion_matrix to CSV file")
        pred_y = np.concatenate(pred_y)
        conf_mat = confusion_matrix(y_true, np.reshape(pred_y, (len(y_true),)))
        df = pd.DataFrame(conf_mat)
        df.to_csv("confusion_matrix.csv")


