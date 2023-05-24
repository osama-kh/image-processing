
"""
auther:
mohammed abbass
osama khwaled
"""


import os
import sys
import numpy as np
from skimage import feature
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV

kernel_models = [[1, 8, ['linear','rbf']],  [3, 24 ,['linear','rbf']]]

testData = []
validData = []
trainData = []
total_time = None
start_time = None


# PART I Loading dataset
def Load_Dataset(train, val, test):
    for gender in os.listdir(train):
        for img in os.listdir(train+"/"+gender):
            trainData.append([img, gender])

    for gender in os.listdir(val):
        for img in os.listdir(val + "/" + gender):
            validData.append([img, gender])

    for gender in os.listdir(test):
        for img in os.listdir(test + "/" + gender):
            testData.append([img, gender])



def GridSearch(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

    # create a grid search classifier
    clf_grid = GridSearchCV(svm.SVC(kernel="rbf"), param_grid, verbose=1)

    # Train the classifier
    clf_grid.fit(X_train, y_train)

    # get the parameters
    best_params = clf_grid.best_params_


    return best_params


# get labels of images
def Get_Lables_Of_Images(imageData, flag):
    images = []
    labels = []
    dir=""
    if flag == 1:
        dir = TRAIN
    elif flag == 0:
        dir = TEST
    else :
        dir = VALIDATION

    for i in imageData:
        img = cv2.imread(dir + "/" +  i[1] + "/" + i[0])
        images.append(img)
        labels.append(i[1])
    return images, labels


# PART II - Feature extraction using  LBP
def LBP_Features(images, radius, pointNum):
    list_LBP = []
    for img in images:
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if type(pointNum) == int and type(radius) == int:
            lbp = feature.local_binary_pattern(gray, int(pointNum), int(radius), method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=range(0, pointNum + 3), range=(0, pointNum + 2))
            eps = 1e-7
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            list_LBP.append(hist)
    return list_LBP


def Data_Extraction(radius, points):
    # training data
    train_image, train_label = Get_Lables_Of_Images(trainData, 1)

    # Validation data
    valid_image, valid_label = Get_Lables_Of_Images(validData, 0)

    # Feature Extraction using LBP
    train_features = LBP_Features(train_image, radius, points)
    valid_features = LBP_Features(valid_image, radius, points)

    return train_image, train_label,valid_image, valid_label,train_features,valid_features

# PART III - training image
def Training_Image(kernel):

    # extracting kernel data
    radius, points, ker = kernel

    # all the data that we need
    train_image, train_label, valid_image, valid_label, train_features, valid_features = Data_Extraction(radius, points)

    #rbf
    params = GridSearch(train_features, train_label)
    model = SVC(kernel=ker[1], C=params['C'], gamma=params['gamma'])
    model.fit(train_features, train_label)
    model_predictions = model.predict(valid_features)
    rbf_accuracy = accuracy_score(valid_label, model_predictions)
    rbf_model = model

    # model train:
    model = SVC(kernel = ker[0])
    model.fit(train_features, train_label)
    # validation on
    model_predictions = model.predict(valid_features)
    # calculating model accuracy
    linear_accuracy = accuracy_score(valid_label, model_predictions)
    linear_model = model
    if(linear_accuracy>rbf_accuracy):
        return linear_accuracy, ker[0], linear_model
    else:
        return rbf_accuracy, ker[1], rbf_model

def Writing_Results(best_E, accuracy, CM, best_points, best_rad):
    f = open("results.txt", "w")
    ker = best_E
    f.write("model parameters :\nkernel= {0}, Number of points= {1}, Radius= {2} \n".format(ker, best_points, best_rad))
    f.write("Accuracy : {:.2f}%".format(accuracy * 100))
    f.write('\nConfusion matrix:')
    f.write(' ' 'male' + '  ' + 'female')
    f.write('\n            male' + '   ' + str(CM[0][0]) + '    ' + str(CM[0][1]))
    f.write('\n            female' + '  '+ str(CM[1][0]) + '    ' + str(CM[1][1]))
    f.close()

# main
if __name__ == "__main__":
    best_kernel = None
    Best_acc = -1
    Best_model = None
    best_rad = None
    best_points = None
    test_features = []

    #get the arguments of paths from the shell code
    if Load_Dataset(sys.argv[1], sys.argv[2], sys.argv[3]) != 0:
        TRAIN = sys.argv[1]
        VALIDATION = sys.argv[3]
        TEST = sys.argv[2]

        for i in kernel_models:
            accuracy, kernel, model = Training_Image(i)
            #get the best accuracy
            if accuracy > Best_acc:
                Best_acc = accuracy
                best_kernel = kernel
                best_rad = i[1]
                best_points = i[0]
                Best_model = model

        test_images, test_labels = Get_Lables_Of_Images(testData, 3)
        test_features = LBP_Features(test_images, best_points, best_rad)
        model_prediction = Best_model.predict(test_features)
        accuracy = accuracy_score(test_labels, model_prediction)
        model_confusion_matrix = confusion_matrix(test_labels, model_prediction)
        Writing_Results(best_kernel, Best_acc, model_confusion_matrix, best_points, best_rad)