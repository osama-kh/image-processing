## authors` : mohammed abbas,osama khwaled
### The purpose of the project is to classify the images by gender (man's handwriting or woman's handwriting)
### For this you will need to train an SVM model, perform experiments with different parameters and kernels, and report which
### A combination of parameters achieves the highest accuracy.
### ******requirements before running the code******
##### OpenCv install by `pip install opencv-contrib-python`
##### Scikit-learn install by pip install scikit-learn
##### Scikit-image install by pip install scikit-image
##### Python 3.11 and older [https://www.python.org/downloads/](https://www.python.org/downloads/)

## How to run
   ##### run by using the CMD where main.py located 
    python [Code Name e.g. main.py]  [Path of 0_HHD of train] [Path of 0_HHD of val] [Path of 0_HHD of test]
   ##### 1) run by absolute path 
    python knn_classifier.py C:\Users\MOHAMMED\PycharmProjects\image_processing\finalProject\HHD_gender\train C:\Users\MOHAMMED\PycharmProjects\image_processing\finalProject\HHD_gender\val C:\Users\MOHAMMED\PycharmProjects\image_processing\finalProject\HHD_gender\test
   ##### 2) run by local file that located in same path of main.py 
    python knn_classifier.py HHD_gender/train HHD_gender/val HHD_gender/test


### after running the program can see the results in results.txt
#### all rights reserved to  Dr. Irina Rabaev 


