import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata

# import custom module
from mnist_helpers import *

mnist = fetch_mldata('MNIST original', data_home='./')

#minist object contains: data, COL_NAMES, DESCR, target fields
#you can check it by running
mnist.keys()

#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target

#---------------- classification begins -----------------
# scale data for [0,255] -> [0,1]
# rand_idx = np.random.choice(images.shape[0],10000)
# X_data =images[rand_idx]/255.0
# Y      = targets[rand_idx]

X_data = images/255.0
Y = targets

# split data to train and test
# training data size: 6000
# testing data size:1000
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y,
                        test_size=1000, train_size=6000)

# Classifier parameters

# rbf(default) kernel
'''
param_C = 5
param_gamma = 0.01
classifier = svm.SVC(kernel='rbf',C=param_C,gamma=param_gamma)
'''
# poly kernel
param_C = 20
param_N = 1
classifier = svm.SVC(kernel='poly',C=param_C,degree=param_N)

# monitor training time
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(X_train, y_train)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

########################################################

# predict the value of the test
expected = y_test
predicted = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

