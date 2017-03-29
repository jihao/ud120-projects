#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess(words_file = "../tools/word_data_unix.pkl", authors_file="../tools/email_authors_unix.pkl")




#########################################################
### your code goes here ###

#########################################################
from sklearn import svm
clf = svm.SVC(kernel='rbf',C=10000,gamma='auto')
#clf = svm.SVC(kernel='linear')

#features_train = features_train[:int(len(features_train)/100)] 
#labels_train = labels_train[:int(len(labels_train)/100)] 

t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
# 139.559s

pred = clf.predict(features_test)
print(pred)
print(sum(pred))
print("predict 10:",pred[10])
print("predict 26:",pred[26])
print("predict 50:",pred[50])

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print(acc)
