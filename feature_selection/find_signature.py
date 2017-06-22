#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"),fix_imports=True)
authors = pickle.load( open(authors_file, "rb"),fix_imports=True )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()

feature_names= vectorizer.get_feature_names()  

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")
acc = accuracy_score(pred, labels_test)
print (acc) 

# How many training points are there, according to the starter code?
print(len(features_train))

# What’s the importance of the most important feature? What is the number of this feature?
importances = clf.feature_importances_
for index, item in enumerate(importances):
    if item > 0.2:        
        print (index, item)       
       
import numpy as np
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(10):
    print ("{} feature no.{} ({})".format(i+1,indices[i],importances[indices[i]]))

# What’s the most powerful word when your decision tree is makeing its classification decisions?
feature_name = vectorizer.get_feature_names()
for index, item in enumerate(feature_name):
    if index == 33614:        
        print (item)

print(vectorizer.get_feature_names()[33614])
# Result: sshacklensf      
        
feature_name = vectorizer.get_feature_names()
for index, item in enumerate(feature_name):
    if index == 14343:        
        print (item) 
        
print(vectorizer.get_feature_names()[14343])
# Result: cgermannsf

