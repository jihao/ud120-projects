#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys_unix.pkl')
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
clf.fit(features, labels)
pred = clf.predict(features)
acc = accuracy_score(pred, labels) 
print(acc)
# 0.989473684211
# 
from sklearn.model_selection import train_test_split
# 使用 sklearn.cross_validation 中的 train_test_split 
# 验证； 将 30% 的数据用于测试，并设置 random_state 参数为 42（random_state 控制哪些点进入训练集，哪些点用于测试；
# 将其设置为 42 意味着我们确切地知道哪些事件在哪个集中； 并且可以检查你得到的结果）。更新后的准确率是多少？
features_train, features_test, labels_train, labels_test  = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
acc = clf.score(features_test,labels_test)
print(acc)