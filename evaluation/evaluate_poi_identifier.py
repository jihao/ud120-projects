#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys_unix.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf_overfit = DecisionTreeClassifier()
clf_overfit.fit(features, labels)
pred1 = clf_overfit.predict(features)
acc1 = accuracy_score(pred1, labels) 
print(acc1)
# 0.989473684211
# 
from sklearn.model_selection import train_test_split
# 使用 sklearn.cross_validation 中的 train_test_split 
# 验证； 将 30% 的数据用于测试，并设置 random_state 参数为 42（random_state 控制哪些点进入训练集，哪些点用于测试；
# 将其设置为 42 意味着我们确切地知道哪些事件在哪个集中； 并且可以检查你得到的结果）。更新后的准确率是多少？
features_train, features_test, labels_train, labels_test  = train_test_split(features, labels, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = clf.score(features_test,labels_test)
print(acc)

# 把你的过拟合模型的预测与真实测试标签比较，你得到 true positive 了吗？（在此情况下，我们定义的 true positive 中实际标签和预测标签均为 1）
# many ?

from sklearn.metrics import precision_score,recall_score
print(precision_score(labels_test, pred))

# 召回率是多少？ 
#（注意：你可能看到过类似于“用户警告：一些标签的精确率和召回率等于零”的消息。 就像其中所显示的，当精确率和/或召回率为零时，计算其他指标（比如 F1 分数）可能会出现问题，而且在问题发生时，警告消息会显示出来。） 
# 显然，这并不是一个优化得非常好的机器学习策略（我们没有尝试过决策树以外的任何算法，或调整过任何参数，也没有进行过任何特征选择）， 现在看来，精确率和召回率要比准确率更直观。
print(recall_score(labels_test, pred))


#此处为一些编造的预测值和假设的测试集的真标签；在以下方框中填空，练习识别 true positive、false positive、true negative 和 false negative。 让我们按照惯例，使用“1”表示正结果，“0”表示负结果。

import numpy as np
预测值 =   [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
真实标签 = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# 有多少 true positive？
# 6

# 此示例中有多少 true negative？
# 9


# 此示例中有多少 false positives？
# 3

# 此示例中有多少 false negatives？
# 2

# 这个分类器的精确率是多少？
precision_s = 6/(6+3) # true positive / true positive + false positive
recall_s = 6/(6+2)    # true positive / true positive + false nagative

# 练习: 理解指标 1
# 我的 True Positive 率很高，这意味着当测试数据中有_POI_时，我能很容易地将他或她标记出来。
# □ POI
# □ 非 POI

