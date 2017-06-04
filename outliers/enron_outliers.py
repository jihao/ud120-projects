#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_unix.pkl", "rb"), fix_imports = True )
data_dict.pop( "TOTAL", 0 )

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
# 去除 TOTAL 之后的数据点，是poi数据，不是要去除的异常值

max_salary = 0
for point in data:
    salary = point[0]
    if salary > max_salary:
        max_salary = salary
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()