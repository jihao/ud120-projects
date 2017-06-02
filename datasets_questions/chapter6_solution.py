# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:16:55 2017

@author: hao
"""

i = 0
for k, v in enron_data.items():
    if v["poi"]==1:
        i=i+1;
# 18

# 25. 这三个人（Lay、Skilling 和 Fastow）当中，谁拿回家的钱最多（“total_payments”特征的最大值）？
# 这个人得到了多少钱？        
enron_data['LAY KENNETH L']['total_payments']
# Out[41]: 103559793

enron_data['SKILLING JEFFREY K']['total_payments']
#Out[42]: 8682716

enron_data['FASTOW ANDREW S']['total_payments']
#Out[43]: 2424083
 
    
# 27. 此数据集中有多少雇员有量化的工资？已知的邮箱地址是否可用？    
i=0
for k, v in enron_data.items():
    if v["email_address"]!= 'NaN':
        i=i+1;

#Out[49]: 111

i=0
for k, v in enron_data.items():
    if v["salary"]!= 'NaN':
        i=i+1;
        
#Out[51]: 95    


"""
30. 缺少的 POI 1（可选）
如你刚才所见，不是每个 POI 在数据集中都有一个条目（比如：Michael Krautz）。那是因为数据集是通过你在 final_project/enron61702insiderpay.pdf 中找到的财务数据所创建的，这些数据中缺少了一些 POI（这些缺失的 POI 被传送至最终的数据集）。另一方面，对于这些“缺少的”POI，我们确实有他们的邮件。

尽管向 E+F 数据集中添加这些 POI 和他们的信息，并且为财务信息设置“NaN”非常简单，但这会带来一个微妙的问题。你将在此处了解到这一问题。

（当前的）E+F 数据集中有多少人的薪酬总额被设置了“NaN”？数据集中这些人的比例占多少？
""" 
feature_list = ["poi", "total_payments"]
data_array = featureFormat( enron_data, feature_list )
label, features = targetFeatureSplit(data_array)

1-len(features)/len(enron_data)

1-125/146

"""
缺少的 POI 2（可选）
E+F 数据集中有多少 POI 的薪酬总额被设置了“NaN”？这些 POI 占多少比例？
"""
0

"""
缺少的 POI 4（可选）
如果你再次添加了全是 POI 的 10 个数据点，并且对这些雇员的薪酬总额设置了“NaN”，你刚才计算的数字会发生变化。

数据集中这些人的数量变成了多少？薪酬总额被设置了“NaN”的雇员数变成了多少？
"""
146+10
31

i=0
for k, v in enron_data.items():
    if v["total_payments"]== 'NaN':
        i=i+1;
# i = 21        
# Correct. Now there are 156 folks in dataset, 31 of whom have "NaN" total_payments. This makes for 20% of them with a "NaN" overall.



"""
缺少的 POI 5（可选）
数据集中的 POI 数量变成了多少？股票总值被设置了“NaN”的 POI 占多少比例？
"""
28
10

#Great work! Now there are 28 POI's, 10 of whom have "NaN" for total_payments
# That's 36% of the POI's who have "NaN" for total_payments, a big jump from before.

"""
缺少的 POI 6（可选）
在添加了新的数据点后，你是否认为，监督式分类算法可将 total_payments 为“NaN”理解为某人是 POI 的线索？
"""
