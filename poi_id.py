#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

labels = ['poi']
financ_ob = ['salary','total_payments']
bonus = ['bonus','expenses','other']
incentive = ["long_term_incentive"]
stock = ['restricted_stock','exercised_stock_options','total_stock_value']
novas = ['frac_messages_to_poi','frac_messages_from_poi']


features_list_old = financ_ob + stock + bonus
features_list = labels + financ_ob + bonus + stock + novas + incentive
features_list2 = labels + financ_ob + bonus + stock + incentive
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)



'''
### PLOT to indentific outliers
for j in features_list_old:
		for i in data_dict:	
			if data_dict[i][j] != "NaN":
				plt.scatter(data_dict[i][j],data_dict[i]["poi"])
				plt.xlabel(j)
				plt.ylabel("poi")
		plt.ylabel("POI")
		plt.xlabel(j)
		name = "POI por" + j + ".png"
		plt.savefig(name)
'''

### Task 2: Remove outliers


# Removendo outliers baseado nos gráficos
OUT_1 = []




for i in data_dict:
	if data_dict[i]["total_payments"] != "NaN":
		if data_dict[i]["total_payments"] > 100000000:
			OUT_1.append(i)

	if data_dict[i]["salary"] != "NaN":
		if data_dict[i]["salary"] > 20000000:
			OUT_1.append(i)
	
	if data_dict[i]["total_stock_value"] != "NaN":
		if data_dict[i]["total_stock_value"] > 300000000:
			OUT_1.append(i)

	if data_dict[i]['bonus'] != "NaN":
		if data_dict[i]["bonus"] > 20000000:
			OUT_1.append(i)

#Removendo pontos sem dados financeiros considerados obrigatórios
for i in data_dict:
	for j in financ_ob:
		if data_dict[i][j] == "NaN":
			OUT_1.append(i)	
#Removendo
for i in OUT_1:
	data_dict.pop(i,0)


### Task 3: Create new feature(s)

#Resto da financia
def dict_value(total,value):
	new_list = []
	for i in data_dict:
		if data_dict[i]["bonus"] != "NaN":
			new_list.append(data_dict[i][total] - data_dict[i][value] - data_dict[i]["bonus"])
		else:
			new_list.append(data_dict[i][total] - data_dict[i][value])
	return new_list

#Emails recebidos por poi e enviados para poi
def dict_fra(email,total):
	new_list = []
	for i in data_dict:
		if data_dict[i][email] == "NaN" or data_dict[i][total] == "NaN":
			new_list.append(0)
		if data_dict[i][email] != "NaN" or data_dict[i][total] != "NaN":
			new_list.append(float(data_dict[i][email]/data_dict[i][total]))
	return new_list



#other_pay = dict_value("total_payments","salary")
fract_to_poi = dict_fra("from_this_person_to_poi","from_messages")
fract_from_poi = dict_fra("from_this_person_to_poi","to_messages")

#
poi_y = 0
poi_n = 0
for i in data_dict:
	if data_dict[i]["poi"] == True:
		poi_y = poi_y + 1
	else:
		poi_n = poi_n + 1



#Criando uma copia 
data_dict2 = data_dict.copy()
### Store to my_dataset for easy export below.


count = 0
for i in data_dict:
	#data_dict[i]["other_pay"] = other_pay[count]
	data_dict[i]["frac_messages_to_poi"] = fract_to_poi[count]
	data_dict[i]["frac_messages_from_poi"] = fract_from_poi[count]
	count = count + 1


my_dataset = data_dict




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


#Para usar LogistcRegression, os atributos devem ser escalonados para que para
#que o código não valorize um atributo em detrimento do outro
skb = SelectKBest(f_classif)
scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)


clf_L = LogisticRegression()
pipeline = Pipeline(steps = [("SKB", skb), ("Logi",clf_L)])
param_grid_L = {"SKB__k":[3,4,5,6,7,8,9,10,11],"Logi__penalty": ['l1','l2'], "Logi__tol":[0.001,0.01,0.1,1], "Logi__C":[10**1,10**4,10**8,10**12],"Logi__class_weight":['balanced']  }
grid_k = GridSearchCV(pipeline, param_grid_L)

'''
clf_gnb = GaussianNB()
pipeline = Pipeline(steps = [("SKB", skb), ("NaiveBayes",clf_gnb)])
param_grid = {"SKB__k":[3,4,5,6,7,8]}
grid_k = GridSearchCV(pipeline,param_grid)
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

grid_k.fit(features, labels)
clf = grid_k.best_estimator_
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
