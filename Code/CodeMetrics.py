#-*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:33:37 2017

Implement RF classifier using sciki-learn package

The RF classifer is trained using code metrics as features. Using code metrics as features is used as the baseline to compare with the method which uses transfer-learned representations as features.  

"""

import numpy as np
import pandas as pd
import time
import csv

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

script_start_time = time.time()

working_dir = '/home/your/user/name/CodeMetrics/vlc/'
print ("Script starts at: " + str(script_start_time))


# 1. Import processed data from CSV files.
#-------------------------------------------
def getData(filePath):
    df = pd.read_csv(filePath, header=None, sep=",")
    
    df_list = df.values.tolist()
    
    return np.asarray(df_list)

def GenerateLabels(input_arr):
    temp_arr = []
    for func_id in input_arr:
        temp_sub_arr = []
        if "cve" in func_id or "CVE" in func_id:
            temp_sub_arr.append(1)
        else:
            temp_sub_arr.append(0)
        temp_arr.append(temp_sub_arr)
    return np.asarray(temp_arr)

def storeOuput(arr, path):
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(arr)

train_set_x = getData(working_dir + 'train_vlc_cm.csv')
train_set_id = getData(working_dir + 'train_vlc_id.csv')

test_set_x = getData(working_dir + 'test_vlc_cm.csv')
test_set_id = getData(working_dir + 'test_vlc_id.csv')

train_set_id = np.ndarray.flatten(np.asarray(train_set_id))
test_set_id = np.ndarray.flatten(np.asarray(test_set_id))

train_set_y = GenerateLabels(train_set_id)
test_set_y = GenerateLabels(test_set_id)

train_set_y = np.ndarray.flatten(np.asarray(train_set_y))
test_set_y = np.ndarray.flatten(np.asarray(test_set_y))

print ("Training set: ")
print (train_set_x)

print ("Testing set: ")
print (test_set_x)

print ("The length of training and testing sets: ")

print (len(train_set_x), len(test_set_x), len(train_set_y), len(test_set_y))

print ("-------------------------")

print ("The shape of the datasets: " + "\r\n")

#print (train_set_x.shape, train_set_y.shape, test_set_x.shape, test_set_y.shape)

print (np.count_nonzero(train_set_y), np.count_nonzero(test_set_y))

# 2. Training RF parameters
# -------------------------------------------------------
param_grid = {'max_depth': [3,4,5,6,10,15,20],
              'min_samples_split': [2,3,4,5,6,10],
              'min_samples_leaf': [1,2,3,4,5,6,10],
              'bootstrap': [True,False],
              'criterion': ['gini','entropy'],
              'n_estimators': [10,20,30,40,50,60,70]}

# 3. Start training the RF model
#--------------------------------------------
clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)
clf = clf.fit(train_set_x, train_set_y)

print("best estimator found by grid search:")
print(clf.best_estimator_)

# Output the feature importance.
feature_importances = clf.best_estimator_.feature_importances_

print ("\r\n")

#evaluate the model on the test set
print("predicting on the test set")
#t0 = time()
y_predict = clf.predict(test_set_x)

y_predict_proba = clf.predict_proba(test_set_x)

np.savetxt("y_predict_vlc_cm.csv", y_predict, delimiter=",")
np.savetxt("y_predict_proba_vlc_cm.csv", y_predict_proba, delimiter=",")
storeOuput(test_set_y, "test_label_vlc.csv")
#np.savetxt("./test_label_3.csv", test_set_y, delimiter=",")
#np.savetxt("./feature_importances_short.csv", feature_importances, delimiter=",")

#y_predict_proba = cross_val_predict(clf, )

#print (y_predict_proba)

# Accuracy
accuracy = np.mean(test_set_y==y_predict)*100
print ("accuracy = " +  str(accuracy))
    
target_names = ["Non-vulnerable","Vulnerable"] #non-vulnerable->0, vulnerable->1
print (confusion_matrix(test_set_y, y_predict, labels=[0,1]))   
print ("\r\n")
print ("\r\n")
print (classification_report(test_set_y, y_predict, target_names=target_names))

print ("\r\n")
print ("--- %s seconds ---" + str((time.time() - script_start_time)))