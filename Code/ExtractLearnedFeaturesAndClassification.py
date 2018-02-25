# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 16:35:39 2017

This file has two functions:
    1. load the trained LSTM network for obtaining the function representations as features.
    2. train a random forest classifier using obtained features.

"""

import time
import numpy as np
import pandas as pd
import os
import csv
import pickle

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing import sequence
from keras import backend as K

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

script_start_time = time.time()

print ("Script starts at: " + str(script_start_time))

# ------------------------------------------------------------ #
# Parameters used
MAX_LEN = 1000 # The Padding Length for each sample.
EMBEDDING_DIM = 100 # The Embedding Dimension for each element within the sequence of a data sample. 

#--------------------------------------------------------#
# 1. Directories of all the needed files.

project_name = "FFmpeg"

working_dir = '/home/your/user/name/TransferRepresentationLearning/ffmpeg/'

w2v_dir = '/home/your/user/name/TransferRepresentationLearning/word2vec/'

model_saved_path = '/home/your/user/name/TransferRepresentationLearning/models/'

#--------------------------------------------------------#
# 2. Load the saved model and compile it.

# The path where the trained models are saved.
model = load_model(model_saved_path + '1st_1000_100_32_90_test_on_ffmpeg.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print (model.summary())

#--------------------------------------------------------#
# 3. Load the data

def LoadSavedData(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

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

training_list = LoadSavedData(working_dir + 'except_ffmpeg_list.pkl')
training_list_id = LoadSavedData(working_dir + 'except_ffmpeg_list_id.pkl')

testing_list = LoadSavedData(working_dir + 'ffmpeg_list.pkl')
testing_list_id = LoadSavedData(working_dir + 'ffmpeg_list_id.pkl')

print ("The number of training functions: " + str(len(training_list)) + "  ID: " + str(len(training_list_id)))
print ("The number of testing functions: " + str(len(testing_list)) + "  ID: " + str(len(testing_list_id)))

#------------------------------------#
# 2. Load pre-trained word2vec and tokens
    
def JoinSubLists(list_to_join):
    new_list = []
    
    for sub_list_token in list_to_join:
        new_line = ','.join(sub_list_token)
        new_list.append(new_line)
    return new_list

new_training_list = JoinSubLists(training_list)
new_testing_list = JoinSubLists(testing_list)

tokenizer = LoadSavedData(w2v_dir + 'tokenizer.pickle')
train_sequences = tokenizer.texts_to_sequences(new_training_list)
test_sequences = tokenizer.texts_to_sequences(new_testing_list)
word_index = tokenizer.word_index
print ('Found %s unique tokens.' % len(word_index))

print ("The length of tokenized sequence: " + str(len(train_sequences)))
print ("The length of tokenized sequence: " + str(len(test_sequences)))

# Load the pre-trained embeddings.
w2v_model_path = w2v_dir + '6_projects_w2v_model_CBOW.txt'
w2v_model = open(w2v_model_path, encoding="latin1")

print ("----------------------------------------")
print ("The trained word2vec model: ")
print (w2v_model)

#------------------------------------#
# 3. Do the paddings.
print ("max_len ", MAX_LEN)
print('Pad sequences (samples x time)')

train_sequences_pad = pad_sequences(train_sequences, maxlen = MAX_LEN, padding ='post')
test_sequences_pad = pad_sequences(test_sequences, maxlen = MAX_LEN, padding ='post')

train_set_x = train_sequences_pad
test_set_x = test_sequences_pad

train_set_y = GenerateLabels(training_list_id)
test_set_y = GenerateLabels(testing_list_id)

print (len(train_set_x), len(train_set_y), len(test_set_x), len(test_set_y))

print ("-------------------------")

print ("The shape of the datasets: " + "\r\n")

print (train_set_x.shape, train_set_y.shape, test_set_x.shape, test_set_y.shape)

print (np.count_nonzero(train_set_y), np.count_nonzero(test_set_y))
        
# ------------------------------------------------------------ #
# 4. Get the activations (outputs of each layer)
def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):

    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    print("Preparing outputs....")
    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs
    
    print ("-----------------")
    print (len(outputs))
    print ("-----------------")
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    print ("--------Layer Ouputs---------")

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

# 4.1 Get the activations (representations) using all the training and testing samples.

print ("Saving the layer outputs...")

repre_testing = get_activations(model, test_set_x, print_shape_only=True)

def storeOuput(arr, path):
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(arr)

# There are five layers, we only care about the third and the fourth layer.

# train_X = train_layer_three = repre_training[2] 
# test_X = test_layer_three = repre_testing[2] 

#train_X = train_layer_three = repre_training
ffmpeg_repre = test_layer_three = repre_testing[4] 

#train_X = train_layer_three = repre_training
#test_X = test_layer_three = repre_testing 

#train_y = np.ndarray.flatten(np.asarray(train_set_y))
#libtiff_label = np.ndarray.flatten(np.asarray(test_set_y))

print ("-------------------------")

print ("The shape of the learned representations: " + "\r\n")

print (ffmpeg_repre.shape)

"""
The obtained representations for project Ffmpeg can be used as features for training a machine learning classifier (here we use random forest).  

Suppose that project FFmpeg has very limited labeled data. To simulate this situation, we divide the total FFmpeg functions into two sets: 
    
25% of total functions are used for training (simulated the labeled data), and 75% of total functions for testing (simulated the unlabeled data)

"""

train_set_x, test_set_x, train_set_y_id, test_set_y_id = train_test_split(ffmpeg_repre, testing_list_id, test_size=0.75, random_state=42) 

train_X = train_set_x
train_y = GenerateLabels(train_set_y_id)
test_X = test_set_x
test_y = GenerateLabels(test_set_y_id)

train_y = np.ndarray.flatten(np.asarray(train_y))
test_y = np.ndarray.flatten(np.asarray(test_y))

print ("-------------------------")

print ("The shape of the datasets: " + "\r\n")

print (len(train_X), len(train_y), len(test_X), len(test_y))

print (train_X.shape, train_y.shape, test_X.shape, test_y.shape)

print (np.count_nonzero(train_y), np.count_nonzero(test_y))

#-------------------------------------------------------------------------------
# Invoke Sklearn tools for classification -- using Random Forest

#train a random forest model
print ("Fitting the classifier to the training set") 
#t0 = time()
param_grid = {'max_depth': [2,3,4,5,9,10,11,15,20],
              'min_samples_split': [2,3,4,5,6,10],
              'min_samples_leaf': [2,3,4,5,6,10],
              'bootstrap': [True,False],
              'criterion': ['gini','entropy'],
              'n_estimators': [10,20,30,40,50,55,60,65,70]}

    #construct the grid search classifier, 10-fold Cross Validation
clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)
clf = clf.fit(train_X, train_y)
#print("finished params search in %0.3fs" % (time() - t0))
print("best estimator found by grid search:")
print(clf.best_estimator_)

#for params, mean_score, scores in clf.grid_scores_:
    #print (mean_score, scores.std()*2, params)
print ("\r\n")

#evaluate the model on the test set
print("predicting on the test set")
#t0 = time()
y_predict = clf.predict(test_X)

y_predict_proba = clf.predict_proba(test_X)

np.savetxt("./new_results/learned_repr_ffmpeg1.csv", ffmpeg_repre, delimiter=",")
np.savetxt("./new_results/y_predict_proba1_w2v_gmp_ffmpeg1.csv", y_predict_proba, delimiter=",")
np.savetxt("./new_results/y_predict1_w2v_gmp_ffmpeg1.csv", y_predict, delimiter=",")
np.savetxt("./new_results/test_label_ffmpeg1.csv", test_set_y, delimiter=",")
storeOuput(test_set_y_id, "./new_results/test_output_ids_ffmpeg1.csv")
storeOuput(testing_list_id, "./new_results/ffmpeg_ids1.csv")

#y_predict_proba = cross_val_predict(clf, )

#print (y_predict_proba)

# Accuracy
accuracy = np.mean(test_y==y_predict)*100
print ("accuracy = " +  str(accuracy))
    
target_names = ["Non-vulnerable","Vulnerable"] #non-vulnerable->0, vulnerable->1
print (confusion_matrix(test_y, y_predict, labels=[0,1]))   
print ("\r\n")
print ("\r\n")
print (classification_report(test_y, y_predict, target_names=target_names))

K.clear_session()
	
print ("\r\n")
print ("--- %s seconds --- " +  str((time.time() - script_start_time)))