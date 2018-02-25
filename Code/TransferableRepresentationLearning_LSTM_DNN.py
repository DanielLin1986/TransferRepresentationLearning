# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:51:39 2017

This file implements a LSTM network that is capable of leveraging the historical vulnerable function data for learning the representations.

"""

import time
import pickle
import csv
import numpy as np

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Bidirectional
from keras.layers import LSTM
from keras.layers import GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard, CSVLogger
from keras import backend as K

script_start_time = time.time()

print ("Script starts at: " + str(script_start_time))

# ------------------------------------------------------------ #
# Parameters used
MAX_LEN = 1000 # The Padding Length for each sample.
EMBEDDING_DIM = 100 # The Embedding Dimension for each element within the sequence of a data sample. 

NUM_TRAIN_SAMPLE = 19326
NUM_VALIDATION_SAMPLE = 8283
NUM_TEST_SAMPLE = 4921
BATCH_SIZE = 32
EPOCHS = 150

working_dir = '/home/your/user/name/TransferRepresentationLearning/ffmpeg/'

w2v_dir = '/home/your/user/name/TransferRepresentationLearning/word2vec/'

log_path = '/home/your/user/name/TransferRepresentationLearning/Logs/'

# The path where the trained models are saved.
model_saved_path = '/home/your/user/name/TransferRepresentationLearning/models/'

saved_model_name = "1st_1000_100_32_90_test_on_ffmpeg"

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

print (train_sequences_pad.shape)
print (test_sequences_pad.shape)

train_set_x, validation_set_x, train_set_y_id, validation_set_id = train_test_split(train_sequences_pad, training_list_id, test_size=0.3, random_state=42) 

print ("Training set: ")

print (train_set_x)

#print test_validation_set_x

print ("The length of the training set: " + str(len(train_set_x)) + "\n" + "The length of the training labels: " +  str(len(train_set_y_id)))

print ("Validation set: ")

print (validation_set_x)

print ("Testing set: ")

test_set_x = test_sequences_pad
test_set_id = testing_list_id

print (test_set_x)

print (len(validation_set_x), len(test_set_x), len(validation_set_id), len(test_set_id))

#print validation_set_x, test_set_x, validation_set_y, test_set_y

# Now we need to convert all the *_set_y to 0 and 1 labels. All the *_set_y lists contain the actual names of all the samples.

# The samples' ids of the train_set should be reserved, so after training we can still use the ids to identify which feature sets belong to which sample.
train_set_y = GenerateLabels(train_set_y_id)
validation_set_y = GenerateLabels(validation_set_id)
test_set_y = GenerateLabels(test_set_id)

print ("-------------------------")

print ("The shape of the datasets: " + "\r\n")

print (train_set_x.shape, train_set_y.shape, validation_set_x.shape, validation_set_y.shape, test_set_x.shape, test_set_y.shape)

print (np.count_nonzero(train_set_y), np.count_nonzero(validation_set_y), np.count_nonzero(test_set_y))

# ----------------------------------------------------- #
# 4. Preparing the Embedding layer

embeddings_index = {} # a dictionary with mapping of a word i.e. 'int' and its corresponding 100 dimension embedding.

# Use the loaded model
for line in w2v_model:
    if not line.isspace():
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
w2v_model.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Get the activations (outputs of each layer)
def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):

    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

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

def storeOuput(arr, path):
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
        wr.writerow(arr)
        
# ------------------------------------------------------------ #
# 5. Define network structure
model = Sequential()
#
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LEN,
                            trainable=False)) # Layer 0: an embedding layer
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True))) # Layer 1: An LSTM layer (tanh)
model.add(GlobalMaxPooling1D())
#model.add(Bidirectional(LSTM(64))) # Layer 2: An LSTM layer
model.add(Dense(64, activation='tanh'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid')) # Layer 3: Dense layer

print ("-------------------------")

print ("strat compiling the model...")

# ------------------------------------------------------------ #
# 6. Configure the learning process
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Save weights of best training epoch: monitor either val_loss or val_acc
callbacks_list = [
        ModelCheckpoint(filepath = model_saved_path + saved_model_name +'_{epoch:02d}_{val_acc:.3f}.h5', monitor='val_loss', verbose=2, save_best_only=True, period=1),
        EarlyStopping(monitor='val_loss', patience=60, verbose=2, mode="min"),
		 TensorBoard(log_dir=log_path, batch_size = BATCH_SIZE,  write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
        CSVLogger(log_path + saved_model_name + '.log')]


print ("start training the model...")

# ------------------------------------------------------------ #
# 7. Train the model. 
model.fit(train_set_x, train_set_y,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
		   shuffle = False, # The data has already been shuffle before, so it is unnessary to shuffle it again. (And also, we need to correspond the ids to the features of the samples.)
          #validation_split=0.5,
          validation_data = (validation_set_x, validation_set_y), # Validation data is not used for training (or development of the model)
          callbacks=callbacks_list, # Get the best weights of the model and stop the first raound training.
          verbose=2)

print ("Model training completed! ")

print ("-----------------------------------------------")

print ("Start predicting....")

predicted_classes = model.predict_classes(test_set_x, batch_size=BATCH_SIZE, verbose=2)

#print (predicted_classes)

test_accuracy = np.mean(np.equal(test_set_y, predicted_classes))

print ("LSTM classification result: ")

target_names = ["Non-vulnerable","Vulnerable"] #non-vulnerable->0, vulnerable->1
print (confusion_matrix(test_set_y, predicted_classes, labels=[0,1]))   
print ("\r\n")
print ("\r\n")
print (classification_report(test_set_y, predicted_classes, target_names=target_names))

print ("LSTM prediction completed.")

K.clear_session()	

print ("\r\n")
print ("--- %s seconds ---" + str(time.time() - script_start_time))