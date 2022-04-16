import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

from keras import callbacks

import tensorflow as tf

import keras.backend as K
import gc

from model import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ======================================================================================================================
# Data Load
df = pd.read_csv('./Datasets/train.tsv', sep='\t')
# df_test=pd.read_csv('./Datasets/test.tsv', sep='\t')
sample_submission = pd.read_csv('./Datasets/sampleSubmission.csv')

# ======================================================================================================================
# Data preprocessing

def data_preprocessing():

    ### Split into train(validation within it) and test
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=3)  # test:train = 1:9

    ### Check if there is null/empty value --> no empty value
    # df_train.info()
    # df_train.isnull().sum()

    ###  Extract useful features and labels
    data = df_train[['Phrase', 'Sentiment']]
    data_test = df_test[['Phrase', 'Sentiment']]
    ### Solve Bug: A value is trying to be set on a copy of a slice from a DataFrame
    data = data.copy()
    data_test = data_test.copy()

    ### Set your model output as categorical and save in new label col
    data['Sentiment_label'] = pd.Categorical(data['Sentiment'])
    data_test['Sentiment_label'] = pd.Categorical(data_test['Sentiment'])

    ### Transform output to numeric
    data['Sentiment'] = data['Sentiment_label'].cat.codes
    data_test['Sentiment'] = data_test['Sentiment_label'].cat.codes

    ### Split the training set into training and validation dataset
    data_train, data_val = train_test_split(data, test_size=0.1, random_state=1)

    return data_train, data_val, data_test

data_train, data_val, data_test = data_preprocessing()

# ======================================================================================================================
# Define some common functions

### Custom callback to save logs for drawing training curves.
class LossAndAccHistory(callbacks.Callback):  # Inherited from the Callback class

    '''
    Define four attributes at the beginning of the model,
    each attribute is a dictionary type, storing the corresponding value and epoch
    '''

    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_accuracy = {'batch': [], 'epoch': []}

    # Record the corresponding value after each batch
    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_accuracy['batch'].append(logs.get('val_accuracy'))

    # Record the corresponding value after each epoch
    def on_epoch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_accuracy['batch'].append(logs.get('val_accuracy'))

### Define tokenizer function for repeatedly use.
def tokenizer_func(data, tokenizer, max_length):
    data_after_tokenizer = tokenizer(
        text=data['Phrase'].to_list(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)
    return data_after_tokenizer

### Define model training function
def model_train(tokenizer, model, learning_rate, max_length,
                epoch, batch_size, history = LossAndAccHistory()):

    # Set an optimizer
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-08, decay=0.01, clipnorm=1.0)

    # Set loss and metrics
    loss = {'Sentiment': CategoricalCrossentropy(from_logits=True)}

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Ready output data for the model
    y_train = to_categorical(data_train['Sentiment'])

    # Tokenize the input (takes some time)
    x_train = tokenizer_func(data_train, tokenizer, max_length)

    y_val = to_categorical(data_val['Sentiment'])

    x_val = tokenizer_func(data_val, tokenizer, max_length)

    # Fit the model
    model.fit(
        x={'input_ids': x_train['input_ids']},
        y={'Sentiment': y_train},
        validation_data=({'input_ids': x_val['input_ids']}, {'Sentiment': y_val}),
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        callbacks=[history])

    return model, history

# ======================================================================================================================
'''
Training by Bert
'''

### Build model object. (model structure in model.py)
tokenizer_bert, model_bert = model_bert(max_length = 45, data_train = data_train)

### Train model based on the training set
learning_rate = 5e-05
max_length = 45
epoch = 2
batch_size = 32
model, history = model_train(tokenizer = tokenizer_bert, model = model_bert, learning_rate = learning_rate, max_length = max_length,
                epoch = epoch, batch_size = batch_size)

### Save the trained model and logs
model.save('./model_trained/bert_epoch{}_batch{}.h5'.format(epoch, batch_size))
np.save('./model_logs/bert_epoch{}_batch{}_train_acc.npy'.format(epoch, batch_size), history.accuracy['batch'])
np.save('./model_logs/bert_epoch{}_batch{}_train_loss.npy'.format(epoch, batch_size), history.loss['batch'])

### Clean up memory/GPU etc.
del model
gc.collect()  # memory
K.clear_session()  # clear a session

# model_A = A(args...)                 # Build model object.
# acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A_test = model_A.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task B
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean up memory/GPU etc...




# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'