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

import tensorflow as tf

import keras.backend as K
import gc





os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ======================================================================================================================
# Data Load
df = pd.read_csv('./Datasets/train.tsv', sep='\t')
# df_test=pd.read_csv('./Datasets/test.tsv', sep='\t')
sample_submission = pd.read_csv('./Datasets/sampleSubmission.csv')

# ======================================================================================================================
# Data preprocessing
from data_preprocessing import *
data_train, data_val, data_test = data_preprocessing(df)

# ======================================================================================================================
'''
Model A: BERT
'''

### Build model object. (model structure in model.py)
from model import *
tokenizer_bert, model_bert = model_bert(max_length = 45, data_train = data_train)

### Train model based on the training set
from train import *
learning_rate = 5e-05
max_length = 45
epoch = 2
batch_size = 32
model, history = model_train(data_train, data_val,
                             tokenizer = tokenizer_bert, model = model_bert,
                             learning_rate = learning_rate, max_length = max_length, epoch = epoch, batch_size = batch_size)

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