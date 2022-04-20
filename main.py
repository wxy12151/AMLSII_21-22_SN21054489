import os
import numpy as np
import pandas as pd
import keras.backend as K
import gc

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ======================================================================================================================
# Data Load
df = pd.read_csv('./Datasets/train.tsv', sep='\t')

# df_test=pd.read_csv('./Datasets/test.tsv', sep='\t')
'''
Tips:
True labels are not available in df_test
We will split df into train/val/test to fine-tune our models based on validation set
and evaluate our models based on test set
'''

# ======================================================================================================================
# Data preprocessing
from function.data_preprocessing import *
data_train, data_val, data_test, submission = data_preprocessing(df)

# ======================================================================================================================
### Choose the model type to train or not:
bert_ = False
RoBERTa_ = False
DistilBERT_ =True
XLNet_ = True
### Hyperparameters
max_length = 45
learning_rate = 5e-05
epoch = 2
batch_size = 32

# ======================================================================================================================
'''
Model 1: bert
'''
if bert_:
    ### Build model object. (model structure in model.py)
    from function.model import *
    tokenizer_bert, model_bert = model_bert(max_length, data_train = data_train)

    ### Train model based on the training set
    from function.train import *
    model, history = model_train(data_train, data_val,
                                 tokenizer = tokenizer_bert, model = model_bert,
                                 learning_rate = learning_rate, max_length = max_length, epoch = epoch, batch_size = batch_size)

    ### Save the trained model and logs
    # model.save('./model_trained/bert_epoch{}_batch{}.h5'.format(epoch, batch_size))
    # np.save('./model_logs/bert_epoch{}_batch{}_train_acc.npy'.format(epoch, batch_size), history.accuracy['batch'])
    # np.save('./model_logs/bert_epoch{}_batch{}_train_loss.npy'.format(epoch, batch_size), history.loss['batch'])

    ### Inference on test dataset and Submission in kaggel competition format
    x_test = tokenizer_func(data_val, tokenizer_bert, max_length)
    label_predicted = model.predict(x={'input_ids': x_test['input_ids']})
    label_pred_max=[np.argmax(i) for i in label_predicted['Sentiment']]
    submission['Sentiment_bert'] = label_pred_max
    # submission.to_csv("./submission/submission_bert.csv", index=False, header=True)
    # print('Submission is ready!')

    ### Evaluation on test dataset

    ### Clean up memory/GPU etc.
    del model
    gc.collect()  # memory
    K.clear_session()  # clear a session

# ======================================================================================================================
'''
Model 2: RoBERTa
'''
if RoBERTa_:
    ### Build model object. (model structure in model.py)
    from function.model import *
    tokenizer_RoBERTa, model_RoBERTa= model_RoBERTa(max_length, data_train = data_train)

    ### Train model based on the training set
    from function.train import *
    model2, history = model_train(data_train, data_val,
                                 tokenizer = tokenizer_RoBERTa, model = model_RoBERTa,
                                 learning_rate = learning_rate, max_length = max_length, epoch = epoch, batch_size = batch_size)

    ### Save the trained model and logs
    # model2.save('./model_trained/RoBERTa_epoch{}_batch{}.h5'.format(epoch, batch_size))
    # np.save('./model_logs/RoBERTa_epoch{}_batch{}_train_acc.npy'.format(epoch, batch_size), history.accuracy['batch'])
    # np.save('./model_logs/RoBERTa_epoch{}_batch{}_train_loss.npy'.format(epoch, batch_size), history.loss['batch'])

    ### Inference on test dataset and Submission in kaggel competition format
    x_test = tokenizer_func(data_val, tokenizer_RoBERTa, max_length)
    label_predicted = model2.predict(x={'input_ids': x_test['input_ids']})
    label_pred_max=[np.argmax(i) for i in label_predicted['Sentiment']]
    submission['Sentiment_RoBERTa'] = label_pred_max
    # submission.to_csv("./submission/submission_RoBERTa.csv", index=False, header=True)
    # print('Submission is ready!')

    ### Evaluation on test dataset

    ### Clean up memory/GPU etc.
    del model2
    gc.collect()  # memory
    K.clear_session()  # clear a session

# ======================================================================================================================
'''
Model 3: DistilBERT
'''
if DistilBERT_:
    ### Build model object. (model structure in model.py)
    from function.model import *
    tokenizer_DistilBERT, model_DistilBERT= model_DistilBERT(max_length, data_train = data_train)

    ### Train model based on the training set
    from function.train import *
    model3, history = model_train(data_train, data_val,
                                 tokenizer = tokenizer_DistilBERT, model = model_DistilBERT,
                                 learning_rate = learning_rate, max_length = max_length, epoch = epoch, batch_size = batch_size)

    ### Save the trained model and logs
    # model3.save('./model_trained/DistilBERT_epoch{}_batch{}.h5'.format(epoch, batch_size))
    # np.save('./model_logs/DistilBERT_epoch{}_batch{}_train_acc.npy'.format(epoch, batch_size), history.accuracy['batch'])
    # np.save('./model_logs/DistilBERT_epoch{}_batch{}_train_loss.npy'.format(epoch, batch_size), history.loss['batch'])

    ### Inference on test dataset and Submission in kaggel competition format
    x_test = tokenizer_func(data_val, tokenizer_DistilBERT, max_length)
    label_predicted = model3.predict(x={'input_ids': x_test['input_ids']})
    label_pred_max=[np.argmax(i) for i in label_predicted['Sentiment']]
    submission['Sentiment_DistilBERT'] = label_pred_max
    # submission.to_csv("./submission/submission_DistilBERT.csv", index=False, header=True)
    # print('Submission is ready!')


    ### Evaluation on test dataset


    ### Clean up memory/GPU etc.
    del model3
    gc.collect()  # memory
    K.clear_session()  # clear a session

# ======================================================================================================================
'''
Model 4: XLNet
'''
if XLNet_:
    ### Build model object. (model structure in model.py)
    from function.model import *
    tokenizer_XLNet, model_XLNet= model_XLNet(max_length, data_train = data_train)

    ### Train model based on the training set
    from function.train import *
    model4, history = model_train(data_train, data_val,
                                 tokenizer = tokenizer_XLNet, model = model_XLNet,
                                 learning_rate = learning_rate, max_length = max_length, epoch = epoch, batch_size = batch_size)

    ### Save the trained model and logs
    # model4.save('./model_trained/XLNet_epoch{}_batch{}.h5'.format(epoch, batch_size))
    # np.save('./model_logs/XLNet_epoch{}_batch{}_train_acc.npy'.format(epoch, batch_size), history.accuracy['batch'])
    # np.save('./model_logs/XLNet_epoch{}_batch{}_train_loss.npy'.format(epoch, batch_size), history.loss['batch'])

    ### Inference on test dataset and Submission in kaggel competition format
    x_test = tokenizer_func(data_val, tokenizer_XLNet, max_length)
    label_predicted = model4.predict(x={'input_ids': x_test['input_ids']})
    label_pred_max=[np.argmax(i) for i in label_predicted['Sentiment']]
    submission['Sentiment_XLNet'] = label_pred_max
    # submission.to_csv("./submission/submission_XLNet.csv", index=False, header=True)
    # print('Submission is ready!')

    ### Evaluation on test dataset

    ### Clean up memory/GPU etc.
    del model4
    gc.collect()  # memory
    K.clear_session()  # clear a session

# ======================================================================================================================
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