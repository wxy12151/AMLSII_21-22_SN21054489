from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow as tf

from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
from transformers import XLNetTokenizer, TFXLNetModel, XLNetConfig
import sentencepiece

def model_bert(max_length, data_train):
    '''
    model setups + build the model
    '''

    # Name of the BERT model to use
    model_name = 'bert-base-uncased'

    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

    # Load the Transformers BERT model
    transformer_bert_model = TFBertModel.from_pretrained(model_name, config=config)

    ### ------- Build the model ------- ###

    # Load the MainLayer
    bert = transformer_bert_model.layers[0]

    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}

    # Load the Transformers BERT model as a layer in a Keras model
    bert_model = bert(inputs)[1]
    dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)

    # Then build your model output
    Sentiments = Dense(units=len(data_train.Sentiment_label.value_counts()),
                       kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Sentiment')(
        pooled_output)
    outputs = {'Sentiment': Sentiments}

    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiClass')

    # Take a look at the model
    model.summary()

    return tokenizer, model

def model_RoBERTa(max_length, data_train):

    # Name of the RoBERTa model to use
    model_name = 'roberta-base'

    # Load transformers config and set output_hidden_states to False
    config = RobertaConfig.from_pretrained(model_name)
    config.output_hidden_states = False

    # Load Roberta tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

    # Load the Roberta model
    transformer_roberta_model = TFRobertaModel.from_pretrained(model_name, config=config)

    ### ------- Build the model ------- ###

    # Load the MainLayer
    roberta = transformer_roberta_model.layers[0]

    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}

    # Load the Transformers RoBERTa model as a layer in a Keras model
    roberta_model = roberta(inputs)[1]
    dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(roberta_model, training=False)

    # Then build your model output
    Sentiments = Dense(units=len(data_train.Sentiment_label.value_counts()),
                       kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Sentiment')(
        pooled_output)
    outputs = {'Sentiment': Sentiments}

    # And combine it all in a model object
    model2 = Model(inputs=inputs, outputs=outputs, name='RoBERTa_MultiClass')

    # Take a look at the model
    model2.summary()

    return tokenizer, model2

def model_DistilBERT(max_length, data_train):

    # Name of the DistilBERT model to use
    model_name = 'distilbert-base-uncased'

    # Load transformers config and set output_hidden_states to False
    config = DistilBertConfig.from_pretrained(model_name)
    config.output_hidden_states = False

    # Load Distilbert tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

    # Load the Distilbert model
    transformer_distilbert_model = TFDistilBertModel.from_pretrained(model_name, config=config)

    ### ------- Build the model ------- ###

    # Load the MainLayer
    distilbert = transformer_distilbert_model.layers[0]

    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}

    # Load the Transformers DistilBERT model as a layer in a Keras model
    distilbert_model = distilbert(inputs)[0][:, 0, :]
    dropout = Dropout(0.1, name='pooled_output')
    pooled_output = dropout(distilbert_model, training=False)

    # Then build your model output
    Sentiments = Dense(units=len(data_train.Sentiment_label.value_counts()),
                       kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Sentiment')(
        pooled_output)
    outputs = {'Sentiment': Sentiments}

    # And combine it all in a model object
    model3 = Model(inputs=inputs, outputs=outputs, name='DistilBERT_MultiClass')

    # Take a look at the model
    model3.summary()

    return tokenizer, model3

def model_XLNet(max_length, data_train):

    # Name of the XLNet model to use
    model_name = 'xlnet-base-cased'

    # Load transformers config and set output_hidden_states to False
    config = XLNetConfig.from_pretrained(model_name)
    config.output_hidden_states = False

    # Load XLNet tokenizer
    tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

    # tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

    # Load the XLNet model
    transformer_xlnet_model = TFXLNetModel.from_pretrained(model_name, config=config)

    ### ------- Build the model ------- ###

    # Load the MainLayer
    xlnet = transformer_xlnet_model.layers[0]

    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}

    # Load the Transformers XLNet model as a layer in a Keras model
    xlnet_model = xlnet(inputs)[0]
    xlnet_model = tf.squeeze(xlnet_model[:, -1:, :], axis=1)
    dropout = Dropout(0.1, name='pooled_output')
    pooled_output = dropout(xlnet_model, training=False)

    # Then build your model output
    Sentiments = Dense(units=len(data_train.Sentiment_label.value_counts()),
                       kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Sentiment')(
        pooled_output)
    outputs = {'Sentiment': Sentiments}

    # And combine it all in a model object
    model4 = Model(inputs=inputs, outputs=outputs, name='XLNet_MultiClass')

    # Take a look at the model
    model4.summary()

    return tokenizer, model4