from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal

from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig


def model_bert(max_length, data_train):

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

