from keras import callbacks

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

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
    '''
        Input: data to be tokenized
               pretrained tokenizer
               token max length
        Output: data after tokenizer
    '''
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

# ======================================================================================================================
### Define model training function
def model_train(data_train, data_val, tokenizer, model, learning_rate, max_length,
                epoch, batch_size, history = LossAndAccHistory()):
    '''
        Input: training and validation dataset
               pretrained tokenizer
               pretrained bert model
               Hyperparameter: lr, max_length. epoch, batch_size
               Customer callback for logs storage
        Output: trained model
                history logs
    '''

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
