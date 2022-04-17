import pandas as pd
from sklearn.model_selection import train_test_split


def data_preprocessing(df):

    ### Split into train(validation within it) and test
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=3)  # test:train = 1:9
    submission = df_test[['PhraseId', 'Sentiment']]
    submission = submission.copy()

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

    return data_train, data_val, data_test, submission
