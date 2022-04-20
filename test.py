from sklearn.metrics import classification_report

def evaluation(model_name, y_true, y_pred):

    report = classification_report(submission['Sentiment'], submission['Sentiment_bert'])
    print('{}\n'.format(model_name), report)