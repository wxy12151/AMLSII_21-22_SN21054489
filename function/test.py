from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

def evaluation(model_name, y_true, y_pred, y_score):
    # classification report
    report = classification_report(y_true, y_pred)
    print('{}\n'.format(model_name), report)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=np.unique(y_true))
    disp.plot(cmap='Blues')
    plt.grid(False)
    # plt.tight_layout()
    plt.savefig('./images/confusion_{}.pdf'.format(model_name))
    plt.show()
    plt.close()

    # Multi-class ROC
    classes = ['Negative', 'Somewhat Negative', 'Neural', 'Somewhat Positive', 'Positive']
    for i in range(0, 5):
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:, i], pos_label=i)
        auc_ = auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr,
                 lw=lw, label='ROC curve of {} (area = %0.2f)'.format(classes[i]) % auc_)
        plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC of {}'.format(model_name))
    plt.tight_layout()
    plt.savefig('./images/ROC_{}.pdf'.format(model_name))
    plt.show()
    plt.close()