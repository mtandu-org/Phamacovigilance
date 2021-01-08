# import important packages
from sklearn import preprocessing
from sklearn.externals import joblib
import pandas as pd

#metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import  f1_score, classification_report
from sklearn.metrics import confusion_matrix

from data import *
from visualize import *
from logs import log
from sklearn.feature_extraction.text import TfidfVectorizer

#set a logger file
logger = log("../logs/test_logs")


# function to test the model performance
def test_model(model_name):

    # name of the classes
    class_names = ['0','1']


    # load the test data and label

    features , labels = load_test_data('../data/test.xlsx' )                                                                               
    #load the model

    clf = joblib.load("../models/{}.{}".format(model_name,"pkl"))
    logger.info("Testing model of upsampled data ") # change the testing model here for the log file
    logger.info("model name:{}".format(model_name))

    y_pred = clf.predict(features)
    cm = confusion_matrix(labels, y_pred)
    print('Confusion Matrix : \n', cm)

    total = sum(sum(cm))

    sensitivity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    logger.info('Sensitivity :{:.3f} '.format(sensitivity1))

    specificity1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    logger.info('Specificity : {:.3f}'.format(specificity1))

    balanced_accuracy = balanced_accuracy_score(labels, y_pred)
    logger.info("balanced Accuracy: {:.3f}".format(balanced_accuracy))

    f1_tra = f1_score(labels, y_pred, average='weighted')
    logger.info("f1 score: {:.3f}".format(f1_tra))
    clf_report = classification_report(labels, y_pred, output_dict=True)
    df = pd.DataFrame(clf_report)
    df = df.transpose()
    print(df)
    logger.info(df)
    df.to_csv('../figures/test_figures/{}_scores.csv'.format(model_name))
 

    # plot the confusion matrix
    plot_confusion_matrix(labels, y_pred, class_names, title= model_name + "-cm")

    # save the confusion matrix
    plt.savefig("../figures/test_figures/{}_{}_cm.pdf".format(model_name, "test_results"), bbox_inches="tight")
    plt.close()
    logger.info("---------Done----------------------")


def main():

    model = "GB-f1-0.917"
    test_model(model_name= model)



if __name__ == "__main__":
    main()
            

    

