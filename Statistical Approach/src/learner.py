# import important packages
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import pandas as pd

#metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import  f1_score, classification_report
from sklearn.metrics import confusion_matrix

from data import *
from classifiers import *
from visualize import *
from logs import log

import re
import nltk
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns
from argparse import Namespace
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#from sklearn.metrics import plot_confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore')
#set a logger file
logger = log("../logs/train_logs")

def process_data():

    df=load_data('../data/data_upsampled.xlsx')
    df,_  = train_test_split(df)
   
    name = re.compile('\[.*\]')
   
    df.text = df.text.str.strip()
    df['text'] = df['text'].apply(lambda x: x.replace(''.join(name.findall(x)), '') if type(x) is str else type(x))

    df['text'] = df['text'].astype(str)
    df['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

    features  = df.loc[:,"text"].values
    labels    = df.loc[:,"Lebels"].values.astype('int')
    nltk.download('stopwords')
    #df.text = df.text.apply(remove_stopwords)
    tfidf_vectorizer = TfidfVectorizer (max_features=600)
   
    tfidf_features = tfidf_vectorizer.fit_transform(features).toarray()
    print(tfidf_features.shape)
    args = Namespace(
    random_state  = 123,
    test_size     = 0.25,
    n_estimators  = 120,
    learning_rate = 0.001,
    max_depth     = 5)

    x_train, x_valid, y_train, y_valid = train_test_split(tfidf_features, labels, 
                                                      test_size=args.test_size, 
                                                      random_state=args.random_state)
    print(x_train.shape)
    return x_train, x_valid, y_train, y_valid 



# function to train the models
def fit_model(parameters = parameters, models = models ):

    # name of the classes
    class_names = ['0','1']
    x_train, x_valid, y_train, y_valid = process_data()


    # load the dataset
  

    logger.info("Fitting| Train model with different classfiers:")

    for model_name, model in models.items():
 

        print("Upsampled class with the model : "+model_name)
        logger.info("Upsampled class with the model : "+model_name) # change train status here for the log file

        #gridsearch for each classifier
        clf = GridSearchCV(model, parameters[model_name], cv=7)
        clf.fit(x_train, y_train)

        print(clf.best_params_)
        logger.info("Best parameters values:{}".format(clf.best_params_))

        y_pred = clf.predict(x_valid)
        cm = confusion_matrix(y_valid, y_pred)
        print('Confusion Matrix : \n', cm)

        total = sum(sum(cm))

        sensitivity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        logger.info('Sensitivity :{:.3f} '.format(sensitivity1))

        specificity1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        logger.info('Specificity : {:.3f}'.format(specificity1))

        balanced_accuracy = balanced_accuracy_score(y_valid, y_pred)
        logger.info("balanced Accuracy: {:.3f}".format(balanced_accuracy))

        f1_tra=f1_score(y_valid, y_pred, average='weighted')
        logger.info("f1 score: {:.3f}".format(f1_tra))
        clf_repot = classification_report(y_valid, y_pred, output_dict=True)
        # print(clt_repot
        #logger.info(clf_repot)
        df = pd.DataFrame(clf_repot)
        df = df.transpose()
        print(df)
        logger.info(df)
        df.to_csv('../figures/train_figures/{}_scores.csv'.format(model_name))
       

        #plot the confusion matrix
        plot_confusion_matrix(y_valid, y_pred, class_names, title= model_name + "-cm")

        #save the confusion matrix
        plt.savefig("../figures/train_figures/{}_{}_cm.pdf".format(model_name,"train_results"), bbox_inches="tight")
        plt.close()

        #save the model
        joblib.dump(clf.best_estimator_, '../models/{}-f1-{:.3f}.pkl'.format(model_name,f1_tra))
        logger.info("-----------Done--------------------")



def main():
    fit_model(parameters = parameters, models = models)

    


if __name__ == "__main__":
    main()
            

    

