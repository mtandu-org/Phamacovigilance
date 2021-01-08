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

#set a logger file
logger = log("../logs/test-single_logs")


# function to test the model performance
def test_single(model_name):

    # name of the classes
    class_names = ['0','1']


    
    #load the model

    clf = joblib.load("../models/{}.{}".format(model_name,"pkl"))
    logger.info("Testing Single row -model upsampled class")
     # change the testing model here for the log file
    logger.info("model name:{}".format(model_name))

   #load data to test 
    df = load_data('../data/data_upsampled.xlsx' )
    
   
    name = re.compile('\[.*\]')
   
    df.text = df.text.str.strip()
    df['text'] = df['text'].apply(lambda x: x.replace(''.join(name.findall(x)), '') if type(x) is str else type(x))

    df['text'] = df['text'].astype(str)
    df['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

    features  = df.loc[:,"text"].values
    labels    = df.loc[:,"Lebels"].values.astype('int')
    nltk.download('stopwords')
    #df.text = df.text.apply(remove_stopwords)
    tfidf_vectorizer = TfidfVectorizer (max_features=1000)
    tfidf_features = tfidf_vectorizer.fit_transform(features).toarray()
    ynew = clf.predict(tfidf_features)
    list1 = []
    print(ynew)
    for i in range(len(ynew)):
        if ynew[i] == 0:
            list1.append("Not ADR")
        elif ynew[i] == 1:
            list1.append("ADR")
        else:
            print( "Error")    
# show the predicted outputs
   
    logger.info((list1))
 
    logger.info("------------Done-------------------")


def main():

    model = "KNN-f1-0.898"
    test_single(model_name= model)



if __name__ == "__main__":
    main()
            

    

