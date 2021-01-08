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


    logger.info("------------Done-------------------")


def main():

    model = "Voting Classifier --f1-1.000"
    test_model(model_name= model)



if __name__ == "__main__":
    main()
            

    

