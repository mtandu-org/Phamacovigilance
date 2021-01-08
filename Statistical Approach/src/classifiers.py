#classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# GridSearch: Set parameters valuaes for each classifier



parameters =  {
           
            "KNN":[{"n_neighbors":[3,3,4,5], "weights":['uniform', 'distance']}],
            "GB" :[{'learning_rate': [0.1,0.01,0.001], "n_estimators":[50,100], "ccp_alpha":[0.001,0.01,0.0001,0.092]}],
            "RF":[{'n_estimators':[10,30,50,100], 'max_features': ['auto','log2',None], 
                      'min_samples_leaf': [0.2,0.4,1]}],
            "MLP":[{'hidden_layer_sizes':[(500,),(600,),(700,),(800,)],                     'solver': [ 'sgd'],
                       'activation':['relu']
                      }],
            "GBN":[{'priors':[None], 
                    #    'var_smoothing':[1e-09]
                       }],
            "DTC": [{}],
            "SVC": [{}],
            "BC":[{}],
            "XGB": [{'classifier__booster':['gbtree','dart'],}],
            "EXT":[{}],
            "LG":[{}]
              
              }

# List of Classifiers to
models = {  

         "KNN": KNeighborsClassifier(),
         "GB":GradientBoostingClassifier(),
         "RF":RandomForestClassifier(),
         "GBN":GaussianNB(),
         "MLP":MLPClassifier(),
         "DTC":DecisionTreeClassifier(),
         "SVC":svm.SVC(),
         "BC":BaggingClassifier(),
         "XGB": XGBClassifier(),
         "EXT":ExtraTreesClassifier(),
         "LG":LogisticRegression() 
         }


