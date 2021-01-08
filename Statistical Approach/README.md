## Phamacovigilance project 

We can run the experment by tunning parameters (adding or changing(edit) parameters) of different classfier that are used on the project.

To change parametters go to the file /src/classfier.py check to the dictionery called parameters

            parameters =  {
           
            "KNN":[{"n_neighbors":[3,3,4,5], "weights":['uniform', 'distance']}],
            "GB" :[{'learning_rate': [0.1,0.01,0.001], "n_estimators":[50,100], "ccp_alpha":[0.001,0.01,0.0001,0.092]}],
            "RF":[{'n_estimators':[10,30,50,100], 'max_features': ['auto','log2',None], 
                      'min_samples_leaf': [0.2,0.4,1]}],
            "MLP":[{'hidden_layer_sizes':[(500,),(600,),(700,),(800,)],'solver': [ 'sgd'],'activation':['relu']
                      }],
            "GBN":[{'priors':[None],  
            #   'var_smoothing':[1e-09] }],
            
The last five classfiers has no parameters on the dictionery so its use the default value you may also add them.


            "DTC": [{}],
            "SVC": [{}],
            "BC":[{}],
            "XGB": [{'classifier__booster':['gbtree','dart'],}],
            "EXT":[{}],
            "LG":[{}]
              
              }
              
List of Classifiers to (Note that the initials are not standard, I created them ,standard name of classfier are in the right side of dictionery )


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


Select the classfier you want to change or tunning like KNN, google it or go direct to its page by checking its full name on list of classfier above , read its parameters, an start to add them on the dictionery 
by creating a new dictionery that contain the parameter name and value 

           "parameters_name":[values] as "n_neighbors":[3,3,4,5] 


As in the format below  
      
     
            "Classfier_name":[{"parameter_name1":[Values1], "parameter_name2":[Values2]....., "parameter_nameN":[ValuesN]}],
            
            as 
            
              "GB" :[{'learning_rate': [0.1,0.01,0.001], "n_estimators":[50,100], "ccp_alpha":[0.001,0.01,0.0001,0.092]}],
              
              
              
After that save the file go to terminal navigate up to /src ,then run the experiment file with command 


            #python experiment --mode mode_name  --arch classfier_name

            
For training mode_name will be training and classfier you will select the classfier you want to train default KNN 

            #python experiment --mode train  --arch KNN

For testing mode test or results classfier are the same you just choose the classfier you want to test 


            #python experiment --mode test  --arch KNN

Then assess the progress of the model by checking on the logs file or see the results on terminal.

To check the figures (comfussion matrix) visit /figures check in train and test figures