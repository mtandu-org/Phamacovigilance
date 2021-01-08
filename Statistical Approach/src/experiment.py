
import argparse 
from data import *
from data import *
from learner import *
from classifiers import *
from visualize import *
from logs import log
from testing import *
import glob




class_names = ['0','1']

def get_arguments():
    parser = argparse.ArgumentParser(description='ADR classfication project')

    parser.add_argument("--mode", type=str, default="train",
                        help="set a module in training or prediction mode")
    parser.add_argument("--arch", type=str, default="GB",
                        help="Classfier we use ")

    parser.add_argument("--model_name", type=str, default="GB",
                        help="type of model to load")
    parser.add_argument("--data_path", type=str, default="../data/",
                        help="type of model to load")
    parser.add_argument("--model_path", type=str, default="../models/",
                        help="type of model to load")
    parser.add_argument("--logs_path", type=str, default="../logs/",
                        help="type of model to load")
    parser.add_argument("--results_path", type=str, default="../figure/",
                        help="type of model to load")
    parser.add_argument("--figure_path", type=str, default="../figure/",
                        help="type of model to load")


    args = parser.parse_args()
    return args
args=get_arguments()



def load_data():
    if args.mode=="train":
        logger = log("../logs/train_logs")
        x_train, x_valid, y_train, y_valid = process_data()
        return x_train, x_valid, y_train, y_valid 


    elif (args.mode=="test") or (args.mode=="results"):
        logger = log("../logs/test_logs")
        test_data=pd.read_excel('../data/test.xlsx')
        return test_data
    

    else:raise AssertionError("Define a correct mode")

    

def load_model(x_train, y_train):
    for model_name, model in models.items():
       
        if args.arch == model_name:
            

            print("Upsampled class with the model : "+model_name)
            logger.info("Upsampled class with the model : "+model_name) # change train status here for the log file

           #gridsearch for each classifier
            clf = GridSearchCV(model, parameters[model_name], cv=7)
            clf.fit(x_train, y_train)

    return clf


def main():

    if args.mode=="train":


        #dataloaders, class_names = load_data()
        x_train, x_valid, y_train, y_valid = load_data()
        

        #print(class_names)
        
        #load model
        clf=load_model(x_train, y_train)
        #print(clf.best_params_)
        #logger.info("Best parameters values:{}".format(clf.best_params_))

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
        df.to_csv('../figures/train_figures/{}_scores.csv'.format(args.arch))
        

        #plot the confusion matrix
        plot_confusion_matrix(y_valid, y_pred, class_names, title= args.arch + "-cm")

        #save the confusion matrix
        plt.savefig("../figures/train_figures/{}_{}_cm.pdf".format(args.arch,"train_results"), bbox_inches="tight")
        plt.close()

        #save the model
        joblib.dump(clf,'../models/{}-f1-{:.3f}.pkl'.format(args.arch,f1_tra))
        logger.info("-----------Done--------------------")


    elif (args.mode=="test") or (args.mode=="results"):
        for model_name, model in models.items():
            if args.arch in model_name:
                if args.arch == model_name:
                    print(model_name)
                    
                    files = glob.glob("../models/*.pkl")
                    for file in files:
                        if model_name in file:
                            print(file)
                            
                            file =  ''.join(file)


                            file= file.split("/")
                            file = file[-1].split(".")
                            file = file[0] + "." + file[1]
                        #print(file)
                            test_model(model_name= file)


    else:raise AssertionError("Define a correct mode")

if __name__ == "__main__":
    main()
