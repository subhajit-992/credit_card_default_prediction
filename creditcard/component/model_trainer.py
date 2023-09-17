import os,sys
import numpy as np
import pandas as pd
from creditcard.exception import CreditCardsException
from creditcard.logger import logging
from creditcard.entity.config_entity import ModelTrainerConfig
from creditcard.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from creditcard.utils import load_numpy_array_data,model_built_select,evaluate_model,save_object

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transform_train_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transform_test_path)

            logging.info("Split data into input and target feature")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info("Start model building")
            models = {
                "Logistic_Regressor": LogisticRegression(),
                #"KNeighbors_Classifier": KNeighborsClassifier(),
                "Decision_Classifier": DecisionTreeClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "AdaBoost_Classifier": AdaBoostClassifier(),
                "GradientBoosting_Classifier":GradientBoostingClassifier(),
                "Random_Forest_Classifier": RandomForestClassifier()
            }

            logging.info("Hyperparameter tuning")
            log_grid={'C':10.0 **np.arange(-2,3),
                      'penalty':['l2'],
                       #'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            }
                 


            random_grid={'n_estimators':[int(x) for x in np.linspace(start=100, stop=1000, num=100)],
                        'max_features':['sqrt','log2'],
                        'criterion':['gini','entropy', 'log_loss'],
                        'max_depth':[2,4,6,10]}

            tree_grid={'criterion':['gini','entropy','log_loss'],
                       'max_features':['sqrt','log2'],
                       'max_depth':[3, 5, 7],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 3]
            }
           
          

            knn_grid={'n_neighbors':[10, 15, 20,],
                      'algorithm':['auto','ball_tree', 'kd_tree',],
                      'weights': ['uniform', 'distance']
            }
         

            xgb_grid={
                 'eta': [0.1, 0.2, 0.3],                  
                 'max_depth': [3, 5, 7],                  
                 'subsample': [0.8, 0.9],                  
                 'learning_rate_decay': [lambda current_round, initial_learning_rate: initial_learning_rate / (1 + current_round * 0.01)]
            }


            adaboost_grid = {
                 'n_estimators': [50, 100, 200],
                 'learning_rate': [0.01, 0.1, 0.5]
            }


            gradientboost_grid = {
                 'n_estimators': [50, 100, 200],
                 'learning_rate': [0.01, 0.1, 0.5],
                 'max_depth': [3, 4, 5]
            }



            parameter={'Logistic_Regressor':log_grid,
                       'Decision_Classifier':tree_grid,
                       'Random_Forest_Classifier':random_grid,
                       #'KNeighbors':knn_grid,
                       'XGBClassifier':xgb_grid,
                       'AdaBoost_Classifier':adaboost_grid,
                       'GradientBoosting_Classifier':gradientboost_grid
            }

            logging.info("fit the model")
            model_report:dict = model_built_select(X_train_ns=x_train,
                                                   y_train_ns=y_train,
                                                   X_test=x_test,
                                                   y_test=y_test,
                                                   models=models,
                                                   parameter=parameter)
            
            logging.info("select best model")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            logging.info(f"best model is {best_model}")           
            
            logging.info("prediction on train and test")
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            logging.info("Model_evalution")
            model_train_accuracy, model_train_cm, model_train_cr, model_train_rs = evaluate_model(y_train, y_train_pred)
            model_test_accuracy , model_test_cm, model_test_cr, model_test_rs = evaluate_model(y_test, y_test_pred)

            logging.info(f"train score:{model_train_accuracy} and test score:{model_test_accuracy}")

            logging.info(f"Checking if our model is underfitting or not")
            if(model_train_accuracy<self.model_trainer_config.expected_score):
                raise CreditCardsException(f"Model is not good as it's not give good accuracy:{model_train_accuracy}")
            
            logging.info("Checking if our model is overfitting or not")
            diff = abs(model_train_accuracy-model_test_accuracy)
            if(diff>self.model_trainer_config.overfitting_threshold):
                raise CreditCardsException(f"difference between train and test accuracy:{diff} is more than overfitting threshold")
            
            logging.info("saving model object")
            save_object(file_path=self.model_trainer_config.model_path, obj=best_model)

            logging.info("prepare artifact")
            model_trainer_artifact = ModelTrainerArtifact(Model_path=self.model_trainer_config.model_path,
                                                          accuracy_train_score=model_train_accuracy,
                                                          accuracy_test_score=model_test_accuracy)
            logging .info(f"model trainer artifact:{model_trainer_artifact}")
            return model_trainer_artifact



        except Exception as e:
            raise CreditCardsException(e,sys)
        
    