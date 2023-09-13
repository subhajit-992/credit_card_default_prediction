import pandas as pd
from .exception import CreditCardsException
import os,sys
import json
import yaml
import dill
import numpy as np
from creditcard.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import boto3


def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise CreditCardsException(e, sys)
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    try:
        def __init__(self):
            pass
    
        def fit(self, X, y=None):
            return self
    
        def transform(self, data):
            transformed_X = data.copy()
            transformed_X = pd.DataFrame(transformed_X)
            numerical_feature = [feature for feature in transformed_X.columns if transformed_X[feature].dtype != 'O']  
        
            for feature in numerical_feature:
                upper_bound = transformed_X[feature].quantile(.75) + 3 * (transformed_X[feature].quantile(.75)-transformed_X[feature].quantile(.25))
                lower_bound = transformed_X[feature].quantile(.25) - 3 * (transformed_X[feature].quantile(.75)-transformed_X[feature].quantile(.25))
                transformed_X.loc[transformed_X[feature] > upper_bound, feature] = upper_bound
                transformed_X.loc[transformed_X[feature] < lower_bound, feature] = lower_bound
        
            return transformed_X
    except Exception as e:
        raise CreditCardsException(e,sys)
    
def save_numpy_array_data(file_path:str, array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise CreditCardsException(e,sys) from e
    
def save_object(file_path:str, obj:object):
    try:
        logging.info("save obj method")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CreditCardsException(e,sys) from e
    
def load_numpy_array_data(file_path: str)->np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditCardsException(e, sys) from e

def evaluate_model(true, predicted):
    try:
        accuracy = accuracy_score(true, predicted)
        cm = confusion_matrix(true, predicted)
        cr = classification_report(true, predicted)
        rs = recall_score(true, predicted)
        return accuracy, cm, cr,rs  
    except Exception as e:
        raise CreditCardsException(e, sys) from e

def model_built_select(X_train_ns,y_train_ns,X_test,y_test,models,parameter):
    try:
        report = {}

        for classifier_name, classifier in models.items():
            params=parameter[classifier_name]
            #clf=GridSearchCV(classifier,param_grid=params,cv=5,scoring='neg_log_loss',n_jobs=-1)
            clf=RandomizedSearchCV(classifier,param_distributions=params,cv=5,scoring='neg_log_loss',n_iter=5)
            clf.fit(X_train_ns,y_train_ns)
            
            classifier.set_params(**clf.best_params_)
            classifier.fit(X_train_ns,y_train_ns)
            
            y_train_pred = classifier.predict(X_train_ns)
            y_test_pred = classifier.predict(X_test)
            
            model_train_accuracy, model_train_cm, model_train_cr, model_train_rs = evaluate_model(y_train_ns, y_train_pred)
            model_test_accuracy , model_test_cm, model_test_cr, model_test_rs = evaluate_model(y_test, y_test_pred)
            
            report[classifier_name] = model_test_accuracy
            logging.info(f"{report}")
        return report

    except Exception as e:
        raise CreditCardsException(e,sys)
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CreditCardsException(e, sys) from e
    
def get_bucket_name_from_secrets():
    try: # Initialize a Secrets Manager client
        secrets_manager_client = boto3.client('secretsmanager', region_name='ap-south-1')

        # Specify the name of your secret containing the bucket name
        secret_name = 'arn:aws:s3:::subhajit-indrani-ml-1'

        # Retrieve the secret value
        response = secrets_manager_client.get_secret_value(SecretId=secret_name)

        # Parse the secret JSON data to extract the bucket name
        secret_data = response['SecretString']
        secret_dict = json.loads(secret_data)
        bucket_name = secret_dict.get('BUCKET_NAME')

        return bucket_name 
    except Exception as e:
        raise CreditCardsException(e,sys)
    



        

      
        

     

    
