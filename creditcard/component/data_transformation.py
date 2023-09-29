import os,sys
import numpy as np
import pandas as pd
from creditcard.exception import CreditCardsException
from creditcard.logger import logging
from creditcard.entity.config_entity import DataTransformationConfig
from creditcard.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from creditcard.utils import OutlierHandler,save_numpy_array_data,save_object
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso    

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='median')
            Scaling = RobustScaler()
            pipeline = Pipeline(steps=[
                ('Imputer',simple_imputer),
                ("Outlier_Handler",OutlierHandler()),
                ('Robustscaler',Scaling),
                ('feature_selector', SelectFromModel(estimator=Lasso(alpha=0.005, random_state=42)))
            ])
            return pipeline
        except Exception as e:
            raise CreditCardsException(e,sys)
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Reading training and testing file")
            train_df = pd.read_csv( self.data_validation_artifact.train_file_path)
            test_df = pd.read_csv( self.data_validation_artifact.test_file_path)

            logging.info(f"Selecting input feature for train and test dataframe")
            input_feature_train_df=train_df.drop("default_payment_next_month",axis=1)
            input_feature_test_df=test_df.drop("default_payment_next_month",axis=1)

            logging.info("Selecting target feature for train and test dataframe")
            target_feature_train_df = train_df["default_payment_next_month"]
            target_feature_test_df = test_df["default_payment_next_month"]

            logging.info("Converting target cat column into numerical column using label encoder")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            logging.info("Transformation on Target columns")
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
        
            
            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df,target_feature_train_df)

            logging.info(f"Transforming input features")
            input_selected_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_selected_test_arr = transformation_pipleine.transform(input_feature_test_df)

            logging.info(f"Balancing imbalance dataset")
            os= RandomOverSampler(sampling_strategy=0.70)

            logging.info(f"Before resampling in training set Input: {input_selected_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_selected_train_arr, target_feature_train_arr = os.fit_resample(input_selected_train_arr,target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_selected_train_arr.shape} Target:{target_feature_train_arr.shape}")

            logging.info(f"Before resampling in testing set Input: {input_selected_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_selected_test_arr, target_feature_test_arr = os.fit_resample(input_selected_test_arr,target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_selected_test_arr.shape} Target:{target_feature_test_arr.shape}")

            train_arr = np.c_[input_selected_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_selected_test_arr, target_feature_test_arr]

            logging.info(f"Saving data")
            save_numpy_array_data(file_path=self.data_transformation_config.transform_train_path,array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path,array=test_arr)

            logging.info(f"Save label encoder")
            save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            logging.info(f"Save transformation pipeline")
            save_object(file_path=self.data_transformation_config.transform_object_path,obj=transformation_pipleine)

            data_transformation_artifact = DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transform_train_path = self.data_transformation_config.transform_train_path,
                transform_test_path = self.data_transformation_config.transform_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path)
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact


        except Exception as e:
            raise CreditCardsException(e,sys)
 
