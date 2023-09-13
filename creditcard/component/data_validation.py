import os
import sys
from creditcard.exception import CreditCardsException
from creditcard.logger import logging
from creditcard.entity.config_entity import DataValidationConfig
from creditcard.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from typing import Optional
from creditcard.utils import write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np


class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig, 
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        try:
            threshold = self.data_validation_config.missing_thresold
            null_report = (df.isna().sum()*100)/df.shape[0]
            logging.info(f"Selecting column name which contains null above to {threshold}")
            drop_column_names = null_report[null_report>threshold].index
            logging.info(f"Columns to Drop: {list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise CreditCardsException(e, sys)
        
    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name)->None:
        try:
            drift_report = dict()
            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")
                same_distribution =ks_2samp(base_data,current_data)

                if same_distribution.pvalue>0.05:
                    #We are accepting null hypothesis
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                else:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }
                    #different distribution
            self.validation_error[report_key_name]=drift_report
        except Exception as e:
            raise CreditCardsException(e, sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(df=train_df,report_key_name="train_missing_value")
            test_df = self.drop_missing_values_columns(df=test_df,report_key_name="test_missing_value")

            if train_df is None:
                logging.info(f"No column left in train df hence stopping this pipeline")
                raise Exception("No column left in train df hence stopping this pipeline")
            
            
            if test_df is None:
                logging.info(f"No column Left in test df hence stopping this pipeline")
                raise Exception("No column left in test df hence stopping this pipeline")
            
            if (len(train_df.columns) != len(test_df.columns)):
                raise Exception(f"Train and test df does not have equal columns")
            
            self.data_drift(base_df=train_df,current_df=test_df,report_key_name="train_test_drift")

            write_yaml_file(file_path=self.data_validation_config.report_file_name,
                            data=self.validation_error)
            os.makedirs(self.data_validation_config.valid_dir,exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)

            data_validation_artifact = DataValidationArtifact(report_file_path=self.data_validation_config.report_file_name,
                                                              train_file_path=self.data_validation_config.valid_train_file_path,
                                                              test_file_path=self.data_validation_config.valid_test_file_path,
                                                              status=True)
            
            return data_validation_artifact
            

        except Exception as e:
            raise CreditCardsException(e,sys)