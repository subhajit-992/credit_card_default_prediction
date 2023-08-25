from creditcard.entity.config_entity import (TrainingPipelineConfig,
                                             DataIngestionConfig,
                                             DataValidationConfig,
                                             DataTransformationConfig,
                                             ModelTrainerConfig,
                                             ModelEvaluationConfig,
                                             ModelPusherConfig)
from creditcard.entity.artifact_entity import (DataIngestionArtifact,
                                               DataValidationArtifact,
                                               DataTransformationArtifact,
                                               ModelTrainerArtifact,
                                               ModelEvaluationArtifact,
                                               ModelPusherArtifact)

from creditcard.exception import CreditCardsException
from creditcard.logger import logging
import os,sys
from creditcard.component.data_ingestion import DataIngestion
from creditcard.component.data_validation import DataValidation
from creditcard.component.data_transformation import DataTransformation
from creditcard.component.model_trainer import ModelTrainer
from creditcard.component.model_evaluation import ModelEvalution
from creditcard.component.model_pusher import ModelPusher


class TrainingPipeline:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            data_ingestion_config =  DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config=data_validation_config,
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config= self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise CreditCardsException(e,sys)
        
    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                         data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def start_model_evaluation(self,model_trainer_artifact:ModelTrainerArtifact,
                               data_validation_artifact:DataValidationArtifact,
                               data_transformation_artifact:DataTransformationArtifact)->ModelEvaluationArtifact:
        try:
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_evaluation = ModelEvalution(model_eval_config= model_evaluation_config,
                                              data_validation_artifact=data_validation_artifact,
                                              data_transformation_artifact=data_transformation_artifact,
                                              model_trainer_artifact=model_trainer_artifact)
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def start_model_pusher(self,model_trainer_artifact:ModelTrainerArtifact,
                           data_transformation_artifact:DataTransformationArtifact)->ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config =model_pusher_config,
                                       data_transformation_artifact=data_transformation_artifact,
                                       model_trainer_artifact=model_trainer_artifact)
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    
    def start(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact = data_ingestion_artifact)
            
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact = data_validation_artifact)
            
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact = data_transformation_artifact)
            
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_artifact=data_transformation_artifact)  

            model_pusher_artifact = self.start_model_pusher(
                model_trainer_artifact = model_trainer_artifact,
                data_transformation_artifact = data_transformation_artifact)                                                  
        
        except Exception as e:
            raise CreditCardsException(e,sys)