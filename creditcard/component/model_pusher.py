import os, sys
from creditcard.exception import CreditCardsException
from creditcard.logger import logging
from creditcard.entity.config_entity import ModelPusherConfig
from creditcard.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact,ModelPusherArtifact
from creditcard.ML.model_resolver import ModelResolver
from creditcard.utils import load_object, save_object

class ModelPusher:
    def __init__(self,model_pusher_config:ModelPusherConfig,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise CreditCardsException(e,sys)
        
    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            logging.info("Load object")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            model = load_object(file_path=self.model_trainer_artifact.Model_path)

            logging.info("save the model in model pusher dir")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)

            logging.info("get the latest path")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()
            model_path = self.model_resolver.get_latest_save_model_path()

            logging.info("save the model in saved model")
            save_object(file_path=transformer_path,obj=transformer)
            save_object(file_path=target_encoder_path, obj=target_encoder)
            save_object(file_path= model_path, obj=model)

            logging.info("prepare model pusher artifact")
            model_pusher_artifact = ModelPusherArtifact(saved_model_dir=self.model_pusher_config.saved_model_dir,
                                                        pusher_model_dir=self.model_pusher_config.pusher_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise CreditCardsException(e,sys)
        