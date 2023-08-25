from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    report_file_path:str
    train_file_path:str 
    test_file_path:str 
    status:bool

@dataclass
class DataTransformationArtifact:
    transform_object_path:str 
    transform_train_path:str 
    transform_test_path:str 
    target_encoder_path:str

@dataclass
class ModelTrainerArtifact:
    Model_path:str
    accuracy_train_score:float
    accuracy_test_score:float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float

@dataclass
class ModelPusherArtifact:
    saved_model_dir:str
    pusher_model_dir:str
    