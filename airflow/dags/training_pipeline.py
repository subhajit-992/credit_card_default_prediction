#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
### DAG Tutorial Documentation
This DAG is demonstrating an Extract -> Transform -> Load pipeline
"""
from __future__ import annotations
from creditcard.pipeline.training_pipeline import TrainingPipeline
from creditcard.entity.config_entity import TrainingPipelineConfig
from creditcard.utils import get_bucket_name_from_secrets,create_folder_s3,download_from_s3,upload_in_s3
from creditcard.logger import logging

# [START tutorial]
# [START import_module]
import json
from textwrap import dedent

import pendulum

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator

import os
#from dotenv import load_dotenv
# Load environment variables from .env file
#load_dotenv()

training_pipeline = TrainingPipeline(training_pipeline_config=TrainingPipelineConfig())

# [END import_module]

# [START instantiate_dag]
with DAG(
    "creditcard_training_pipeline",
    # [START default_args]
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={"retries": 2},
    # [END default_args]
    description="DAG tutorial",
    schedule="@weekly",
    start_date=pendulum.datetime(2023, 9, 2, tz="UTC"),
    catchup=False,
    tags=["Machine_Learning","Classification","CreditCard"],
) as dag:
    # [END instantiate_dag]
    # [START documentation]
    dag.doc_md = __doc__
    # [END documentation]

    # Add this code to print the current directory
    def get_dir():
        current_directory = os.getcwd()
        logging.info(f"Current Directory: {current_directory}")


    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = training_pipeline.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifact",data_ingestion_artifact.__dict__)

    
    def data_validation(**kwargs):
        ti = kwargs["ti"]
        from creditcard.entity.artifact_entity import DataIngestionArtifact
        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifact")
        data_ingestion_artifact = DataIngestionArtifact(**data_ingestion_artifact)
        data_validation_artifact = training_pipeline.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        ti.xcom_push("data_validation_artifact", data_validation_artifact.__dict__)


    

    def data_transformation(**kwargs):
        ti = kwargs["ti"]
        from creditcard.entity.artifact_entity import DataValidationArtifact
        data_validation_artifact = ti.xcom_pull(task_ids="data_validation", key="data_validation_artifact")
        data_validation_artifact = DataValidationArtifact(**(data_validation_artifact))
        data_transformation_artifact = training_pipeline.start_data_transformation(data_validation_artifact=data_validation_artifact)
        ti.xcom_push("data_transformation_artifact", data_transformation_artifact.__dict__)
   
    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        from creditcard.entity.artifact_entity import DataTransformationArtifact
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifact")
        data_transformation_artifact = DataTransformationArtifact(**(data_transformation_artifact))
        model_trainer_artifact = training_pipeline.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        ti.xcom_push("model_trainer_artifact", model_trainer_artifact.__dict__)

    def model_evaluation(**kwargs):
        ti = kwargs["ti"]
        from creditcard.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact,ModelTrainerArtifact
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifact")
        data_transformation_artifact = DataTransformationArtifact(**(data_transformation_artifact))
        
        data_validation_artifact = ti.xcom_pull(task_ids="data_validation", key="data_validation_artifact")
        data_validation_artifact = DataValidationArtifact(**(data_validation_artifact))

        model_trainer_artifact = ti.xcom_pull(task_ids="model_trainer", key="model_trainer_artifact")
        model_trainer_artifact = ModelTrainerArtifact(**(model_trainer_artifact))
        
        model_eval_artifact = training_pipeline.start_model_evaluation(
            data_validation_artifact=data_validation_artifact, 
            data_transformation_artifact=data_transformation_artifact, 
            model_trainer_artifact=model_trainer_artifact)
        ti.xcom_push("model_eval_artifact", model_eval_artifact.__dict__)   
   
    def model_pusher(**kwargs):
        ti = kwargs["ti"]
        from creditcard.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifact")
        data_transformation_artifact = DataTransformationArtifact(**(data_transformation_artifact))
        
        model_trainer_artifact = ti.xcom_pull(task_ids="model_trainer", key="model_trainer_artifact")
        model_trainer_artifact = ModelTrainerArtifact(**(model_trainer_artifact))
        
        model_pusher_artifact = training_pipeline.start_model_pusher(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact)
        ti.xcom_push("model_pusher_artifact", model_pusher_artifact.__dict__)


    def push_data_to_s3(**kwargs):
        #import os
        bucket_name = os.getenv("BUCKET_NAME")
        #bucket_name = os.environ.get("BUCKET_NAME")
        #bucket_name = get_bucket_name_from_secrets()
        logging.info(f"bucket_name:{bucket_name}")
        artifact_folder = "/application/artifact"
        saved_model = "/application/saved_model"
        #os.system(f"aws s3 sync {artifact_folder} s3://{bucket_name}/artifact/")
        # os.system(f"aws s3 sync /application/artifact s3://{bucket_name}/artifact/")
        #os.system(f"aws s3 sync {saved_model} s3://{bucket_name}/saved_model/")
        # os.system(f"aws s3 sync /application/saved_model s3://{bucket_name}/saved_model/")
        #create_folder_s3(bucket_name=bucket_name,folder_name="artifact")
        #create_folder_s3(bucket_name=bucket_name,folder_name="saved_model")
        upload_in_s3(local_folder_path=artifact_folder ,bucket_name=bucket_name,s3_folder_prefix="artifact")
        upload_in_s3(local_folder_path=saved_model ,bucket_name=bucket_name,s3_folder_prefix="saved_model")
        logging.info("Done with upload a file in S3 bucket")




    




    # [START main_flow]

    get_dir_task = PythonOperator(
        task_id="get_dir",
        python_callable=get_dir,
    )

    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
    #### Ingestion task
    This task created train and test file
    """
    )

    data_validation_task = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation,
    )
    data_validation_task.doc_md = dedent(
        """\
    #### Validation task
    This task validate the data.
    """
    )

    data_transformation_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation,
    )
    data_transformation_task.doc_md = dedent(
        """\
    #### Transformation task
    This task transform the data.
    """
    )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """\
    #### Model trainer task
    This task train the model  on data.
    """
    )


    model_evaluation_task = PythonOperator(
        task_id="model_evalution",
        python_callable=model_evaluation,
    )
    model_evaluation_task.doc_md = dedent(
        """\
    #### Model eval task
    This task evaluate train the model  on data.
    """
    )
    push_data_to_s3_task = PythonOperator(task_id="push_data_to_s3",python_callable=push_data_to_s3)


    model_pusher_task = PythonOperator(
        task_id="model_pusher",
        python_callable=model_pusher,
    )
    model_pusher_task.doc_md = dedent(
        """\
    #### Model Pusher task
    Push the model
    """
    )

    get_dir_task >> data_ingestion_task >> data_validation_task  >> data_transformation_task >> model_trainer_task >> model_evaluation_task >> model_pusher_task >> push_data_to_s3_task

# [END main_flow]


