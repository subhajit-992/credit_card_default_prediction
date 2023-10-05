## Description:
•	The "Credit Card Default Prediction" repository houses a system designed to identify clients who are likely to default on their credit card    payments in the upcoming month. This project employs ML algorithms to analyse relevant financial data, enabling early detection of potential defaults and allowing for proactive measures to be taken. This predictive model serves as a valuable tool for financial institutions in managing and mitigating credit risk effectively.
## Dataset:
•	Past financial data of credit card holder.
•	Data provide personal, financial, historical information about credit card holder


## Install
### Requirements:
•	Python 3.11
•	Pandas
•	numpy
•	seaborn
•	matplotlib
•	scikit-learn
•	scipy
•	imblearn
•	xgboost
•	dill
•	PyYAML
•	cassandra-driver
•	apache-airflow
•	boto3


## Setup:
### Docker setup in EC2:
•	sudo apt-get update -y
•	sudo apt-get upgrade
•	sudo apt-get install awscli
•	curl -fsSL https://get.docker.com -o get-docker.sh
•	sudo sh get-docker.sh
•	sudo usermod -aG docker ubuntu
•	newgrp docker
### Run Self-Hosted runner :
•	In GitHub action you need to run Self-hosted runner in ubuntu(Linux).
 
## Demo Video:
https://drive.google.com/file/d/1W04N1OCaCwD9TlM1G8c0XvpDUGbLnZXw/view?usp=drive_link

## Run in local system: 
docker run -p 8080:8080 -v %cd%\airflow\dags:/application/airflow/dags creditcard_fault_prediction:latest

## Contributor:
•	Subhajit Das 
•	Indrani Das
