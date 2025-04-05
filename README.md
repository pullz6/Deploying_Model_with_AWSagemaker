# Deploying_Model_with_AWSagemaker

This project is to understand and provide a template on how to take a model in Mlflow to deployment with AWS Sagemaker. We will be using a Kinematic Movement Dataset from Kaggle, which is used to predict activite from the phone's sensors. 

---

## Table of Contents

1. [Project Overview]
2. [Installation]
3. [Usage]
4. [Project Structure]
5. [Acknowledgments]

---

## Project Overview

### About

This project is an end to end model training to deployment pipeline, below objectives are achieved: 

- Creating Linear Regression, XGBoost and a stacked model with a Linear regressor and the XGBoost models.
- Testing and evalution of these models.
- Deploying the model with Amazon Sagemaker. 


## Installation

### Prerequisites

1. Python
2. Pandas
3. SKlearn
4. XGBoost
5. Docker
6. AWS - Sagemaker
7. AWS - ECR
8. AWS - S3

### Setup

This project can be cloned, but it is highly recommend to use it as a guideline. The sets are listed under "Running the Application"

## Usage

### Running the Application

1. Step 1 - Create ENV
2. Step 2 - Download all neccessary libaries
3. Step 3 - Run run_training.py file with python run_training.py 
4. Step 4 - To view the MlFlow dashboard on port 500 - mlflow ui --port 5000
5. Step 5 - Upload model artifact to S3
6. Step 6 - Create Docker file
7. Step 7 - Upload to ECR
8. Step 8 - Create sagemaker endpoint

### Configuration

Please ensure to set up your AWS credentials with 'AWS configure'. 


## Project Structure

```bash
project-root/
│
├── check_kaggle.py/     # Accesing the dataset from Kaggle
├── data_ingestion.py/   # Faciliating the data ingestion from the check_kaggle.py
├── data_cleaning.py/    # Cleaning the data after the data ingestion.
├── log_df.py/           # Log the cleaned dataframe into MLFlow.
├── model_tuning.py/     # Parameter search for XGBoost model's parameter.
├── model_stacking.py/   # The training of all the models as well as model evaluation.
├── run_training.py/     # Running the training pipeline.
├── run_inference.py/    # Running the inference pipeline.
├── deploy_sagemaker_initialrun.py/ # Running deployment at the first time.
├── deploy_sagemaker.py/ # Running the deployment after first time, this replaces the model in the endpoint.
├── dockerfile/          # The docker file of the model selected
└── README.md            # Details about the project

```


## Acknowledgments

Shout out to mlflow, aws sagemaker documentations. 
