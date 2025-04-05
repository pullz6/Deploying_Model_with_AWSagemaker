# Deploying_Model_with_AWSagemaker

This project is to understand and provide a template on how to take a model in Mlflow to deployment with AWS Sagemaker. We will be using a Kinematic Movement Dataset from Kaggle, which is used to predict activite from the phone's sensors. 

---

## Table of Contents

1. [Project Overview]
2. [Installation]
4. [Usage]
5. [Examples]
6. [Project Structure]
7. [Contributing]
8. [License]
9. [Acknowledgments]

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


```

---

## Usage

### Running the Application

Step 1 - Create ENV
Step 2 - Download all neccessary libaries
Step 3 - Run run_training.py file with python run_training.py 
Step 4 - To view the MlFlow dashboard on port 500 - mlflow ui --port 5000
Step 5 - Upload model artifact to S3 
Step 6 - Create Docker file 
Step 7 - Upload to ECR
Step 8 - Create sagemaker endpoint

### Configuration

Please ensure to set up your AWS credentials with AWS configure. 


---


## Project Structure

```bash
project-root/
│
├── check_kaggle.py/     # Accesing the dataset from Kaggle
├── data_ingestion.py/   # Faciliating the data ingestion from the check_kaggle.py
├── data_cleaning.py/    # Cleaning the data after the data ingestion.
├── log_df.py/           # Log the cleaned dataframe into MLFlow. 
├── tests/               # Unit and integration tests
├── README.md            # Project documentation
└── requirements.txt     # Project dependencies

```


---

## Contributing

Guidelines for contributing to the project:

- How to submit issues, bug reports, or feature requests.
- Guidelines for pull requests, including coding standards and testing.

---


---

## Acknowledgments

Shout out to resources, libraries, people, or tutorials that helped you build the project.
