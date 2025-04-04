# Deploying_Model_with_AWSagemaker

This project is to understand and provide a template on how to take a model in Mlflow to deployment with AWS Sagemaker. 

---

## Table of Contents

1. [Project Overview](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
2. [Features](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
3. [Installation](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
4. [Usage](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
5. [Examples](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
6. [Project Structure](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
7. [Contributing](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
8. [License](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)
9. [Acknowledgments](https://www.notion.so/readme-md-Template-in-Github-12c81ba51b198002bc79e96f749cafd6?pvs=21)

---

## Project Overview

### About

This project is an end to end model training to deployment pipeline. 

- We will develop several versions of a model
- We will also develop a model with Stacking ensemble method. 
- Explain the problem it addresses and why it’s useful.
- Mention any key tools, libraries, or frameworks used.

### Screenshots or Demos

If available, add images or animated GIFs that showcase the application. This adds visual appeal and context.

---

## Features

- List out the key features of your project.
- Focus on unique functionalities, user experience highlights, or specific innovations.

---

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

This project can be cloned, but it is highly recommend to use it as a guideline. 


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

Explain any environment variables or configuration steps:

- For example, set up `.env` files, database settings, or API keys.

---

## Examples


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

Provide brief explanations for each major directory/file if the structure is complex.

---

## Contributing

Guidelines for contributing to the project:

- How to submit issues, bug reports, or feature requests.
- Guidelines for pull requests, including coding standards and testing.

---

## License

Specify the license type (e.g., MIT, Apache, etc.) and include a link to the full license file.

---

## Acknowledgments

Shout out to resources, libraries, people, or tutorials that helped you build the project.
