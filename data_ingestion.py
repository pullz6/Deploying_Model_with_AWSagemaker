# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import pandas as pd
import kagglehub
import os

def loading_df(): 
    """This function is used to ingest the data from kaggle"""
    
    # Download latest version
    path = kagglehub.dataset_download("yasserh/kinematics-motion-data")

    print("Path to dataset files:", path)

    # List files in the directory
    files = os.listdir(path)
    print("Files in dataset:", files)

    # Finding the first CSV file
    csv_files = [file for file in files if file.endswith(".csv")]

    if csv_files:
        file_path = os.path.join(path, csv_files[0])  
        # Use the first CSV file found
        df = pd.read_csv(file_path)
    else:
        print("No CSV file found in the dataset directory.")
    
    #Returning the dataframe
    return df 

        
