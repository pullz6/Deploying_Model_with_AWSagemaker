# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import pandas as pd
import kagglehub
import os

def loading_df(): 
    # Download latest version
    path = kagglehub.dataset_download("yasserh/kinematics-motion-data")

    print("Path to dataset files:", path)

    # List files in the directory
    files = os.listdir(path)
    print("Files in dataset:", files)

    # Assuming the dataset contains a CSV file, find the first CSV file
    csv_files = [file for file in files if file.endswith(".csv")]

    if csv_files:
        file_path = os.path.join(path, csv_files[0])  # Use the first CSV file found
        df = pd.read_csv(file_path)
        #print("DataFrame loaded successfully!")
        #print(df.head())  # Display the first few rows
    else:
        print("No CSV file found in the dataset directory.")
        
    return df 

        
