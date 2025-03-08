from data_ingestion import * 
import pandas as pd 

def cleaning_df(): 
    """This function is used to clean the data after calling data ingestion"""
    #Calling for data ingestion
    df = loading_df()
    #Conveting date and time that is in object format into datetime format. 
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S:%f")
    #Dropping the username as it is required here. 
    df = df.drop(["username"], axis=1)
    #Returning the dataframe
    return df