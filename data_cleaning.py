from data_ingestion import * 
import pandas as pd 

def cleaning_df(): 
    df = loading_df()
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S:%f")
    df = df.drop(["username"], axis=1)
    return df