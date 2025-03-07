from data_ingestion import * 
import pandas as pd 

df = loading_df()
print(df.info())
df["date"] = pd.to_datetime(df["date"])
df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S:%f")
print(df.info())