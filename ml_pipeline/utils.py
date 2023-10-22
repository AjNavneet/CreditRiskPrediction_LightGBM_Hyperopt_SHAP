# Importing libraries
import pandas as pd
import numpy as np




# Function to read and drop unnecessary columns
def process_data(path, drop_columns):
    """Read data, drop columns and do processing
    Parameters
    ----------
    path : String
    drop_columns : List
    
    Returns
    -------
    df :  DataFrame
    """
    try:
        df = pd.read_csv(path).drop(columns = drop_columns)
    except Exception as e:
        print(e)
    else:
        return df


# Function to split the data 
def data_split(df):
    """Split data in train, val, hold_out
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    train :  DataFrame,
    val :  DataFrame,
    hold_out :  DataFrame
    """
    try:
        train = df[df.yearmo<=202203]
        val = df[df.yearmo==202204]
        hold_out = df[df.yearmo==202205]
    except Exception as e:
        print(e)

    else:
        return train.reset_index(drop = True), val.reset_index(drop = True), hold_out.reset_index(drop = True)



