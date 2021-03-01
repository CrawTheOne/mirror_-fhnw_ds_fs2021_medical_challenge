import pandas as pd
import regex as re

def rename(df, path):
    '''
    Renames columns according to a given list (from Excel)
    
    Arguments
    ---------
    df: df, Original DataFrame
    path: str, Path to excel-file with new column names
    
    Returns
    -------
    df: df, Returns original DataFrame with new column names
    '''
    # get new column names
    col = pd.read_excel(path)['new col name']
    
    col = col.str.strip().str.lower() 
    col = [c.replace(' ', '_') for c in col] # remove whitespace
    col = [re.sub(r"\([^()]*\)", "", c) for c in col] # remove all text in parantheses
    
    # return df with new column names
    df = df.rename(columns=dict(zip(df.columns,col)))
    
    return df