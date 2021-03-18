import pandas as pd
import regex as re

def read_data(rel_path, **kwargs):
    """
    Generalized function that reads a xlsx, csv, json or html data structure into a pandas dataframe

    :argument rel_path: relative path to data to be read
    :argument kwargs:  parameters for pd.read_"extension" functions
    """
    filename, file_extension = os.path.splitext(rel_path)
    print(os.path.basename(rel_path) + " will be read")
    if file_extension == ".xlsx": #when reading excel a usefull kwargs will be na_values = "dict of values to consider na"
        df = pd.read_excel(rel_path, **kwargs)
        print("reading filetype "+file_extension)
    elif file_extension == ".csv":
        df = pd.read_csv(rel_path, **kwargs)
        print("reading filetype "+file_extension)
    elif file_extension == ".json":
        df = pd.read_json(rel_path, **kwargs)
        print("reading filetype "+file_extension)
    elif file_extension == ".html":
        df = pd.read_html(rel_path, **kwargs)
        print("reading filetype "+file_extension)
    else:
        print("Filetype not supported by function")
        df = 0

    if df.size != 0:
        print("\nSuccessfully created table with ", df.size, "values and loaded as df")
        print("The table is",df.shape[1], "wide and",df.shape[0],"long \n")

    return df

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


def preprocessing(cat_features, num_features, imputer):
    '''
    Creates preprocesser object that scales numeric features and onehotencodes categorical features
    
    Arguments
    ---------
    cat_features: list, list of categorical features
    num_features: list, list of numerical features
    imputer: dict, defines imputation method of SimpleImputer in Form {'categorical':{'strategy':'METHOD', 'fill_value'='METHOD'}, 'numerical':{'strategy':'METHOD'}}
    
    Returns
    -------
    df: df with encoded numeric and categorical features
    '''
    if num_features is not None:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=imputer['numeric']['strategy'])),
            ('scaler', StandardScaler())])

    if cat_features is not None:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    if cat_features is None and num_features is not None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features)])
    elif cat_features is not None and num_features is  None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, cat_features)])
    elif cat_features is not None and num_features is not None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)])
    return preprocessor

def extract_num(df, column, errors='coerce', verbose=False):
    """
    Extracts numerical values out of a column and returns the DataFrame with said column in numerical dtype
    
    Arguments
    ---------
    df: pandas.core.frame.DataFrame, Pandas DataFrame
    column: str, Name of column whos numerical values should be extracted
    errors: str, Determines how the function should handle errors (i.e nan's etc.), handled by pandas.to_numeric
            - If 'raise', then invalid parsing will raise an exception.
            - If 'coerce', then invalid parsing will be set as NaN.
            - If 'ignore', then invalid parsing will return the input.
        
    Returns
    -------
    df: pandas.core.frame.DataFrame
    """
    if verbose:
        print("Unique values before transformation:")
        print(df[column].unique())
    
    # filter out decimal numbers of column
    df[column] = df[column].astype('string')
    df[column] = df[column].str.extract(r"(\d{1,}[.|,]\d{1,}|\d*)", expand=False)

    # change feat to uniform uom (unit of measurment)
    df[column] = pd.to_numeric(df[column], errors=errors)
                    
    if verbose:
        print("Unique values after transformation:")
        print(df[column].unique())
    
    return df

def drop_nan_columns(df, nan_percentage, verbose = False):
    """
    Drop collumns if nan or missing values >= to nan_percentage
    
    Arguments
    ---------
    df:             pandas DataFrame
    nan_percentage: float, in range [0,1]
    verbose:        bool, default False, prints list of collumns that have been dropped if true
    
    
    Return
    ------
    df:             pandas DataFrame without collumns that have nan_percentage missing values
    """
    assert 0 < nan_percentage <= 1, "nan_percentage not between 0 and 1"
    
    # get list of collumns and corresponding ratio of missing values
    percent_missing = (df.isnull().sum() / len(df)).reset_index().rename(columns={0:'ratio'})
    
    #remove uom and range lists (these collumns are important for later preprocessing)
    searchfor = ['uom', 'range']
    percent_missing = percent_missing[~percent_missing['index'].str.contains('|'.join(searchfor))]
    
    # get list of collumns which have more than nan_percantage of missing values
    drop_list = percent_missing.loc[percent_missing.ratio >= nan_percentage, 'index'].tolist()
    
    if verbose:
        print('The following columns have been removed from the dataset:\n')
        print(percent_missing.loc[percent_missing.ratio >= nan_percentage].sort_values(by='ratio', ascending = False))
        
    # drop columns
    df = df.drop(drop_list, axis=1)
    
    return df

def drop_via_filter(df, filter_str ,verbose = False):
    """
    Drop collumns if nan or missing values >= to nan_percentage
    
    Arguments
    ---------
    df:             pandas DataFrame
    filter_str:     str, String that filters for a subset of colums to drop. eg. 'range' drops all columns containing the string 'range'
    verbose:        bool, default False, prints list of collumns that have been dropped if true
    
    Return
    ------
    df:             pandas DataFrame without collumns that have nan_percentage missing values
    """
    ranges = df.filter(like=filter_str).columns # get list of columns with range 
    df = df.drop(ranges, axis=1)
    
    if verbose:
        print('Dropped the following columns:\n')
        print(ranges)
        
    return df