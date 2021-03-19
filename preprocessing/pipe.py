import pandas as pd
import regex as re
import numpy as np
import os

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
    df[column] = df[column].str.extract(r"(\d+(?:\.\d+)?)", expand=False)

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

def preprocessing_loc(df, approach='multi', verbose= False):
    """
    Arguments
    ---------
    df: 
    approach:  if 'multi': consolidates location feature into multiple locations ('anterior','posterior','pan...',etc.)
               elif 'binary': collapses location feature into categories 'posterior_segment', 'anterior_segment'
    
    Return
    ------
    df:        df with collapsed loc-column
    """
    # assert approach parameter
    assert approach in ['multi','binary'], "approach parameter is not in list ['multi','binary']"
    
    df['loc'] = df['loc'].str.lower().str.strip()
    df['loc'] = df['loc'].replace({'pan':'panuveitis'})

    
    # collapse according to approach-parameter
    if approach == 'multi':
        pass
    elif approach == 'binary':
        df['loc'] = df['loc'].replace({'intermediate':'posterior_segment',
                                       'posterior':'posterior_segment',
                                       'panuveitis':'posterior_segment',
                                       'anterior':'anterior_segment',
                                       'scleritis':'anterior_segment'})
        
        # assert len(df['loc'].unique()) == 2, 'not all categories have been collapsed'

        
    df['loc'] = df['loc'].astype('category')
    
    if verbose:
        print('Categories: \n')
        print(df['loc'].value_counts())
    
    return df

def num_to_binary(df, column, cutoff):
    """
    Creates a binary feature (0 and 1 values) out of a numerical feature
    
    Arguments
    ---------
    df:       pd.dataframe, DataFrame
    column:   str, A string to select the column to transform
    cutoff:   numeric, values at or below the cutoff will be 0, otherwise 1
    
    Returns
    -------
    df:       pd.dataframe, DataFrame where 'column' is replaced with a binarized version of said 'column'
    """ 
    # extract numerical values (strings with no digits = np.nan)
    if(df[column].dtype != np.float64 or df[column].dtype != np.int64):
        df = extract_num(df, column, errors='coerce', verbose=False)
    df[column]= np.where(df[column].isna(), np.nan, np.where(df[column] <=20, 0, 1))
    return df


#function to return list of columns to convert to which data type (with choice) according to given list
def list_of_totype(list_path, col_index_name, col_data_type_name="data_type", data_type="numerical"):
    """
    Return list of columns, selected based on their data types from custom excel list
    -----
    :param list_path: path of excel document to read data type from
    :param col_index_name: index column to use (column name of data type)
    :param col_data_type_name: name of column where data type is stored
    :param data_type: type of data desired. Should be numerical, categorical, char or both (appelation for edge cases)

    :return: returns a list of column names with the desired data type
    """
    col = pd.read_excel(list_path, index_col = col_index_name)[col_data_type_name] #get content of column with name data_type
    col = col.reset_index()

    new_col = col[col_index_name].str.strip().str.lower()
    new_col = [c.replace(' ', '_') for c in new_col] # remove whitespace
    new_col = [re.sub(r"\([^()]*\)", "", c) for c in new_col] # remove all text in parantheses
    col[col_index_name] = new_col

    col_to_type = col.where(col.data_type == data_type).dropna()[col_index_name].tolist()
    return col_to_type


#this function is pretty useless alone
def problem_columns(matches, desired_dtype):
    """
    Return all columns and their content that couldn't be correctly coerced to a desired dtype
    -----
    :param matches: feeded from another function
    :param desired_dtype: a list or string that contains the desired data_types that were transformed correctly

    :return: returns a dataframe with all the problematic columns
    """
    for key, value in matches.iteritems():
        #print(key, value)
        if value.dtype in desired_dtype:
            print("\n", key, "is desired", value.dtype, "and will be popped from problematic list \n")
            matches.pop(key)
        else:
            print(key, "has unwanted dtype, keeping for transformation")
    return matches


#function to coerce columns to desired datatype
def coerce_then_problems(dframe, list_path, col_index_name, col_data_type_name, data_type, desired_dtype):
    """
    Coerce with convert_dtypes pandas function all columns from the list_of_totype. Those that fail to
    be converted into the desired dtype will be compiled into a dataframe for next steps
    -----
    :param dframe: dataframe to be checked and wrecked
    :param list_path: path of excel document to read data type from
    :param col_index_name: index column to use (column name of data type)
    :param col_data_type_name: name of column where data type is stored
    :param data_type: type of data desired. Should be numerical, categorical, char or both (appelation for edge cases)
    :param desired_dtype: a list or string that contains the desired data_types that were transformed correctly

    :return: a dataframe with values that couldn't be coerced to the desired data_type automagically via convert_dtypes()
    """

    item_filter = list_of_totype(list_path, col_index_name, col_data_type_name, data_type)
    matches = dframe.filter(items = item_filter).convert_dtypes()
    print(matches)
    matches = problem_columns(matches, desired_dtype)
    return matches


