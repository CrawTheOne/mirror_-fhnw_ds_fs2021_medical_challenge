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
