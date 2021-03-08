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
    df[column] = df[column].str.extract(r"(\d{1,}[.|,]\d{1,})", expand=False)

    # change feat to uniform uom (unit of measurment)
    df[column] = pd.to_numeric(df[column], errors=errors)
                    
    if verbose:
        print("Unique values after transformation:")
        print(df[column].unique())
    
    return df