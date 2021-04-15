import pandas as pd
import regex as re
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def read_data(rel_path, verbose=False, **kwargs):
    """
    Generalized function that reads a xlsx, csv, json or html data structure into a pandas dataframe

    :argument rel_path: relative path to data to be read
    :argument kwargs:  parameters for pd.read_"extension" functions
    """
    filename, file_extension = os.path.splitext(rel_path)
    if verbose:
        print(os.path.basename(rel_path) + " will be read")
    if file_extension == ".xlsx": #when reading excel a usefull kwargs will be na_values = "dict of values to consider na"
        df = pd.read_excel(rel_path, **kwargs)
        if verbose:
            print("reading filetype "+file_extension)
    elif file_extension == ".csv":
        df = pd.read_csv(rel_path, **kwargs)
        if verbose:
            print("reading filetype "+file_extension)
    elif file_extension == ".json":
        df = pd.read_json(rel_path, **kwargs)
        if verbose:
            print("reading filetype "+file_extension)
    elif file_extension == ".html":
        df = pd.read_html(rel_path, **kwargs)
        if verbose:
            print("reading filetype "+file_extension)
    else:
        if verbose:
            print("Filetype not supported by function")
        df = 0

    if df.size != 0:
        if verbose:
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


def impute_and_encode(cat_features, num_features, imputer, one_hot=True):
    '''
    Creates preprocesser object that scales numeric features and onehotencodes categorical features
    
    Arguments
    ---------
    cat_features: list, list of categorical features
    num_features: list, list of numerical features
    imputer: dict, defines imputation method of SimpleImputer in Form {'categorical':{'strategy':'METHOD', 'fill_value'='METHOD'}, 'numerical':{'strategy':'METHOD'}}
    one_hot: bool, default True, if false prohibits one_hot_encoding
    
    Returns
    -------
    preprocessor: returns preprocesser object
    '''
    if num_features is not None:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=imputer['numerical']['strategy'], fill_value=imputer['numerical']['fill_value'])),
            ('scaler', StandardScaler())])

    if cat_features is not None:
        if one_hot:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=imputer['categorical']['strategy'], fill_value=imputer['categorical']['fill_value'])),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        else:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=imputer['categorical']['strategy'], fill_value=imputer['categorical']['fill_value']))])

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
    df[column] = df[column].astype('str')
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

def num_to_binary(df, column, cutoff = None, verbose=False):
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
    if column == 'rheumatoid_factor':
        df = extract_num(df, column, errors='coerce', verbose=False)
        df = extract_num(df, df.iloc[:, df.columns.get_indexer([column])+2].columns[0], errors='coerce', verbose=False) # select range column and extract_num
        df[column] = np.where(df[column] <= df.iloc[:, df.columns.get_indexer([column])+2].iloc[:, 0] , 0, 1)
        df[column].where(df.iloc[:, df.columns.get_indexer([column])+2].iloc[:,0].notna(),np.nan,inplace=True) # keeps NaN's
        df[column] = df[column].astype('category')
        
        if verbose:
            print(df[column].dtype)
            print(df[column].value_counts(dropna=False))
        
        return df
    # extract numerical values (strings with no digits = np.nan)
    elif(df[column].dtype != np.float64 or df[column].dtype != np.int64):
        df = extract_num(df, column, errors='coerce', verbose=False)
        df[column]= np.where(df[column].isna(), np.nan, np.where(df[column] <=cutoff, 0, 1))
    
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
def problem_columns(matches, desired_dtype, verbose=False):
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
            if verbose:
                print("\n", key, "is desired", value.dtype, "and will be popped from problematic list \n")
            matches.pop(key)
        else:
            if verbose: 
                print(key, "has unwanted dtype, keeping for transformation")
    return matches


#function to coerce columns to desired datatype
def coerce_then_problems(dframe, list_path, col_index_name, col_data_type_name, data_type, desired_dtype, verbose=False):
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
    if verbose:
        print(matches)
    matches = problem_columns(matches, desired_dtype)
    return matches

def preprocessing_neg_col_to_cat(df, columns, verbose=False):
    """
    Transform columns with majority of 'NEG' values into binary, categorical columns
    -----
    :param df: a pandas DataFrame
    :param columns: list of strings, containing column names in df
    :param verbose: bool, default False, prints value counts of transformed columns
    
    :return: input DataFrame with transformed columns
    """
    for col in columns:
        if col == 'anti-ccp_ab':
            df = num_to_binary(df, 'anti-ccp_ab', 20)
            df[col] = df[col].astype('bool')
            if verbose:
                print(df[col].value_counts(dropna=False))
        elif col == 'rheumatoid_factor':
            df = num_to_binary(df, 'rheumatoid_factor', verbose=verbose)
        else:
            df[col].replace(r'(see)', np.nan)
            df[col] = np.where(df[col].isna(), np.nan, np.where(df[col] =='NEG', 0, 1))
            df[col] = df[col].astype('str').astype('bool')
            if verbose:
                print(df[col].value_counts(dropna=False))

    return df

def drop_uom_and_range(df, verbose=False):
    """
    Drop all columns which names contain 'uom' or 'range'
    
    :param df: pandas DataFrame
    :param verbose: bool, default=False, prints list of remaining columns after transformation
    
    :return df: Returns input dataframe without columns containing str 'uom' or 'range'
    """
    df = df[df.columns.drop(list(df.filter(regex='uom|range')))]
    if verbose: 
        print(df.columns.tolist())

    return df

#iterate trough problem columns and extract numeric features
def iter_columns_extract_num(df):
    """

    :param df:
    :return:
    """
    for c in df.columns:
        #print(c)
        df = extract_num(df, str(c))
        #print(problem_df.columns.dtype)
        df = df.convert_dtypes()
        #print(df[c].dtypes)

    return df

#merge corrected columns onto dataframe
def merge_corrected(original_df, corrected_df):
    for c in corrected_df.columns:
        #print(c, df1[c])
        original_df[c] = corrected_df[c]

    return original_df

def range_var_to_cat(df, columns, verbose =False):
    """
    Transform features with associated range column to a categorical variable with the following categories:
     - 0 = 'below range'
     - 1 = 'in range'
     - 2 = 'above range'
    -----
    :param df: a pandas DataFrame
    :param columns: list of strings, containing column names in df
    :param verbose: bool, default False, prints value_count of columns
    
    :return: input DataFrame with transformed columns
    """
    for col in columns:
        if verbose:
            print(f'Current column: {col}')
        t = df.loc[:,col:].iloc[:,:3] # select range and uom of feature
        ran_col =  [i for i in t.columns if 'range' in i]
        expand_range = t[ran_col[0]].str.split('-', expand = True).rename(columns={0:'lower',1:'upper'}) # create column with upper and lower limit of range

        # handle special range2 case of '<0.80' entries
        expand_range.loc[expand_range.lower == '<0.80', 'upper'] = 0.8
        expand_range.replace('<0.80', 0.0, inplace=True)

        t['lower'], t['upper'] = expand_range.lower.astype('float'), expand_range.upper.astype('float')
        t[col] = t[col].astype('float') # makes sure all numeric columns are of same type (important for np.where)
        df[col] = np.where(t[col] <= t.lower, 0, np.where(t[col] >= t.upper, 2, 1))
        df[col].where(t[col].notna(),np.nan,inplace=True) # keeps NaN's
        df[col] = df[col].astype('category')
        
        if verbose:
            print(df[col].value_counts(dropna=False))
            
    return df

# TODO: Add description
def preprocessing_numeric(df, num_to_cat = False, verbose=False):
    """
    TODO: Describe function
    """
    list_path = "../data/col_names_and_data_type.xlsx"
    col_index_name = "new col name"
    col_data_type_name = "data_type"
    data_type = "numerical"

    desired_dtype = ["int64", "float64"]
    
    #return list of all columns with specific dtype
    num_columns = list_of_totype(list_path, col_index_name, col_data_type_name, data_type)

    # filter already dropped columns
    num_columns = list(set(num_columns) & set(df.columns.tolist()))
    # filter out columns with dtype category
    num_columns = [i for i in num_columns if df[i].dtype.name != 'category']
    #create dataframe with columns that contain a mix of strings and numerical values
    problem_df = coerce_then_problems(df, list_path, col_index_name, col_data_type_name, data_type, desired_dtype, verbose=False)
    problem_columns = list(problem_df)

    corrected_df = iter_columns_extract_num(problem_df)

    #foo = pipe.coerce_then_problems(df, list_path, col_index_name, col_data_type_name, data_type, desired_dtype)

    df = merge_corrected(df, corrected_df)
    
    if num_to_cat:
        # get list of columns with uom and range
        ranges_list = []
        list_cols = num_columns
    
        for _, i in enumerate(list_cols):
            if 'uom' in df.iloc[:, df.columns.get_indexer([i])+1].columns[0] \
            and 'range' in df.iloc[:, df.columns.get_indexer([i])+2].columns[0] \
            and 'range' in df.iloc[:, df.columns.get_indexer([i])-1].columns[0]:
                ranges_list.append(i)
            if _ >= len(list_cols)-2:
                break
        df = range_var_to_cat(df, ranges_list, verbose =verbose)
    
    return df

def preprocessing_cat(df):
    df.cat = df.cat.str.lower().str.strip().astype('category')
    if df.cat.isna().sum() == 1:
        df.cat = df.cat.fillna(value='not_uveitis')
    df.loc[df['cat'].str.contains('masquerade', case=False), 'cat'] = 'not_uveitis'
    df.drop(df[df.cat == 'scleritis'].index, inplace = True)
    
    return df

def preprocessing_specific(df):
    df.specific_diagnosis = df.specific_diagnosis.str.lower().astype('category')
    df.loc[df['specific_diagnosis'].str.contains('masquerade', case=False), 'specific_diagnosis'] = 'not_uveitis'

    count = df.specific_diagnosis.value_counts().reset_index().rename(columns={'index':'diagnosis','specific_diagnosis':'count'})
    diag_less_10 = count[count['count'] <= 10].diagnosis.tolist()
    df.specific_diagnosis = df.specific_diagnosis.replace({x:'other' for x in diag_less_10})
    df.specific_diagnosis = df.specific_diagnosis.astype('category')
    return df

def uom_fix(df, verbose=False):
    """
    Fixes 'mulitple uoms' per feature problem   
    Part of the code is from Riccard Nef: https://gitlab.fhnw.ch/riccard.nef/medicalchallenge/-/blob/master/NordStream.html

    -----
    :param df: a pandas DataFrame
    :param verbose: bool, default False, prints value_count of columns
    
    :return: input DataFrame with transformed columns
    """
    # replacing mg/dL with mg/L and changing value accordingly
    condition = df['uom2'[:]] == df['uom2'].value_counts().idxmax()
    df['c-reactive_protein,_normal_and_high_sensitivity'] = pd.to_numeric(df['c-reactive_protein,_normal_and_high_sensitivity'],errors='coerce')
    df['c-reactive_protein,_normal_and_high_sensitivity'].where(cond = condition, other = df['c-reactive_protein,_normal_and_high_sensitivity']*10.0, inplace = True)
    df['uom2'].where(cond = condition, other = df['uom2'].value_counts().idxmax(), inplace = True)
    if verbose:
        print("units left in uom2: ",df['uom2'].unique())

    #uom27
    condition = df['uom27'[:]] == df['uom27'].value_counts().idxmax()
    df['dna_double-stranded_ab'] = pd.to_numeric(df['dna_double-stranded_ab'],errors='ignore')
    try:
        other = df['dna_double-stranded_ab']*1000.0
    except TypeError:
        other = df['dna_double-stranded_ab']
        if verbose:
            print('Some values are not float')
    df['dna_double-stranded_ab'].where(cond = condition, other = other, inplace = True)
    df['uom27'].where(cond = condition, other = df['uom27'].value_counts().idxmax(), inplace = True)

    if verbose:
        print("units left in uom27: ",df['uom27'].unique())

    # change ranges
    df['range2'] = df['range2'].replace('<0.80', '0.00-8.00')
    df['range2'] = df['range2'].replace('0.020-0.800', '0.20-8.00')
    
    return df

def preprocessing_hepatitis(df, col=['hbc__ab', 'hbs__ag', 'hcv__ab'], verbose=False):
    for c in col:
        df[c] = df[c].str.lower()
        df.loc[df[c] == 'negative', c] = 0
        df.loc[df[c] == 'see note | positive result s/co ratio is >5.0.  confirmatory testing i', c] = 1
        df.loc[df[c] == 'see below | positive result s/co ratio is >5.0.  confirmatory testing', c] = 1
        df.loc[df[c] == 'reactive', c] = 1
        df.loc[df[c] == 'repeat reactive', c] = 1
        df.loc[df[c] == 'invalid result', c] = np.nan
        df.loc[df[c] == 'note:', c] = np.nan
        df[c] = df[c].astype('bool')
        if verbose:
            print(df[c].value_counts())
    return df

def preprocessing_inflammation(df, col = ['ac_abn_od_cells', 'ac_abn_os_cells', 'vit_abn_od_cells',
       'vit_abn_os_cells', 'vit_abn_od_haze', 'vit_abn_os_haze']):
    for c in col: 
        # replace 'C' (for missing) with NaN
        df[c] = df[c].replace('C',np.nan)
        df[c] = df[c].astype('float')
        df[c] = pd.Categorical(values=df[c], categories=df[c].unique().sort(), ordered=True)
    return df

def preprocessing_race(df):
    df.race = df.race.replace({'Race or Ethnic Group Data Not Provided by Source':'unknown', 
                               'Unknown Race':'unknown'})
    df.race = df.race.fillna(value='unknown')
    df.race = df.race.astype('category')
    assert df.race.isna().sum() == 0, 'Not all missing values are treated'
    return df

def preprocessing_gender_dtype(df):
    df.gender = df.gender.astype('category')
    return df

def define_uveitis(x):
    if x == 'not_uveitis':
        return False
    else:
        return True
    
def preprocessing_pipe(rel_path="../data/uveitis_data.xlsx", 
                       rename_path="../data/col_names_and_data_type.xlsx",
                       num_to_cat = False, 
                       nan_percentage=.5, 
                       drop_columns=['ehr_diagnosis', 'id'], 
                       verbose = False,
                       drop_filter = ['hla'], 
                       loc_approach = 'multi', 
                       save_as_csv = False,
                       binary_cat = False,
                       hotencode_gender = False,
                       neg_col_as_cat = ['anti-ccp_ab','anti-ena_screen','antinuclear_antibody','dna_double-stranded_ab', 'rheumatoid_factor']):
    """
    Preprocessing_pipe combines the preprocessing functions of the pipe.py script. It returns a cleaned dataset (note that missing values can still exist)
    that is ready to be piped into a ml-pipeline (see impute_and_encode-function in pipe.py)
    
    Arguments
    ---------
    :param rel_path:       str, relative path to the input data (default: path to uveitis_data.xlsx)
    :param rename_path:    str, relative path an excel file containing cleaned up column names (default: path to col_names_and_data_type.xlsx) 
    :param num_to_cat:     bool, default False, if True numeric features (with uoms and ranges) get encoded into a categorical feature 
                               with categories 0 = 'below range', 1 = 'in range', 2 = 'above range'. 
                               If false, numeric features will only get cleaned up (brought into uniform data type) 
    :param nan_percentage: float, in range 0 to 1, default 0.5 (aka 50%). Features with more than nan_percentage missing values will get droppep
    :param drop_columns:   list, a list containing column names to be dropped.
    :param drop_filter:    list, a list containing strings. If the name of a feature contains one of the strings it will be dropped.
    :param loc_approach:   if 'multi' (default): consolidates location feature into multiple locations ('anterior','posterior','pan...',etc.) 
                               elif 'binary': collapses location feature into categories 'posterior_segment', 'anterior_segment' 
    :param binary_cat:     bool, default False, if True adds uveitis feature where True =  'uveitis' and False = 'not_uveitis'
    :param neg_col_as_cat: list, a list of column names of features to transform into binary variables. 0 = 'Negative', 1 = 'Positive'
    :param save_as_csv:    bool, default True, if true saves the transformed dataframe named 'cleaned_uveitis_data' to data folder 
                                (Attention! If true, it is possible that already existing files get overwritten)
    
    Return
    ------
    :return:               input DataFrame with transformed columns
    """
    # load dataset
    df = read_data(rel_path=rel_path)
    
    df = (df.pipe(rename, path=rename_path) # rename columns
        .pipe(pd.DataFrame.applymap, lambda x: x.strip() if isinstance(x, str) else x) # strip leading or trailing whitespace

        # dropping columns
        .pipe(drop_nan_columns, nan_percentage=nan_percentage, verbose = verbose) # drop columns with above nan_percantage missing values
        .pipe(pd.DataFrame.drop, columns=drop_columns)
         )
          
    
        
    df = (df.pipe(preprocessing_gender_dtype)                              # change dtype from 'gender' to catgory
        .pipe(preprocessing_race)                                          # collapse 'race' feature
        .pipe(preprocessing_loc, approach=loc_approach, verbose=verbose)   # use approach ='binary' for binary classification
        .pipe(preprocessing_cat)                                           # collapes 'cat' feature
        .pipe(preprocessing_specific)                                      # collapse 'specific_diagnosis'
        .pipe(preprocessing_inflammation)                                  # transform columns that contain information about severeness of inlamation
        .pipe(preprocessing_hepatitis)                                     # clean and binarize hepatitis-columns
        .pipe(preprocessing_neg_col_to_cat, columns=neg_col_as_cat)        # transform columns into binary (negative, postive) features
        .pipe(preprocessing_numeric, num_to_cat=num_to_cat)                # clean up numeric and if num_to_cat true transform into categorical features
        .pipe(drop_uom_and_range, verbose=False)                           # drop 'uom' amd 'range' columns after use
        ) 
    
    if binary_cat:
        df['uveitis'] = df['cat'].apply(define_uveitis)
        df['uveitis'] = df['uveitis'].astype('bool')

    if hotencode_gender:
        df['gender'] = df['gender'].cat.codes

    for i in drop_filter:
        df = drop_via_filter(df, filter_str=i, verbose=verbose)

    if save_as_csv:
        df.to_csv('../data/cleaned_uveitis_data.csv')

    return df

