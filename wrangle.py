# Wrangling functions


################ Imports ################
import pandas as pd
import numpy as np
from env import host, password, user
import os
import scipy.stats as stats


from sklearn.model_selection import train_test_split

###################### Getting database Url ################
def get_db_url(db_name, user=user, host=host, password=password):
    """
        This helper function takes as default the user host and password from the env file.
        You must input the database name. It returns the appropriate URL to use in connecting to a database.
    """
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url


######################### get generic data #########################
def get_any_data(database, sql_query):
    '''
    put in the query and the database and get the data you need in a dataframe
    '''

    return pd.read_sql(sql_query, get_db_url(database))

################ train test split helper function ################
def banana_split(df):
    '''
    args: df
    This function take in the telco_churn data data acquired by aquire.py, get_telco_data(),
    performs a split.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=713)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=713)
    print(f'train --> {train.shape}')
    print(f'validate --> {validate.shape}')
    print(f'test --> {test.shape}')
    return train, validate, test



################ Scaler helper function ################
def my_scaler(train, validate, test, col_names, scaler, scaler_name):
    
    '''
    This function takes in the train validate and test dataframes, columns you want to scale (as a list), a scaler (i.e. MinMaxScaler(), with whatever paramaters you need),
    scaler_name as a string.
    col_names: list of columns to scale
    Scaler_name, should be what you want in the name of your new dataframe columns.
    Adds columns to the train validate and test dataframes. 
    Outputs scaler for doing inverse transforms.
    Ouputs a list of the new column names (what you can use to create the X_train).
    
    example: min_max_scaler, scaled_cols_list = my_scaler(train, validate, test, MinMaxScaler(), 'scaled_min_max')
    
    '''
    
    #create the scaler (input here should be minmax scaler)
    mm_scaler = scaler
    
    # make empty list for return
    scaled_cols_list = []
    
    # loop through columns in col names
    for col in col_names:
        
        #fit and transform to train, add to new column on train df
        train[f'{col}_{scaler_name}'] = mm_scaler.fit_transform(train[[col]]) 
        
        #df['col'].values.reshape(-1, 1)
        
        #transform cols from validate and test (only fit on train)
        validate[f'{col}_{scaler_name}']= mm_scaler.transform(validate[[col]])
        test[f'{col}_{scaler_name}']= mm_scaler.transform(test[[col]])
        
        #add new column name to the list that will get returned
        scaled_cols_list.append(f'{col}_{scaler_name}')
    
    #confirmation print
    print('Your scaled columns have been added to your train validate and test dataframes.')
    
    #returns scaler, and a list of column names that can be used in X_train, X_validate and X_test.
    return scaler, scaled_cols_list   

    
#################################### get ZILLOW data ####################################
def get_zillow_data():
    '''
    This function reads in Zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    sql_query = '''
                SELECT p.parcelid AS parcel_id,
                taxvaluedollarcnt AS tax_value,
                bathroomcnt AS bathroom_cnt,
                bedroomcnt AS bedroom_cnt,
                calculatedfinishedsquarefeet AS sqft_calculated,
                poolcnt AS has_pool,
                garagecarcnt AS garage_car_count,
                p.fips AS fips,
                taxamount AS tax_amount,
                transactiondate AS transaction_date
                FROM properties_2017 AS p
                JOIN predictions_2017 AS pred ON p.`parcelid` = pred.`parcelid`
                WHERE p.`propertylandusetypeid` IN (261) 
                	AND pred.`transactiondate` BETWEEN '2017-05-01' AND '2017-08-31';
                '''
    if os.path.isfile('zillow_data.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = pd.read_sql(sql_query, get_db_url('zillow'))
        
        # Cache data
        df.to_csv('zillow_data.csv')

    return df

# old query sql_query = ''' SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
   # FROM properties_2017
   # WHERE propertylandusetypeid = 261; '''

#################################### Look at nulls ####################################

# I saw this on the afore mentioned kaggle site. This is the credit that author gave.
# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 
# One of the best notebooks on getting started with a ML problem.

def missing_values_table(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values, 
    and the percent of that column that has missing values
    '''
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values.")
        
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


#################################### handle NaNs ####################################

def handle_NaNs(df):
    '''
    This function handles the NaN values for the Zillow data (could be used for other dataframes). 
    Takes in a dataframe and returns a dataframe with the unneeded rows dropped.
    '''
    # drop the duplicated rows
    df = df.drop_duplicates(keep = 'first')
    
    # drop any rows that have NaN values
    df = df.drop(df[df.isna().any(axis=1)].index)
    
    return df

#################################### drop the cols ####################################
def drop_the_cols(df):
    '''
    This function takes in the Zillow dataframe from get_zillow_data and gets rid of the rows with NaN values, 
    The rows with a bathroom count of 0.0
    and returns the data frame with the rows dropped.
    '''
    
    # drop NaNs
    df = df.drop(df[df.isna().any(axis=1)].index)
    
    # drop bathroom count of 0
    df = df.drop(df[df.bathroom_cnt == 0].index)
    
    return df

#################################### remove outliers ####################################

def remove_outlier(df):
    '''
    This function takes in a dataframe
    It outputs a the dataframe with the outliers that have a Z score above 3 or below -3 removed
    '''
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df

#################################### pools and garages ####################################
def pool_and_garage(df):
    '''
    This function fixes the NaNs in these two columns (changes them to 0s). 
    In the end two boolian/categorical, values has_pool and has_garage.
    '''
    df['has_pool'] = df.has_pool.fillna(value=0)

    df.garage_car_count.fillna(value=0, inplace=True)

    df['has_garage'] = (df.garage_car_count != 0).astype(int)

    df = df.drop(columns = 'garage_car_count')

    return df


#################################### Function to get Zillow Data ####################################

def wrangle_zillow():
    '''
    This function handels getting the data from the zillow database and getting rid of the unneeded rows.
    It returns the dataframe ready to work with.
    Uses other helper functions in wrangle.py to get this done. 
    '''

    df = get_zillow_data()

    df = pool_and_garage(df)

    df = drop_the_cols(df)

    df['transaction_date'] = pd.to_datetime(df.transaction_date)

    # Remove outliers from columns that need to be removed
    df_outliers = remove_outlier(df[['sqft_calculated', 'bedroom_cnt', 'bathroom_cnt']])

    # only put back the ones that had outliers in it 
    df = df[df.index.isin(df_outliers.index)]

    return df



