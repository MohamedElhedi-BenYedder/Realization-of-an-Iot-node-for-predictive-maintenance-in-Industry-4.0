# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#!pip3 install requests
#!pip3 install pandas 


# %%
import json as js
import requests as rqs
import pandas as pd
import numpy as np
import Config 


# %%
### Data Source ###
apiUrl = Config.api_server_main_url
url=apiUrl+'/engineCycle'
engines = apiUrl+'/engine'
r_engines = rqs.get(engines)
data_engines = pd.json_normalize(r_engines.json())

# %% [markdown]
# ### Load Data:

# %%
r = rqs.get(url)
data = pd.json_normalize(r.json())
data

# %% [markdown]
# ### Data Columns
# 
# •	__engine__: | engine.id : is the engine ID, ranging from 1 to Ne | engine.description : is the description of the engine | engine.state : True if the engine reach the last cycle else False |    
# •	__cycle__: per engine sequence, starts from 1 to the cycle number where failure had happened    
# •	__settings__: engine operational settings  
# •	__sensors__: sensors measurements  
# 

# %%
df = data[data['lastCycleReached']==False]
df.fillna(0,inplace=True)
df.columns

# %% [markdown]
# There are Ne engines. each engine has between 1 to Nc cycles.
# The last cycle for each engine represents the cycle when failure had happened.

# %%
# check the data types
df.dtypes
isinstance(df,pd.DataFrame)

# %% [markdown]
# All data columns are numeric.

# %%
# check for NaN values

df.isnull().sum()

# %% [markdown]
# No missing values. This is a clean dataset!
# now let us add some features to smooth the sensors reading: rolling average and rolling standard deviation.
# %% [markdown]
# ### Feature Extraction:
# %% [markdown]
# Create helper function to create features based on smoothing the time series for sensors by adding rolling mean and rolling standard deviation

# %%

def add_features(df_in, rolling_win_size,columns_to_treat):
    
    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.
    
    Args:
            df_in (dataframe)     : The input dataframe to be proccessed (training or test) 
            rolling_win_size (int): The window size, number of cycles for applying the rolling function
        
    Reurns:
            dataframe: contains the input dataframe with additional rolling mean and std for each sensor
    
    """
    
    av_cols = [nm+'__av' for nm in columns_to_treat]
    sd_cols = [nm+'__sd' for nm in columns_to_treat]
    min_cols =[nm+'__min' for nm in columns_to_treat]
    max_cols =[nm+ '__max' for nm in columns_to_treat]
    
    df_out = pd.DataFrame()
    
    ws = rolling_win_size
    
    #calculate rolling stats for each engine (engine.id)
    
    for m_id in pd.unique(df_in['id.engine.id']):
    
        # get a subset for each engine sensors
        df_engine = df_in[df_in['id.engine.id'] == m_id]
        df_sub = df_engine[columns_to_treat]

    
        # get rolling mean for the subset
        av = df_sub.rolling(ws, min_periods=1).mean()
        av.columns = av_cols
    
        # get the rolling standard deviation for the subset
        sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)
        sd.columns = sd_cols

        # get rolling rolling max for the subset
        max = df_sub.rolling(ws, min_periods=1).max()
        max.columns = max_cols
        
        # get the rolling standard deviation for the subset
        min = df_sub.rolling(ws, min_periods=1).min().fillna(0)
        min.columns = min_cols
    
        # combine the two new subset dataframes columns to the engine subset
        new_ftrs = pd.concat([df_engine,av,sd,min,max], axis=1)
    
        # add the new features rows to the output dataframe
        df_out = pd.concat([df_out,new_ftrs])
        
    return df_out

# %% [markdown]
# create helper function to add the regression and classification labels to the training data

# %%

def find_labels(df_in, period):
    
    """Add regression and classification labels to the training data.

        Regression label: label.ttf (time-to-failure) = each cycle# for an engine subtracted from the last cycle# of the same engine
        Binary classification label: label.bnc = if ttf is <= parameter period then 1 else 0 (values = 0,1)
        Multi-class classification label: label.mcc = 2 if ttf <= 0.5* parameter period , 1 if ttf<= parameter period, else 2
        
      Args:
          df_in (dataframe): The input training data
          period (int)     : The number of cycles for TTF segmentation. Used to derive classification labels
          
      Returns:
          dataframe: The input dataframe with regression and classification labels added
          
    """
    
    #create regression label
    
    #make a dataframe to hold the last cycle for each enginge in the dataset
    df_max_cycle = pd.DataFrame(df_in.groupby(['id.engine.id','id.maintenanceIndex'])['id.cycle'].max())
    df_max_cycle.reset_index(inplace=True)
    df_max_cycle.columns = ['id.engine.id','id.maintenanceIndex', 'lastCycle']
    
    #add time-to-failure ttf as a new column - regression label
    df_in = pd.merge(df_in, df_max_cycle, on=['id.engine.id','id.maintenanceIndex'])
    df_in['labels.ttf'] = df_in['lastCycle'] - df_in['id.cycle']
    #df_in.drop(['lastCycleReached'], axis=1, inplace=True)
    
    #create binary classification label
    df_in['labels.bnc'] = df_in['labels.ttf'].apply(lambda x: 1 if x <= period else 0)
    
    #create multi-class classification label
    df_in['labels.mcc'] = df_in['labels.ttf'].apply(lambda x: 2 if x <= period/2 else 1 if x <= period else 0)
    
    return df_in
    

# %% [markdown]
# create helper function to add the regression and classification labels to the training data
# %% [markdown]
# With the help of these functions, let us prepare training and test data by adding features and labels
# %% [markdown]
# ### Prepare the Data:

# %%
# choose parameters
period = 1
columns_to_treat = [k for k in df.columns if 'sensors' in k]
columns_to_treat


# %%
# add extracted features to training data
df_fx = add_features(df, period,columns_to_treat)
df_fx.head()


# %%
#add labels to training data using period of 30 cycles for classification

df_labels = find_labels (df_fx, period)
df_labels.head()


# %%
df_labels.dtypes

# %% [markdown]
# Rolling average, rolling standard deviation, regression labels, and classification labels have been added to the data.  
# 
# Let us save the dataframe for later use in data exploration and modeling phases.

# %%
# save the training data to csv file for later use

df_labels.to_csv(Config.data_wrangling, index=False)


# %%



