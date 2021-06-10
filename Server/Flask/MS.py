# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#!pip3 install matplotlib
#!pip3 install sklearn
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline  

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn import model_selection #import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn import metrics  # mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.feature_selection import SelectFromModel, RFECV
import Config 
import EDA
#binary classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn import model_selection

# %% [markdown]
# ### Load Data:

# %%
# load training data prepared previously
# Load training data prepared previuosly
filepath =Config.data_wrangling
df = pd.read_csv(filepath)
df.fillna(0,inplace=True)
df.head()

# %% [markdown]
# ### Regression Modelling:
# %% [markdown]
# Segment training and test data into features dataframe and labels series.  
# 
# To make it easy to train models on different set of features, a variable to hold the set of features required was used to subset the original dataframes

# %%
#Prepare data for regression model
features =EDA.L


X_df = df[features]
y_df = df['labels.ttf']


# %%
def get_regression_metrics(model, actual, predicted):
    
    """Calculate main regression metrics.
    
    Args:
        model (str): The model name identifier
        actual (series): Contains the test label values
        predicted (series): Contains the predicted values
        
    Returns:
        dataframe: The combined metrics in single dataframe
    
    
    """
    regr_metrics = {
                        'Root Mean Squared Error' : metrics.mean_squared_error(actual, predicted)**0.5,
                        'Mean Absolute Error' : metrics.mean_absolute_error(actual, predicted),
                        'R^2' : metrics.r2_score(actual, predicted),
                        'Explained Variance' : metrics.explained_variance_score(actual, predicted)
                   }

    #return reg_metrics
    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
    df_regr_metrics.columns = [model]
    return df_regr_metrics

# %% [markdown]
# Using the above functions let us model and evaluate some regression algorithms

# %%
ttf_models =[linear_model.LinearRegression(),linear_model.Lasso(alpha=0.001),linear_model.Ridge(alpha=0.01),DecisionTreeRegressor(max_depth=7, random_state=123),RandomForestRegressor(n_estimators=100, max_features=3, max_depth=4, n_jobs=-1, random_state=1)]
ttf_models_names =["LinearRegression","LASSO","RIDGE","DecisionTree","RandomForest"]


# %%
def consideredData(df):
    return df[df['lastCycleReached']==False]
def toPredictData(df):
    return df[df['lastCycleReached']==False]
def tryModelRegression(model,df,features,label):
    data = consideredData(df)
    X_df = data[features]
    Y_df = data[label]
    X_train,X_test,Y_train,Y_test =train_test_split(X_df,Y_df,test_size=0.33,random_state=42)
    trainedModel = model.fit(X_train,Y_train)
    Y_test_predicted = trainedModel.predict(X_test)
    modelScore = get_regression_metrics(trainedModel,Y_test,Y_test_predicted)
    return modelScore
    


# %%
score_ttf = pd.DataFrame()
for model in ttf_models :
    score_ttf = pd.concat([score_ttf,tryModelRegression(model,df,EDA.L,'labels.ttf')],axis=1)
score_ttf.columns = ttf_models_names
score_ttf.reset_index(inplace=True)
score_ttf


# %%
def selectmodel(score,modelnames):
    m = score[modelnames].abs().mean().max()
    avg = score[modelnames].abs().mean()
    selectedModel = list(avg [avg == m].index)[0]
    return selectedModel

# %% [markdown]
# ### Binary Classifcation:

# %%
bnc_models = [LogisticRegression(random_state=123),DecisionTreeClassifier(random_state=123),RandomForestClassifier(n_estimators=50, random_state=123),SVC(kernel='rbf', random_state=123),LinearSVC(random_state=123),GaussianNB()]
bnc_models_names=['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','SVC','LinearSVC','GaussianNB']


# %%
def bin_class_metrics(modelname, y_test, y_pred):
    
    """Calculate main binary classifcation metrics, plot AUC ROC and Precision-Recall curves.
    
    Args:
        model (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        y_score (series): Contains the predicted scores
        print_out (bool): Print the classification metrics and thresholds values
        plot_out (bool): Plot AUC ROC, Precision-Recall, and Threshold curves
        
    Returns:
        dataframe: The combined metrics in single dataframe
        
        
    """
      
    binclass_metrics = {
                        'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                        'Precision' : metrics.precision_score(y_test, y_pred),
                        'Recall' : metrics.recall_score(y_test, y_pred),
                        'F1 Score' : metrics.f1_score(y_test, y_pred),
                       }

    df_metrics = pd.DataFrame.from_dict(binclass_metrics, orient='index')
    df_metrics.columns = [modelname]  


    

    return  df_metrics


# %%
def tryModelBNC(model,modelname,df,features,label,thresh=0.5):
    data = consideredData(df)
    X_df = df[features]
    Y_df = df[label]
    X_train,X_test,Y_train,Y_test =train_test_split(X_df,Y_df,test_size=0.33,random_state=42)
    trainedModel = model.fit(X_train,Y_train)
    Y_test_predicted = trainedModel.predict(X_test)
    Y_test_predicted[Y_test_predicted<thresh] = 0
    Y_test_predicted[Y_test_predicted>=thresh] = 1
    modelScore = bin_class_metrics(modelname, Y_test, Y_test_predicted)
    return modelScore


# %%
score_bnc = pd.DataFrame()
for i in range(len(bnc_models)) :

    score_bnc = pd.concat([score_bnc,tryModelBNC(bnc_models[i],bnc_models_names[i],df,EDA.L,'labels.bnc')],axis=1)
score_bnc.columns = bnc_models_names
score_bnc.reset_index(inplace=True)
score_bnc


# %%
score_bnc.mean(axis=1).max()

# %% [markdown]
# ### Mutli Classifcation:

# %%
def multiclass_metrics(modelname, y_test, y_pred):
    
    """Calculate main multiclass classifcation metrics
    Args:
        model (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        
        
    Returns:
        dataframe: The combined metrics in single dataframe
        
  
    
    """
    multiclass_metrics = {
                            'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                            'macro F1' : metrics.f1_score(y_test, y_pred, average='macro'),
                            'micro F1' : metrics.f1_score(y_test, y_pred, average='micro'),
                            'macro Precision' : metrics.precision_score(y_test, y_pred,  average='macro'),
                            'micro Precision' : metrics.precision_score(y_test, y_pred,  average='micro'),
                            'macro Recall' : metrics.recall_score(y_test, y_pred,  average='macro'),
                            'micro Recall' : metrics.recall_score(y_test, y_pred,average='macro'),
                        }
    
    df_metrics = pd.DataFrame.from_dict(multiclass_metrics, orient='index')
    df_metrics.columns = [model]

   
    
    return df_metrics


# %%
def tryModelMCC(model,modelname,df,features,label,thresh=0.5):
    data = consideredData(df)
    X_df = df[features]
    Y_df = df[label]
    X_train,X_test,Y_train,Y_test =train_test_split(X_df,Y_df,test_size=0.33,random_state=42)
    trainedModel = model.fit(X_train,Y_train)
    Y_test_predicted = trainedModel.predict(X_test)
    Y_test_predicted = np.round(Y_test_predicted)
    modelScore = multiclass_metrics(modelname, Y_test, Y_test_predicted)
    return modelScore 


# %%
score_mcc = pd.DataFrame()
for i in range(len(bnc_models)) :

    score_mcc = pd.concat([score_mcc,tryModelMCC(bnc_models[i],bnc_models_names[i],df,EDA.L,'labels.bnc')],axis=1)
score_mcc.columns = bnc_models_names
score_mcc.reset_index(inplace=True)
score_mcc


# %%
SelectedModels = [selectmodel(score_ttf,ttf_models_names),selectmodel(score_bnc,bnc_models_names),selectmodel(score_mcc,bnc_models_names)]
SelectedModels


# %%
SelectedModelsindexs = [ttf_models_names.index(SelectedModels[0]),bnc_models_names.index(SelectedModels[1]),bnc_models_names.index(SelectedModels[2])]
SelectedModelsindexs


# %%
def RunModels(models,df,features):
    datatrain = consideredData(df)
    dataToPredict = toPredictData(df)
    X_train = datatrain[features]
    #ttf
    Y_train_ttf = datatrain['labels.ttf']
    trainedModel_ttf = ttf_models[models[0]].fit(X_train,Y_train_ttf)
    dataToPredict['labels.ttf']= trainedModel_ttf.predict(dataToPredict[features])
    #bnc 
    Y_train_bnc = datatrain['labels.bnc']
    trainedModel_bnc = bnc_models[models[1]].fit(X_train,Y_train_bnc)
    dataToPredict['labels.bnc'] = trainedModel_bnc.predict(dataToPredict[features])
    #mcc 
    Y_train_mcc = datatrain['labels.mcc']
    trainedModel_mcc = bnc_models[models[2]].fit(X_train,Y_train_mcc)
    dataToPredict['labels.mcc']= trainedModel_mcc.predict(dataToPredict[features])
    return dataToPredict
    


# %%
cols = []
for k in df.columns.tolist():
    if (('id.' in k) or ('labels.' in k)):
        cols.append(k)
predictions =RunModels(SelectedModelsindexs,df,EDA.L)[cols]
predictions


