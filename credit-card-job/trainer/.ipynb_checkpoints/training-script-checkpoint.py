import argparse
import os
from google.cloud import storage, bigquery
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import requests
from google.auth import compute_engine
import joblib

def preprocessing(df):
    
    oEncode = OrdinalEncoder()
    ssEncode = StandardScaler()
    
    # Drop id and predicted score columns
    df = df.drop(['id', 'predicted_default_payment_next_month'], axis=1)
    
    # For pay, bill_amt and pay_amt columns, trailing number indicates months previous to current captured month
    # 0 = September 2005, 1 = August 2005 etc, 6 = April 2005 
    
    # Bill amount preprocess
    billAmountFeat = [name for name in df.columns.tolist() if 'bill_amt' in name]
    for feat in billAmountFeat:
        df[[feat]] = ssEncode.fit_transform(df[[feat]])
        
    # Pay amount preprocess
    payAmountFeat = [name for name in df.columns.tolist() if 'pay_amt' in name]
    for feat in payAmountFeat:
        df[[feat]] = ssEncode.fit_transform(df[[feat]])
        
    # Limit balance preprocess 
    df['limit_balance'] = ssEncode.fit_transform(df[['limit_balance']])
    
    # Age preprocess (binerize and convert to categorical)
    bin_edges = [0, 20, 30, 40, 50, 60, 70, 150]  # Define the age group boundaries
    bin_labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '>=70']
    df['age'] = pd.cut(df['age'], bins = bin_edges, labels = bin_labels, right=False)
    df[['age']] = oEncode.fit_transform(df[['age']]).astype(int)
        
#     Categorical features preprocess (sex, education_level, marital_status, pay_0 to pay_6)
#     Ordinal for education_level, marital_status, pay columns, one hot for sex
    payDelayedMonth = [name for name in df.columns.tolist() if 'pay_' in name if 'pay_amt' not in name]
    for feat in payDelayedMonth:
        df[[feat]] = oEncode.fit_transform(df[[feat]]).astype(int)
        df[[feat]] = df[[feat]].apply(lambda x: x.astype('category'))
    df[['education_level']] = oEncode.fit_transform(df[['education_level']]).astype(int)
    df[['marital_status']] = oEncode.fit_transform(df[['marital_status']]).astype(int)
    df[['education_level', 'marital_status', 'age']] = df[['education_level', 'marital_status', 'age']].apply(lambda x:x.astype('category'))
    
    df = pd.get_dummies(df, columns = ['sex'])

    # Reorder the dataframe so that labels at end
    df = df[[col for col in df if col not in ['default_payment_next_month']] + ['default_payment_next_month']]
    
    # Extract labels, remove from data dataframe
    dfLabels = df[['default_payment_next_month']].copy()
    df = df.drop(['default_payment_next_month'], axis = 1)
                                                                
    return df, dfLabels

def persist_trained_model(model, modelPath, X, Y, storageClient):
    model = model.fit(X, Y)
    joblib.dump(model, modelPath)

    modelDir = os.environ['AIP_MODEL_DIR']
    localStoragePath = os.path.join(modelDir, modelPath)
    blob = storage.blob.Blob.from_string(localStoragePath, client = storageClient)
    blob.upload_from_filename(modelPath)

def getData(dataURL, projectID) -> pd.DataFrame:
    
    client = bigquery.Client(project = projectID)
    sqlQuery = 'SELECT * FROM `{}`'.format(dataURL)

    df = client.query(sqlQuery).to_dataframe()
    return df
    # return pd.read_gbq(query, project_id = projectID, progress_bar_type = 'tqdm')

if __name__ == '__main__':
    # os.system('gcloud auth login')
    storageClient = storage.Client()
    PROJECT_ID_INT = os.environ["CLOUD_ML_PROJECT_ID"]

    PROJECT_ID = storageClient.project
    LOCATION = 'us-central1-a'
    REGION = LOCATION[:-2]
 
    BQ_DATA_URL = '{}.public_credit_card_data_sandbox.credit_card_default'.format(PROJECT_ID_INT)
    BUCKET_NAME = 'preprocessed_credit_card_data'
    BUCKET_URL = 'gs://{}'.format(BUCKET_NAME)
    BUCKET_STAGING_FOLDER_URL = 'gs://public-credit-card-data-output/vertex-pipeline'
    MODEL_PATH = 'model.joblib'

    # Get data from cloud bucket
    rawData = getData(BQ_DATA_URL, PROJECT_ID_INT)

    # Preprocess data
    preprocessedData, preprocessedLabels = preprocessing(rawData)

    # Train test split, and train model
    xTrain, xTest, yTrain, yTest = train_test_split(preprocessedData, preprocessedLabels, test_size = 0.20, random_state = 42)

    # Train model and ensure model persists, saved to local path
    persist_trained_model(SVC(), MODEL_PATH, xTrain, yTrain, storageClient)
