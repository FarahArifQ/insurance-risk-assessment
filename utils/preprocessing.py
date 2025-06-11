import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def preprocess_data(df, is_train=True):
    # Handle missing values (-1)
    df.replace(-1, np.nan, inplace=True)
    
    # Drop columns with high missingness (>60%)
    missing_perc = df.isnull().mean()
    cols_to_drop = missing_perc[missing_perc > 0.6].index.tolist()
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    # Fill remaining missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        elif df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Feature selection (only for training)
    if is_train:
        X = df.drop(['id', 'target'], axis=1)
        y = df['target']
        
        # Select top 20 features using ANOVA F-value
        selector = SelectKBest(f_classif, k=20)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()]
        selected_cols = selected_cols.tolist() + ['id', 'target']
        df = df[selected_cols]
    
    return df

def save_processed_data(train, test):
    train.to_csv('data/train_processed.csv', index=False)
    test.to_csv('data/test_processed.csv', index=False)