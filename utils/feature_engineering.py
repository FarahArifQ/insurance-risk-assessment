import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
import joblib
import os

def load_raw_data():
    """Load raw datasets"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def basic_cleaning(df, is_train=True):
    """Handle missing values and basic cleaning"""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Replace -1 with NaN
    df = df.replace(-1, np.nan)
    
    # Drop columns with high missingness (>60%)
    missing_perc = df.isnull().mean()
    cols_to_drop = missing_perc[missing_perc > 0.6].index.tolist()
    df = df.drop(columns=cols_to_drop)
    
    # Fill remaining numerical missing values with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    medians = {col: df[col].median() for col in num_cols}
    df = df.fillna(medians)
    
    return df

def remove_low_variance_features(df, is_train=True):
    """Remove features with low variance"""
    cols_to_keep = ['id']
    if is_train:
        cols_to_keep.append('target')
    
    features = [col for col in df.columns if col not in cols_to_keep]
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df[features])
    
    # Get columns to keep
    selected_features = df[features].columns[selector.get_support()]
    cols_to_keep += selected_features.tolist()
    
    return df[cols_to_keep]

def select_top_features(df, k=20):
    """Select top k features using mutual information"""
    if 'target' not in df.columns:
        return df
    
    X = df.drop(['id', 'target'], axis=1)
    y = df['target']
    
    # Select top features
    selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    
    # Get selected columns
    selected_cols = X.columns[selector.get_support()]
    cols_to_keep = ['id', 'target'] + selected_cols.tolist()
    
    return df[cols_to_keep]

def analyze_feature_correlation(df):
    """Analyze and remove highly correlated features"""
    features = [col for col in df.columns if col not in ['id', 'target']]
    if len(features) < 2:
        return df
    
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    cols_to_keep = [col for col in df.columns if col not in to_drop]
    
    return df[cols_to_keep]

def save_processed_data(train, test):
    """Save processed datasets"""
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    train.to_csv('data/train_optimized.csv', index=False)
    test.to_csv('data/test_optimized.csv', index=False)
    print("Optimized datasets saved successfully")
    
    # Save the list of selected features for prediction
    selected_features = [col for col in train.columns if col not in ['id', 'target']]
    joblib.dump(selected_features, 'model/selected_features.pkl')

def process_data():
    """Full data processing pipeline"""
    train, test = load_raw_data()
    
    # Clean both datasets
    train = basic_cleaning(train, is_train=True)
    test = basic_cleaning(test, is_train=False)
    
    # Apply same transformations to both datasets
    train = remove_low_variance_features(train, is_train=True)
    test_features = [col for col in train.columns if col not in ['id', 'target']]
    test = test[['id'] + test_features]
    
    # Feature selection only on training data
    train = select_top_features(train)
    test_features = [col for col in train.columns if col not in ['id', 'target']]
    test = test[['id'] + test_features]
    
    # Remove correlated features
    train = analyze_feature_correlation(train)
    test_features = [col for col in train.columns if col not in ['id', 'target']]
    test = test[['id'] + test_features]
    
    # Save processed data
    save_processed_data(train, test)

if __name__ == "__main__":
    process_data()