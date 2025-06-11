# model/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import sys
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import gc
from sklearn.utils import shuffle

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import preprocess_data

# Load and preprocess data more efficiently
print("\nLoading and preprocessing data...")
df_train = pd.read_csv(r'D:\Farah Arif\AILAB\AI_PROJECT\data\train.csv')
X, y = preprocess_data(df_train)
del df_train
gc.collect()

# Split data into training and validation sets (80-20 split) using stratified sampling
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
del X, y
gc.collect()

# Calculate class distribution
class_counts = np.bincount(y_train)
print(f"\nOriginal class distribution: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")

# Calculate class weights
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
print(f"\nClass weights: {class_weights}")

# Create SMOTE instance with better parameters for severe imbalance
print("\nApplying SMOTE sampling...")
smote = SMOTE(
    sampling_strategy=0.5,  # Increased ratio for better minority representation
    random_state=42,
    n_jobs=4  # Parallel processing for SMOTE
)

# Process in smaller chunks if needed
chunk_size = 100000
n_chunks = len(X_train) // chunk_size + 1

X_resampled_list = []
y_resampled_list = []

for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(X_train))
    
    if end_idx <= start_idx:
        break
        
    print(f"\nProcessing chunk {i+1}/{n_chunks}...")
    X_chunk = X_train[start_idx:end_idx]
    y_chunk = y_train[start_idx:end_idx]
    
    X_res_chunk, y_res_chunk = smote.fit_resample(X_chunk, y_chunk)
    X_resampled_list.append(X_res_chunk)
    y_resampled_list.append(y_res_chunk)
    
    del X_chunk, y_chunk, X_res_chunk, y_res_chunk
    gc.collect()

# Combine resampled chunks
X_train_resampled = np.vstack(X_resampled_list)
y_train_resampled = np.concatenate(y_resampled_list)

# Clean up
del X_resampled_list, y_resampled_list
gc.collect()

# Shuffle the combined resampled data
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

# Print new class distribution
resampled_class_counts = np.bincount(y_train_resampled)
print(f"\nResampled class distribution: Class 0: {resampled_class_counts[0]}, Class 1: {resampled_class_counts[1]}")

# Logistic Regression configuration optimized for imbalanced data
classifiers = {
    'Logistic Regression': (
        LogisticRegression(
            random_state=42,
            max_iter=2000,
            solver='saga',
            n_jobs=4,
            class_weight='balanced'  # Set class weights directly
        ),
        {
            'C': [0.001, 0.01, 0.1],  # Focusing on stronger regularization
            'penalty': ['l2'],
            'l1_ratio': [0.5]  # Add elastic net mixing parameter
        }
    )
}

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)

# Dictionary to store results
results = {}

print("\nTraining and evaluating multiple models...")
for name, (classifier, param_grid) in classifiers.items():
    print(f"\nTraining {name}...")
    # Perform GridSearchCV with F1 scoring for better balance between precision and recall
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=3,  # Reduced cross-validation folds
        scoring='f1',
        n_jobs=4,  # Limit number of parallel jobs
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train_scaled, y_train_resampled)
    
    # Make predictions
    y_val_pred = grid_search.predict(X_val_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'best_params': grid_search.best_params_,
        'model': grid_search.best_estimator_
    }
    
    print(f"{name} - Validation Accuracy: {accuracy:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))

# Find the best model
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest performing model: {best_model_name}")
print(f"Best validation accuracy: {best_accuracy:.4f}")

# Use the best model for final predictions
print("\nTraining the final model...")
clf = best_model

# Load test data (without target)
df_test = pd.read_csv(r'D:\Farah Arif\AILAB\AI_PROJECT\data\test.csv')
X_test = preprocess_data(df_test)

# Predictions (if test set has target, you'd compare. If not, just save predictions)
y_pred = clf.predict(X_test)

# Create results directory if it doesn't exist
results_dir = r'D:\Farah Arif\AILAB\AI_PROJECT\results'
os.makedirs(results_dir, exist_ok=True)

# Save predictions (optional)
pd.DataFrame({'id': df_test['id'] if 'id' in df_test.columns else range(len(y_pred)), 'target': y_pred}).to_csv(
    os.path.join(results_dir, 'test_predictions.csv'), index=False
)

# Evaluate on training set
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred) * 100
print("\n=== Model Performance ===")
print(f"✅ Training Accuracy: {train_accuracy:.2f}%")

# Evaluate on validation set
y_val_pred = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred) * 100
print(f"✅ Validation Accuracy: {val_accuracy:.2f}%")

# Print detailed classification report for validation set
print("\n=== Validation Set Classification Report ===")
print(classification_report(y_val, y_val_pred))

# Check for overfitting
if train_accuracy - val_accuracy > 5:
    print("\n⚠️ Warning: Model might be overfitting (training accuracy is significantly higher than validation accuracy)")

# Save model
model_dir = r'D:\Farah Arif\AILAB\AI_PROJECT\model'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(clf, os.path.join(model_dir, 'model.pkl'))

print("✅ Model trained and saved at:", model_dir)
