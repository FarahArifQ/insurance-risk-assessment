import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import time

# Removed circular import

def load_optimized_data():
    """Load the preprocessed optimized data"""
    train = pd.read_csv('data/train_optimized.csv')
    return train

def train_models():
    # Load optimized data
    train = load_optimized_data()
    
    X = train.drop(['id', 'target'], axis=1)
    y = train['target']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models with optimized parameters
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=500,
            class_weight='balanced',
            solver='liblinear',
            penalty='l1',
            C=0.1
        ),        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            class_weight='balanced',
            n_jobs=-1  # Use all cores
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=10,
            weights='distance',
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'ANN': MLPClassifier(
            hidden_layer_sizes=(30,),
            max_iter=200,
            early_stopping=True,
            alpha=0.01
        )
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probas = model.predict_proba(X_val)[:,1]
        
        training_time = time.time() - start_time
        
        results[name] = {
            'accuracy': accuracy_score(y_val, preds),
            'f1': f1_score(y_val, preds),
            'roc_auc': roc_auc_score(y_val, probas),
            'training_time': training_time,
            'model': model
        }
        
        print(f"{name} Results:")
        print(f"- Accuracy: {results[name]['accuracy']:.4f}")
        print(f"- F1 Score: {results[name]['f1']:.4f}")
        print(f"- ROC AUC: {results[name]['roc_auc']:.4f}")
        print(f"- Training Time: {training_time:.2f} seconds")
      # Select best model based on accuracy (must be >90%) and F1 score
    valid_models = {name: data for name, data in results.items() 
                   if data['accuracy'] >= 0.90}  # Filter models with >90% accuracy
    
    if valid_models:
        best_model_name = max(valid_models, key=lambda x: valid_models[x]['f1'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
        print(f"Training Time: {results[best_model_name]['training_time']:.2f}s")
    else:
        print("\nNo model achieved the required 90% accuracy threshold")
    
      # Save best model, feature columns, and metrics
    joblib.dump(best_model, 'model/model.pkl')
    joblib.dump(X.columns.tolist(), 'model/train_columns.pkl')
    
    # Save metrics from best model
    metrics = {
        'accuracy': results[best_model_name]['accuracy'],
        'f1': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc']
    }
    joblib.dump(metrics, 'model/metrics.pkl')
    print("\nModel, feature columns, and metrics saved successfully")    # Save metrics from best model
    metrics = {
        'accuracy': results[best_model_name]['accuracy'],
        'f1': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc']
    }
    joblib.dump(metrics, 'model/metrics.pkl')

if __name__ == "__main__":
    train_models()