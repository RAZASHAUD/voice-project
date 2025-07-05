import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_evaluate():
    try:
        # Load data
        X = np.load("features.npy")
        y = np.load("labels.npy")
        
        # Check data
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data found in features.npy or labels.npy")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data - ensure at least 1 sample per class in test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        
        # Create custom cross-validation that ensures at least 2 samples per class in each fold
        min_samples = min(np.bincount(y_train))
        cv_splits = min(5, min_samples)  # Don't exceed the minimum number of samples
        
        if cv_splits < 2:
            raise ValueError(f"Some classes have only {min_samples} samples - too few for cross-validation")
            
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        # Simplified Random Forest parameters (reduced complexity)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }
        
        clf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        print("Training model...")
        clf.fit(X_train, y_train)
        
        # Save best model
        best_model = clf.best_estimator_
        joblib.dump(best_model, 'voice_classifier_rf.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        
        # Evaluation
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n=== Best Model Parameters ===")
        print(clf.best_params_)
        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Feature Importance
        print("\nTop 10 Important Features:")
        importances = best_model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        for idx in top_indices:
            print(f"Feature {idx}: {importances[idx]:.4f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you've run feature_extraction.py first")

if __name__ == "__main__":
    train_and_evaluate()