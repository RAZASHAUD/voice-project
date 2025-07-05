import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import joblib

def evaluate_model():
    try:
        # Load the trained model and scaler
        model = joblib.load('voice_classifier_rf.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Load features and labels
        X = np.load("features.npy")
        y = np.load("labels.npy")
        
        # Check data
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data found in features.npy or labels.npy")
        
        # Normalize features
        X = scaler.transform(X)
        
        # Cross-validation accuracy
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-validation accuracy: {np.mean(cv_scores):.2%} (±{np.std(cv_scores):.2%})")
        
        # Train-test split evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.2%}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y), 
                    yticklabels=np.unique(y))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Feature importance visualization
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        plt.figure(figsize=(10, 6))
        plt.title('Top 20 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), indices)
        plt.xlabel('Relative Importance')
        plt.show()
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    evaluate_model()