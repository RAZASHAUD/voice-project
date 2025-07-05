import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

def evaluate_filters(features_csv='filtered_audio/filtered_features.csv'):
    # Load data
    df = pd.read_csv(features_csv)
    
    # Prepare results storage
    results = []
    feature_cols = [col for col in df.columns if col not in ['speaker', 'file', 'filter']]
    
    # Scale features
    scaler = StandardScaler()
    
    for filter_name in df['filter'].unique():
        print(f"\nEvaluating {filter_name} filter...")
        
        # Filter data
        filter_df = df[df['filter'] == filter_name]
        X = filter_df[feature_cols].values
        y = filter_df['speaker'].values
        
        # Scale features
        X = scaler.fit_transform(X)
        
        # Split data (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        
        # Cross-validation setup
        unique_classes, counts = np.unique(y_train, return_counts=True)
        n_splits = min(5, min(counts))  # Adjust based on smallest class
        
        if n_splits < 2:
            print(f"Skipping {filter_name} - not enough samples per class")
            continue
            
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Train model with cross-validation
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        )
        
        # Fit model
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results.append({
            'filter': filter_name,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        })
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
        disp.plot(xticks_rotation=90)
        plt.title(f"Confusion Matrix - {filter_name}")
        plt.tight_layout()
        plt.savefig(f"{filter_name}_confusion_matrix.png")
        plt.close()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('filter_performance.csv', index=False)
    
    # Plot comparison
    results_df.plot(x='filter', y='accuracy', kind='bar', legend=False)
    plt.title('Filter Performance Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('filter_comparison.png')
    
    return results_df

if __name__ == "__main__":
    results = evaluate_filters()
    print("\nFilter Performance Summary:")
    print(results)