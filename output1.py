import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load filtered features
X = np.load("filtered_features.npy")
y = np.load("filtered_labels.npy")

# Encode labels (ignoring filter type for speaker ID)
speaker_ids = [label.split("_")[0] for label in y]
le = LabelEncoder()
y_encoded = le.fit_transform(speaker_ids)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train Random Forest with optimized parameters
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
joblib.dump(clf, "speaker_classifier_filtered.joblib")
joblib.dump(le, "label_encoder.joblib")