import os
import numpy as np
import librosa
from scipy.signal import wiener
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from collections import Counter

class AudioFilterEvaluator:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.methods = ['original', 'noisy', 'wiener', 'spectral', 'kalman', 'dae']
        self.features_dict = {method: [] for method in self.methods}
        self.labels = []
        self.le = LabelEncoder()
        
    def add_noise(self, signal, noise_factor=0.05):
        noise = np.random.randn(len(signal))
        return signal + noise_factor * noise
    
    def wiener_filter(self, signal):
        return wiener(signal)
    
    def spectral_subtraction(self, signal, n_fft=1024):
        stft = librosa.stft(signal, n_fft=n_fft)
        magnitude, phase = librosa.magphase(stft)
        noise_estimate = np.mean(magnitude[:, :5], axis=1, keepdims=True)
        magnitude_clean = np.maximum(magnitude - noise_estimate, 0)
        stft_clean = magnitude_clean * phase
        return librosa.istft(stft_clean, length=len(signal))
    
    def kalman_filter(self, signal, process_var=1e-4, measurement_var=0.05):
        n = len(signal)
        x = np.zeros(n)
        p = np.zeros(n)
        x[0] = signal[0]
        p[0] = 1.0
        
        for k in range(1, n):
            x_pred = x[k-1]
            p_pred = p[k-1] + process_var
            k_gain = p_pred / (p_pred + measurement_var)
            x[k] = x_pred + k_gain * (signal[k] - x_pred)
            p[k] = (1 - k_gain) * p_pred
        return x
    
    def extract_enhanced_features(self, signal, sr):
        features = []
        
        # Time-domain features
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.median(signal),
            np.max(signal),
            np.min(signal),
            librosa.feature.zero_crossing_rate(signal)[0].mean(),
            librosa.feature.rms(y=signal)[0].mean()
        ])
        
        # Frequency-domain features
        fft_vals = np.abs(fft(signal))
        features.extend([
            np.mean(fft_vals),
            np.std(fft_vals),
            np.median(fft_vals)
        ])
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
        
        features.extend([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth)
        ])
        
        # MFCCs (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        for i in range(13):
            features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
        
        return np.array(features)
    
    def process_audio_files(self):
        file_paths = []
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            print("No WAV files found!")
            return False
        
        print(f"Found {len(file_paths)} audio files to process")
        
        for file_path in tqdm(file_paths, desc="Processing audio files"):
            try:
                signal, sr = librosa.load(file_path, sr=None)
                signal = librosa.util.normalize(signal)
                
                label = os.path.basename(file_path).split('_')[0]
                
                # Generate noisy signal
                noisy_signal = self.add_noise(signal)
                
                # Apply all filters
                filtered_signals = {
                    'original': signal,
                    'noisy': noisy_signal,
                    'wiener': self.wiener_filter(noisy_signal),
                    'spectral': self.spectral_subtraction(noisy_signal),
                    'kalman': self.kalman_filter(noisy_signal)
                }
                
                # Extract features for each method
                for method in self.methods:
                    if method == 'dae':
                        self.features_dict[method].append(self.extract_enhanced_features(filtered_signals['wiener'], sr))
                    else:
                        self.features_dict[method].append(self.extract_enhanced_features(filtered_signals[method], sr))
                
                self.labels.append(label)
                
            except Exception as e:
                print(f"\nError processing {file_path}: {str(e)}")
                continue
        
        # Check and remove classes with insufficient samples
        label_counts = Counter(self.labels)
        min_samples = 2  # Minimum samples required per class
        valid_classes = [label for label, count in label_counts.items() if count >= min_samples]
        
        if len(valid_classes) < len(label_counts):
            print(f"\nRemoving {len(label_counts)-len(valid_classes)} classes with insufficient samples")
            
            # Filter out samples from invalid classes
            valid_indices = [i for i, label in enumerate(self.labels) if label in valid_classes]
            
            for method in self.methods:
                self.features_dict[method] = [self.features_dict[method][i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
        
        if not self.labels:
            print("Error: No valid samples left after filtering!")
            return False
        
        # Encode labels
        self.y = self.le.fit_transform(self.labels)
        return True
    
    def evaluate_methods(self):
        results = {}
        scaler = StandardScaler()
        
        # Check if we have enough samples for stratified CV
        min_samples_per_class = min(np.bincount(self.y))
        n_splits = min(5, min_samples_per_class)
        
        if n_splits < 2:
            print("\nWarning: Insufficient samples for stratified CV. Using simple train-test split.")
            for method in self.methods:
                X = np.array(self.features_dict[method])
                X_scaled = scaler.fit_transform(X)
                
                # Use stratification only if possible, otherwise don't
                stratify = self.y if min_samples_per_class >= 2 else None
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, self.y, test_size=0.3, random_state=42, stratify=stratify)
                
                rf = RandomForestClassifier(n_estimators=200, 
                                          max_depth=15,
                                          min_samples_split=5,
                                          class_weight='balanced',
                                          random_state=42)
                rf.fit(X_train, y_train)
                
                y_pred = rf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[method] = {
                    'accuracy': accuracy,
                    'model': rf,
                    'scaler': scaler
                }
                
                print(f"{method} Accuracy: {accuracy:.4f}")
        else:
            for method in self.methods:
                print(f"\nEvaluating {method} features...")
                X = np.array(self.features_dict[method])
                X_scaled = scaler.fit_transform(X)
                
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                accuracies = []
                
                for train_idx, test_idx in skf.split(X_scaled, self.y):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                    
                    rf = RandomForestClassifier(n_estimators=200, 
                                              max_depth=15,
                                              min_samples_split=5,
                                              class_weight='balanced',
                                              random_state=42)
                    rf.fit(X_train, y_train)
                    
                    y_pred = rf.predict(X_test)
                    accuracies.append(accuracy_score(y_test, y_pred))
                
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                
                rf_final = RandomForestClassifier(n_estimators=200, 
                                                max_depth=15,
                                                min_samples_split=5,
                                                class_weight='balanced',
                                                random_state=42)
                rf_final.fit(X_scaled, self.y)
                
                results[method] = {
                    'accuracy': mean_accuracy,
                    'std': std_accuracy,
                    'model': rf_final,
                    'scaler': scaler
                }
                
                print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return results
    
    def save_results(self, results):
        os.makedirs('models', exist_ok=True)
        
        for method in self.methods:
            joblib.dump(results[method]['model'], f'models/rf_{method}.pkl')
            joblib.dump(results[method]['scaler'], f'models/scaler_{method}.pkl')
        
        joblib.dump(self.le, 'models/label_encoder.pkl')
        
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(self.methods))
        accuracies = [results[m]['accuracy'] for m in self.methods]
        
        if 'std' in results[self.methods[0]]:
            errors = [results[m]['std'] for m in self.methods]
            plt.bar(x_pos, accuracies, yerr=errors, align='center', alpha=0.7, capsize=10)
        else:
            plt.bar(x_pos, accuracies, align='center', alpha=0.7)
        
        plt.xticks(x_pos, self.methods)
        plt.title('Speaker Recognition Accuracy by Filter Method')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig('accuracy_comparison.png')
        plt.show()

if __name__ == "__main__":
    dataset_dir = r'C:\Users\razas\Downloads\voice project\dataset'
    
    evaluator = AudioFilterEvaluator(dataset_dir)
    if evaluator.process_audio_files():
        results = evaluator.evaluate_methods()
        evaluator.save_results(results)
import os
import numpy as np
import librosa
from scipy.signal import wiener
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from collections import Counter

class AudioFilterEvaluator:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.methods = ['original', 'noisy', 'wiener', 'spectral', 'kalman', 'dae']
        self.features_dict = {method: [] for method in self.methods}
        self.labels = []
        self.le = LabelEncoder()
        
    def add_noise(self, signal, noise_factor=0.05):
        noise = np.random.randn(len(signal))
        return signal + noise_factor * noise
    
    def wiener_filter(self, signal):
        return wiener(signal)
    
    def spectral_subtraction(self, signal, n_fft=1024):
        stft = librosa.stft(signal, n_fft=n_fft)
        magnitude, phase = librosa.magphase(stft)
        noise_estimate = np.mean(magnitude[:, :5], axis=1, keepdims=True)
        magnitude_clean = np.maximum(magnitude - noise_estimate, 0)
        stft_clean = magnitude_clean * phase
        return librosa.istft(stft_clean, length=len(signal))
    
    def kalman_filter(self, signal, process_var=1e-4, measurement_var=0.05):
        n = len(signal)
        x = np.zeros(n)
        p = np.zeros(n)
        x[0] = signal[0]
        p[0] = 1.0
        
        for k in range(1, n):
            x_pred = x[k-1]
            p_pred = p[k-1] + process_var
            k_gain = p_pred / (p_pred + measurement_var)
            x[k] = x_pred + k_gain * (signal[k] - x_pred)
            p[k] = (1 - k_gain) * p_pred
        return x
    
    def extract_enhanced_features(self, signal, sr):
        features = []
        
        # Time-domain features
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.median(signal),
            np.max(signal),
            np.min(signal),
            librosa.feature.zero_crossing_rate(signal)[0].mean(),
            librosa.feature.rms(y=signal)[0].mean()
        ])
        
        # Frequency-domain features
        fft_vals = np.abs(fft(signal))
        features.extend([
            np.mean(fft_vals),
            np.std(fft_vals),
            np.median(fft_vals)
        ])
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
        
        features.extend([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth)
        ])
        
        # MFCCs (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        for i in range(13):
            features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
        
        return np.array(features)
    
    def process_audio_files(self):
        file_paths = []
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            print("No WAV files found!")
            return False
        
        print(f"Found {len(file_paths)} audio files to process")
        
        for file_path in tqdm(file_paths, desc="Processing audio files"):
            try:
                signal, sr = librosa.load(file_path, sr=None)
                signal = librosa.util.normalize(signal)
                
                label = os.path.basename(file_path).split('_')[0]
                
                # Generate noisy signal
                noisy_signal = self.add_noise(signal)
                
                # Apply all filters
                filtered_signals = {
                    'original': signal,
                    'noisy': noisy_signal,
                    'wiener': self.wiener_filter(noisy_signal),
                    'spectral': self.spectral_subtraction(noisy_signal),
                    'kalman': self.kalman_filter(noisy_signal)
                }
                
                # Extract features for each method
                for method in self.methods:
                    if method == 'dae':
                        self.features_dict[method].append(self.extract_enhanced_features(filtered_signals['wiener'], sr))
                    else:
                        self.features_dict[method].append(self.extract_enhanced_features(filtered_signals[method], sr))
                
                self.labels.append(label)
                
            except Exception as e:
                print(f"\nError processing {file_path}: {str(e)}")
                continue
        
        # Check and remove classes with insufficient samples
        label_counts = Counter(self.labels)
        min_samples = 2  # Minimum samples required per class
        valid_classes = [label for label, count in label_counts.items() if count >= min_samples]
        
        if len(valid_classes) < len(label_counts):
            print(f"\nRemoving {len(label_counts)-len(valid_classes)} classes with insufficient samples")
            
            # Filter out samples from invalid classes
            valid_indices = [i for i, label in enumerate(self.labels) if label in valid_classes]
            
            for method in self.methods:
                self.features_dict[method] = [self.features_dict[method][i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
        
        if not self.labels:
            print("Error: No valid samples left after filtering!")
            return False
        
        # Encode labels
        self.y = self.le.fit_transform(self.labels)
        return True
    
    def evaluate_methods(self):
        results = {}
        scaler = StandardScaler()
        
        # Check if we have enough samples for stratified CV
        min_samples_per_class = min(np.bincount(self.y))
        n_splits = min(5, min_samples_per_class)
        
        if n_splits < 2:
            print("\nWarning: Insufficient samples for stratified CV. Using simple train-test split.")
            for method in self.methods:
                X = np.array(self.features_dict[method])
                X_scaled = scaler.fit_transform(X)
                
                # Use stratification only if possible, otherwise don't
                stratify = self.y if min_samples_per_class >= 2 else None
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, self.y, test_size=0.3, random_state=42, stratify=stratify)
                
                rf = RandomForestClassifier(n_estimators=200, 
                                          max_depth=15,
                                          min_samples_split=5,
                                          class_weight='balanced',
                                          random_state=42)
                rf.fit(X_train, y_train)
                
                y_pred = rf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[method] = {
                    'accuracy': accuracy,
                    'model': rf,
                    'scaler': scaler
                }
                
                print(f"{method} Accuracy: {accuracy:.4f}")
        else:
            for method in self.methods:
                print(f"\nEvaluating {method} features...")
                X = np.array(self.features_dict[method])
                X_scaled = scaler.fit_transform(X)
                
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                accuracies = []
                
                for train_idx, test_idx in skf.split(X_scaled, self.y):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                    
                    rf = RandomForestClassifier(n_estimators=200, 
                                              max_depth=15,
                                              min_samples_split=5,
                                              class_weight='balanced',
                                              random_state=42)
                    rf.fit(X_train, y_train)
                    
                    y_pred = rf.predict(X_test)
                    accuracies.append(accuracy_score(y_test, y_pred))
                
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                
                rf_final = RandomForestClassifier(n_estimators=200, 
                                                max_depth=15,
                                                min_samples_split=5,
                                                class_weight='balanced',
                                                random_state=42)
                rf_final.fit(X_scaled, self.y)
                
                results[method] = {
                    'accuracy': mean_accuracy,
                    'std': std_accuracy,
                    'model': rf_final,
                    'scaler': scaler
                }
                
                print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return results
    
    def save_results(self, results):
        os.makedirs('models', exist_ok=True)
        
        for method in self.methods:
            joblib.dump(results[method]['model'], f'models/rf_{method}.pkl')
            joblib.dump(results[method]['scaler'], f'models/scaler_{method}.pkl')
        
        joblib.dump(self.le, 'models/label_encoder.pkl')
        
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(self.methods))
        accuracies = [results[m]['accuracy'] for m in self.methods]
        
        if 'std' in results[self.methods[0]]:
            errors = [results[m]['std'] for m in self.methods]
            plt.bar(x_pos, accuracies, yerr=errors, align='center', alpha=0.7, capsize=10)
        else:
            plt.bar(x_pos, accuracies, align='center', alpha=0.7)
        
        plt.xticks(x_pos, self.methods)
        plt.title('Speaker Recognition Accuracy by Filter Method')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig('accuracy_comparison.png')
        plt.show()

if __name__ == "__main__":
    dataset_dir = r'C:\Users\razas\Downloads\voice project\dataset'
    
    evaluator = AudioFilterEvaluator(dataset_dir)
    if evaluator.process_audio_files():
        results = evaluator.evaluate_methods()
        evaluator.save_results(results)