import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import wiener, get_window
import librosa
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function for Spectral Subtraction
def spectral_subtraction(noisy_signal, noise_estimate, window='hamming', alpha=1.0):
    win = get_window(window, len(noisy_signal))
    noisy_signal_win = noisy_signal * win
    noise_estimate_win = noise_estimate * win

    noisy_spectrum = fft(noisy_signal_win)
    noise_spectrum = fft(noise_estimate_win)

    noisy_magnitude = np.abs(noisy_spectrum)
    noise_magnitude = np.abs(noise_spectrum)

    cleaned_magnitude = noisy_magnitude - alpha * noise_magnitude
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0)

    cleaned_spectrum = cleaned_magnitude * np.exp(1j * np.angle(noisy_spectrum))
    cleaned_signal = np.real(ifft(cleaned_spectrum))

    return cleaned_signal

# Function for Kalman Filter
def kalman_filter(noisy_signal, A=1, H=1, Q=1e-5, R=0.1**2, initial_estimate=0, initial_error=1):
    n = len(noisy_signal)
    estimated_signal = np.zeros(n)
    estimate = initial_estimate
    error_covariance = initial_error

    for i in range(n):
        # Prediction step
        estimate = A * estimate
        error_covariance = A * error_covariance * A + Q

        # Update step
        kalman_gain = error_covariance * H / (H * error_covariance * H + R)
        estimate = estimate + kalman_gain * (noisy_signal[i] - H * estimate)
        error_covariance = (1 - kalman_gain * H) * error_covariance

        estimated_signal[i] = estimate

    return estimated_signal

# Denoising Autoencoder (DAE) Model
def build_dae(input_shape):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_shape[0], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train DAE and apply it to the signal
def apply_dae(noisy_signal, original_signal, epochs=50, batch_size=32):
    dae = build_dae((len(noisy_signal),))
    dae.fit(noisy_signal[np.newaxis, :], original_signal[np.newaxis, :],
            epochs=epochs, batch_size=batch_size, verbose=0)
    cleaned_signal = dae.predict(noisy_signal[np.newaxis, :])[0]
    return cleaned_signal

# Load and preprocess audio files
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    audio = audio / np.max(np.abs(audio))
    return audio, sr

# Generate features for classification
def extract_features(signal, sr):
    features = []
    
    # Time-domain features
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.median(signal))
    
    # Frequency-domain features
    fft_vals = np.abs(fft(signal))
    features.append(np.mean(fft_vals))
    features.append(np.std(fft_vals))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))
    
    return features

# Process audio and save waveform images
def process_and_plot_audio(file_path, output_dir):
    original_signal, sr = load_audio(file_path)
    noise = np.random.normal(0, 0.1, original_signal.shape)
    noisy_signal = original_signal + noise
    noise_estimate = noise

    # Apply filters
    wiener_filtered = wiener(noisy_signal)
    spectral_subtracted = spectral_subtraction(noisy_signal, noise_estimate)
    kalman_filtered = kalman_filter(noisy_signal)
    dae_filtered = apply_dae(noisy_signal, original_signal)

    # Create time axis
    t = np.arange(len(original_signal)) / sr
    
    # Plot and save waveform
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(t, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 2)
    plt.plot(t, noisy_signal, color='orange')
    plt.title('Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 3)
    plt.plot(t, wiener_filtered, color='green')
    plt.title('Wiener Filtered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 4)
    plt.plot(t, spectral_subtracted, color='purple')
    plt.title('Spectral Subtracted')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 5)
    plt.plot(t, kalman_filtered, color='blue')
    plt.title('Kalman Filtered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 6)
    plt.plot(t, dae_filtered, color='red')
    plt.title('DAE Filtered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    
    # Save plot
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f'{base_name}_waveforms.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved waveform plot: {output_path}")
    
    # Extract features for classification
    features = {
        'original': extract_features(original_signal, sr),
        'noisy': extract_features(noisy_signal, sr),
        'wiener': extract_features(wiener_filtered, sr),
        'spectral': extract_features(spectral_subtracted, sr),
        'kalman': extract_features(kalman_filtered, sr),
        'dae': extract_features(dae_filtered, sr)
    }
    
    return features, os.path.basename(file_path).split('_')[0]  # Assuming filename format "speaker_X_..."

# Main processing function
def main():
    dataset_dir = r'C:\Users\razas\Downloads\voice project\dataset'
    output_dir = os.path.join(dataset_dir, 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all WAV files
    file_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        print("No WAV files found!")
        return
    
    # Process files and collect features
    features_list = []
    labels_list = []
    
    for file_path in file_paths:
        print(f"\nProcessing: {file_path}")
        try:
            features, label = process_and_plot_audio(file_path, output_dir)
            # Add all filtered versions to our dataset
            for method in ['original', 'noisy', 'wiener', 'spectral', 'kalman', 'dae']:
                features_list.append(features[method])
                labels_list.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Prepare data for classification
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nClassification Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model
    model_path = os.path.join(output_dir, 'speaker_classifier_rf.pkl')
    joblib.dump(rf, model_path)
    print(f"\nSaved trained model to: {model_path}")

if __name__ == "__main__":
    main()