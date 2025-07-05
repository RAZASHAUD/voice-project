import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import wiener, get_window
import librosa
import csv
from sklearn.decomposition import NMF

# Set up environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def add_noise(signal, noise_factor=0.05):
    """Add random noise to the signal"""
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

def wiener_filter(signal, mysize=None):
    """Apply Wiener filter to the signal"""
    return wiener(signal, mysize=mysize)

def spectral_subtraction(signal, n_fft=1024):
    """Apply spectral subtraction noise reduction"""
    stft = librosa.stft(signal, n_fft=n_fft)
    magnitude, phase = librosa.magphase(stft)
    
    # Estimate noise from the first few frames
    noise_estimate = np.mean(magnitude[:, :5], axis=1, keepdims=True)
    
    # Subtract noise estimate with flooring at zero
    magnitude_clean = np.maximum(magnitude - noise_estimate, 0)
    
    # Reconstruct the signal
    stft_clean = magnitude_clean * phase
    return librosa.istft(stft_clean, length=len(signal))

def kalman_filter(signal, process_var=1e-4, measurement_var=0.05):
    """Simplified Kalman filter implementation"""
    n = len(signal)
    x = np.zeros(n)  # Estimated state
    p = np.zeros(n)  # Estimation error covariance
    
    # Initial guesses
    x[0] = signal[0]
    p[0] = 1.0
    
    for k in range(1, n):
        # Prediction
        x_pred = x[k-1]
        p_pred = p[k-1] + process_var
        
        # Update
        k_gain = p_pred / (p_pred + measurement_var)
        x[k] = x_pred + k_gain * (signal[k] - x_pred)
        p[k] = (1 - k_gain) * p_pred
    
    return x

def dae_denoise(signal, n_components=10, max_iter=500):
    """Denoise using NMF (simplified DAE approach)"""
    stft = librosa.stft(signal)
    magnitude, phase = librosa.magphase(stft)
    
    # Apply NMF with increased iterations
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=max_iter)
    W = model.fit_transform(magnitude)
    H = model.components_
    magnitude_clean = W @ H
    
    # Reconstruct the signal
    stft_clean = magnitude_clean * phase
    return librosa.istft(stft_clean, length=len(signal))

def save_results_to_csv(results, t, filename):
    """Save all signals to a CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        headers = ['time'] + list(results.keys())
        writer.writerow(headers)
        
        # Write data
        for i in range(len(t)):
            row = [t[i]] + [results[key][i] if key != 'sr' else results[key] for key in results]
            writer.writerow(row)

def process_audio(file_path, noise_factor=0.05):
    """Process an audio file through all filtering methods"""
    try:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=None)
        
        # Create noisy signal
        noisy_signal = add_noise(signal, noise_factor)
        
        # Apply various filters
        wiener_filtered = wiener_filter(noisy_signal)
        
        # Ensure all filtered signals have the same length as original
        spectral_subtracted = spectral_subtraction(noisy_signal)
        kalman_filtered = kalman_filter(noisy_signal)
        dae_filtered = dae_denoise(noisy_signal)
        
        # Truncate all signals to the shortest length
        min_length = min(len(signal), len(noisy_signal), len(wiener_filtered),
                      len(spectral_subtracted), len(kalman_filtered), len(dae_filtered))
        
        return {
            'original_signal': signal[:min_length],
            'noisy_signal': noisy_signal[:min_length],
            'wiener_filtered_signal': wiener_filtered[:min_length],
            'spectral_subtracted_signal': spectral_subtracted[:min_length],
            'kalman_filtered_signal': kalman_filtered[:min_length],
            'dae_filtered_signal': dae_filtered[:min_length],
            'sr': sr
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def plot_waveforms(results, filename):
    """Plot all waveforms for a single audio file"""
    if results is None:
        print(f"No results to plot for {filename}")
        return
    
    t = np.arange(len(results['original_signal'])) / results['sr']
    
    plt.figure(figsize=(16, 10))
    plt.suptitle(f'Waveform Analysis: {os.path.basename(filename)}', fontsize=14)
    
    # Plot 1: Original vs Noisy
    plt.subplot(2, 2, 1)
    plt.plot(t, results['original_signal'], label='Original')
    plt.plot(t, results['noisy_signal'], label='Noisy', alpha=0.7)
    plt.title('Original vs Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot 2: Filtered Signals
    plt.subplot(2, 2, 2)
    plt.plot(t, results['wiener_filtered_signal'], label='Wiener')
    plt.plot(t, results['spectral_subtracted_signal'], label='Spectral Subtraction')
    plt.title('Filtered Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot 3: Advanced Filters
    plt.subplot(2, 2, 3)
    plt.plot(t, results['kalman_filtered_signal'], label='Kalman')
    plt.plot(t, results['dae_filtered_signal'], label='DAE')
    plt.title('Advanced Filtering')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot 4: All Signals Comparison
    plt.subplot(2, 2, 4)
    for key, signal in results.items():
        if key != 'sr':
            plt.plot(t, signal, label=key.replace('_', ' ').title())
    plt.title('All Signals Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_dir = r'C:\Users\razas\Downloads\voice project\dataset'
    
    # Verify dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        exit(1)
    
    # Find all WAV files recursively
    file_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        print("No WAV files found in the dataset directory")
        exit(1)
    
    print(f"Found {len(file_paths)} audio files to process")
    
    for file_path in file_paths:
        print(f"\nProcessing {os.path.basename(file_path)}...")
        
        try:
            results = process_audio(file_path)
            if results:
                # Save results to CSV
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                t = np.arange(len(results['original_signal'])) / results['sr']
                save_results_to_csv(results, t, f'results_{base_name}.csv')
                
                # Plot waveforms
                plot_waveforms(results, file_path)
                
                print(f"Successfully processed {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print("\nProcessing complete!")