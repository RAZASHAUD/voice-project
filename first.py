import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
import soundfile as sf
from tqdm import tqdm

# --- Enhanced Filter Functions ---
def apply_wiener(audio):
    """Enhanced Wiener filter with noise estimation"""
    noise = np.mean(audio[:500])  # Better noise estimation
    return signal.wiener(audio, noise=noise)

def apply_lowpass(audio, sr, cutoff=4000):
    """Butterworth lowpass filter with better parameters"""
    b, a = signal.butter(4, cutoff / (sr / 2), btype='low')  # Reduced order for stability
    return signal.filtfilt(b, a, audio)

def apply_spectral_subtraction(audio, sr):
    """Improved spectral subtraction with FFT-based noise reduction"""
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    
    # Estimate noise from first 10% of signal
    noise_est = np.mean(magnitude[:len(magnitude)//10])
    
    # Subtract noise while preserving phase
    clean_magnitude = np.maximum(magnitude - noise_est, 0)
    clean_fft = clean_magnitude * np.exp(1j * phase)
    return np.fft.ifft(clean_fft).real

def apply_kalman(audio):
    """More robust Kalman filter implementation"""
    n_iter = len(audio)
    xhat = np.zeros(n_iter)
    P = np.zeros(n_iter)
    K = np.zeros(n_iter)
    
    # Tuned parameters
    Q = 1e-6  # Process variance
    R = 0.01  # Measurement variance
    
    xhat[0] = audio[0]
    P[0] = 1.0
    
    for k in range(1, n_iter):
        # Prediction
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + Q
        
        # Update
        K[k] = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K[k] * (audio[k] - xhat_minus)
        P[k] = (1 - K[k]) * P_minus
    
    return xhat

def apply_deep_filter_sim(audio):
    """Enhanced smoothing filter"""
    return uniform_filter1d(audio, size=15)

FILTERS = {
    "wiener": apply_wiener,
    "lowpass": apply_lowpass,
    "spectral": apply_spectral_subtraction,
    "kalman": apply_kalman,
    "deep": apply_deep_filter_sim
}

# --- Enhanced Feature Extraction ---
def extract_features(audio, sr):
    """Extract comprehensive audio features"""
    features = {}
    
    # Time-domain features
    features['zcr'] = librosa.feature.zero_crossing_rate(audio)[0].mean()
    features['rms'] = librosa.feature.rms(y=audio)[0].mean()
    
    # Frequency-domain features
    features['centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
    features['bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0].mean()
    features['flatness'] = librosa.feature.spectral_flatness(y=audio)[0].mean()
    
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    features['pitch_mean'] = np.mean(pitches[pitches > 0])
    features['pitch_std'] = np.std(pitches[pitches > 0])
    
    # MFCCs (more coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    for i in range(mfccs.shape[0]):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
    
    return features

def process_dataset(dataset_path, output_dir):
    """Process all audio files with all filters"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Get sorted list of speakers
    speakers = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    for speaker in tqdm(speakers, desc="Processing Speakers"):
        speaker_path = os.path.join(dataset_path, speaker)
        
        for file in os.listdir(speaker_path):
            if file.endswith('.wav'):
                file_path = os.path.join(speaker_path, file)
                
                try:
                    audio, sr = librosa.load(file_path, sr=16000)  # Fixed sample rate
                    
                    # Process with each filter
                    for filter_name, filter_func in FILTERS.items():
                        if filter_name == 'lowpass':
                            filtered = filter_func(audio, sr)
                        else:
                            filtered = filter_func(audio)
                        
                        # Save filtered audio
                        filtered_path = os.path.join(
                            output_dir, 
                            f"{speaker}_{file.replace('.wav', '')}_{filter_name}.wav"
                        )
                        sf.write(filtered_path, filtered, sr)
                        
                        # Extract features
                        features = extract_features(filtered, sr)
                        features['speaker'] = speaker
                        features['file'] = file
                        features['filter'] = filter_name
                        results.append(features)
                
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'filtered_features.csv'), index=False)
    return df

if __name__ == "__main__":
    dataset_path = 'dataset'  # Path to your dataset
    output_dir = 'filtered_audio'  # Where to save processed files
    process_dataset(dataset_path, output_dir)