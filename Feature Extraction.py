import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # Enhanced feature set
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increased from 13 to 20
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        
        return np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1),
            np.mean(zcr),
            np.mean(rms),
            np.std(mfccs, axis=1),  # Adding standard deviation
            librosa.feature.spectral_flatness(y=y)[0].mean(),
            librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
        ])
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def create_feature_dataset(dataset_path="dataset"):
    features = []
    labels = []
    
    # Get sorted list of speaker folders (a01, a02, etc.)
    speaker_folders = sorted([f for f in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, f))])
    
    for speaker_id, speaker_folder in enumerate(tqdm(speaker_folders, desc="Processing Speakers")):
        speaker_path = os.path.join(dataset_path, speaker_folder)
        
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                feature_vector = extract_features(file_path)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(speaker_id)  # Use numerical labels
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Save with checks
    if len(features) > 0:
        np.save("features.npy", features)
        np.save("labels.npy", labels)
        print(f"\nSuccessfully processed {len(features)} samples from {len(speaker_folders)} speakers")
        print(f"Feature vector shape: {features.shape}")
    else:
        print("Error: No features were extracted!")

if __name__ == "__main__":
    create_feature_dataset()