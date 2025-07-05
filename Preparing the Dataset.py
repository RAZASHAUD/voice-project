import sounddevice as sd
import numpy as np
import wave
import os

# Function to record audio
def record_audio(filename, duration=3, samplerate=16000):
    print(f"Recording: {filename}...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished

    # Save the recorded audio as a WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print(f"Saved: {filename}")

# Main function to record multiple samples for each speaker
def record_dataset(num_speakers=20, samples_per_speaker=0, duration=3):
    dataset_path = "dataset"
    os.makedirs(dataset_path, exist_ok=True)

    for speaker_id in range(1, num_speakers + 1):
        speaker_folder = os.path.join(dataset_path, f"speaker_{speaker_id}")
        os.makedirs(speaker_folder, exist_ok=True)

        print(f"\nRecording for Speaker {speaker_id}:")
        for sample_num in range(1, samples_per_speaker + 1):
            filename = os.path.join(speaker_folder, f"sample_{sample_num}.wav")
            record_audio(filename, duration)

    print("\nRecording Complete!")

# Run the recording function
record_dataset(num_speakers=15, samples_per_speaker=1, duration=1)
