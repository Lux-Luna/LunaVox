import os
import numpy as np
import librosa
import soundfile as sf

file_path = r"Data/audio_resources/English/First get into position like this, then move like that. Yep, that's it..mp3"

print(f"Testing file: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")

try:
    print("--- Librosa Test ---")
    y, sr = librosa.load(file_path, sr=32000)
    print(f"Loaded with librosa. SR: {sr}, Shape: {y.shape}, Dtype: {y.dtype}")
    print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean()}")
    print(f"Has NaNs: {np.isnan(y).any()}")
    if np.isnan(y).any():
        print("NaNs detected!")
except Exception as e:
    print(f"Librosa error: {e}")

try:
    print("\n--- Soundfile Test ---")
    y_sf, sr_sf = sf.read(file_path)
    print(f"Loaded with soundfile. SR: {sr_sf}, Shape: {y_sf.shape}, Dtype: {y_sf.dtype}")
    print(f"Min: {y_sf.min()}, Max: {y_sf.max()}, Mean: {y_sf.mean()}")
except Exception as e:
    print(f"Soundfile error: {e}")

