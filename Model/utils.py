import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D

def load_data(data_path, metadata_path):
   features = []
   labels = []

   metadata = pd.read_csv(metadata_path)

   for index, row in metadata.iterrows():
     file_path = os.path.join(data_path, f"fold{row['fold']}", f"{row['slice_file_name']}")

     # Load the audio file and resample it
     target_sr = 22050
     audio, sample_rate = librosa.load(file_path, sr=target_sr)

     # Extract MFCC features
     mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
     mfccs_scaled = np.mean(mfccs.T, axis=0)

     # Append features and labels
     features.append(mfccs_scaled)
     labels.append(row['class'])

     return np.array(features), np.array(labels)
