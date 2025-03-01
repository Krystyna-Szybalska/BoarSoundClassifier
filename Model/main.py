import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import tensorflow as tf
import random
np.random.seed(12)
tf.random.set_seed(12)
random.seed(12)

def load_data(data_path, metadata_path):
    features = []
    labels = []

    metadata = pd.read_csv(metadata_path, sep=';')

    for index, row in metadata.iterrows():
        file_path = os.path.join(data_path, f"{row['FileName']}.wav")

        # Load the audio file and resample it
        target_sr = 22050
        audio, sample_rate = librosa.load(file_path, sr=target_sr)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        # print(mfccs.shape)
        # print(mfccs_scaled)
        # Append features and labels
        features.append(mfccs_scaled)
        labels.append(row['Label'])

    return np.array(features), np.array(labels)

def get_dataset_partitions(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

def predict_audio_class(file_path, model, le):
    # Load the audio file and resample it
    target_sr = 22050
    audio, sample_rate = librosa.load(file_path, sr=target_sr)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    # Reshape the features to fit the input shape of the model
    features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)

    # Predict the class
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    # Decode the class index to its corresponding label
    predicted_class = le.inverse_transform(predicted_class_index)

    return predicted_class[0]


data_path = "./SoundData/PreparedData/"
metadata_path = "./SoundData/Metadata1.csv"
features, labels = load_data(data_path, metadata_path)
# print("Test data size: " + str(len(labels)))
# print(features)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels)

train_dataset = tf.data.Dataset.from_tensor_slices((features, labels_onehot))

train_ds, val_ds, test_ds = get_dataset_partitions(ds=train_dataset, ds_size=len(features))

# that's CNN designed for a multiclass problem - better change that
input_shape = (40, n_timesteps, 1)  # Adjust based on MFCC dimensions
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
]
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)

# test it
correct_predictions = 0
total_predictions = 0
metadata = pd.read_csv("C:\\Users\Krysia\Desktop\BoarSoundClassifier\Model\SoundData\Metadata0.csv", sep=';')
for index, row in metadata.iterrows():
    file_name = row['FileName'] + ".wav"  # Add the file extension
    # Construct the full path to the audio file
    audio_file_path = os.path.join("C:\\Users\Krysia\Desktop\BoarSoundClassifier\Model\SoundData\Quality_0_Files\\",
                                   file_name)
    # Call your predict_audio_class function to get the predicted class
    predicted_class = predict_audio_class(audio_file_path, model, le)
    # Compare the predicted class with the correct label
     correct_label = row['Label']

    if predicted_class == correct_label:
        correct_predictions += 1
    else:
        print(f"File: {file_name}, Predicted: {predicted_class}, Actual: {correct_label}")

    total_predictions += 1

print(f"Total: {correct_predictions}/{total_predictions}")

loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
