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

data_path = "./SoundData/PreparedData/"
metadata_path = "./SoundData/Metadata1.csv"
features, labels = load_data(data_path, metadata_path)
# print("Test data size: " + str(len(labels)))
# print(features)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2, random_state=42,
                                                    stratify=labels_onehot)
# random state set to specific number ensures that the data will always be split the same way

# thats CNN designed for a multiclass problem - better change that
input_shape = (X_train.shape[1], 1)
model = Sequential()
model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))  # prevents overfitting by randomly setting some input values to 0
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=1)


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
    print(predicted_vector)
    # Decode the class index to its corresponding label
    predicted_class = le.inverse_transform(predicted_class_index)

    return predicted_class[0]


test_file_path = "./SoundData/76796__robinhood76__01161-boar-oink-3.wav"
predicted_class1 = predict_audio_class(test_file_path, model, le)
print("Correct output is: Dzik (1)")
print("Predicted class:", predicted_class1)
