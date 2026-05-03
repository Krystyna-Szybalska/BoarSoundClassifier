import os
import sys
from SplitWavAudioIn4SekChunks import SplitWavAudioIn4SekChunks  # Import your class definition

if len(sys.argv) != 2:
    print("Usage: python split_files.py <folder_path>")
    sys.exit(1)

folder_path = sys.argv[1]  # Get the folder path from command-line argument

if not os.path.exists(folder_path):

    print("Error: Folder does not exist")
    sys.exit(1)

# Get a list of all WAV files in the folder
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

for wav_file in wav_files:
    file_path = os.path.join(folder_path, wav_file)
    print("Cur file path: " + file_path)
    splitter = SplitWavAudioIn4SekChunks(file_path)
    splitter.split_file()
