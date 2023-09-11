from pydub import AudioSegment
from pydub.utils import make_chunks
import os


class SplitWavAudioIn4SekChunks:
    def __init__(self, file_path):
        self.filepath = file_path
        self.filename = os.path.splitext(os.path.basename(file_path))[0]
        self.audio = AudioSegment.from_wav(self.filepath)
        self.to_folder = 'PreparedData'

    def get_duration(self):
        return self.audio.duration_seconds

    def split_file(self):
        chunk_length_ms = 4000  # pydub calculates in millisec
        chunks = make_chunks(self.audio, chunk_length_ms)

        for i, chunk in enumerate(chunks):
            chunk_name = self.filename + "_{0}.wav".format(i)
            print("exporting", chunk_name)
            chunk.export(f'C:\\Users\Krysia\Desktop\SoundData\PreparedData\\'+os.path.basename(chunk_name), format="wav")
