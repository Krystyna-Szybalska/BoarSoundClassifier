from pydub import AudioSegment
from pydub.silence import detect_leading_silence
import os


def pad_audio_to_4_seconds(input_filepath, output_filepath):
    """
    Pads an audio file with silence to ensure it is at least 4 seconds long.
    If the audio is shorter than 4 seconds, equal amounts of silence are added
    at the beginning and end.

    Args:
        input_filepath (str): Path to the input audio file.
        output_filepath (str): Path to save the processed audio file.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_filepath)

        # Target duration in milliseconds
        target_duration_ms = 4 * 1000  # 4 seconds

        current_duration_ms = len(audio)

        final_audio_segment = audio  # Initialize with original audio

        if current_duration_ms < target_duration_ms:
            # Calculate the amount of silence needed
            silence_needed_ms = target_duration_ms - current_duration_ms

            padding_before_ms = silence_needed_ms // 2
            padding_after_ms = silence_needed_ms - padding_before_ms

            silence_before = AudioSegment.silent(duration=padding_before_ms)
            silence_after = AudioSegment.silent(duration=padding_after_ms)

            padded_audio = silence_before + audio + silence_after
            final_audio_segment = padded_audio  # Update to padded audio

            # Print the duration that pydub calculates for the padded audio
            print(f"Padded '{input_filepath}'. Pydub calculated duration: {len(padded_audio)} ms.")

            original_format = input_filepath.split('.')[-1]
            padded_audio.export(output_filepath, format=original_format)
            print(f"Successfully padded '{input_filepath}' and saved to '{output_filepath}'")
        else:
            # Print the duration for files that are not padded
            print(f"Audio '{input_filepath}' is {current_duration_ms} ms. Pydub calculated duration: {len(audio)} ms.")
            original_format = input_filepath.split('.')[-1]
            audio.export(output_filepath, format=original_format)
            print(
                f"Audio '{input_filepath}' is already {current_duration_ms / 1000:.2f}s long (>= 4s). Copied to '{output_filepath}'.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred while processing '{input_filepath}': {e}")


def process_files_from_txt(txt_filepath):
    """
    Reads audio file paths from a text file, pads them, and saves them
    in their original directories with a '_padded' suffix.
    """
    try:
        with open(txt_filepath, 'r') as f:
            audio_paths = [line.strip()[1:-2] for line in f if line.strip()]

        if not audio_paths:
            print(f"No audio paths found in '{txt_filepath}'.")
            return

        for input_path in audio_paths:
            if not os.path.exists(input_path):
                print(f"Warning: File not found at '{input_path}', skipping.")
                continue

            try:
                '''directory = os.path.dirname(input_path)
                filename, extension = os.path.splitext(os.path.basename(input_path))
                output_filename = f"{filename}_padded{extension}"
                output_path = os.path.join(directory, output_filename)'''

                pad_audio_to_4_seconds(input_path, input_path)
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

    except FileNotFoundError:
        print(f"Error: The file list '{txt_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the list file: {e}")


txt_file = r"C:\Users\Krysia\Desktop\To pad and 0.txt"
txt_file2 = r"C:\Users\Krysia\Desktop\To pad.txt"

#process_files_from_txt(txt_file)
process_files_from_txt(txt_file2)
