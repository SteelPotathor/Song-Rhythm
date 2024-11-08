import librosa
import json
import argparse
import os

import numpy as np


def main(audio_path, output_path):
    """
    Analyzes an audio file to extract the tempo, beat timings, and intense moments,
    then saves this information to a JSON file.

    :param audio_path: Path to the audio file (supported formats by librosa, such as .wav, .mp3, .ogg).
    :param output_path: Path to the output JSON file. This file will be created or overwritten if it already exists.
    """
    # Check if the audio file exists
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"The specified audio file does not exist: {audio_path}")

    # Load the audio file
    print(f"Loading '{audio_path}' for analysis...")
    y, sr = librosa.load(audio_path)

    # Analyze the rhythm and extract beat timings
    print("Analyzing rhythm and extracting beat timings...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Calculate the onset strength of the signal
    print("Calculating the onset strength of the signal...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Detect intense moments (onsets)
    print("Detecting intense moments...")
    intense_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='frames')

    # Convert frames to time (seconds)
    intense_times = librosa.frames_to_time(intense_frames, sr=sr)

    # Prepare data for output
    output_data = {
        "tempo": tempo[0],
        "beat_times": [f"{time:.3f}" for time in beat_times],
        "intense_moments": [f"{time:.3f}" for time in intense_times]
    }

    # Save beat timings and intense moments to the output JSON file
    print(f"Saving beat timings and intense moments to '{output_path}'...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Analysis complete. Estimated tempo: {tempo[0]:.2f} BPM")
    print(f"Beat timings and intense moments successfully saved in '{output_path}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze an audio file to extract beat timings, tempo, and intense moments, then save them to a JSON file."
    )
    parser.add_argument(
        '--audio', type=str, help="Path to the audio file to be analyzed.", required=True
    )
    parser.add_argument(
        '--output', type=str, help="Path to the output JSON file to save beat timings and intense moments.", required=True
    )
    args = parser.parse_args()
    try:
        main(args.audio, args.output)
    except Exception as e:
        print(f"Error: {e}")
