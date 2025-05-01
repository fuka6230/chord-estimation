# Full preprocessing pipeline for MP3 + chord annotation (24-chord classification)

import json
import librosa
import numpy as np
import re


# --- Config ---
SR = 22050  # Sampling rate
CHORD_LABELS_24 = [
    'C', 'Cm', 'C#', 'C#m', 'D', 'Dm', 'D#', 'D#m',
    'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G', 'Gm',
    'G#', 'G#m', 'A', 'Am', 'A#', 'A#m', 'B', 'Bm'
]
CHORD_LABELS_25 = CHORD_LABELS_24 + ['N']
label_to_index = {label: i for i, label in enumerate(CHORD_LABELS_25)}

# --- Chord normalization ---
def normalize_chord(chord_name):
    if chord_name == "N":
        return "N"
    match = re.match(r"^([A-G][#b]?)(m)?", chord_name)
    if not match:
        return "N"
    root = match.group(1)
    quality = match.group(2)
    return root + 'm' if quality == 'm' else root

# --- Load chord annotation ---
def load_chord_segments(json_path):
    with open(json_path) as f:
        data = json.load(f)
    segments = []
    for chord in data["chords"]:
        start = chord["start"] / 1000
        end = (chord["start"] + chord["duration"]) / 1000
        name = normalize_chord(chord["name"])
        if name not in label_to_index:
            name = 'N'
        label = label_to_index[name]
        segments.append((start, end, label))
    return segments

# --- Match timestamp to chord label ---
def get_chord_label_at_time(t, segments):
    for start, end, label in segments:
        if start <= t < end:
            return label
    return label_to_index['N']

# --- Preprocessing pipeline ---
def preprocess(mp3_path, chord_json_path):
    y, sr = librosa.load(mp3_path, sr=SR)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    chord_segments = load_chord_segments(chord_json_path)

    X = []  # Chroma features
    y_labels = []  # Corresponding chord labels

    for i in range(len(times) - 1):
        t_start = times[i]
        t_end = times[i + 1]
        mid_time = (t_start + t_end) / 2

        start_frame = librosa.time_to_frames(t_start, sr=sr)
        end_frame = librosa.time_to_frames(t_end, sr=sr)

        chroma_avg = chroma[:, start_frame:end_frame].mean(axis=1)
        X.append(chroma_avg)

        chord_label = get_chord_label_at_time(mid_time, chord_segments)
        y_labels.append(chord_label)

    return np.array(X), np.array(y_labels)

def preprocess_dataset(ids):
    X_all, y_all = [], []
    for id in ids:
        X, y = preprocess(f'data/audio/{id}.mp3', f'data/chord/{id}.json')
        X_all.append(X)
        y_all.append(y)
    return np.vstack(X_all), np.hstack(y_all)
