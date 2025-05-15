import json
import librosa
import numpy as np
import re
import itertools
import csv
import os


# 定数
SR = 22050  # Sampling rate
CHORD_LABELS_24 = [
    'C', 'Cm', 'C#', 'C#m', 'D', 'Dm', 'D#', 'D#m',
    'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G', 'Gm',
    'G#', 'G#m', 'A', 'Am', 'A#', 'A#m', 'B', 'Bm'
]
CHORD_LABELS_25 = CHORD_LABELS_24 + ['N']
label_to_index = {label: i for i, label in enumerate(CHORD_LABELS_25)}

# コードのルートとメジャー,マイナーを抽出
def normalize_chord(chord_name):
    if chord_name == "N":
        return "N"
    match = re.match(r"^([A-G][#b]?)(m)?", chord_name)
    if not match:
        return "N"
    root = match.group(1)
    quality = match.group(2)
    return root + 'm' if quality == 'm' else root

# 一つのコードごとに開始時間, 終了時間, コードのlabel_to_indexでのラベルを取得
def load_chord_segments(json_path):
    with open(json_path) as f:
        data = json.load(f)
        if data == {}:
            return None
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

# 時間とコード列をマッチさせる
def get_chord_label_at_time(t, segments):
    for start, end, label in segments:
        if start <= t < end:
            return label
    return label_to_index['N']

# このファイルでのメイン処理
def preprocess(mp3_path, chord_json_path):
    chord_segments = load_chord_segments(chord_json_path)
    if chord_segments is None:
        return None, None
    y, sr = librosa.load(mp3_path, sr=SR)
    _, beats = librosa.beat.beat_track(y=y, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    X = []
    y_labels = []

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

def preprocess_cnn(mp3_path, chord_json_path, patch_len=16, hop_len=8):
    y, sr = librosa.load(mp3_path, sr=SR)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # shape: (12, T)
    chord_segments = load_chord_segments(chord_json_path)
    if chord_segments is None:
        return None, None

    # --- chroma → パッチ (B, 1, 12, patch_len) ---
    chroma_patches = []
    labels = []

    total_frames = chroma.shape[1]
    for start in range(0, total_frames - patch_len + 1, hop_len):
        patch = chroma[:, start:start + patch_len]  # (12, patch_len)
        chroma_patches.append(patch)

        # 対応するラベルを取得（中央時刻）
        patch_times = librosa.frames_to_time(np.arange(start, start + patch_len), sr=sr)
        mid_time = patch_times[patch_len // 2]
        label = get_chord_label_at_time(mid_time, chord_segments)
        labels.append(label)

    X = np.stack(chroma_patches)[:, np.newaxis, :, :]  # shape: (B, 1, 12, patch_len)
    y_labels = np.array(labels)

    return X, y_labels


def preprocess_dataset(ids, is_cnn=False):
    X_all, y_all = [], []
    for idx, id in enumerate(ids):
        print(idx)
        if is_cnn:
            X, y = preprocess_cnn(f'data/audio/{id}.mp3', f'data/chord/{id}.json')
        else:
            X, y = preprocess(f'data/audio/{id}.mp3', f'data/chord/{id}.json')
        if X is None:
            continue
        X_all.append(X)
        y_all.append(y)
    return np.vstack(X_all), np.hstack(y_all)

def main():
    chord_dir = "data/chord"

    ids = [os.path.splitext(f)[0] for f in os.listdir(chord_dir) if os.path.isfile(os.path.join(chord_dir, f))]
    ids_idx = int(len(ids) * 0.7)

    # 汎用
    X_train, y_train = preprocess_dataset(ids[:ids_idx])
    X_test, y_test = preprocess_dataset(ids[ids_idx:])
    np.savez('data/audio_and_chord', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # CNN
    X_train, y_train = preprocess_dataset(ids[:ids_idx], True)
    X_test, y_test = preprocess_dataset(ids[ids_idx:], True)
    np.savez('data/audio_and_chord_cnn', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

if __name__ == '__main__':
    main()
