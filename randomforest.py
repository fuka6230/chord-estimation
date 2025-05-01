from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocessing import preprocess_dataset
import csv
import itertools

CHORD_LABELS_24 = [
    'C', 'Cm', 'C#', 'C#m', 'D', 'Dm', 'D#', 'D#m',
    'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G', 'Gm',
    'G#', 'G#m', 'A', 'Am', 'A#', 'A#m', 'B', 'Bm'
]
CHORD_LABELS_25 = CHORD_LABELS_24 + ['N']
label_to_index = {label: i for i, label in enumerate(CHORD_LABELS_25)}
index_to_label = {i: label for label, i in label_to_index.items()}

def randomforest(ids):
    X_train, y_train = preprocess_dataset(ids[:7])
    X_test, y_test = preprocess_dataset(ids[7:])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    unique_labels = sorted(set(y_test))
    target_names = [index_to_label[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

with open('data/id.csv') as f:
    ids = list(itertools.chain.from_iterable(csv.reader(f)))
randomforest(ids)
