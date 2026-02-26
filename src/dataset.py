import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_image_paths_and_labels(root_dir, exclude=None):
    '''
    Scans the dataset folder
    Input: root directory of caltech 101 dataset

    output:  paths -> (list/numpy arrays of strings), lables - > (list/numpy array of ints), label2idx -> (dictionary)
    '''
    root = Path(root_dir)
    exclude = set(exclude or [])
    classes = sorted([d.name for d in root.iterdir() if d.is_dir() and d.name not in exclude])
    label2idx = {c: i for i, c in enumerate(classes)}

    paths = []
    labels = []
    for c in classes:
        class_dir = root / c
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            paths.append(str(img_path.resolve()))
            labels.append(label2idx[c])

    return paths, labels, label2idx


def create_split(paths, labels, test_size = 0.3, seed=42):
    '''
    Creates stratified train test split
    Input: paths and labels
    Output: train_paths, train_labels, test_paths, test_labels
    '''
    paths = np.array(paths)
    labels = np.array(labels)

    # sanity check
    if len(paths) != len(labels):
        raise ValueError("paths and labels must have the same length")

    # Stratified split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    return train_paths, test_paths, train_labels, test_labels

def save_split(train_paths, test_paths, train_labels, test_labels, label2idx, out_dir, seed=42, test_size=0.3):
    '''
    Saves train-test-split to disk 
    '''
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "train_paths.npy", train_paths)
    np.save(out / "test_paths.npy", test_paths)
    np.save(out / "train_labels.npy", train_labels)
    np.save(out / "test_labels.npy", test_labels)

    meta = {
        "label2idx": label2idx,
        "idx2label": {v: k for k, v in label2idx.items()},
        "seed": seed,
        "test_size": test_size
    }

    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

def load_split(out_dir):
    '''
    Returns train and test labels and paths
    '''
    out = Path(out_dir)
    train_paths = np.load(out / "train_paths.npy", allow_pickle=True)
    test_paths = np.load(out / "test_paths.npy", allow_pickle=True)
    train_labels = np.load(out / "train_labels.npy", allow_pickle=True)
    test_labels = np.load(out / "test_labels.npy", allow_pickle=True)

    with open(out / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    return train_paths, test_paths, train_labels, test_labels, meta

