import csv
import os

import numpy as np

DATA_PATH = os.path.join("data", "records.csv")
OUT_DIR = "outputs"


def load_data():
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.reader(f))
    arr = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    X = arr[:, :7]
    y = arr[:, 7].astype(int)
    value = arr[:, 8]
    return X, y, value


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return f"{float(v):.4f}"


def write_results(name, results):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="\n") as f:
        for k, v in results.items():
            f.write(f"{k} = {fmt(v)}\n")


def write_csv(name, header, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def sample_sd(a, axis=None):
    a = np.asarray(a, dtype=float)
    n = a.shape[0] if axis == 0 else a.size
    m = a.mean(axis=axis)
    return np.sqrt(((a - m) ** 2).sum(axis=axis) / (n - 1))


def stratified_split(y):
    train, test = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        cut = 7 * len(idx) // 10
        train.extend(idx[:cut])
        test.extend(idx[cut:])
    return np.sort(np.array(train)), np.sort(np.array(test))


def prepare():
    X, y, value = load_data()
    tr, te = stratified_split(y)
    mu = X[tr].mean(axis=0)
    sd = sample_sd(X[tr], axis=0)
    Xs = (X - mu) / sd
    return {
        "X": X, "y": y, "value": value, "train": tr, "test": te,
        "Xtr": Xs[tr], "ytr": y[tr], "Xte": Xs[te], "yte": y[te],
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(p, y):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_logistic(X, y, eta, steps, record=()):
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(sigmoid(X @ w + b), y)
    n = len(y)
    for step in range(1, steps + 1):
        err = sigmoid(X @ w + b) - y
        w = w - eta * (X.T @ err) / n
        b = b - eta * err.sum() / n
        if step in record:
            losses[step] = bce(sigmoid(X @ w + b), y)
    return w, b, losses


def confusion(pred, y):
    pred = np.asarray(pred).astype(bool)
    y = np.asarray(y).astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    tn = int(np.sum(~pred & ~y))
    fn = int(np.sum(~pred & y))
    return tp, fp, tn, fn


def div(a, b):
    return a / b if b else 0.0


def scores(tp, fp, tn, fn):
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "specificity": div(tn, tn + fp),
        "fpr": div(fp, fp + tn),
    }


def knn_votes(ref_X, ref_y, query_X, k):
    votes = np.empty(len(query_X), dtype=int)
    for i, q in enumerate(query_X):
        d = np.sqrt(((ref_X - q) ** 2).sum(axis=1))
        nearest = np.argsort(d, kind="stable")[:k]
        votes[i] = ref_y[nearest].sum()
    return votes


def knn_predict(ref_X, ref_y, query_X, k):
    votes = knn_votes(ref_X, ref_y, query_X, k)
    return votes >= (k + 1) / 2, votes / k


def fit_main_model(d):
    return train_logistic(d["Xtr"], d["ytr"], 0.1, 1000, record=(0, 1, 10, 100, 1000))


def choose_k(Xtr, ytr):
    folds = np.arange(len(ytr)) % 5
    cv = []
    for k in range(1, 22, 2):
        f1s = []
        for f in range(5):
            held = folds == f
            pred, _ = knn_predict(Xtr[~held], ytr[~held], Xtr[held], k)
            f1s.append(scores(*confusion(pred, ytr[held]))["f1"])
        cv.append((k, float(np.mean(f1s))))
    best_k = max(cv, key=lambda r: (r[1], -r[0]))[0]
    return best_k, cv
