import math

import numpy as np

from common import (choose_k, confusion, fit_main_model, knn_predict, prepare, scores,
                    sigmoid, write_csv, write_results)


def entropy(pos, n):
    if n == 0:
        return 0.0
    h = 0.0
    for p in (pos / n, 1 - pos / n):
        if p > 0:
            h -= p * math.log2(p)
    return h


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1 - p ** 2 - (1 - p) ** 2


def gain(func, y, mask):
    n = len(y)
    parent = func(y.sum(), n)
    yes, no = y[mask], y[~mask]
    child = (len(yes) * func(yes.sum(), len(yes)) + len(no) * func(no.sum(), len(no))) / n
    return parent - child


def split_masks(X):
    return {
        "S1": X[:, 5] == 1,
        "S2": X[:, 6] == 1,
        "S3": X[:, 2] > 0.5,
        "S4": X[:, 0] <= 24,
    }


def adaboost():
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    rules = [
        np.array([1, 1, -1, -1, -1, -1, -1, -1]),
        np.array([-1, 1, 1, -1, 1, -1, 1, -1]),
    ]
    w = np.full(8, 1 / 8)
    rows, errors, alphas = [], [], []
    for t, h in enumerate(rules, start=1):
        err = w[h != y].sum()
        alpha = 0.5 * math.log((1 - err) / err)
        raw = w * np.exp(-alpha * y * h)
        new = raw / raw.sum()
        for p in range(8):
            rows.append([t, p + 1, w[p], raw[p], new[p]])
        errors.append(err)
        alphas.append(alpha)
        w = new
    combined = np.sign(alphas[0] * rules[0] + alphas[1] * rules[1])
    return rows, errors, alphas, float(np.mean(combined == y))


def conv_out(size, f, pad, stride):
    return (size - f + 2 * pad) // stride + 1


def dense(n_in, n_out):
    return n_in * n_out + n_out


def conv(f, depth_in, filters):
    return f * f * depth_in * filters + filters


def architecture():
    mlp = [("mlp_hidden1", dense(784, 16)),
           ("mlp_hidden2", dense(16, 16)),
           ("mlp_output", dense(16, 10))]

    side = conv_out(28, 5, 2, 1)
    side = conv_out(side, 2, 0, 2)
    side = conv_out(side, 5, 0, 1)
    side = conv_out(side, 2, 0, 2)
    flat = side * side * 16
    cnn = [("cnn_conv1", conv(5, 1, 6)),
           ("cnn_conv2", conv(5, 6, 16)),
           ("cnn_dense1", dense(flat, 32)),
           ("cnn_dense2", dense(32, 10))]
    return mlp, cnn


def main():
    d = prepare()
    X, train, test = d["X"], d["train"], d["test"]
    ytr, yte = d["ytr"], d["yte"]
    res = {}

    res["root_entropy"] = entropy(ytr.sum(), len(ytr))
    res["root_gini"] = gini(ytr.sum(), len(ytr))
    split_rows = []
    best_ig, best_gg = None, None
    for name, mask in split_masks(X[train]).items():
        ig = gain(entropy, ytr, mask)
        gg = gain(gini, ytr, mask)
        split_rows.append([name, int(mask.sum()), int(ytr[mask].sum()),
                           int((~mask).sum()), int(ytr[~mask].sum()), ig, gg])
        if best_ig is None or ig > best_ig[1]:
            best_ig = (name, ig)
        if best_gg is None or gg > best_gg[1]:
            best_gg = (name, gg)
    write_csv("part4_splits.csv",
              ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
              split_rows)
    res["best_split_by_info_gain"] = best_ig[0]
    res["best_split_by_gini_gain"] = best_gg[0]

    stump_tr = split_masks(X[train])[best_ig[0]]
    stump_te = split_masks(X[test])[best_ig[0]]
    rate_yes = ytr[stump_tr].mean()
    rate_no = ytr[~stump_tr].mean()
    res["stump_rate_yes"] = rate_yes
    res["stump_rate_no"] = rate_no

    w, b, _ = fit_main_model(d)
    p_log = sigmoid(d["Xte"] @ w + b)
    best_k, _ = choose_k(d["Xtr"], ytr)
    knn_pred, p_knn = knn_predict(d["Xtr"], ytr, d["Xte"], best_k)
    p_stump = np.where(stump_te, rate_yes, rate_no)

    log_pred = p_log >= 0.3
    stump_pred = p_stump >= 0.3
    hard = (log_pred.astype(int) + knn_pred.astype(int) + stump_pred.astype(int)) >= 2
    soft = (p_log + p_knn + p_stump) / 3 >= 0.3

    vote_rows = []
    for name, pred in [("logistic", log_pred), ("knn", knn_pred), ("stump", stump_pred),
                       ("hard_vote", hard), ("soft_vote", soft)]:
        cells = confusion(pred, yte)
        s = scores(*cells)
        vote_rows.append([name, *cells, s["precision"], s["recall"], s["f1"]])
    write_csv("part4_voting.csv",
              ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vote_rows)

    ada_rows, errors, alphas, ens_acc = adaboost()
    res["round1_error"] = errors[0]
    res["round1_alpha"] = alphas[0]
    res["round2_error"] = errors[1]
    res["round2_alpha"] = alphas[1]
    res["ensemble_accuracy"] = ens_acc
    write_csv("part4_adaboost.csv",
              ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)

    res["oob_fraction"] = (1 - 1 / 560) ** 560

    mlp, cnn = architecture()
    write_csv("part4_architecture.csv", ["layer", "parameters"], mlp + cnn)
    res["mlp_total"] = sum(p for _, p in mlp)
    res["cnn_total"] = sum(p for _, p in cnn)
    res["fc_layer_params"] = dense(128 * 128 * 3, 1024)
    res["conv_layer_params"] = conv(3, 3, 64)
    res["rnn_total"] = 64 * 128 + 128 * 128 + 128 + 128 * 16 + 16

    write_results("part4_results.txt", res)


if __name__ == "__main__":
    main()
