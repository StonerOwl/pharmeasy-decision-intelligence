import numpy as np

from common import (bce, choose_k, confusion, fit_main_model, knn_predict, prepare,
                    scores, sigmoid, train_logistic, write_csv, write_results)

FALSE_ALARM_COST = 50
MISSED_COST = 400


def cost(tp, fp, tn, fn):
    return FALSE_ALARM_COST * fp + MISSED_COST * fn


def main():
    d = prepare()
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    res = {
        "train_rows": len(ytr),
        "train_positives": int(ytr.sum()),
        "test_rows": len(yte),
        "test_positives": int(yte.sum()),
    }

    w, b, losses = fit_main_model(d)
    for step, loss in losses.items():
        res[f"loss_epoch_{step}"] = loss
    for i, wi in enumerate(w):
        res[f"weight_X{i + 1}"] = wi
    res["bias"] = b
    p_tr = sigmoid(Xtr @ w + b)
    p_te = sigmoid(Xte @ w + b)
    res["test_bce"] = bce(p_te, yte)

    for eta in (0.01, 0.1, 1.0):
        _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
        res[f"loss_epoch_200_eta_{eta}"] = l[200]

    for i, wi in enumerate(w):
        res[f"odds_ratio_X{i + 1}"] = np.exp(wi)

    for tau in (0.5, 0.3):
        cells = confusion(p_te >= tau, yte)
        s = scores(*cells)
        for name, v in zip(("tp", "fp", "tn", "fn"), cells):
            res[f"tau_{tau}_{name}"] = v
        for name in ("accuracy", "precision", "recall", "f1", "specificity"):
            res[f"tau_{tau}_{name}"] = s[name]
    res["blind_accuracy"] = float(np.mean(yte == 0))

    sweep = []
    for k in range(1, 10):
        tau = k / 10
        cells = confusion(p_te >= tau, yte)
        s = scores(*cells)
        sweep.append((tau, *cells, s["recall"], s["fpr"]))
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], sweep)

    best_k, cv = choose_k(Xtr, ytr)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], cv)
    res["knn_best_k"] = best_k
    pred, _ = knn_predict(Xtr, ytr, Xte, best_k)
    cells = confusion(pred, yte)
    s = scores(*cells)
    for name, v in zip(("tp", "fp", "tn", "fn"), cells):
        res[f"knn_{name}"] = v
    for name in ("accuracy", "precision", "recall", "f1"):
        res[f"knn_{name}"] = s[name]

    best_tau, best_cost = None, None
    for k in range(1, 20):
        tau = k / 20
        c = cost(*confusion(p_tr >= tau, ytr))
        if best_cost is None or c < best_cost:
            best_tau, best_cost = tau, c
    res["cost_tau"] = best_tau
    res["cost_train_at_tau"] = best_cost
    res["cost_test_at_tau"] = cost(*confusion(p_te >= best_tau, yte))
    res["cost_test_at_0.5"] = cost(*confusion(p_te >= 0.5, yte))
    res["cost_test_flag_nobody"] = MISSED_COST * int(yte.sum())

    write_results("part2_results.txt", res)


if __name__ == "__main__":
    main()
