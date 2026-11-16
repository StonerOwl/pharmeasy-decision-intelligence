import numpy as np

from common import load_data, sample_sd, write_csv, write_results

NAMES = ["E1", "E2", "E3", "E4", "E5", "E6"]
DIST = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
])


def kmeans(Z, k):
    step = len(Z) // k
    centroids = Z[[i * step for i in range(k)]].copy()
    labels = None
    while True:
        d2 = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = Z[labels == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    wcss = ((Z - centroids[labels]) ** 2).sum()
    return labels, wcss


def linkage_distance(a, b, rule):
    d = DIST[np.ix_(a, b)]
    if rule == "single":
        return d.min()
    if rule == "complete":
        return d.max()
    return d.mean()


def agglomerate(rule):
    clusters = [[i] for i in range(len(NAMES))]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                h = linkage_distance(clusters[i], clusters[j], rule)
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        a, b = sorted([clusters[i], clusters[j]], key=min)
        merges.append((a, b, h))
        clusters = [c for n, c in enumerate(clusters) if n not in (i, j)] + [sorted(a + b)]
    return merges


def label(cluster):
    return "+".join(NAMES[i] for i in sorted(cluster))


def clusters_at_cut(merges):
    heights = [m[2] for m in merges]
    gaps = np.diff(heights)
    below = int(np.argmax(gaps)) + 1
    return len(NAMES) - below


def main():
    X, y, value = load_data()
    C = X[:, :5]
    Z = (C - C.mean(axis=0)) / sample_sd(C, axis=0)
    res = {}

    wcss = {}
    labels_by_k = {}
    for k in range(2, 7):
        labels_by_k[k], wcss[k] = kmeans(Z, k)
        res[f"wcss_k{k}"] = wcss[k]

    elbow = None
    for k in (3, 4, 5):
        if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
            elbow = k
            break
    res["elbow_k"] = elbow if elbow is not None else "none"

    labels = labels_by_k[3]
    order = sorted(range(3), key=lambda c: C[labels == c, 2].mean())
    segment_of = {c: "ABC"[i] for i, c in enumerate(order)}
    seg_rows = []
    for c in order:
        m = labels == c
        name = segment_of[c]
        res[f"size_segment_{name}"] = int(m.sum())
        seg_rows.append([name, int(m.sum()), *C[m].mean(axis=0), y[m].mean(), value[m].mean()])
    write_csv("part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
               "outcome_rate", "mean_value"], seg_rows)

    hier_rows = []
    for rule in ("single", "complete", "average"):
        merges = agglomerate(rule)
        for no, (a, b, h) in enumerate(merges, start=1):
            hier_rows.append([rule, no, label(a), label(b), h])
        res[f"{rule}_clusters_at_cut"] = clusters_at_cut(merges)
    write_csv("part3_hierarchical.csv",
              ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hier_rows)

    centred = Z - Z.mean(axis=0)
    S = centred.T @ centred / (len(Z) - 1)
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    for j in range(vecs.shape[1]):
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] = -vecs[:, j]

    total = vals.sum()
    ratios = vals / total
    cumulative = np.cumsum(ratios)
    res["eigenvalue_sum"] = total
    for i in range(5):
        res[f"eigenvalue_{i + 1}"] = vals[i]
    for i in range(5):
        res[f"explained_ratio_{i + 1}"] = ratios[i]
    for i in range(5):
        res[f"cumulative_ratio_{i + 1}"] = cumulative[i]
    res["components_for_80pct"] = int(np.argmax(cumulative >= 0.8)) + 1

    W = vecs[:, :2]
    for pc in range(2):
        for i in range(5):
            res[f"pc{pc + 1}_loading_X{i + 1}"] = W[i, pc]
    proj = centred @ W
    for r in range(3):
        res[f"row{r + 1}_pc1"] = proj[r, 0]
        res[f"row{r + 1}_pc2"] = proj[r, 1]
    for c in order:
        res[f"mean_pc1_segment_{segment_of[c]}"] = proj[labels == c, 0].mean()

    write_results("part3_results.txt", res)


if __name__ == "__main__":
    main()
