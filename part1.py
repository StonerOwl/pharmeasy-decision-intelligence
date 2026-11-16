import math

import numpy as np

from common import load_data, sample_sd, write_results


def simple_ols(x, y):
    n = len(x)
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum() / (n - 1)
    var = ((x - mx) ** 2).sum() / (n - 1)
    slope = cov / var
    intercept = my - slope * mx
    fitted = intercept + slope * x
    tss = ((y - my) ** 2).sum()
    ess = ((fitted - my) ** 2).sum()
    rss = ((y - fitted) ** 2).sum()
    return slope, intercept, fitted, tss, ess, rss


def half_spread(fitted, resid):
    order = np.argsort(fitted, kind="stable")
    half = len(order) // 2
    lower = sample_sd(resid[order[:half]])
    upper = sample_sd(resid[order[half:]])
    return lower, upper, upper / lower


def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    X, _, y = load_data()
    x4 = X[:, 3]
    n = len(y)
    res = {}

    mean = y.mean()
    sd = sample_sd(y)
    se = sd / math.sqrt(n)
    res["n"] = n
    res["mean_value"] = mean
    res["sd_value"] = sd
    res["se_value"] = se
    res["ci95_low"] = mean - 1.96 * se
    res["ci95_high"] = mean + 1.96 * se

    z = (mean - 8.0) / se
    res["z_stat"] = z
    res["reject_h0"] = abs(z) > 1.96

    res["n_required"] = math.ceil((1.96 * sd / 0.5) ** 2)

    slope, intercept, fitted, tss, ess, rss = simple_ols(x4, y)
    res["reg_slope"] = slope
    res["reg_intercept"] = intercept
    res["tss"] = tss
    res["ess"] = ess
    res["rss"] = rss
    res["tss_equals_ess_plus_rss"] = abs(tss - ess - rss) < 1e-6 * tss
    res["r2"] = 1 - rss / tss
    res["rmse"] = math.sqrt(rss / n)
    res["mae"] = np.abs(y - fitted).mean()

    x4_mean, x4_sd = x4.mean(), sample_sd(x4)
    zx = (x4 - x4_mean) / x4_sd
    m = c = 0.0
    res["gd_J0"] = np.mean(y ** 2)
    for step in range(1, 501):
        e = y - (m * zx + c)
        grad_m = -2 / n * (zx * e).sum()
        grad_c = -2 / n * e.sum()
        m -= 0.1 * grad_m
        c -= 0.1 * grad_c
        if step in (1, 10, 100, 500):
            res[f"gd_J{step}"] = np.mean((y - (m * zx + c)) ** 2)
    res["gd_slope"] = m / x4_sd
    res["gd_intercept"] = c - m * x4_mean / x4_sd
    res["gd_slope_gap"] = abs(res["gd_slope"] - slope)

    lo, hi, ratio = half_spread(fitted, y - fitted)
    res["resid_sd_lower_raw"] = lo
    res["resid_sd_upper_raw"] = hi
    res["resid_sd_ratio_raw"] = ratio

    lx, ly = np.log(x4), np.log(y)
    l_slope, l_int, l_fit, l_tss, _, l_rss = simple_ols(lx, ly)
    loglog_r2 = 1 - l_rss / l_tss
    res["loglog_intercept"] = l_int
    res["loglog_slope"] = l_slope
    res["loglog_r2"] = loglog_r2
    res["loglog_rmse"] = math.sqrt(l_rss / n)
    lo, hi, ratio = half_spread(l_fit, ly - l_fit)
    res["resid_sd_lower_log"] = lo
    res["resid_sd_upper_log"] = hi
    res["resid_sd_ratio_log"] = ratio

    D = np.column_stack([np.ones(n), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    beta = np.linalg.solve(D.T @ D, D.T @ ly)
    for i, b in enumerate(beta):
        res[f"mr_beta_{i}"] = b
    mr_rss = ((ly - D @ beta) ** 2).sum()
    mr_r2 = 1 - mr_rss / l_tss
    res["mr_r2"] = mr_r2
    res["mr_adj_r2"] = adj_r2(mr_r2, n, 7)
    res["loglog_adj_r2"] = adj_r2(loglog_r2, n, 1)

    write_results("part1_results.txt", res)


if __name__ == "__main__":
    main()
