import numpy as np
from collections import OrderedDict
from scipy.stats import wasserstein_distance


def sparse_histogram(x):
    n = len(x)
    hist = OrderedDict()
    for k in np.sort(x):
        hist[k] = hist.get(k, 0) + 1
    for k in hist:
        hist[k] /= n
    return hist


def emd_naive(p: OrderedDict, q: OrderedDict):
    """
    naive implementation of Earth Mover's Distance (EMD)
    linear time complexity but the performance is compromised due to python loop
    """
    sorted_values = np.sort(np.concatenate(
        [list(p.keys()), list(q.keys())], axis=0))
    total_cost = curr_cost = 0
    for i in range(len(sorted_values) - 1):
        v, v_next = sorted_values[i], sorted_values[i + 1]
        curr_cost += p.get(v, 0) - q.get(v, 0)
        total_cost += abs(curr_cost) * (v_next - v)
    return total_cost


def test_emd_naive():
    # test case 1
    p = sparse_histogram([1, 2, 3])
    q = sparse_histogram([101, 102, 103])
    print("Test 1: passed!" if emd_naive(p, q) == 100. else "Test 1: failed!")
    # test case 2
    p = sparse_histogram([1, 2, 3])
    q = sparse_histogram([1, 4, 9])
    print("Test 2: passed!" if emd_naive(p, q) == 8./3 else "Test 2: failed!")

    print(emd_naive(p, q))
    print(wasserstein_distance(
        np.array(list(p.keys())),
        np.array(list(q.keys())),
        np.array(list(p.values())),
        np.array(list(q.values())),
    ))


def sparse_hist_to_vw(p):
    values = np.array(list(p.keys()))
    weights = np.array(list(p.values()))
    return values, weights


def data_emd(x, y):
    p, q = sparse_histogram(x), sparse_histogram(y)
    pv, pw = sparse_hist_to_vw(p)
    qv, qw = sparse_hist_to_vw(q)
    return wasserstein_distance(pv, qv, pw, qw)


if __name__ == "__main__":
    from timeit import timeit
    np.random.seed(1234)
    x = np.random.randn(1000) + 1
    y = np.random.randn(1000) + 3
    p = sparse_histogram(x)
    q = sparse_histogram(y)
    u_values, u_weights = sparse_hist_to_vw(p)
    v_values, v_weights = sparse_hist_to_vw(q)

    print(timeit(stmt="emd_naive(p, q)", globals={"p": p, "q": q, "emd_naive": emd_naive}, number=100))
    print(timeit(stmt="wasserstein_distance(u_values, v_values, u_weights, v_weights)", globals={
        "wasserstein_distance": wasserstein_distance,
        "u_values": u_values, "v_values": v_values,
        "u_weights": u_weights, "v_weights": v_weights
    }, number=100))
    print(timeit(stmt="wasserstein_distance(x, y)", globals={
        "x": x, "y": y, "wasserstein_distance": wasserstein_distance}, number=100))
