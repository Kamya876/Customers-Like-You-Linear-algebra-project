"""ML Engineer tasks ML1-ML5, ML7."""
import time
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from config import PROCESSED, RANDOM_SEED, K_VALUES, K_CHOSEN, TOP_N_RECS


def load_matrix():
    A = sp.load_npz(PROCESSED / "matrix.npz")
    cust_ids = np.loadtxt(PROCESSED / "customer_ids.csv", skiprows=1, dtype=int)
    prod_ids = np.loadtxt(PROCESSED / "product_ids.csv", skiprows=1, dtype=str)
    return A, cust_ids, prod_ids


METHOD_JUSTIFICATION = "Truncated SVD: sparse-native, gives U*S and V together."


def fit_svd(A, k, seed=RANDOM_SEED):
    t0 = time.time()
    svd = TruncatedSVD(n_components=k, random_state=seed)
    svd.fit(A)
    return svd, time.time() - t0


def variance_curve_svd(A, ks=K_VALUES):
    A_norm = np.sqrt((A.data ** 2).sum())
    full_svd = TruncatedSVD(n_components=min(min(A.shape) - 1, max(ks)),
                            random_state=RANDOM_SEED)
    full_svd.fit(A)
    sv = full_svd.singular_values_
    rows = []
    for k in ks:
        err = np.sqrt(np.sum(sv[k:] ** 2)) / A_norm if k < len(sv) else 0.0
        rows.append({"k": int(k), "rel_recon_error": round(float(err), 5)})
    return rows


def fit_pca(A, k):
    t0 = time.time()
    pca = PCA(n_components=k, svd_solver="arpack", random_state=RANDOM_SEED)
    pca.fit(A)
    return pca, time.time() - t0


def product_factors_svd(svd):
    return normalize(svd.components_.T)


def score_customer(U_S, bought_idx, P_vec, top_n=TOP_N_RECS):
    """Recommend products using the factors for products the customer bought.

    U_S is retained for backwards compatibility; the profile is built from
    product factors, so bought_idx must index rows of P_vec.
    """
    bought_idx = np.asarray(bought_idx, dtype=int)
    if bought_idx.size == 0:
        return np.array([], dtype=int)
    bought_idx = bought_idx[(bought_idx >= 0) & (bought_idx < P_vec.shape[0])]
    if bought_idx.size == 0:
        return np.array([], dtype=int)
    profile = P_vec[bought_idx].mean(axis=0)
    scores = P_vec @ profile
    scores[bought_idx] = -np.inf
    return np.argsort(-scores)[:top_n]


def hit_rate_at_k(A_train, U_S, P_vec, test_df, cust_map, prod_map, k=TOP_N_RECS):
    held = {}
    for _, row in test_df.iterrows():
        c = int(row["Customer ID"]); p = row["StockCode"]
        if c in cust_map and p in prod_map:
            held.setdefault(c, set()).add(prod_map[p])
    hits, total = 0, 0
    bought_counts = []
    for c, prods in held.items():
        ci = cust_map[c]
        bought = A_train[ci].indices
        bought_counts.append(len(bought))
        recs = score_customer(U_S, bought, P_vec, k)
        total += 1
        if any(r in prods for r in recs):
            hits += 1
    if bought_counts:
        print(f"  [diagnostic] {total} test customers | "
              f"median bought={int(np.median(bought_counts))} | "
              f"mean bought={np.mean(bought_counts):.1f} | "
              f"cold-start (0 bought)={sum(1 for x in bought_counts if x == 0)} | "
              f"sparse (<=5 bought)={sum(1 for x in bought_counts if x <= 5)}")
    return (hits / total if total else 0.0), total


def baseline_hit_rate(A_train, test_df, cust_map, prod_map, k=TOP_N_RECS):
    pop = np.asarray(A_train.sum(axis=0)).ravel()
    top_pop = np.argsort(-pop)[: max(k * 20, 200)]
    held = {}
    for _, row in test_df.iterrows():
        c = int(row["Customer ID"]); p = row["StockCode"]
        if c in cust_map and p in prod_map:
            held.setdefault(c, set()).add(prod_map[p])
    hits, total = 0, 0
    for c, prods in held.items():
        ci = cust_map[c]
        bought = set(A_train[ci].indices)
        recs = [p for p in top_pop if p not in bought][:k]
        total += 1
        if any(r in prods for r in recs):
            hits += 1
    return (hits / total if total else 0.0), total


def ndcg_at_k(recs, relevant, k=TOP_N_RECS):
    """Normalised discounted cumulative gain at k for a single customer."""
    dcg = 0.0
    for i, r in enumerate(recs[:k]):
        if r in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_over_test(A_train, U_S, P_vec, test_df, cust_map, prod_map, k=TOP_N_RECS):
    held = {}
    for _, row in test_df.iterrows():
        c = int(row["Customer ID"]); p = row["StockCode"]
        if c in cust_map and p in prod_map:
            held.setdefault(c, set()).add(prod_map[p])
    scores = []
    for c, prods in held.items():
        ci = cust_map[c]
        bought = A_train[ci].indices
        recs = score_customer(U_S, bought, P_vec, k)
        scores.append(ndcg_at_k(recs, prods, k))
    return float(np.mean(scores)) if scores else 0.0, len(scores)
