"""Core ML evaluation: component curves and PCA vs Truncated SVD comparison."""
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from config import TABLES, FIGURES, PROCESSED, K_CHOSEN, K_VALUES, RANDOM_SEED
from src.modeling import load_matrix, product_factors_svd, hit_rate_at_k, baseline_hit_rate, ndcg_over_test


def measured_fit(model, A):
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(A)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1e6


def main():
    A, cust_ids, prod_ids = load_matrix()
    test_df = pd.read_csv(PROCESSED / "test.csv")
    cust_map = {int(c): i for i, c in enumerate(cust_ids)}
    prod_map = {str(p): i for i, p in enumerate(prod_ids)}
    A_norm = float(np.sqrt(np.sum(A.data.astype(np.float64) ** 2)))
    max_k = max(K_VALUES)
    curve_rows = []

    # Fit up to max k once per method; use cumulative metrics for each requested k.
    print("[ML2] Fitting SVD for k curve...")
    svd_curve = TruncatedSVD(n_components=max_k, random_state=RANDOM_SEED)
    svd_seconds, svd_peak = measured_fit(svd_curve, A)
    sv = np.asarray(svd_curve.singular_values_, dtype=float)
    # This is a truncated-spectrum estimate; clearly label it as approximate.
    for k in K_VALUES:
        tail_energy = max(0.0, A_norm**2 - float(np.sum(sv[:k]**2)))
        err = np.sqrt(tail_energy) / A_norm if A_norm else np.nan
        curve_rows.append({"method":"Truncated SVD", "k":k,
                           "relative_reconstruction_error_estimate":err,
                           "explained_variance_ratio":np.nan})

    print("[ML2] Fitting PCA for k curve...")
    pca_curve = PCA(n_components=max_k, svd_solver="arpack", random_state=RANDOM_SEED)
    pca_seconds, pca_peak = measured_fit(pca_curve, A)
    cumulative = np.cumsum(pca_curve.explained_variance_ratio_)
    for k in K_VALUES:
        curve_rows.append({"method":"PCA", "k":k,
                           "relative_reconstruction_error_estimate":np.nan,
                           "explained_variance_ratio":float(cumulative[k-1])})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(TABLES / "component_curve.csv", index=False)
    plt.figure(figsize=(8,5))
    svd_rows = curve_df[curve_df.method=="Truncated SVD"]
    pca_rows = curve_df[curve_df.method=="PCA"]
    plt.plot(svd_rows.k, svd_rows.relative_reconstruction_error_estimate, marker="o", label="Truncated SVD: relative error estimate")
    plt.plot(pca_rows.k, pca_rows.explained_variance_ratio, marker="s", label="PCA: cumulative explained variance")
    plt.xlabel("Number of components (k)")
    plt.ylabel("Metric value (different scales; see legend)")
    plt.title("Compression quality by component count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "component_curve.png", dpi=160)
    plt.close()

    # Comparison at the selected k. Measure independently for fair timings.
    results=[]
    for name in ["Truncated SVD","PCA"]:
        print(f"[ML5] Fitting {name} at k={K_CHOSEN} for evaluation...")
        model = (TruncatedSVD(n_components=K_CHOSEN, random_state=RANDOM_SEED)
                 if name=="Truncated SVD" else
                 PCA(n_components=K_CHOSEN, svd_solver="arpack", random_state=RANDOM_SEED))
        fit_s, peak_mb = measured_fit(model, A)
        U = model.transform(A)
        P = normalize(model.components_.T)
        factor_mb = (U.nbytes + P.nbytes)/1e6
        hr, n = hit_rate_at_k(A, U, P, test_df, cust_map, prod_map)
        ndcg, _ = ndcg_over_test(A, U, P, test_df, cust_map, prod_map)
        if name=="Truncated SVD":
            sv_local=np.asarray(model.singular_values_, dtype=float)
            err=np.sqrt(max(0.0,A_norm**2-float(np.sum(sv_local**2))))/A_norm if A_norm else np.nan
            var=np.nan
            comp1 = "Primarily high-volume purchase patterns; verify with product loadings."
        else:
            var=float(np.sum(model.explained_variance_ratio_))
            err=np.nan
            # describe PCA component using the largest absolute product loadings
            load=np.argsort(np.abs(model.components_[0]))[-5:][::-1]
            products=pd.read_csv(PROCESSED/"product_ids.csv")["StockCode"].astype(str).tolist()
            comp1="Strongest product loadings: "+", ".join(products[i] for i in load)
        results.append({
            "method":name,"k":K_CHOSEN,"fit_seconds":round(fit_s,3),
            "peak_memory_MB_tracemalloc":round(peak_mb,2),
            "stored_factors_MB":round(factor_mb,3),
            "explained_variance":round(var,5) if np.isfinite(var) else "",
            "relative_reconstruction_error_estimate":round(float(err),5) if np.isfinite(err) else "",
            "hit_rate_at_10":round(hr,4),"ndcg_at_10":round(ndcg,4),
            "eligible_test_customers":n,"component_1_interpretation":comp1
        })

    base_hr, base_n = baseline_hit_rate(A, test_df, cust_map, prod_map)
    results.append({"method":"Best-seller baseline","k":"","fit_seconds":"",
                    "peak_memory_MB_tracemalloc":"","stored_factors_MB":"",
                    "explained_variance":"","relative_reconstruction_error_estimate":"",
                    "hit_rate_at_10":round(base_hr,4),"ndcg_at_10":"",
                    "eligible_test_customers":base_n,"component_1_interpretation":"Not applicable"})
    pd.DataFrame(results).to_csv(TABLES/"pca_vs_svd.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print("\n[ML2] Curve written to", TABLES/"component_curve.csv")
    print("[ML2] Plot written to", FIGURES/"component_curve.png")
    print("Note: tracemalloc measures Python-traced allocations and may not include all native-library memory.")
    print("Note: SVD reconstruction error is an estimate based on retained singular values, not a full explicit reconstruction.")


if __name__ == "__main__":
    main()
