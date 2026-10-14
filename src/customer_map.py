"""ML6: customer map - project on 2 components, cluster, name segments (no IDs)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from config import PROCESSED, FIGURES, TABLES, K_CHOSEN, N_CLUSTERS, RANDOM_SEED
from src.modeling import load_matrix


def main():
    A, cust_ids, prod_ids = load_matrix()
    train = pd.read_csv(PROCESSED / "train.csv")
    svd = TruncatedSVD(n_components=K_CHOSEN, random_state=RANDOM_SEED).fit(A)
    U = svd.transform(A)
    svd2 = TruncatedSVD(n_components=2, random_state=RANDOM_SEED).fit(A)
    X2 = svd2.transform(A)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=20, init="k-means++")
    labels = km.fit_predict(normalize(U))

    n_active = N_CLUSTERS
    print(f"[ML6] Active clusters: {n_active} "
          f"(sizes: {np.bincount(labels, minlength=n_active).tolist()})")
    cust_map = {c: i for i, c in enumerate(cust_ids)}
    train["cluster"] = train["Customer ID"].map(
        {c: labels[i] for c, i in cust_map.items()}
    )
    # Use product descriptions where available, rather than opaque stock codes.
    desc_map = (train.dropna(subset=["Description"])
                .groupby("StockCode")["Description"]
                .agg(lambda x: x.value_counts().index[0]).to_dict())
    rows = []
    segment_names = {}
    for c in range(n_active):
        sub = train[train["cluster"] == c]
        top_codes = (sub.groupby("StockCode")["Quantity"].sum()
                     .sort_values(ascending=False).head(5).index.tolist())
        top_names = [str(desc_map.get(code, code)) for code in top_codes]
        top_country = (sub.groupby("Country")["Customer ID"].nunique()
                       .sort_values(ascending=False).head(3).index.tolist())
        # Ground the segment label in the leading product descriptions.
        label_words = " ".join(top_names[:2]).upper()
        if "BAG" in label_words or "SHOPPER" in label_words:
            seg_name = "Bag & Packaging Buyers"
        elif "HEART" in label_words or "CANDLE" in label_words or "HOLDER" in label_words:
            seg_name = "Candle & Homeware Buyers"
        elif "PAPER" in label_words or "NAPKIN" in label_words:
            seg_name = "Paper Goods Buyers"
        elif "CHRISTMAS" in label_words or "DECORATION" in label_words:
            seg_name = "Seasonal Decor Buyers"
        else:
            seg_name = f"Retail Mix Segment {c+1}"
        if seg_name in segment_names.values():
            seg_name = f"{seg_name} {c+1}"
        segment_names[c] = seg_name
        rows.append({
            "segment": seg_name,
            "n_customers": int(sub["Customer ID"].nunique()),
            "top_products": " | ".join(top_names),
            "top_countries": ", ".join(map(str, top_country)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "segments.csv", index=False)
    print(summary.to_string(index=False))
    plt.figure(figsize=(9, 6))
    for c in range(n_active):
        m = labels == c
        plt.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.5, label=segment_names[c])
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.title("GiftLane customer map (anonymised)")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "customer_map.png", dpi=150)
    plt.close()
    print(f"[ML6] Saved {FIGURES / 'customer_map.png'}")


if __name__ == "__main__":
    main()
