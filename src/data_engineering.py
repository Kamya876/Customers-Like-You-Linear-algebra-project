"""Data Engineer tasks DE1-DE5."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from config import (RAW_FILE, PROCESSED, TEST_MONTHS, PURCHASE_VALUE,
                    MIN_CUSTOMERS_PER_PRODUCT, RANDOM_SEED)

NON_PRODUCT_CODES = {
    "POST", "D", "M", "BANK CHARGES", "C2", "DOT", "PADS",
    "CRUK", "S", "AMAZONFEE", "B", "SP1002",
    "gift_0001_20", "gift_0001_40", "gift_0001_50", "gift_0001_30", "gift_0001_10",
    "TEST001", "TEST002", "ADJUST", "ADJUST2",
}


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    rename = {
        "InvoiceNo": "Invoice", "UnitPrice": "Price",
        "CustomerID": "Customer ID",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    print(f"[DE1] Loaded {len(df):,} rows from {len(sheets)} sheets")
    print(f"[DE1] Columns: {df.columns.tolist()}")
    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    profile.to_csv(PROCESSED / "data_profile.csv")
    print(profile)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    log = []
    n0 = len(df)
    mask = ~df["Invoice"].astype(str).str.startswith("C")
    log.append(("Cancellation invoices", int(n0 - mask.sum())))
    df = df[mask]
    n = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    log.append(("Non-positive quantity or price", int(n - len(df))))
    n = len(df)
    df = df[~df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)]
    df = df[~df["StockCode"].astype(str).str.contains(r"[A-Za-z]{3,}", regex=True, na=False)]
    log.append(("Non-product codes", int(n - len(df))))
    n = len(df)
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    log.append(("Missing Customer ID", int(n - len(df))))
    n = len(df)
    df = df.drop_duplicates()
    log.append(("Duplicate rows", int(n - len(df))))
    log_df = pd.DataFrame(log, columns=["Rule", "Rows_removed"])
    log_df.loc[len(log_df)] = ["TOTAL removed", int(n0 - len(df))]
    log_df.loc[len(log_df)] = ["Remaining rows", int(len(df))]
    log_df.to_csv(PROCESSED / "cleaning_log.csv", index=False)
    print(log_df.to_string(index=False))
    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame):
    df = df.sort_values("InvoiceDate").copy()
    cutoff = df["InvoiceDate"].max() - pd.DateOffset(months=TEST_MONTHS)
    train_all = df[df["InvoiceDate"] <= cutoff].copy()
    test_all = df[df["InvoiceDate"] > cutoff].copy()
    # Vectorized customer-product pair lookup avoids a slow Python row-wise apply
    # over tens of thousands of test rows.
    train_all["StockCode"] = train_all["StockCode"].astype(str)
    test_all["StockCode"] = test_all["StockCode"].astype(str)
    train_pairs = pd.MultiIndex.from_frame(train_all[["Customer ID", "StockCode"]])
    test_pairs = pd.MultiIndex.from_frame(test_all[["Customer ID", "StockCode"]])
    test_new = test_all.loc[~test_pairs.isin(train_pairs)].copy()
    train_all.to_csv(PROCESSED / "train.csv", index=False)
    test_new.to_csv(PROCESSED / "test.csv", index=False)
    print(f"[DE3] train rows={len(train_all):,} | test rows={len(test_new):,}")
    print(f"[DE3] cutoff={cutoff} | test customers={test_new['Customer ID'].nunique():,}")
    return train_all, test_new


def build_sparse_matrix(train: pd.DataFrame):
    """DE4: sparse customer x product matrix from TRAINING only."""
    # Normalise StockCode to string so mixed str/int codes sort consistently
    train = train.copy()
    train["StockCode"] = train["StockCode"].astype(str)

    prod_counts = train.groupby("StockCode")["Customer ID"].nunique()
    keep_products = prod_counts[prod_counts >= MIN_CUSTOMERS_PER_PRODUCT].index
    train = train[train["StockCode"].isin(keep_products)]

    cust_ids = sorted(train["Customer ID"].unique())
    prod_ids = sorted(train["StockCode"].unique())
    cust_map = {c: i for i, c in enumerate(cust_ids)}
    prod_map = {p: j for j, p in enumerate(prod_ids)}

    rows = train["Customer ID"].map(cust_map).values
    cols = train["StockCode"].map(prod_map).values

    if PURCHASE_VALUE == "raw":
        vals = train["Quantity"].values.astype(np.float32)
    elif PURCHASE_VALUE == "binary":
        vals = np.ones(len(train), dtype=np.float32)
    else:
        vals = np.log1p(train["Quantity"].values).astype(np.float32)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(cust_ids), len(prod_ids)))
    A.sum_duplicates()

    sp.save_npz(PROCESSED / "matrix.npz", A)
    pd.Series(cust_ids, name="Customer ID").to_csv(PROCESSED / "customer_ids.csv", index=False)
    pd.Series(prod_ids, name="StockCode").to_csv(PROCESSED / "product_ids.csv", index=False)

    dense_mb = A.shape[0] * A.shape[1] * 8 / 1e6
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    sparsity = 1 - A.nnz / (A.shape[0] * A.shape[1])
    report = pd.DataFrame([{
        "shape": str(A.shape), "non_zeros": int(A.nnz),
        "sparsity_pct": round(sparsity * 100, 4),
        "dense_MB": round(dense_mb, 2), "sparse_MB": round(sparse_mb, 3),
    }])
    report.to_csv(PROCESSED / "matrix_report.csv", index=False)
    print(report.to_string(index=False))
    return A, cust_map, prod_map

def main():
    df = load_raw()
    df = clean(df)
    train, test = time_split(df)
    build_sparse_matrix(train)


if __name__ == "__main__":
    main()
