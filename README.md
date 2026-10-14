# Customers Like You — PCA or Truncated SVD

**Student:** Kamya Kanara  
**Submission date:** October 13, 2026

## Business problem
GiftLane wants product-page recommendations ("Customers who bought this also bought...") and an anonymised picture of customer groups. This project uses UCI Online Retail II transaction data to build a customer–product matrix, compare Truncated SVD with PCA, evaluate recommendations against a best-seller baseline, and cluster customers.

## Dataset
- Source: UCI Machine Learning Repository, Online Retail II (CC BY 4.0).
- Raw workbook: `data/raw/online_retail_II.xlsx`
- Original invoice-line count: **1,067,371** across two sheets.
- Time coverage in the brief: December 2009–December 2011.
- Customer IDs are internal processing keys and must not be included in public memo, slides, or figures.

## Success criteria
The project is assessed against the brief's four goals:
1. Compressed factors are materially smaller than a dense customer–product matrix.
2. The model fit/scoring work is fast enough to be considered for a nightly job.
3. Recommendation Hit Rate@10 is compared fairly with the 10-best-seller baseline on held-out purchases; any loss must be reported honestly.
4. The customer map shows 3–6 groups described from observed product and country patterns.

These are evaluation criteria, not a claim that all goals have been achieved. Current measured results are documented below and in `outputs/tables/`.

## Current data-processing results
- Cleaning leaves 776,577 transaction lines after the listed cleaning rules.
- Matrix shape: (5473, 3687) (customers × products).
- Non-zero cells: 413,927.
- Sparsity: 97.9487%.
- Dense 8-byte estimate: 161.43 MB.
- CSR sparse storage: 3.333 MB.
- Purchase value: `log(1 + Quantity)` per transaction line, summed for repeated customer–product pairs.
- Product filter: keep products bought by at least 10 distinct training customers.
- Test period: final two months; held-out products are products not seen in the customer's training history.

## Model results at k = 50
| Measure | Truncated SVD | PCA | Best-seller baseline |
|---|---:|---:|---:|
| Fit time (seconds) | 0.211 | 0.164 | — |
| Peak traced memory (MB) | 4.89 | 4.51 | — |
| Stored customer + product factors (MB) | 1.832 | 1.832 | — |
| Explained variance | Not reported for SVD | 0.65056 | — |
| Relative reconstruction error estimate | 0.58405 | Not reported here | — |
| Hit Rate@10 | 0.2002 | 0.2013 | 0.1929 |
| NDCG@10 | 0.0336 | 0.0340 | Not calculated in comparison row |

**Interpretation:** With the corrected factor-based scoring, both models slightly exceed the current best-seller baseline in Hit Rate@10 on the 1,913 eligible test customers. The difference is modest, so it should not be described as a large improvement. SVD is the preferred primary method because the input is sparse and SVD works without centering. PCA is retained as the required comparison.

Memory note: `tracemalloc` reports Python-traced allocations and may not capture all native NumPy/SciPy/scikit-learn allocations. SVD reconstruction error is an estimate based on retained singular values, not a full explicit matrix reconstruction. Treat these limitations when interpreting the numbers.

## Current customer segments
See `outputs/tables/segments.csv` and `outputs/figures/customer_map.png`. Four clusters are formed from normalized 50-component customer representations and shown on a two-component SVD projection. Segment labels are descriptive summaries of leading product descriptions, not formal customer types.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run from the project root
```bash
python build_matrix.py
python run_pipeline.py
```

`build_matrix.py` loads both workbook sheets, profiles and cleans transactions, creates the time split, and rebuilds the sparse matrix and index maps. `run_pipeline.py` generates the component-curve outputs, PCA/SVD comparison table, and customer map.

## Important output files
- `data/processed/data_profile.csv` — column types, missing values, unique counts.
- `data/processed/cleaning_log.csv` — row counts removed by each cleaning rule.
- `data/processed/matrix_report.csv` — matrix shape, nonzeros, sparsity and storage estimate.
- `outputs/tables/component_curve.csv` — requested k values and compression-quality metrics.
- `outputs/figures/component_curve.png` — compression-quality plot.
- `outputs/tables/pca_vs_svd.csv` — comparison table for k=50 plus baseline.
- `outputs/tables/segments.csv` — anonymised customer-segment summary.
- `outputs/figures/customer_map.png` — customer map without customer IDs.
- `notebooks/analysis.ipynb` — documented, reproducible walkthrough.

## Method decisions and limitations
- Truncated SVD is the primary candidate because it operates directly on sparse input without centering.
- PCA centers columns implicitly; the comparison uses scikit-learn's `arpack` solver.
- The test set only evaluates held-out products that exist in the training product vocabulary. New products unseen by the entire training matrix cannot be recommended by this latent-factor model.
- The current SVD reconstruction error is an approximation based on retained singular values; a full exact reconstruction would require a more expensive explicit calculation.
- Peak memory is an approximate Python-traced measurement.
- Customer clustering is sensitive to representation and distance choices. Inspect cluster sizes and product summaries before making marketing decisions.

## Roles
This is an individual submission; 
