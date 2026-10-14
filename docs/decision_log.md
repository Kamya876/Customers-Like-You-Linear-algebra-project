# Decision Log (R1)

| Decision | Reason |
|---|---|
| Use UCI Online Retail II | The assignment specifies this real retail transaction dataset. Both workbook sheets are loaded. |
| Remove cancellation invoices, non-positive quantity/price, non-product codes, missing customer IDs and exact duplicate rows | These records do not represent a clean positive purchase by an identifiable customer for this task. Rules and row counts are saved in `data/processed/cleaning_log.csv`. |
| Time split: last two months | Mimics the real use case: learn from past transactions and evaluate on later purchases. |
| Test labels: customer–product pairs not seen in that customer's training history | Measures whether recommendations can identify a new-to-that-customer product. Products absent from the filtered training vocabulary are not recommendable and are excluded from scored eligible labels. |
| Purchase value: `log(1 + Quantity)` per transaction line, summed for repeated customer/product pairs | Dampens the effect of large wholesale quantities while preserving positive purchase-strength information. This is not the same as applying log after summing quantities. |
| Product filter: at least 10 distinct training customers | Reduces very rare product columns and focuses on products with some repeat-customer evidence. |
| Primary method: Truncated SVD | The matrix is sparse; Truncated SVD works directly without centering and provides customer and product factors useful for recommendation. |
| Compare PCA at the same k=50 | Required by ML5; PCA centers columns and provides explained variance, while SVD is evaluated with an estimated relative reconstruction error. |
| Main comparison uses k=50; k curve tests 10, 20, 50, 100, 200 | k=50 is the selected working point for the required side-by-side comparison; the curve is supplied as evidence for revisiting this choice. Do not claim k=50 is optimal without discussing the curve and business metric. |
| Customer segmentation: K-means with 4 clusters on normalized 50-component customer representations | Four clusters satisfy the assignment's 3–6 range. Normalization reduces the dominance of customers with large factor magnitudes. A 2-component SVD projection is used only for display. |
| Fixed random seed 42 | Supports repeatability of randomized decomposition and clustering. |

## Reproducibility note
The comparison reports measured results from the current run. `tracemalloc` may not include all native-library allocations, and the SVD reconstruction error is an estimate based on retained singular values rather than an explicit full reconstruction.
