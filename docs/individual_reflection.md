# Individual Reflection

**Student:** Kamya Kanara  
**Submission date:** October 13, 2026

In this individual project, I worked through the data-engineering and machine-learning workflow for the Customers Like You recommendation problem. The workflow loads and profiles the UCI Online Retail II data, applies cleaning rules, creates a time-based training and test split, and builds a sparse customer–product matrix. I then compared Truncated SVD with PCA, evaluated top-10 recommendations against a best-seller baseline, and created a four-segment customer map using anonymised summaries.

A key lesson from this project is that model output alone is not enough to establish usefulness. The evaluation needs a held-out time period and a simple baseline, and recommendation scoring must correctly align customer and product factors with the product index. The results show that both models are only modestly above the best-seller baseline in Hit Rate@10, so the measured difference should not be presented as a large business improvement. Dimensionality reduction also involves a trade-off: the factors are compact, but some purchase information is lost.

If I continued this work, I would evaluate additional time windows, compare alternative purchase-value definitions, measure memory at the process level, and investigate whether the customer segments remain stable and useful to marketing. I would also review the generated outputs after each run to ensure the written conclusions continue to match the measured results.
