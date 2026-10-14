"""Run the full pipeline: matrix -> SVD -> evaluation -> customer map."""
from src import evaluation, customer_map

if __name__ == "__main__":
    print("=== ML5: PCA vs SVD comparison ===")
    evaluation.main()
    print("\n=== ML6: customer map ===")
    customer_map.main()
    print("\nDone. See outputs/ for figures and tables.")
