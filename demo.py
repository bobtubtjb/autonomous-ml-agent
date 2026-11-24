"""Generate a synthetic regression case without downloading any data."""
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    destination = Path("demo-data")
    destination.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    n_train, n_test = 400, 80
    area = rng.uniform(40, 300, n_train + n_test)
    category = rng.choice(["office", "retail", "school"], size=len(area))
    table = pd.DataFrame({"id": np.arange(len(area)), "area": area, "category": category})
    target = 1.7 * area + 30 * (category == "retail") + rng.normal(0, 8, len(area))
    table.iloc[:n_train].assign(target=target[:n_train]).to_csv(destination / "train.csv", index=False)
    test = table.iloc[n_train:]
    test.to_csv(destination / "test.csv", index=False)
    test[["id"]].assign(target=0.0).to_csv(destination / "sample_submission.csv", index=False)
    (destination / "task.md").write_text("Synthetic regression task. Metric: RMSE.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
