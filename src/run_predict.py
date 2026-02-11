import pandas as pd
from tqdm import tqdm

from predictor import ConsistencyPredictor


def main():
    df = pd.read_csv("data/test.csv")

    model = ConsistencyPredictor(
        k_per_claim=8,
        max_claims=10,
        contradiction_threshold=0.65,
    )

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        out = model.predict(
            book_name=r["book_name"],
            char_name=r["char"],
            backstory=r["content"],
            caption=None if ("caption" not in df.columns or pd.isna(r.get("caption", None))) else str(r["caption"]),
        )
        rows.append({
            "id": r["id"],
            "label": out.label,                     # string label
            "label_num": 1 if out.label == "consistent" else 0,
            "confidence": out.confidence,
        })

    pd.DataFrame(rows).to_csv("results.csv", index=False)
    print("Wrote results.csv")


if __name__ == "__main__":
    main()
