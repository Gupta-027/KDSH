import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from predictor import ConsistencyPredictor


def main():
    df = pd.read_csv("data/train.csv")

    model = ConsistencyPredictor(
        k_per_claim=8,
        max_claims=10,
        contradiction_threshold=0.65,
    )

    preds = []
    confs = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        out = model.predict(
            book_name=r["book_name"],
            char_name=r["char"],
            backstory=r["content"],
            caption=None if pd.isna(r["caption"]) else str(r["caption"]),
        )
        preds.append(out.label)
        confs.append(out.confidence)

    y_true = df["label"].tolist()
    y_pred = preds

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 (macro):", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred))

    df_out = df.copy()
    df_out["pred"] = y_pred
    df_out["pred_conf"] = confs
    df_out.to_csv("train_predictions.csv", index=False)
    print("Wrote train_predictions.csv")


if __name__ == "__main__":
    main()
