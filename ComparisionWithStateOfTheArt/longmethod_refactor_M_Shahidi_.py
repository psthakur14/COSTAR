import re
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, average_precision_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


INPUT_CSV = "input_dataset.csv"
ALL_FOLD_CSV = "all_fold_results.csv"
MEAN_CSV = "mean_result.csv"
N_SPLITS = 10
MAX_FEATURES = 5000
MAX_EPOCHS = 6
BATCH_SIZE = 32


def fetch_code_from_github(link, timeout=10):
    
    if not isinstance(link, str) or link.strip() == "":
        return None
    try:
        raw_link = link.replace("/blob/", "/raw/")
        start, end = None, None
        if "#" in link:
            part = link.split("#", 1)[1]
            nums = re.findall(r"\d+", part)
            if len(nums) >= 2:
                start, end = int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                start = int(nums[0])
        resp = requests.get(raw_link, timeout=timeout)
        if resp.status_code != 200:
            return None
        lines = resp.text.splitlines()
        if start:
            segment = lines[start - 1:end] if end else lines[start - 1:]
            return "\n".join(segment)
        return resp.text
    except Exception:
        return None



def prepare_dataset(df):
    # Filter only Long Method smell
    df = df[df["smell"].str.lower() == "long method"].reset_index(drop=True)

    df['severity'] = df.apply(lambda x: 4 if x['severity'] in ['critical'] else 3 if x['severity'] in ['major'] else 2 if x['severity'] in ['minor'] else 1, axis = 1)
    df['severity'] = df.apply(lambda x: 1 if x['severity'] in [3,4,2] else 0, axis = 1)
    df = df.rename(columns={'severity':'label'})

    # Fetch code from GitHub
    codes = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fetching code"):
        code = fetch_code_from_github(row["link"])
        codes.append(code if code else None)
    df["code"] = codes

    
    df = df.dropna(subset=["code"]).reset_index(drop=True)

   
    return df


def vectorize_code(codes):
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, token_pattern=r"[A-Za-z_]\w+")
    X = tfidf.fit_transform(codes)
    return X, tfidf


def build_model(input_dim):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_cross_validation(X, y):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    all_results = []
    fold = 1
    for train_idx, test_idx in kf.split(X, y):
        print(f"\n=== Fold {fold} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(X.shape[1])
        model.fit(X_train.toarray(), to_categorical(y_train, 2),
                  epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        probs = model.predict(X_test.toarray(), verbose=0)
        preds = np.argmax(probs, axis=1)

        fold_res = {
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "CohenKappa": cohen_kappa_score(y_test, preds),
            "MCC": matthews_corrcoef(y_test, preds),
            "AUCPR": average_precision_score(y_test, probs[:, 1])
        }
        print(fold_res)
        all_results.append(fold_res)
        fold += 1

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(ALL_FOLD_CSV, index=False)
    mean_df = pd.DataFrame([results_df.drop(columns=["Fold"]).mean()])
    mean_df.to_csv(MEAN_CSV, index=False)
    print(f"\n Saved results to {ALL_FOLD_CSV} and {MEAN_CSV}")


def main():
    df = pd.read_csv(INPUT_CSV)
    df = prepare_dataset(df)

    X, _ = vectorize_code(df["code"].tolist())
    y = df["label"].values

    run_cross_validation(X, y)


if __name__ == "__main__":
    main()
