

import os
import re
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, average_precision_score
)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

INPUT_CSV = "Input_dataset.csv"      
FILTER_SMELL = "feature envy"
MAX_WORDS = 10000
MAX_LEN = 300
EMBEDDING_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 10
N_SPLITS = 10
RANDOM_STATE = 42

def get_java_code(github_link):
    """Return raw code snippet from GitHub blob link. If invalid or fails, returns None."""
    if not isinstance(github_link, str) or github_link.strip() == "":
        return None
    try:
        raw_link = github_link.replace("/blob/", "/raw/")
        
        if "#" in github_link:
           
            line_part = github_link.split("#")[1]
            nums = re.findall(r"\d+", line_part)
            if len(nums) >= 2:
                start_line, end_line = int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                start_line = int(nums[0]); end_line = None
            else:
                start_line, end_line = 1, None
        else:
            start_line, end_line = 1, None

        resp = requests.get(raw_link, timeout=10)
        if resp.status_code != 200:
            return None
        lines = resp.text.splitlines()
        if end_line:
            snippet_lines = lines[max(0, start_line - 1):end_line]
        else:
            snippet_lines = lines[start_line - 1:]
        
        return " ".join(snippet_lines).strip() if snippet_lines else None
    except Exception:
        return None

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_context", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b) 
        ait = tf.tensordot(uit, self.u, axes=1)  
        a = tf.nn.softmax(ait, axis=1) 
        a_expanded = tf.expand_dims(a, axis=-1)  
        weighted_input = inputs * a_expanded 
        output = tf.reduce_sum(weighted_input, axis=1)  
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_model(max_words=MAX_WORDS, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN, lstm_units=LSTM_UNITS):
    seq_in = Input(shape=(max_len,), name="seq_in")
    x = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(seq_in)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = SelfAttention()(x)  
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=seq_in, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

df = pd.read_csv(INPUT_CSV)


df = df[df["smell"].str.lower() == FILTER_SMELL].reset_index(drop=True)


df['severity'] = df.apply(lambda x: 4 if x['severity'] in ['critical'] else 3 if x['severity'] in ['major'] else 2 if x['severity'] in ['minor'] else 1, axis = 1)

df['severity'] = df.apply(lambda x: 1 if x['severity'] in [3,4,2] else 0, axis = 1)
df = df.rename(columns={'severity':'label'})


if "label" not in df.columns:
    raise RuntimeError("Input CSV must have 'label' column with 0/1 values.")
df["label"] = df["label"].astype(int)


CACHE_COL = "_extracted_code"
if CACHE_COL not in df.columns:
    df[CACHE_COL] = df["link"].apply(get_java_code)
else:
   
    missing = df[df[CACHE_COL].isna()].index
    for i in missing:
        df.at[i, CACHE_COL] = get_java_code(df.at[i, "link"])


df = df.dropna(subset=[CACHE_COL]).reset_index(drop=True)
print(f"[Data] examples after filter & extraction: {len(df)}")


tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=None)
tokenizer.fit_on_texts(df[CACHE_COL].astype(str).tolist())
sequences = tokenizer.texts_to_sequences(df[CACHE_COL].astype(str).tolist())
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = df["label"].astype(int).values

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold = 1
all_results = []

for train_idx, test_idx in kf.split(X, y):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_model()
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    
    fold_res = {
        "Fold": fold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "CohenKappa": cohen_kappa_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUCPR": average_precision_score(y_test, y_pred_prob),
    }
    all_results.append(fold_res)
    print(f"Fold {fold} metrics: {fold_res}")
    fold += 1


results_df = pd.DataFrame(all_results)
results_df.to_csv("all_fold_result.csv", index=False)

mean_results = results_df.drop(columns=["Fold"]).mean().to_dict()
pd.DataFrame([mean_results]).to_csv("mean_result.csv", index=False)

print("\n Saved all_fold_result.csv and mean_result.csv")
