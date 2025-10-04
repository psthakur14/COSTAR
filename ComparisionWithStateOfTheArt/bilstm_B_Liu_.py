# Paper 54: B. Liu et al., “Deep learning based feature envy detection boosted by real-world examples,” ESEC/FSE 2023

import pandas as pd
from sklearn.model_selection import train_test_split

import os, re, requests
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, average_precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("Input_dataset.csv")   # CSV must contain "severity" and "link"

def df_cleaning(df_):
  print("data frame shape ",df_.shape)
  df_ = df_.dropna(subset=['link'])
  print("After removing NA - data frame shape ",df_.shape)

  df_.reset_index(inplace=True, drop=True)

  columns_to_keep = ['smell', 'severity', 'path','link']
  df_new= df_[columns_to_keep]

  return df_new

df_dir_path_name=df_cleaning(df)

import requests
import re
def get_java_code(code_link):
  
    github_link = code_link
    # github_link = 'https://github.com/apache/syncope/blob/114c412afbfba24ffb4fbc804e5308a823a16a78/client/idrepo/ui/src/main/java/org/apache/syncope/client/ui/commons/ConnIdSpecialName.java/#L35-L37'

    
    raw_link = github_link.replace('/blob/', '/raw/')
    
    line_range = github_link.split('#')[1]
    
    start_line, end_line = map(int, re.findall(r'\d+', line_range))
    print("start_line ",start_line)
    print("end_line ",end_line)
   
    response = requests.get(raw_link)

   
    if response.status_code == 200:
       
        lines = response.text.split('\n')
        
        relevant_lines = lines[start_line-1:end_line]
     
        
        code_content = " ".join(relevant_lines)
        
    else:
        print("Failed to fetch content from the GitHub link")
        
        code_content=None
    return code_content


print("before DF shape ",df_dir_path_name.shape)

df_dir_path_name = df_dir_path_name[~df_dir_path_name["smell"].isin(["blob", "data class","long method"])].reset_index(drop=True)

print("after DF shape ",df_dir_path_name.shape)

extracted_code = []
# count=0
for idx, row in df_dir_path_name.iterrows():
    print(f" Processing row {idx+1}/{len(df_dir_path_name)} | Link: {row['link']}")
    code = get_java_code(row["link"])
    
    if code is not None:
        extracted_code.append(code)
        print(f" Code extracted successfully for row {idx}")
    else:
        extracted_code.append(None)
        print(f"Failed to extract code for row {idx}")
    count=count+1
    # if(count==50):
    #     break

df_dir_path_name["code"] = extracted_code


df_dir_path_name = df_dir_path_name.dropna(subset=["code"]).reset_index(drop=True)

print(" Final dataset shape after extraction:", df_dir_path_name.shape)



print("df_dir_path_name \n",df_dir_path_name)


df = df_dir_path_name.dropna(subset=["code"]).reset_index(drop=True)

df['severity'] = df.apply(lambda x: 4 if x['severity'] in ['critical'] else 3 if x['severity'] in ['major'] else 2 if x['severity'] in ['minor'] else 1, axis = 1)
df = df.rename(columns={'smell':'dir_name'})

print("DF shape ",df.shape)


df['severity'] = df.apply(lambda x: 1 if x['severity'] in [3,4,2] else 0, axis = 1)
df = df.rename(columns={'severity':'label'})



print(df)


MAX_WORDS = 10000
MAX_LEN = 300

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df["code"])
X = tokenizer.texts_to_sequences(df["code"])
X = pad_sequences(X, maxlen=MAX_LEN)
y = df["label"].astype(int).values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def build_bilstm_model():
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_results = []
fold = 1

for train_idx, test_idx in kf.split(X, y):
    print(f"\n=== Fold {fold} ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_bilstm_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    fold_results = {
        "Fold": fold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "CohenKappa": cohen_kappa_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUCPR": average_precision_score(y_test, y_pred_prob),
    }
    all_results.append(fold_results)
    fold += 1

results_df = pd.DataFrame(all_results)
results_df.to_csv("all_fold_result.csv", index=False)

mean_results = results_df.drop(columns=["Fold"]).mean().to_dict()
pd.DataFrame([mean_results]).to_csv("mean_result.csv", index=False)

print(" Results saved to all_fold_result.csv and mean_result.csv")