# [53] H. Liu, Z. Xu, and Y. Zou,
# “Deep learning based feature envy detection,” ASE 2018

import os
import re
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, matthews_corrcoef,
    average_precision_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
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

    # Convert GitHub link to raw content link
    raw_link = github_link.replace('/blob/', '/raw/')

    # Extract the starting and ending line numbers from the link
    line_range = github_link.split('#')[1]
    
    start_line, end_line = map(int, re.findall(r'\d+', line_range))
    print("start_line ",start_line)
    print("end_line ",end_line)
    # Send a GET request to fetch the content
    response = requests.get(raw_link)

    # Check if the request was successful
    if response.status_code == 200:
        # Split the content into lines
        lines = response.text.split('\n')

        # Extract lines within the specified range
        relevant_lines = lines[start_line-1:end_line]

        # Join the lines to get the code content
        
        code_content = " ".join(relevant_lines)
        
    else:
        print("Failed to fetch content from the GitHub link")
        # code_values.append(None)
        code_content=None
    return code_content

# -----------------------------
# Step 2: Severity → Label
# -----------------------------
# df["label"] = df["severity"].map(
#     lambda x: 0 if str(x).lower() in ["none", "minor"] else 1
# )
# df["code"] = df["link"].apply(
#     lambda url: get_java_code(url) if isinstance(url, str) else None
# )

# -----------------------------
# Step 3: GitHub Code Extraction
# -----------------------------

# valid_smells = ["feature envy", "long method"]
# df_dir_path_name = df_dir_path_name[df_dir_path_name["severity"].isin(valid_smells)].reset_index(drop=True)

# --- fetch Java code only for these rows ---
count=0;
print("before DF shape ",df_dir_path_name.shape)

df_dir_path_name = df_dir_path_name[~df_dir_path_name["smell"].isin(["blob", "data class", "long method"])].reset_index(drop=True)

print("after DF shape ",df_dir_path_name.shape)

# for index, row in df_dir_path_name.iterrows():
#         print("------Extracting code No ", index," ---- Name of the codeSmell ", df_dir_path_name['smell'].loc[index]," -------")
#         java_code = get_java_code(row['link'])
#         df_dir_path_name["code"]=java_code
#         count=count+1
#         if(count==50):
#             break
#         # df_dir_path_name["code"] = df_dir_path_name["link"].apply(lambda url: get_java_code(url))


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

# Add extracted code as a new column
df_dir_path_name["code"] = extracted_code

# Drop rows where code could not be extracted
df_dir_path_name = df_dir_path_name.dropna(subset=["code"]).reset_index(drop=True)

print(" Final dataset shape after extraction:", df_dir_path_name.shape)


print("df_dir_path_name \n",df_dir_path_name)
# # Apply code extraction (replace link with actual code text)
# df["code"] = df["link"].apply(lambda url: get_java_code(url))

# Drop rows with failed extraction
df = df_dir_path_name.dropna(subset=["code"]).reset_index(drop=True)

df['severity'] = df.apply(lambda x: 4 if x['severity'] in ['critical'] else 3 if x['severity'] in ['major'] else 2 if x['severity'] in ['minor'] else 1, axis = 1)
df = df.rename(columns={'smell':'dir_name'})

print("DF shape ",df.shape)



# df = df[df['0'].notna()]

df['severity'] = df.apply(lambda x: 1 if x['severity'] in [3,4,2] else 0, axis = 1)
df = df.rename(columns={'severity':'label'})



print(df)

# -----------------------------
# Step 4: Tokenization
# -----------------------------
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["code"])
X = tokenizer.texts_to_sequences(df["code"])
X = pad_sequences(X, maxlen=200)   # same as Liu et al.
y = df["label"].values
print(X)
print("y=\n",y)
# -----------------------------
# Step 5: 10-Fold Cross Validation
# -----------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_results = []

fold = 1
for train_idx, test_idx in kf.split(X, y):
    print(f"\n===== Fold {fold} =====")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # -----------------------------
    # Step 6: CNN model (as per Liu et al.)
    # -----------------------------
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=200),
        Conv1D(64, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=6, batch_size=32, verbose=0)

    # -----------------------------
    # Step 7: Predictions & Metrics
    # -----------------------------
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype("int32")

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

# -----------------------------
# Step 8: Save Results
# -----------------------------
results_df = pd.DataFrame(all_results)
results_df.to_csv("all_fold_result.csv", index=False)

mean_results = results_df.drop(columns=["Fold"]).mean().to_dict()
mean_results_df = pd.DataFrame([mean_results])
mean_results_df.to_csv("mean_result.csv", index=False)

print("all_fold_result.csv and mean_result.csv generated successfully")
