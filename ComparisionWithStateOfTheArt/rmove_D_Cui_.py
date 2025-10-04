
import os
import re
import requests
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, average_precision_score
)

import javalang
import torch
from transformers import AutoTokenizer, AutoModel

INPUT_CSV = "Input_dataset.csv"
TEXT_CACHE_COL = "_snippet"
CODEBERT_MODEL = "microsoft/codebert-base"
MAX_LEN_TOKENS = 256  
BATCH_SIZE_EMB = 16
N_SPLITS = 10
RANDOM_STATE = 42
CLASSIFIER = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

def get_java_code(github_link, timeout=10):
    """Return raw code snippet from a GitHub blob link with optional #Lstart-Lend"""
    if not isinstance(github_link, str) or github_link.strip() == "":
        return None
    try:
        raw_link = github_link.replace("/blob/", "/raw/")
        
        start_line, end_line = None, None
        if "#" in github_link:
            part = github_link.split("#", 1)[1]
            nums = re.findall(r"\d+", part)
            if len(nums) >= 2:
                start_line, end_line = int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                start_line = int(nums[0])
        r = requests.get(raw_link, timeout=timeout)
        if r.status_code != 200:
            
            r = requests.get(github_link, timeout=timeout)
            if r.status_code != 200:
                return None
        text = r.text
        lines = text.splitlines()
        if start_line is not None:
            if end_line is not None:
                lines = lines[max(0, start_line - 1): end_line]
            else:
                lines = lines[max(0, start_line - 1):]
        return "\n".join(lines).strip()
    except Exception:
        return None

def extract_structural_features(java_snippet):
    
    feats = {
        "method_length": 0,
        "num_calls": 0,
        "num_field_access": 0,
        "num_conditions": 0,
        "num_loops": 0,
        "num_params": 0
    }
    if not isinstance(java_snippet, str) or java_snippet.strip() == "":
        return feats

    lines = java_snippet.splitlines()
    feats["method_length"] = len(lines)

    parsed = None
    try:
        parsed = javalang.parse.parse_member_declaration(java_snippet)
    except Exception:
        try:
            parsed = javalang.parse.parse(java_snippet)
        except Exception:
            parsed = None

   
    if parsed is not None:
       
        for path, node in parsed:
            nodename = type(node).__name__
            if nodename == "MethodInvocation":
                feats["num_calls"] += 1
            elif nodename in ("MemberReference", "FieldAccess"):
                feats["num_field_access"] += 1
            elif nodename in ("IfStatement", "TernaryExpression", "SwitchStatement"):
                feats["num_conditions"] += 1
            elif nodename in ("ForStatement", "WhileStatement", "DoStatement", "EnhancedForControl"):
                feats["num_loops"] += 1
            elif nodename == "FormalParameter":
                pass

        try:
            for _, node in parsed:
                if isinstance(node, javalang.tree.MethodDeclaration):
                    params = getattr(node, "parameters", None)
                    if params is not None:
                        feats["num_params"] = len(params)
                    break
        except Exception:
            feats["num_params"] = 0
    else:
        
        text = java_snippet
        feats["num_calls"] = len(re.findall(r"\w+\s*\(", text))
        feats["num_field_access"] = len(re.findall(r"\bthis\.\w+|\w+\.\w+", text))
        feats["num_conditions"] = len(re.findall(r"\bif\b|\bswitch\b|\?:", text))
        feats["num_loops"] = len(re.findall(r"\bfor\b|\bwhile\b|\bdo\b", text))
        m = re.search(r"\b[\w\<\>\[\]]+\s+(\w+)\s*\(([^)]*)\)", text)
        if m:
            params_text = m.group(2).strip()
            feats["num_params"] = 0 if params_text == "" else len([p for p in params_text.split(",") if p.strip() != ""])
    return feats

print("Loading CodeBERT tokenizer & model (this may download >400MB)...")
tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
codebert = AutoModel.from_pretrained(CODEBERT_MODEL)
codebert.eval()
if torch.cuda.is_available():
    codebert.to("cuda")

@torch.no_grad()
def get_codebert_embedding(batch_texts):
   
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN_TOKENS, return_tensors="pt")
    if torch.cuda.is_available():
        for k in inputs:
            inputs[k] = inputs[k].to("cuda")
    outputs = codebert(**inputs)
    last_hidden = outputs.last_hidden_state  
    
    cls_emb = last_hidden[:, 0, :].cpu().numpy()
    return cls_emb


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df[~df["smell"].isin(["blob", "data class","long method"])].reset_index(drop=True)
    df['severity'] = df.apply(lambda x: 4 if x['severity'] in ['critical'] else 3 if x['severity'] in ['major'] else 2 if x['severity'] in ['minor'] else 1, axis = 1)

    df['severity'] = df.apply(lambda x: 1 if x['severity'] in [3,4,2] else 0, axis = 1)
    df = df.rename(columns={'severity':'label'})

    if "link" not in df.columns or "label" not in df.columns:
        raise RuntimeError("Input CSV must contain 'link' and 'label' columns.")

    df = df.dropna(subset=["link", "label"]).reset_index(drop=True)
    print("Loaded", len(df), "rows.")

    if TEXT_CACHE_COL not in df.columns:
        df[TEXT_CACHE_COL] = df["link"].apply(get_java_code)
    else:
        miss = df[df[TEXT_CACHE_COL].isna()].index
        for i in miss:
            df.at[i, TEXT_CACHE_COL] = get_java_code(df.at[i, "link"])

    df = df.dropna(subset=[TEXT_CACHE_COL]).reset_index(drop=True)
    print("Snippets available:", len(df))

    print("Extracting structural features...")
    struct_feats = []
    for s in tqdm(df[TEXT_CACHE_COL].tolist()):
        f = extract_structural_features(s)
        struct_feats.append([f["method_length"], f["num_calls"], f["num_field_access"],
                             f["num_conditions"], f["num_loops"], f["num_params"]])
    struct_feats = np.array(struct_feats, dtype=np.float32)

    print("Computing CodeBERT embeddings (batched)...")
    texts = df[TEXT_CACHE_COL].astype(str).tolist()
    emb_list = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE_EMB)):
        batch = texts[i:i+BATCH_SIZE_EMB]
        emb = get_codebert_embedding(batch)  
        emb_list.append(emb)
    emb_all = np.vstack(emb_list).astype(np.float32)  

    X = np.hstack([struct_feats, emb_all])  
    y = df["label"].astype(int).values

    np.save("_features.npy", X)
    np.save("_labels.npy", y)
    print("Saved features to _features.npy and _labels.npy")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold = 1
    results = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        print(f"=== Fold {fold} ===")
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = CLASSIFIER  
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te)
        y_pred = (y_prob > 0.5).astype(int)

        fold_res = {
            "Fold": fold,
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, zero_division=0),
            "Recall": recall_score(y_te, y_pred, zero_division=0),
            "F1": f1_score(y_te, y_pred, zero_division=0),
            "CohenKappa": cohen_kappa_score(y_te, y_pred),
            "MCC": matthews_corrcoef(y_te, y_pred),
            "AUCPR": average_precision_score(y_te, y_prob)
        }
        print("Fold", fold, ":", fold_res)
        results.append(fold_res)
        fold += 1

    results_df = pd.DataFrame(results)
    results_df.to_csv("all_fold_result.csv", index=False)
    mean_results = results_df.drop(columns=["Fold"]).mean().to_dict()
    pd.DataFrame([mean_results]).to_csv("mean_result.csv", index=False)
    print("Saved all_fold_result.csv and mean_result.csv")

if __name__ == "__main__":
    main()
