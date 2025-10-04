import os, re, sys
import requests
import javalang
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, average_precision_score

import torch
import torch.nn.functional as F
from torch import nn, optim


try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:
    print("ERROR: torch_geometric not available. Install PyTorch Geometric per instructions.")
    raise e


INPUT_CSV = "Input_dataset.csv"
FILTER_SMELL = "feature envy"
MAX_TOKENS = 10000  
NODE_EMBED_DIM = 64
GNN_HIDDEN = 128
NUM_GCN_LAYERS = 2
FC_HIDDEN = 128
BATCH_SIZE = 32
EPOCHS = 20
N_SPLITS = 10
RANDOM_STATE = 42
MAX_SEQ_LEN = 10  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_java_code(github_link):
    if not isinstance(github_link, str) or github_link.strip() == "":
        return None
    try:
        raw_link = github_link.replace("/blob/", "/raw/")
        if "#" in github_link:
            line_part = github_link.split("#")[1]
            nums = re.findall(r"\d+", line_part)
            if len(nums) >= 2:
                start, end = int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                start, end = int(nums[0]), None
            else:
                start, end = 1, None
        else:
            start, end = 1, None
        r = requests.get(raw_link, timeout=10)
        if r.status_code != 200:
            return None
        lines = r.text.splitlines()
        if end:
            seg = lines[max(0, start-1):end]
        else:
            seg = lines[start-1:]
        return "\n".join(seg)
    except Exception:
        return None


def build_ast_graph(source_code):
    
    try:
        tree = javalang.parse.parse_member_declaration(source_code)
    except Exception:
        
        try:
            tree = javalang.parse.parse(source_code)
        except Exception:
            return None

    nodes = []
    edges = []

    
    node_stack = [(tree, -1)] 
    idx = 0
    parent_of = {}
    children_of = {}

    while node_stack:
        node, parent_idx = node_stack.pop()
        cur_idx = idx
        idx += 1
       
        label = type(node).__name__
        
        node_text = label
        
        try:
            if hasattr(node, "name") and node.name:
                node_text = f"{label}:{node.name}"
            elif hasattr(node, "member") and getattr(node, "member", None):
                node_text = f"{label}:{getattr(node, 'member')}"
            elif hasattr(node, "qualifier") and getattr(node, "qualifier", None):
                node_text = f"{label}:{getattr(node, 'qualifier')}"
        except Exception:
            pass

        nodes.append(node_text)
        parent_of[cur_idx] = parent_idx
        if parent_idx != -1:
            children_of.setdefault(parent_idx, []).append(cur_idx)

        
        try:
            children = []
            for child in node.children:
                
                if isinstance(child, list):
                    for c in reversed(child):
                        if isinstance(c, javalang.ast.Node):
                            children.append(c)
                elif isinstance(child, javalang.ast.Node):
                    children.append(child)
        except Exception:
            children = []

        for c in children:
            node_stack.append((c, cur_idx))

    
    for v, p in parent_of.items():
        if p != -1:
            edges.append((p, v))
            edges.append((v, p))  

    
    for p, childs in children_of.items():
        for i in range(len(childs)-1):
            a, b = childs[i], childs[i+1]
            edges.append((a, b))
            edges.append((b, a))

    return nodes, edges

from collections import Counter
def build_tokenizer(node_texts_list, max_vocab=MAX_TOKENS):
    
    token_counts = Counter()
    for node_texts in node_texts_list:
        for nt in node_texts:
           
            toks = re.findall(r"[A-Za-z0-9_]+", nt)
            token_counts.update([t.lower() for t in toks])
    most_common = token_counts.most_common(max_vocab-2)  
    vocab = {tok: i+2 for i, (tok, _) in enumerate(most_common)}
    vocab["<PAD>"] = 0
    vocab["<OOV>"] = 1
    return vocab

def node_texts_to_feature_matrix(node_texts, vocab, max_seq_len=MAX_SEQ_LEN):
   
    out = []
    for nt in node_texts:
        toks = re.findall(r"[A-Za-z0-9_]+", nt)
        toks = [t.lower() for t in toks]
        idxs = [vocab.get(t, vocab["<OOV>"]) for t in toks][:max_seq_len]
        # pad
        if len(idxs) < max_seq_len:
            idxs = idxs + [vocab["<PAD>"]] * (max_seq_len - len(idxs))
        out.append(idxs)
    return np.array(out, dtype=np.int64)


def sample_to_pygdata(code_text, label, vocab):
    be = build_ast_graph(code_text)
    if be is None:
        return None
    nodes, edges = be
    if len(nodes) == 0:
        return None

  
    node_token_idxs = node_texts_to_feature_matrix(nodes, vocab, MAX_SEQ_LEN) 
    num_nodes = node_token_idxs.shape[0]

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T  

    data = {
        "x_token_idxs": torch.tensor(node_token_idxs, dtype=torch.long),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "y": torch.tensor([label], dtype=torch.float)
    }
    return Data(**data)

class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, token_emb_dim=NODE_EMBED_DIM, node_emb_dim=NODE_EMBED_DIM,
                 gnn_hidden=GNN_HIDDEN, num_gcn=NUM_GCN_LAYERS, fc_hidden=FC_HIDDEN):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, token_emb_dim, padding_idx=0)
      
        self.node_proj = nn.Linear(token_emb_dim, node_emb_dim)
        
        self.convs = nn.ModuleList()
        for i in range(num_gcn):
            in_ch = node_emb_dim if i == 0 else gnn_hidden
            out_ch = gnn_hidden
            self.convs.append(GCNConv(in_ch, out_ch))
        
        self.fc1 = nn.Linear(gnn_hidden, fc_hidden)
        self.out = nn.Linear(fc_hidden, 1)

    def forward(self, data):
        x_tokens = data.x_token_idxs.to(device)  
        
        token_emb = self.token_emb(x_tokens) 
       
        node_feats = token_emb.mean(dim=1)
        node_feats = self.node_proj(node_feats)  

        x = node_feats
        edge_index = data.edge_index.to(device)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
       
        batch = data.batch.to(device)
        g = global_mean_pool(x, batch) 
        
        g = F.relu(self.fc1(g))
        logits = self.out(g).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, logits

def main():
    df = pd.read_csv(INPUT_CSV)
    # filter
    df = df[df["smell"].str.lower() == FILTER_SMELL].reset_index(drop=True)
    if df.shape[0] == 0:
        print("No rows for smell", FILTER_SMELL)
        return
    df['severity'] = df.apply(lambda x: 4 if x['severity'] in ['critical'] else 3 if x['severity'] in ['major'] else 2 if x['severity'] in ['minor'] else 1, axis = 1)

    df['severity'] = df.apply(lambda x: 1 if x['severity'] in [3,4,2] else 0, axis = 1)
    df = df.rename(columns={'severity':'label'})

   
    cache_col = "_code_snippet"
    if cache_col not in df.columns:
        print("Extracting the codes!!!...")
        df[cache_col] = df["link"].apply(get_java_code)
    else:
        missing = df[df[cache_col].isna()].index
        for i in missing:
            df.at[i, cache_col] = get_java_code(df.at[i, "link"])

    df = df.dropna(subset=[cache_col]).reset_index(drop=True)
    print(f"Samples after extraction: {len(df)}")

   
    node_texts_list = []
    codes = df[cache_col].tolist()
    for code in tqdm(codes, desc="Parsing ASTs for vocab"):
        be = build_ast_graph(code)
        if be is None:
            node_texts_list.append([])
        else:
            nodes, _ = be
            node_texts_list.append(nodes)

    
    vocab = build_tokenizer(node_texts_list, max_vocab=MAX_TOKENS)
    vocab_size = max(vocab.values()) + 1  # id max +1

    data_list = []
    labels = []
    for i, code in enumerate(tqdm(codes, desc="Building graphs")):
        d = sample_to_pygdata(code, int(df.loc[i, "label"]), vocab)
        if d is not None and d.edge_index.shape[1] >= 0 and d.x_token_idxs.shape[0] > 0:
            data_list.append(d)
            labels.append(int(df.loc[i, "label"]))
    print("Total graphs:", len(data_list))
    if len(data_list) == 0:
        print("No graphs created successfully.")
        return

    # CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_all = np.array(labels)
    all_results = []
    fold = 1

    for train_idx, test_idx in skf.split(np.zeros(len(y_all)), y_all):
        print(f"=== Fold {fold} ===")
        train_list = [data_list[i] for i in train_idx]
        test_list = [data_list[i] for i in test_idx]

        model = GNNClassifier(vocab_size=vocab_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        
        for epoch in range(1, EPOCHS+1):
            model.train()
            
            perm = np.random.permutation(len(train_list))
           
            for i in range(0, len(perm), BATCH_SIZE):
                batch_idx = perm[i:i+BATCH_SIZE]
                batch_graphs = [train_list[j] for j in batch_idx]
                batch = Batch.from_data_list(batch_graphs).to(device)
                optimizer.zero_grad()
                probs, logits = model(batch)
                loss = criterion(probs, batch.y.to(device).float())
                loss.backward()
                optimizer.step()
            
        model.eval()
        y_true = []
        y_prob = []
        with torch.no_grad():
            for i in range(0, len(test_list), BATCH_SIZE):
                batch_graphs = test_list[i:i+BATCH_SIZE]
                batch = Batch.from_data_list(batch_graphs).to(device)
                probs, logits = model(batch)
                y_prob.extend(probs.cpu().numpy().tolist())
                y_true.extend(batch.y.cpu().numpy().tolist())

        y_pred = (np.array(y_prob) > 0.5).astype(int)
        y_true = np.array(y_true).astype(int)

        fold_res = {
            "Fold": fold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "CohenKappa": cohen_kappa_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "AUCPR": average_precision_score(y_true, np.array(y_prob)),
        }
        print(f"Fold {fold} metrics:", fold_res)
        all_results.append(fold_res)
        fold += 1

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("all_fold_result.csv", index=False)
    mean_results = results_df.drop(columns=["Fold"]).mean().to_dict()
    pd.DataFrame([mean_results]).to_csv("mean_result.csv", index=False)
    print("Saved all_fold_result.csv and mean_result.csv")

if __name__ == "__main__":
    main()
