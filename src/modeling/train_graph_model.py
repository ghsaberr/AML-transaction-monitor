import torch
import pandas as pd
import numpy as np
from torch_geometric.nn import SAGEConv
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"


# --------------------------------------------------
# Account-level GNN
# --------------------------------------------------
class AccountGNN(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, 64)
        self.conv2 = SAGEConv(64, 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# --------------------------------------------------
# Transaction-level classifier
# --------------------------------------------------
class TxClassifier(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x).view(-1)


def train():
    print("Loading data...")
    df = pd.read_parquet(DATA_DIR / "train_graph_model_input.parquet")
    test = pd.read_parquet(DATA_DIR / "test_graph_model_input.parquet")

    # --------------------------------------------------
    # 0. Data cleaning & scaling
    # --------------------------------------------------
    tab_cols = [
        "log_amount",
        "tx_count_1h",
        "tx_count_24h",
        "tx_count_7d",
        "pagerank",
    ]

    df[tab_cols] = df[tab_cols].fillna(0)
    test[tab_cols] = test[tab_cols].fillna(0)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    test.replace([np.inf, -np.inf], 0, inplace=True)

    scaler = StandardScaler()
    df[tab_cols] = scaler.fit_transform(df[tab_cols])
    test[tab_cols] = scaler.transform(test[tab_cols])

    # --------------------------------------------------
    # 1. Account index mapping
    # --------------------------------------------------
    all_accounts = pd.unique(
        df[["from_account", "to_account"]].values.ravel("K")
    )
    acc2idx = {acc: i for i, acc in enumerate(all_accounts)}
    print(f"Graph nodes: {len(all_accounts):,}")

    df["from_idx"] = df["from_account"].map(acc2idx)
    df["to_idx"] = df["to_account"].map(acc2idx)

    test["from_idx"] = test["from_account"].map(acc2idx).fillna(0).astype(int)
    test["to_idx"] = test["to_account"].map(acc2idx).fillna(0).astype(int)

    # --------------------------------------------------
    # 2. Node feature matrix (train-only)
    # --------------------------------------------------
    emb_cols = [c for c in df.columns if c.startswith("n2v_")]

    node_features_df = df.groupby("from_idx")[emb_cols].mean()

    x_np = np.zeros((len(all_accounts), len(emb_cols)), dtype=np.float32)
    x_np[node_features_df.index.values] = node_features_df.values
    x_np = np.nan_to_num(x_np)

    x = torch.tensor(x_np, dtype=torch.float)

    # --------------------------------------------------
    # 3. Graph edges
    # --------------------------------------------------
    edges = df[["from_idx", "to_idx"]].drop_duplicates()
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    # --------------------------------------------------
    # 4. Model setup
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    x = x.to(device)
    edge_index = edge_index.to(device)

    gnn = AccountGNN(x.size(1)).to(device)
    clf = TxClassifier(32 * 2 + len(tab_cols)).to(device)

    train_tab = torch.tensor(df[tab_cols].values, dtype=torch.float).to(device)
    train_y = torch.tensor(df["label"].values, dtype=torch.float).to(device)

    train_from_idx = torch.tensor(df["from_idx"].values, dtype=torch.long).to(device)
    train_to_idx = torch.tensor(df["to_idx"].values, dtype=torch.long).to(device)

    # --------------------------------------------------
    # 5. Loss weighting (class imbalance handling)
    # --------------------------------------------------
    num_neg = (train_y == 0).sum()
    num_pos = (train_y == 1).sum()
    pos_weight_value = num_neg / (num_pos + 1e-5)

    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Class imbalance weight: {pos_weight.item():.2f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(clf.parameters()),
        lr=1e-3
    )

    # --------------------------------------------------
    # 6. Training loop
    # --------------------------------------------------
    print("Starting weighted training...")
    gnn.train()
    clf.train()

    for epoch in range(1, 21):
        optimizer.zero_grad()

        h = gnn(x, edge_index)
        h_src = h[train_from_idx]
        h_dst = h[train_to_idx]

        tx_features = torch.cat([h_src, h_dst, train_tab], dim=1)

        logits = clf(tx_features)
        loss = criterion(logits, train_y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)

        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    # --------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------
    print("Evaluating on test set...")
    gnn.eval()
    clf.eval()

    with torch.no_grad():
        h = gnn(x, edge_index)

        test_tab = torch.tensor(test[tab_cols].values, dtype=torch.float).to(device)
        test_from_idx = torch.tensor(test["from_idx"].values, dtype=torch.long).to(device)
        test_to_idx = torch.tensor(test["to_idx"].values, dtype=torch.long).to(device)

        h_src = h[test_from_idx]
        h_dst = h[test_to_idx]

        test_features = torch.cat([h_src, h_dst, test_tab], dim=1)
        scores = torch.sigmoid(clf(test_features)).cpu().numpy()

        scores = np.nan_to_num(scores)

        pr_auc = average_precision_score(test["label"].values, scores)

        print("=== Graph Model Results ===")
        print(f"PR-AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    train()
