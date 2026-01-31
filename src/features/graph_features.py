import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from pathlib import Path
import numpy as np
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"


def build_train_graph(train_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, row in train_df.iterrows():
        G.add_edge(
            row["from_account"],
            row["to_account"],
            weight=row["amount"]
        )

    return G


def compute_structural_features(G: nx.DiGraph) -> pd.DataFrame:
    df = pd.DataFrame({
        "account": list(G.nodes),
        "out_degree": dict(G.out_degree()).values(),
        "in_degree": dict(G.in_degree()).values(),
    })
    df["total_degree"] = df["out_degree"] + df["in_degree"]
    return df


def compute_pagerank(G: nx.DiGraph) -> pd.DataFrame:
    pr = nx.pagerank(G, weight="weight")
    return pd.DataFrame({
        "account": pr.keys(),
        "pagerank": pr.values()
    })


def compute_ego_features(train_df: pd.DataFrame) -> pd.DataFrame:
    grp = train_df.groupby("from_account")

    ego = grp["amount"].agg(
        ego_tx_count="count",
        ego_amount_sum="sum",
        ego_amount_mean="mean"
    ).reset_index()

    ego = ego.rename(columns={"from_account": "account"})
    return ego


def compute_node2vec_embeddings(G: nx.DiGraph, dim=16) -> pd.DataFrame:
    n2v = Node2Vec(
        G,
        dimensions=dim,
        walk_length=10,
        num_walks=50,
        workers=2,
        weight_key="weight"
    )

    model = n2v.fit(window=5, min_count=1)

    embeddings = []
    for node in G.nodes:
        vec = model.wv[node]
        row = {"account": node}
        for i in range(dim):
            row[f"n2v_{i}"] = vec[i]
        embeddings.append(row)

    return pd.DataFrame(embeddings)


def build_account_graph_features():
    train_df = pd.read_parquet(DATA_DIR / "train_base.parquet")

    print("Building train-only account graph...")
    G = build_train_graph(train_df)

    print("Structural features...")
    structural = compute_structural_features(G)

    print("PageRank...")
    pr = compute_pagerank(G)

    print("Ego-network features...")
    ego = compute_ego_features(train_df)

    print("Node2Vec embeddings...")
    emb = compute_node2vec_embeddings(G)

    features = (
        structural
        .merge(pr, on="account", how="left")
        .merge(ego, on="account", how="left")
        .merge(emb, on="account", how="left")
    )

    hash_raw = "|".join(sorted(features.columns))
    features.attrs["graph_feature_hash"] = hashlib.sha1(hash_raw.encode()).hexdigest()

    features.to_parquet(DATA_DIR / "graph_features_accounts.parquet", index=False)

    print("Account-level graph features saved")
    print(f"Feature hash: {features.attrs['graph_feature_hash']}")


def attach_graph_features(split: str):
    assert split in {"train", "test"}

    base = pd.read_parquet(DATA_DIR / f"{split}_base.parquet")
    graph = pd.read_parquet(DATA_DIR / "graph_features_accounts.parquet")

    merged = base.merge(
        graph,
        left_on="from_account",
        right_on="account",
        how="left"
    ).drop(columns=["account"])

    merged.fillna(0, inplace=True)

    merged.to_parquet(
        DATA_DIR / f"{split}_graph_features.parquet",
        index=False
    )

    print(f"Graph features attached to {split}")


if __name__ == "__main__":
    build_account_graph_features()
    attach_graph_features("train")
    attach_graph_features("test")
