"""
cluster.py

TF-IDF + K-Means clustering pipeline over keyword/context text, plus a
separate numeric clustering pass over document-level exposure statistics.

Note on the PCA axes: PC1/PC2 here are the two directions of maximum
variance in the TF-IDF term-frequency space of the keyword+context text.
They are reported and plotted as exactly that -- "PC1" / "PC2" -- rather
than being relabeled as "sentiment" or "intensity". PCA on a bag-of-words
matrix has no inherent sentiment meaning; claiming it does without
separately validating that a component correlates with labeled sentiment
would be a methodological overstatement. If you want an actual sentiment
axis, that requires either a labeled sentiment score to correlate against
component loadings, or a purpose-built sentiment model (e.g. VADER, or a
finetuned classifier) -- PCA of TF-IDF is not a substitute for that.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 6
CUSTOM_STOPWORDS = {
    "question", "questions", "limit", "signal", "clarify", "clarification",
    "respond", "response", "chance", "addressed", "hedging", "cover",
}


def cluster_keywords(kw_df: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    """
    TF-IDF vectorize keyword+context text, K-Means cluster, and project
    to 2D via PCA for visualization. Returns the enriched dataframe plus
    the fraction of variance the 2D projection explains (report this
    alongside any plot -- a low number means the 2D picture is a lossy
    view of a higher-dimensional cluster structure).
    """
    df = kw_df[~kw_df["keyword"].isin(CUSTOM_STOPWORDS)].copy()
    df["text"] = df["keyword"] + " " + df["context"].fillna("")

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    X = vectorizer.fit_transform(df["text"])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    explained = pca.explained_variance_ratio_.sum()
    return df, vectorizer, kmeans, explained


def enrich_with_exposure(kw_clustered: pd.DataFrame, exposure_df: pd.DataFrame) -> pd.DataFrame:
    """Join keyword-level clusters onto document-level exposure stats and
    derive risk_score / match_intensity."""
    exposure_df = exposure_df.rename(columns={"id": "exposure_run_id"})
    merged = kw_clustered.merge(exposure_df, on="exposure_run_id", how="left")
    merged["risk_score"] = merged["pct_sentences_with_match"]
    merged["match_intensity"] = (
        merged["total_direct_matches"]
        / (merged["total_direct_matches"] + merged["total_cosine_matches"] + 1e-9)
    )
    return merged


def cluster_summary(merged: pd.DataFrame) -> pd.DataFrame:
    stats = merged.groupby("cluster").agg(
        mean_risk=("risk_score", "mean"),
        std_risk=("risk_score", "std"),
        median_risk=("risk_score", "median"),
        mean_intensity=("match_intensity", "mean"),
        count=("cluster", "count"),
    ).reset_index()
    stats["z_score"] = (stats["mean_risk"] - stats["mean_risk"].mean()) / stats["mean_risk"].std()
    return stats


def cluster_exposure_runs(exposure_df: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    """
    Separate clustering pass on document-level numeric features only
    (no text) -- clusters documents by their overall match volume/shape
    rather than by keyword theme.
    """
    cluster_cols = [
        "total_direct_matches", "total_cosine_matches", "total_sentences_in_document",
        "total_sentences_with_match", "pct_sentences_with_match",
    ]
    df = exposure_df.dropna(subset=cluster_cols).copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[cluster_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    stats = df.groupby("cluster").agg(
        mean_pct_match=("pct_sentences_with_match", "mean"),
        mean_direct_matches=("total_direct_matches", "mean"),
        mean_sentences=("total_sentences_in_document", "mean"),
        count=("cluster", "count"),
    ).reset_index()
    stats["z_score"] = (stats["mean_pct_match"] - stats["mean_pct_match"].mean()) / stats["mean_pct_match"].std()

    return df, stats
