import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Amazon Music Clustering")
st.caption("Unsupervised clustering of songs using audio features")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"],
)

# Default columns expected from your dataset
DEFAULT_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

k_value = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=3, step=1)
use_extra_features = st.sidebar.checkbox(
    "Include popularity features (experiment only)",
    value=False,
)
run_button = st.sidebar.button("Run Clustering")

# ---------------------------
# Load data
# ---------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    default_path = Path("data/single_genre_artists.csv")
    if default_path.exists():
        df = pd.read_csv(default_path)
        st.info("Loaded default dataset from data/single_genre_artists.csv")
    else:
        st.warning("Upload a CSV file or place single_genre_artists.csv inside the data/ folder.")
        st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Basic Information")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing values", int(df.isnull().sum().sum()))

with st.expander("Show column names"):
    st.write(df.columns.tolist())

# ---------------------------
# Feature selection
# ---------------------------
audio_features = [c for c in DEFAULT_AUDIO_FEATURES if c in df.columns]
if len(audio_features) < 3:
    st.error("Not enough audio features found in the uploaded file.")
    st.stop()

extra_features = ["popularity_songs", "popularity_artists", "followers"]
if use_extra_features:
    selected_features = audio_features + [c for c in extra_features if c in df.columns]
else:
    selected_features = audio_features

st.subheader("Selected Features")
st.write(selected_features)

# ---------------------------
# Clean data
# ---------------------------
work_df = df.copy()
work_df = work_df.drop_duplicates()
work_df = work_df.dropna(subset=selected_features)

X = work_df[selected_features].copy()

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Clustering
# ---------------------------
if run_button:
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    work_df["cluster"] = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, work_df["cluster"])
    dbi = davies_bouldin_score(X_scaled, work_df["cluster"])
    inertia = kmeans.inertia_

    st.subheader("Model Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette Score", f"{sil:.4f}")
    m2.metric("Davies-Bouldin Index", f"{dbi:.4f}")
    m3.metric("Inertia", f"{inertia:.2f}")

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["cluster"] = work_df["cluster"].values

    st.subheader("PCA Cluster Visualization")
    fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color=pca_df["cluster"].astype(str),
        title="Songs grouped by cluster",
        labels={"color": "Cluster"},
        opacity=0.75,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster profile
    st.subheader("Cluster Profile")
    cluster_profile = work_df.groupby("cluster")[audio_features].mean().round(2)
    st.dataframe(cluster_profile, use_container_width=True)

    # Cluster sizes
    st.subheader("Cluster Distribution")
    cluster_counts = work_df["cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["cluster", "count"]
    fig2 = px.bar(cluster_counts, x="cluster", y="count", title="Number of songs in each cluster")
    st.plotly_chart(fig2, use_container_width=True)

    # Sample songs
    st.subheader("Sample Songs from Each Cluster")
    name_col = "name_song" if "name_song" in work_df.columns else None
    artist_col = "name_artists" if "name_artists" in work_df.columns else None

    for cluster_id in sorted(work_df["cluster"].unique()):
        st.markdown(f"### Cluster {cluster_id}")
        cluster_df = work_df[work_df["cluster"] == cluster_id].copy()
        cols = [c for c in [name_col, artist_col, "cluster"] if c is not None and c in cluster_df.columns]
        st.dataframe(cluster_df[cols].head(10), use_container_width=True)

    # Download outputs
    st.subheader("Export Results")
    export_df = work_df.copy()
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download clustered CSV",
        data=csv_data,
        file_name="clustered_music.csv",
        mime="text/csv",
    )

    metrics_text = (
        f"Silhouette Score: {sil:.4f}\n"
        f"Davies-Bouldin Index: {dbi:.4f}\n"
        f"Inertia: {inertia:.2f}\n"
        f"k: {k_value}\n"
    )
    st.download_button(
        label="Download metrics.txt",
        data=metrics_text.encode("utf-8"),
        file_name="metrics.txt",
        mime="text/plain",
    )

    if st.sidebar.checkbox("Show raw clustered data", value=False):
        st.subheader("Clustered Data Preview")
        st.dataframe(export_df.head(200), use_container_width=True)

else:
    st.info("Set k in the sidebar and click 'Run Clustering' to generate results.")
