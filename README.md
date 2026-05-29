# 🎵 Amazon Music Clustering using K-Means

## 📌 Project Overview

With millions of songs available on music streaming platforms, manually categorizing tracks based on their characteristics is inefficient. This project uses **Unsupervised Machine Learning (K-Means Clustering)** to automatically group songs with similar audio characteristics.

The model analyzes audio features such as danceability, energy, loudness, acousticness, tempo, and valence to discover meaningful patterns and organize songs into distinct clusters.

---

## 🎯 Objectives

* Perform Exploratory Data Analysis (EDA) on Amazon Music song data.
* Select relevant audio features for clustering.
* Normalize features using StandardScaler.
* Determine the optimal number of clusters using the Elbow Method and Silhouette Score.
* Apply K-Means Clustering to group similar songs.
* Visualize clusters using PCA.
* Interpret cluster characteristics based on audio features.
* Export the final clustered dataset.

---

## 📂 Dataset Information

The dataset contains song-level and artist-level information, including:

### Audio Features

* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo
* Duration

### Additional Information

* Song Name
* Artist Name
* Popularity Metrics
* Followers
* Release Date
* Genre Information

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* PCA (Principal Component Analysis)
* Streamlit

---

## 🔍 Project Workflow

### 1. Data Exploration

* Loaded dataset
* Inspected data structure
* Checked missing values
* Removed duplicates
* Visualized feature distributions

### 2. Feature Selection

Selected the following audio features:

```python
audio_features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms'
]
```

### 3. Data Preprocessing

* Handled missing values
* Standardized numerical features using StandardScaler

### 4. Clustering

* Applied K-Means Clustering
* Used Elbow Method to identify suitable K values
* Evaluated clusters using Silhouette Score

### 5. Dimensionality Reduction

* Applied PCA for 2D visualization of clusters

### 6. Cluster Interpretation

Analyzed average feature values within each cluster to understand the musical characteristics represented by each group.

---

## 📊 Evaluation Metrics

### Silhouette Score

Measures how similar a song is to its own cluster compared to other clusters.

### Davies-Bouldin Index

Measures cluster separation and compactness.

### Inertia

Measures within-cluster variation.

---

## 📈 Results

The model successfully grouped songs into meaningful clusters based on audio characteristics.

Example cluster patterns discovered:

### Cluster 0

* High speechiness
* Shorter duration
* Spoken-word / speech-heavy tracks

### Cluster 1

* High acousticness
* Lower energy
* Chill and acoustic-oriented songs

### Cluster 2

* High energy
* High danceability
* Fast tempo
* Dance and party-oriented tracks

These clusters demonstrate that songs with similar audio properties can be automatically grouped without using genre labels.

---

## 📁 Project Structure

```text
amazon-music-clustering/
│
├── Amazon_Music_Clustering.ipynb
├── app.py
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── clustered_music.csv
│   └── metrics.txt
│
└── data/
    └── single_genre_artists.csv
```

---

## 🚀 Streamlit Application

The project also includes a Streamlit application that allows users to:

* Upload a dataset
* Select the number of clusters
* Visualize PCA cluster plots
* View cluster profiles
* Explore clustered songs
* Download clustered results

Run the application:

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

* Song recommendation system
* Similar song search
* Interactive Plotly visualizations
* Genre inference analysis
* Model comparison (K-Means vs DBSCAN)
* Cloud deployment

---

## 👨‍💻 Author

**Gokulraj V**

AI/ML Enthusiast | Machine Learning | Deep Learning | NLP | Generative AI

GitHub: https://github.com/gokulraj-5
