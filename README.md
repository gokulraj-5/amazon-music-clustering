# 🎵 Amazon Music Clustering using K-Means

## 📖 Project Overview

Music streaming platforms contain millions of songs with diverse audio characteristics. Manually organizing these songs into meaningful groups is difficult and time-consuming.

This project applies **Unsupervised Machine Learning (K-Means Clustering)** to automatically group songs based on their audio features such as danceability, energy, loudness, acousticness, tempo, and valence. The resulting clusters help identify songs with similar musical characteristics and listening experiences.

---

## 🎯 Problem Statement

The goal of this project is to automatically discover patterns in music data and group similar songs together without using predefined labels or genres.

By analyzing audio features, the model creates clusters that can be used for:

* Personalized playlist generation
* Music recommendation systems
* Song discovery
* Artist and music trend analysis

---

## 📂 Dataset

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
* Duration (ms)

### Additional Features

* Song Popularity
* Artist Popularity
* Followers
* Release Date
* Song Name
* Artist Name
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
* Jupyter Notebook

---

## 🔍 Project Workflow

### 1. Data Exploration (EDA)

* Loaded dataset
* Examined dataset structure
* Checked missing values
* Removed duplicate records
* Visualized feature distributions
* Generated correlation heatmap

---

### 2. Feature Selection

Selected the following audio features for clustering:

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

These features describe the rhythm, mood, energy, and acoustic characteristics of songs.

---

### 3. Data Preprocessing

* Removed missing values
* Standardized numerical features using StandardScaler
* Prepared data for clustering

---

### 4. K-Means Clustering

* Applied K-Means clustering
* Used the Elbow Method to identify suitable cluster counts
* Evaluated cluster quality using Silhouette Score

---

### 5. Dimensionality Reduction

Applied Principal Component Analysis (PCA) to visualize clusters in two dimensions.

---

### 6. Cluster Evaluation

The clustering model was evaluated using:

* Silhouette Score
* Davies-Bouldin Index
* Inertia

### Results

| Metric               | Value       |
| -------------------- | ----------- |
| Silhouette Score     | 0.2423      |
| Davies-Bouldin Index | 1.5702      |
| Inertia              | 658335.0813 |

---

## 📊 Cluster Analysis

The model successfully identified three meaningful song groups:

### Cluster 0 – Speech-Oriented Tracks

Characteristics:

* Very high speechiness
* Shorter duration
* Spoken-word and dialogue-heavy content

---

### Cluster 1 – Acoustic & Chill Songs

Characteristics:

* High acousticness
* Lower energy
* Relaxed and mellow musical style

---

### Cluster 2 – High-Energy Dance Tracks

Characteristics:

* High energy
* High danceability
* Faster tempo
* Suitable for workouts, parties, and mainstream listening

---

## 📈 Visualizations

The project includes:

* Feature Distribution Plots
* Correlation Heatmap
* Elbow Method Curve
* Silhouette Analysis
* PCA Cluster Visualization
* Cluster Distribution Chart
* Cluster Feature Comparison Heatmap

---

## 📁 Project Structure

```text
amazon-music-clustering/
│
├── Amazon_Music_Clustering.ipynb
├── README.md
├── requirements.txt
│
├── outputs/
│   ├── clustered_music.csv
│   └── metrics.txt
│
└── data/
    └── single_genre_artists.csv
```

---

## 🚀 How to Run

### Clone Repository

```bash
git clone <repository-url>
cd amazon-music-clustering
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Amazon_Music_Clustering.ipynb
```

and run all cells.

---

## 💡 Key Learnings

Through this project, I gained hands-on experience in:

* Exploratory Data Analysis (EDA)
* Feature Selection
* Data Standardization
* K-Means Clustering
* Elbow Method
* Silhouette Score Analysis
* PCA Visualization
* Cluster Interpretation
* Unsupervised Machine Learning

---

## 🔮 Future Improvements

* Streamlit dashboard for interactive exploration
* Song recommendation engine
* Similar song search using cosine similarity
* Interactive Plotly visualizations
* Comparison with DBSCAN and Hierarchical Clustering
* Cloud deployment

---

## 👨‍💻 Author

**Gokulraj V**

Machine Learning | Deep Learning | NLP | Generative AI

GitHub: https://github.com/gokulraj-5
