import os
import matplotlib
matplotlib.use('Agg')  # headless mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PLOT_DIR = 'static/plots'

def process_data(df):
    os.makedirs(PLOT_DIR, exist_ok=True)
    features = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'Rating']
    df = df.drop_duplicates().dropna()
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, features

def elbow_method(X_scaled, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Plot elbow inertia vs K
    fig1 = plt.figure()
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method (Inertia vs K)')
    elbow_path = os.path.join(PLOT_DIR, 'elbow_inertia_vs_k.png')
    fig1.savefig(elbow_path, bbox_inches='tight')
    plt.close(fig1)
    print(f"[INFO] Elbow inertia plot saved to {elbow_path}")

    # Plot elbow WCSS vs K (sama dengan inertia)
    fig2 = plt.figure()
    plt.plot(range(1, max_k + 1), inertias, marker='o', color='orange')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method (WCSS vs K)')
    wcss_path = os.path.join(PLOT_DIR, 'elbow_wcss_vs_k.png')
    fig2.savefig(wcss_path, bbox_inches='tight')
    plt.close(fig2)
    print(f"[INFO] Elbow WCSS plot saved to {wcss_path}")

def run_kmeans(df, X_scaled, features, k=4):
    os.makedirs(PLOT_DIR, exist_ok=True)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = labels

    # Scatter plot Total vs Rating dengan warna cluster
    fig3 = plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Total'], y=df['Rating'], hue=labels, palette='Set1', s=60)
    plt.title('K-Means Clustering (Total vs Rating)')
    plt.xlabel('Total')
    plt.ylabel('Rating')
    plt.legend(title='Cluster')
    scatter_path = os.path.join(PLOT_DIR, 'scatter_clusters.png')
    fig3.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig3)
    print(f"[INFO] Scatter plot saved to {scatter_path}")

    # Plot distribusi cluster (line plot dengan marker 'o')
    fig4 = plt.figure()
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    plt.plot(cluster_counts.index, cluster_counts.values, marker='o', linestyle='-')
    plt.xlabel('Cluster')
    plt.ylabel('Jumlah Data')
    plt.title('Distribusi Jumlah Data per Cluster')
    dist_path = os.path.join(PLOT_DIR, 'cluster_distribution.png')
    fig4.savefig(dist_path, bbox_inches='tight')
    plt.close(fig4)
    print(f"[INFO] Cluster distribution plot saved to {dist_path}")

    return df, labels, kmeans.cluster_centers_

if __name__ == "__main__":
    data_path = 'dataset/preprocessed_data.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        exit(1)

    print("[INFO] Loading data...")
    df = pd.read_csv(data_path)
    print(f"[INFO] Data loaded. Shape: {df.shape}")

    print("[INFO] Processing data...")
    df, X_scaled, features = process_data(df)

    print("[INFO] Running elbow method...")
    elbow_method(X_scaled)

    print("[INFO] Running KMeans clustering...")
    df, labels, centers = run_kmeans(df, X_scaled, features, k=4)

    print("[INFO] Clustering process completed.")
