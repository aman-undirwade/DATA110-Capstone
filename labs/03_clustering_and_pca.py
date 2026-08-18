"""DATA110 Lab 03 - K-Means, hierarchical clustering and PCA practice."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

DATA_PATH = "data.csv"
df = pd.read_csv(DATA_PATH)
X = df.select_dtypes(include="number").dropna()
X_scaled = StandardScaler().fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)
print("K-Means centroids:\n", kmeans.cluster_centers_)
print("Cluster counts:\n", pd.Series(labels_kmeans).value_counts())

# Hierarchical / agglomerative clustering
agg = AgglomerativeClustering(n_clusters=2)
labels_agg = agg.fit_predict(X_scaled)
print("Agglomerative counts:\n", pd.Series(labels_agg).value_counts())

# PCA: reduce to two principal components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Components:\n", pca.components_)
