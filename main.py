import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("auto-mpg.csv")

print("Data Awal:")
print(df.head())

print("\nInfo Data:")
print(df.info())

print("\nCek Missing Value:")
print(df.isnull().sum())

# =========================
# 2. Cleaning Data
# =========================
df = df.replace('?', None)
df = df.dropna()

df['horsepower'] = pd.to_numeric(df['horsepower'])

print("\nSetelah Cleaning:")
print(df.isnull().sum())

# =========================
# 3. Pilih Fitur
# =========================
features = ["mpg", "horsepower", "weight", "acceleration"]
X = df[features]

# =========================
# 4. Normalisasi
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. Elbow Method
# =========================
inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# =========================
# 5B. Silhouette Score
# =========================
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure()
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Silhouette Score')
plt.title('Metode Silhouette Score')
plt.show()

# =========================
# 6. K-Means Clustering
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nHasil Clustering:")
print(df[['mpg', 'horsepower', 'weight', 'acceleration', 'Cluster']].head())

# =========================
# 7. Visualisasi
# =========================
plt.figure()
sns.scatterplot(x=df['mpg'], y=df['weight'], hue=df['Cluster'])
plt.title("Hasil Clustering Kendaraan")
plt.xlabel("MPG")
plt.ylabel("Weight")
plt.show()