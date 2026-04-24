from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    table_data = None
    sil_score = None
    error_msg = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            try:
                df = pd.read_csv(file)

                # =========================
                # Rapikan nama kolom
                # =========================
                df.columns = df.columns.str.strip().str.lower()

                # =========================
                # Cleaning
                # =========================
                df = df.replace('?', None)
                df = df.dropna()

                # =========================
                # Ubah ke numerik
                # =========================
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='ignore')

                # =========================
                # Ambil kolom numerik
                # =========================
                X = df.select_dtypes(include=['int64', 'float64'])

                if X.shape[1] < 2:
                    error_msg = "Dataset harus punya minimal 2 kolom numerik!"
                    return render_template("index.html", error_msg=error_msg)

                # =========================
                # Normalisasi
                # =========================
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # =========================
                # KMeans
                # =========================
                kmeans = KMeans(n_clusters=3, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X_scaled)

                # =========================
                # Silhouette Score
                # =========================
                sil_score = round(silhouette_score(X_scaled, df['Cluster']), 3)

                # =========================
                # PCA (INI YANG BARU 🔥)
                # =========================
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                df['PCA1'] = X_pca[:, 0]
                df['PCA2'] = X_pca[:, 1]

                # =========================
                # Plot PCA
                # =========================
                plt.figure()
                sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'])
                plt.title("Clustering dengan PCA")

                plot_path = os.path.join("static", "plot.png")
                plt.savefig(plot_path)
                plt.close()

                plot_url = "static/plot.png"

                # =========================
                # Tabel
                # =========================
                table_data = df.head(10).to_html(classes='table table-striped', index=False)

            except Exception as e:
                error_msg = f"Terjadi error: {str(e)}"

    return render_template("index.html",
                           plot_url=plot_url,
                           table_data=table_data,
                           sil_score=sil_score,
                           error_msg=error_msg)


if __name__ == "__main__":
    app.run(debug=True)