from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    error_msg = None
    sil_score = None
    plot_url = None
    table_data = None
    elbow_plot_url = None
    silhouette_plot_url = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                # =========================
                # 1. Load Dataset
                # =========================
                df = pd.read_csv(filepath)
                
                # Preview data
                table_data = df.head(10).to_html(classes='table table-striped', index=False)
                
                # =========================
                # 2. Cleaning Data
                # =========================
                # Handle '?' and missing values
                df = df.replace('?', pd.NA)
                df = df.dropna()
                
                # Convert horsepower to numeric if needed
                if 'horsepower' in df.columns:
                    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
                    df = df.dropna(subset=['horsepower'])
                
                # =========================
                # 3. Pilih Fitur Numerik
                # =========================
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Pilih fitur yang relevan (prioritaskan yang umum)
                priority_features = ['mpg', 'horsepower', 'weight', 'acceleration', 
                                    'displacement', 'cylinders', 'model year']
                
                features = [f for f in priority_features if f in numeric_cols]
                
                # Jika tidak ada, ambil kolom numerik pertama
                if len(features) < 2:
                    features = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                
                if len(features) >= 2:
                    X = df[features].copy()
                    
                    # =========================
                    # 4. Normalisasi
                    # =========================
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # =========================
                    # 5. K-Means Clustering
                    # =========================
                    # Optimal K = 3 (default)
                    n_clusters = 3
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df['Cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # =========================
                    # 6. Silhouette Score
                    # =========================
                    if len(set(df['Cluster'])) > 1:
                        sil_score = round(silhouette_score(X_scaled, df['Cluster']), 4)
                    
                    # =========================
                    # 7. Elbow Method Plot
                    # =========================
                    inertias = []
                    k_range = range(1, 11)
                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(X_scaled)
                        inertias.append(km.inertia_)
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
                    plt.xlabel('Jumlah Cluster (K)', fontsize=12)
                    plt.ylabel('Inertia', fontsize=12)
                    plt.title('Elbow Method for Optimal K', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    
                    elbow_img = BytesIO()
                    plt.savefig(elbow_img, format='png', bbox_inches='tight', dpi=100)
                    elbow_img.seek(0)
                    elbow_plot_url = base64.b64encode(elbow_img.getvalue()).decode()
                    plt.close()
                    
                    # =========================
                    # 8. Silhouette Score Plot
                    # =========================
                    sil_scores = []
                    k_range_sil = range(2, 11)
                    for k in k_range_sil:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(X_scaled)
                        if len(set(labels)) > 1:
                            score = silhouette_score(X_scaled, labels)
                            sil_scores.append(score)
                        else:
                            sil_scores.append(0)
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(k_range_sil, sil_scores, marker='s', linewidth=2, markersize=8, color='green')
                    plt.xlabel('Jumlah Cluster (K)', fontsize=12)
                    plt.ylabel('Silhouette Score', fontsize=12)
                    plt.title('Silhouette Score Analysis', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    
                    sil_img = BytesIO()
                    plt.savefig(sil_img, format='png', bbox_inches='tight', dpi=100)
                    sil_img.seek(0)
                    silhouette_plot_url = base64.b64encode(sil_img.getvalue()).decode()
                    plt.close()
                    
                    # =========================
                    # 9. Clustering Visualization
                    # =========================
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Plot 1: MPG vs Weight
                    if 'mpg' in df.columns and 'weight' in df.columns:
                        scatter1 = axes[0].scatter(df['mpg'], df['weight'], 
                                                  c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
                        axes[0].set_xlabel('MPG', fontsize=11)
                        axes[0].set_ylabel('Weight', fontsize=11)
                        axes[0].set_title('Clustering: MPG vs Weight', fontsize=12)
                        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
                    
                    # Plot 2: Horsepower vs Acceleration
                    if 'horsepower' in df.columns and 'acceleration' in df.columns:
                        scatter2 = axes[1].scatter(df['horsepower'], df['acceleration'], 
                                                  c=df['Cluster'], cmap='plasma', s=50, alpha=0.7)
                        axes[1].set_xlabel('Horsepower', fontsize=11)
                        axes[1].set_ylabel('Acceleration', fontsize=11)
                        axes[1].set_title('Clustering: Horsepower vs Acceleration', fontsize=12)
                        plt.colorbar(scatter2, ax=axes[1], label='Cluster')
                    
                    plt.tight_layout()
                    
                    # Save plot
                    img = BytesIO()
                    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode()
                    plt.close()
                    
                    # Tampilkan hasil clustering di tabel
                    result_cols = features + ['Cluster']
                    if all(col in df.columns for col in result_cols):
                        table_data = df[result_cols].head(10).to_html(classes='table table-striped', index=False)
                    
                else:
                    error_msg = "Dataset membutuhkan minimal 2 kolom numerik untuk clustering"
                    
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
        else:
            error_msg = "Harap upload file CSV yang valid"
    
    return render_template('index.html', 
                         sil_score=sil_score,
                         plot_url=f'data:image/png;base64,{plot_url}' if plot_url else None,
                         elbow_plot_url=f'data:image/png;base64,{elbow_plot_url}' if elbow_plot_url else None,
                         silhouette_plot_url=f'data:image/png;base64,{silhouette_plot_url}' if silhouette_plot_url else None,
                         table_data=table_data,
                         error_msg=error_msg)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)