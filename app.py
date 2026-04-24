from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os
import traceback
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def detect_vehicle_columns(df):
    """Deteksi kolom yang relevan untuk kendaraan (akurasi tinggi)"""
    # Mapping keywords untuk fitur kendaraan
    vehicle_keywords = {
        'mpg': ['mpg', 'fuel', 'consumption', 'economy', 'fuel consumption', 'l/100km'],
        'horsepower': ['horsepower', 'hp', 'power', 'tenaga', 'daya'],
        'weight': ['weight', 'berat', 'mass', 'bobot', 'kerb'],
        'acceleration': ['acceleration', 'accel', '0-60', 'percepatan'],
        'displacement': ['displacement', 'cc', 'engine', 'silinder', 'cylinders', 'cylinder'],
        'model_year': ['year', 'model year', 'tahun', 'year model'],
        'cylinders': ['cylinders', 'cylinder', 'silinder', 'cyl']
    }
    
    detected_cols = {}
    used_cols = set()
    
    for feature, keywords in vehicle_keywords.items():
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in used_cols:
                continue
            for keyword in keywords:
                if keyword in col_lower:
                    # Coba konversi ke numeric
                    try:
                        test_series = pd.to_numeric(df[col], errors='coerce')
                        if test_series.notna().sum() > 0:
                            detected_cols[feature] = col
                            used_cols.add(col_lower)
                            break
                    except:
                        pass
            if feature in detected_cols:
                break
    
    return detected_cols

def prepare_features(df, detected_cols):
    """Siapkan fitur untuk clustering"""
    feature_list = []
    feature_names = []
    
    # Prioritas fitur kendaraan
    priority_features = ['mpg', 'horsepower', 'weight', 'acceleration', 'displacement', 'cylinders', 'model_year']
    
    for feature in priority_features:
        if feature in detected_cols:
            col_name = detected_cols[feature]
            try:
                series = pd.to_numeric(df[col_name], errors='coerce')
                if series.notna().sum() > 10:  # Minimal 10 data valid
                    feature_list.append(series)
                    feature_names.append(feature.upper() if feature == 'mpg' else feature.capitalize())
            except:
                pass
    
    # Jika tidak ada fitur kendaraan, ambil semua kolom numerik
    if len(feature_list) < 2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:4]:  # Maksimal 4 fitur
            feature_list.append(df[col])
            feature_names.append(col)
    
    if len(feature_list) < 2:
        return None, None
    
    # Gabungkan jadi DataFrame
    X = pd.concat(feature_list, axis=1)
    X.columns = feature_names
    
    # Hapus baris dengan NaN
    X = X.dropna()
    
    return X, feature_names

@app.route('/', methods=['GET', 'POST'])
def index():
    error_msg = None
    sil_score = None
    plot_url = None
    table_data = None
    elbow_plot_url = None
    silhouette_plot_url = None
    cluster_summary = None
    summary_stats = None
    use_vehicle_mode = False
    
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                # =========================
                # 1. LOAD CSV
                # =========================
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                df = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=enc)
                        break
                    except:
                        continue
                
                if df is None:
                    error_msg = "Tidak bisa membaca file CSV"
                    return render_template('index.html', error_msg=error_msg)
                
                if df.empty:
                    error_msg = "File CSV kosong!"
                    return render_template('index.html', error_msg=error_msg)
                
                # Preview data
                table_data = df.head(10).to_html(classes='table table-striped', index=False)
                summary_stats = f"📊 {len(df):,} baris × {len(df.columns)} kolom"
                
                # =========================
                # 2. DETEKSI FITUR KENDARAAN
                # =========================
                detected_cols = detect_vehicle_columns(df)
                
                if detected_cols:
                    use_vehicle_mode = True
                    summary_stats += f" | 🚗 Mode Kendaraan: {', '.join(detected_cols.keys())}"
                else:
                    summary_stats += " | 📊 Mode General (analisis semua data numerik)"
                
                # =========================
                # 3. PREPARE FITUR
                # =========================
                X, feature_names = prepare_features(df, detected_cols)
                
                if X is None or len(feature_names) < 2:
                    error_msg = f"Tidak cukup data numerik untuk clustering. Minimal 2 kolom numerik dengan data valid. Kolom yang ada: {list(df.columns)}"
                    return render_template('index.html', error_msg=error_msg, table_data=table_data)
                
                if len(X) < 10:
                    error_msg = f"Data setelah cleaning hanya {len(X)} baris. Minimal 10 baris untuk clustering yang valid."
                    return render_template('index.html', error_msg=error_msg, table_data=table_data)
                
                # =========================
                # 4. NORMALISASI
                # =========================
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # =========================
                # 5. TENTUKAN K OPTIMAL (ELBOW + SILHOUETTE)
                # =========================
                max_k = min(10, len(X_scaled) - 1)
                if max_k < 2:
                    n_clusters = 2
                else:
                    # Elbow Method
                    inertias = []
                    k_range = range(1, max_k + 1)
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(X_scaled)
                        inertias.append(kmeans.inertia_)
                    
                    # Silhouette Score
                    silhouette_scores = []
                    for k in range(2, max_k + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(X_scaled)
                        if len(set(labels)) > 1:
                            score = silhouette_score(X_scaled, labels)
                            silhouette_scores.append(score)
                        else:
                            silhouette_scores.append(0)
                    
                    # Pilih K terbaik berdasarkan Silhouette Score
                    if silhouette_scores:
                        best_k_idx = np.argmax(silhouette_scores)
                        n_clusters = best_k_idx + 2
                    else:
                        n_clusters = 3
                    
                    # =========================
                    # 6. PLOT ELBOW METHOD
                    # =========================
                    plt.figure(figsize=(10, 5))
                    plt.plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='#667eea')
                    plt.axvline(x=n_clusters, color='red', linestyle='--', label=f'K terpilih = {n_clusters}')
                    plt.xlabel('Jumlah Cluster (K)', fontsize=12)
                    plt.ylabel('Inertia', fontsize=12)
                    plt.title('Elbow Method untuk Menentukan K Optimal', fontsize=14)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    elbow_img = BytesIO()
                    plt.savefig(elbow_img, format='png', bbox_inches='tight', dpi=100)
                    elbow_img.seek(0)
                    elbow_plot_url = base64.b64encode(elbow_img.getvalue()).decode()
                    plt.close()
                    
                    # =========================
                    # 7. PLOT SILHOUETTE SCORE
                    # =========================
                    plt.figure(figsize=(10, 5))
                    k_range_sil = range(2, max_k + 1)
                    plt.plot(k_range_sil, silhouette_scores, marker='s', linewidth=2, markersize=8, color='#e74c3c')
                    plt.axvline(x=n_clusters, color='red', linestyle='--', label=f'K terpilih = {n_clusters}')
                    plt.xlabel('Jumlah Cluster (K)', fontsize=12)
                    plt.ylabel('Silhouette Score', fontsize=12)
                    plt.title('Silhouette Score untuk Evaluasi K', fontsize=14)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    sil_img = BytesIO()
                    plt.savefig(sil_img, format='png', bbox_inches='tight', dpi=100)
                    sil_img.seek(0)
                    silhouette_plot_url = base64.b64encode(sil_img.getvalue()).decode()
                    plt.close()
                
                # =========================
                # 8. K-MEANS CLUSTERING FINAL
                # =========================
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                X['Cluster'] = labels
                
                # =========================
                # 9. SILHOUETTE SCORE FINAL
                # =========================
                if len(set(labels)) > 1:
                    final_silhouette = silhouette_score(X_scaled, labels)
                    sil_score = f"{final_silhouette:.4f}"
                    
                    # Interpretasi
                    if final_silhouette >= 0.7:
                        sil_score += " (Sangat Baik)"
                    elif final_silhouette >= 0.5:
                        sil_score += " (Baik)"
                    elif final_silhouette >= 0.25:
                        sil_score += " (Cukup)"
                    else:
                        sil_score += " (Lemah, perlu periksa data)"
                else:
                    sil_score = "N/A (Hanya 1 cluster)"
                
                # =========================
                # 10. CLUSTER SUMMARY
                # =========================
                cluster_counts = X['Cluster'].value_counts().sort_index()
                cluster_summary = "<div style='display: flex; gap: 15px; flex-wrap: wrap; justify-content: center;'>"
                for cluster_id in range(n_clusters):
                    count = cluster_counts.get(cluster_id, 0)
                    percentage = (count / len(X)) * 100
                    cluster_summary += f"""
                    <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                                color: white; padding: 15px; border-radius: 15px; 
                                text-align: center; min-width: 100px;'>
                        <div style='font-size: 24px; font-weight: bold;'>Cluster {cluster_id}</div>
                        <div>{count} kendaraan</div>
                        <div style='font-size: 12px; opacity: 0.8;'>({percentage:.1f}%)</div>
                    </div>
                    """
                cluster_summary += "</div>"
                
                # =========================
                # 11. VISUALISASI CLUSTERING (AKURAT)
                # =========================
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot 1: 2D Clustering (2 fitur pertama)
                scatter1 = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                          c=labels, cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
                axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                               c='red', marker='X', s=250, linewidths=3, edgecolors='white', label='Centroids')
                axes[0].set_xlabel(feature_names[0], fontsize=11, fontweight='bold')
                axes[0].set_ylabel(feature_names[1], fontsize=11, fontweight='bold')
                axes[0].set_title(f'Clustering: {feature_names[0]} vs {feature_names[1]}', fontsize=12)
                axes[0].legend()
                plt.colorbar(scatter1, ax=axes[0], label='Cluster')
                
                # Plot 2: Distribution plot per cluster
                if len(feature_names) >= 1:
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    for i in range(min(n_clusters, 5)):
                        cluster_data = X_scaled[labels == i]
                        if len(cluster_data) > 0:
                            axes[1].scatter([i] * len(cluster_data), cluster_data[:, 0], 
                                          alpha=0.5, c=colors[i % len(colors)], label=f'Cluster {i}' if i < 5 else '')
                    axes[1].set_xlabel('Cluster', fontsize=11)
                    axes[1].set_ylabel(feature_names[0], fontsize=11)
                    axes[1].set_title(f'Distribusi {feature_names[0]} per Cluster', fontsize=12)
                    axes[1].set_xticks(range(n_clusters))
                    axes[1].legend()
                
                plt.tight_layout()
                
                img = BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
                
                # Tampilkan hasil clustering dalam tabel
                result_cols = feature_names + ['Cluster']
                display_df = X[result_cols].head(15).round(3)
                table_data = display_df.to_html(classes='table table-striped', index=False)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(traceback.format_exc())
        else:
            error_msg = "Harap upload file CSV (ekstensi .csv)"
    
    return render_template('index.html', 
                         sil_score=sil_score,
                         plot_url=f'data:image/png;base64,{plot_url}' if plot_url else None,
                         elbow_plot_url=f'data:image/png;base64,{elbow_plot_url}' if elbow_plot_url else None,
                         silhouette_plot_url=f'data:image/png;base64,{silhouette_plot_url}' if silhouette_plot_url else None,
                         table_data=table_data,
                         cluster_summary=cluster_summary,
                         summary_stats=summary_stats,
                         use_vehicle_mode=use_vehicle_mode,
                         error_msg=error_msg)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)