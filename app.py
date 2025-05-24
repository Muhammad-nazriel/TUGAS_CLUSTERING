from flask import Flask, render_template, request
import os
import pandas as pd
from model.clustering import process_data, elbow_method, run_kmeans

DATA_PATH = 'dataset/preprocessed_data.csv'

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_k = 4  # default jumlah cluster
    table_html = None
    elbow_inertia_path = 'static/plots/elbow_inertia_vs_k.png'
    elbow_wcss_path = 'static/plots/elbow_wcss_vs_k.png'
    scatter_path = 'static/plots/scatter_clusters.png'
    cluster_dist_path = 'static/plots/cluster_distribution.png'

    # Load data
    if not os.path.exists(DATA_PATH):
        return f"Data file not found at {DATA_PATH}", 404
    df = pd.read_csv(DATA_PATH)

    # Preprocess data dan scaling
    df, X_scaled, features = process_data(df)

    # Jalankan elbow method (bikin plot)
    elbow_method(X_scaled)

    # Jika user kirim form dengan nilai k cluster
    if request.method == 'POST' and 'k_value' in request.form:
        try:
            selected_k = int(request.form['k_value'])
        except ValueError:
            selected_k = 4

    # Jalankan kmeans clustering dengan k terpilih
    df_clustered, labels, centers = run_kmeans(df, X_scaled, features, k=selected_k)

    # Buat tabel HTML hasil clustering
    table_html = df_clustered.to_html(classes='table table-striped table-hover', index=False, border=0)

    return render_template('index.html',
                           selected_k=selected_k,
                           table_html=table_html,
                           elbow_inertia_path=elbow_inertia_path,
                           elbow_wcss_path=elbow_wcss_path,
                           scatter_path=scatter_path,
                           cluster_dist_path=cluster_dist_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
