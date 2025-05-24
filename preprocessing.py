import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path='dataset/Supermarket Sales Cleaned.csv', 
                    output_path='dataset/preprocessed_data.csv'):
    df = pd.read_csv(input_path)

    # Drop kolom yang tidak dibutuhkan
    df.drop(['Invoice ID', 'Date', 'Time'], axis=1, inplace=True)

    # Encode kolom kategorikal dengan LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ['Gender', 'Customer type', 'Payment']:
        df[col] = le.fit_transform(df[col])

    # Fitur numerik untuk dinormalisasi
    num_features = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'Rating']
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data telah diproses dan disimpan di {output_path}")

if __name__ == '__main__':
    preprocess_data()
