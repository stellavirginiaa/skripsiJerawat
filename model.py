import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Membaca dataset dari file CSV
file_path = r"D:\skripsi_stella\acne_streamlit\extraction_features_value.csv"
df = pd.read_csv(file_path)

# Hapus kolom 'filename' jika ada
if 'filename' in df.columns:
    df = df.drop('filename', axis=1)

# Tentukan fitur dan label
X = df.drop('category', axis=1)   # Fitur (semua kolom kecuali kolom target)
y = df['category']                # Label (kolom target)

# Normalisasi fitur menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Ubah kembali ke DataFrame agar tetap mempertahankan nama kolom
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Membagi data menjadi 80% untuk training dan 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, stratify=y, random_state=42)

# Cek bentuk data hasil split
print("Jumlah data training:", X_train.shape[0])
print("Jumlah data testing:", X_test.shape[0])

# Inisialisasi model KNN dengan jumlah tetangga sebanyak 1
knn = KNeighborsClassifier(n_neighbors=28)

# Latih model
knn.fit(X_train, y_train)

# Prediksi dan evaluasi model pada data uji
y_pred = knn.predict(X_test)

# Evaluasi: Akurasi dan laporan klasifikasi
print(f'Akurasi Model KNN: {accuracy_score(y_test, y_pred):.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Simpan model KNN yang telah dilatih menggunakan joblib
joblib.dump(knn, 'knn_model.joblib')
print("\nModel KNN disimpan sebagai 'knn_model.joblib'.")

# Simpan scaler yang telah digunakan untuk normalisasi fitur
joblib.dump(scaler, 'scaler.joblib')
print("\nScaler disimpan sebagai 'scaler.joblib'.")
