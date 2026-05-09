from google_play_scraper import Sort, reviews
import pandas as pd

# 1. Gunakan ID aplikasi yang benar (bukan nama aplikasi) untuk menarik data review.
app_id = 'id.dana' 

print(f"Sedang menarik data dari {app_id}...")

# 2. Ambil data
result, _ = reviews(
    app_id,
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=500
)

# 3. Cek apakah data berhasil ditarik
if not result:
    print("Wah, datanya kosong! Coba cek koneksi internet atau App ID-nya lagi.")
else:
    df = pd.DataFrame(result)
    
    # 4. Tampilkan daftar kolom yang tersedia 
    print("Kolom yang tersedia di data ini adalah:", df.columns.tolist())
    
    # 5. Filter kolom yang dibutuhkan
    df = df[['userName', 'score', 'at', 'content']]
    
    # 6. Simpan
    df.to_csv('reviews_dana.csv', index=False)
    print(f"Berhasil! {len(df)} review telah disimpan ke reviews_dana.csv")
    
    # Intip 5 data teratas
    print(df.head())