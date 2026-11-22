# Klasifikasi Kematangan Pisang dengan Ekstraksi Fitur GLCM

Aplikasi web yang mengklasifikasikan tingkat kematangan pisang dari sebuah foto ke dalam 6 kelas
(freshripe, freshunripe, overripe, ripe, rotten, unripe).

Dataset: Banana Ripeness Classification (Roboflow), 5.616 citra, dibagi `train` / `valid` / `test`.

## Isi folder

| File / folder | Fungsi |
|---|---|
| `dataset_citra.ipynb` | Eksperimen dan pelatihan model |
| `app.py` | Aplikasi Streamlit |
| `models/` | Model hasil pelatihan (dibaca oleh aplikasi) |
| `Dataset/` | Data citra (`train`, `valid`, `test`, masing-masing berisi folder per kelas) |
| `requirements.txt` | Daftar library |

## Cara menjalankan

1. Ekstrak dataset ke folder `Dataset/` sehingga ada `Dataset/train`, `Dataset/valid`, `Dataset/test`.
2. Buka terminal di folder ini, lalu pasang library:
   ```
   python -m pip install -r requirements.txt
   python -m pip install jupyter
   ```
3. Latih model: jalankan `python -m notebook`, buka `dataset_citra.ipynb`, pilih **Kernel > Restart & Run All**, dan tunggu sampai selesai. Model tersimpan di `models/`.
4. Jalankan aplikasi:
   ```
   python -m streamlit run app.py
   ```
   Aplikasi terbuka di `http://localhost:8501`. 
   
   Unggah foto pisang, atau pilih contoh di panel kiri.
