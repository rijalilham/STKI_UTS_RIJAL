# UTS STKI – Sistem Temu Kembali Informasi Parfum HMNS
## README (Dokumentasi Lengkap)

---

# 1. Deskripsi Project
Project ini merupakan implementasi Sistem Temu Kembali Informasi (STKI) yang dibangun untuk memenuhi UTS mata kuliah STKI. Sistem ini mengimplementasikan dua model pencarian utama:

1. Boolean Retrieval Model  
2. Vector Space Model (TF-IDF + Cosine Similarity)

Korpus dibuat manual berupa 10 deskripsi parfum HMNS tanpa web crawling.

---

# 2. Struktur Folder Project

```
stki-uts-A11.2022.14113-Alrijal Nur Ilham/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── boolean_ir.py
│   ├── vsm_ir.py
│   ├── eval.py
│   ├── analyze_results.py
│   └── search.py
│
├── app/
│   └── main.py
│
├── notebooks/
│   └── UTS_STKI_A11.2022.14113.ipynb
│
├── reports/
│   ├── laporan.pdf
│   └── readme.md
│
└── requirements.txt
```

---

# 3. Cara Menjalankan Project

## 3.1 Install Dependency
```
pip install -r requirements.txt
```

## 3.2 Preprocessing
```
python src/preprocess.py
```

## 3.3 Boolean Retrieval
```
python src/boolean_ir.py
```

## 3.4 VSM (TF-IDF)
```
python src/vsm_ir.py
```

## 3.5 Menu Utama (CLI)
```
python src/search.py
```

---

# 4. Alur Proses Sistem
- Preprocessing (case folding, tokenizing, stopword, stemming)
- Boolean Retrieval (AND/OR/NOT)
- VSM TF-IDF
- Evaluasi IR

---

# 5. Asumsi
- Dokumen hanya 10 deskripsi parfum
- Stopword list sederhana
- Tidak ada operator kurung dalam boolean

---

# 6. Teknologi
- Python 3
- Numpy
- Sastrawi
- Scikit-learn
- Jupyter Notebook

---

# 7. Pengembang
Alrijal Nur Ilham – A11.2022.14113  
UTS Sistem Temu Kembali Informasi
