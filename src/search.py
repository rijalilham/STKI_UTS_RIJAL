

import os
import sys
from preprocess import process_folder
from boolean_ir import build_inverted_index, eval_boolean_query
from vsm_ir import load_documents, build_vocabulary, rank_documents
from evaluation import evaluate_system
from analyze_results import evaluate as eval_query_case   # ← tambahan dari analyze_results.py

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_PATH, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_PATH, "data", "processed")

# -------------------------------------------------
# Helper: cek folder data
# -------------------------------------------------
def check_data():
    if not os.path.exists(RAW_DIR):
        print("Folder data/raw tidak ditemukan. Pastikan korpus sudah dibuat.")
        sys.exit(1)
    if len(os.listdir(RAW_DIR)) == 0:
        print("Folder data/raw kosong. Masukkan dokumen .txt terlebih dahulu.")
        sys.exit(1)
    print("Folder data/raw siap digunakan.")

# -------------------------------------------------
# Menu utama CLI
# -------------------------------------------------
def main_menu():
    while True:
        print("\n=== SISTEM PENCARIAN PARFUM HMNS ===")
        print("1. Preprocessing Dokumen")
        print("2. Pencarian Boolean Retrieval")
        print("3. Pencarian Vector Space Model (TF-IDF)")
        print("4. Evaluasi Sistem (Precision, Recall, F1)")
        print("5. Analisis Hasil Evaluasi (Analyze Results)")
        print("0. Keluar")

        choice = input("\nPilih menu [0-5]: ").strip()

        if choice == "1":
            run_preprocessing()
        elif choice == "2":
            run_boolean_search()
        elif choice == "3":
            run_vsm_search()
        elif choice == "4":
            run_evaluation()
        elif choice == "5":
            run_analyze_results()
        elif choice == "0":
            print("Keluar dari program.")
            break
        else:
            print("Pilihan tidak valid. Coba lagi.")

# -------------------------------------------------
# 1️⃣ Preprocessing
# -------------------------------------------------
def run_preprocessing():
    print("\nMenjalankan Preprocessing...")
    check_data()
    process_folder(RAW_DIR, PROCESSED_DIR)
    print("Preprocessing selesai. File tersimpan di data/processed/")

# -------------------------------------------------
# 2️⃣ Boolean Retrieval
# -------------------------------------------------
def run_boolean_search():
    print("\nMODE: BOOLEAN RETRIEVAL")
    inverted_index, all_docs = build_inverted_index(PROCESSED_DIR)
    print(f"Inverted Index terbentuk ({len(all_docs)} dokumen)")

    while True:
        query = input("\nMasukkan query (gunakan AND/OR/NOT, ketik 'exit' untuk kembali): ").lower()
        if query == "exit":
            break
        results = eval_boolean_query(query, inverted_index, all_docs)
        if results:
            print(f"Dokumen cocok: {results}")
        else:
            print("Tidak ada dokumen yang cocok dengan query.")

# -------------------------------------------------
# 3️⃣ Vector Space Model (TF-IDF)
# -------------------------------------------------
def run_vsm_search():
    print("\nMODE: VECTOR SPACE MODEL (TF-IDF)")
    docs = load_documents(PROCESSED_DIR)
    vocab, df = build_vocabulary(docs)
    print(f" {len(docs)} dokumen dimuat, {len(vocab)} kosakata unik.")

    while True:
        query = input("\nMasukkan query pencarian (ketik 'exit' untuk kembali): ").lower()
        if query == "exit":
            break

        results = rank_documents(query, docs, vocab, df)
        print("\nHasil Ranking (Cosine Similarity):")
        for doc, score in results[:5]:
            if score > 0:
                print(f"{doc:10s} → skor: {score:.4f}")
        print("-" * 40)

# -------------------------------------------------
# 4️⃣ Evaluasi Sistem (Manual)
# -------------------------------------------------
def run_evaluation():
    print("\nMODE: EVALUASI SISTEM (Manual Input)")
    query = input("Masukkan query evaluasi: ").lower()
    relevan = input("Masukkan dokumen relevan (pisahkan dengan koma, contoh: doc6,doc1): ").split(",")
    relevan = [r.strip() for r in relevan if r.strip() != ""]
    evaluate_system(query, relevan)
    print("\nEvaluasi selesai.")

# -------------------------------------------------
# 5️⃣ Analisis Hasil Evaluasi (Otomatis)
# -------------------------------------------------
def run_analyze_results():
    print("\nMODE: ANALISIS HASIL EVALUASI OTOMATIS")
    print("Menjalankan evaluasi untuk beberapa query parfum...\n")

    import analyze_results as ar

    docs = ar.load_documents()
    vocab, df = ar.build_vocabulary(docs)

    test_cases = [
        ("vanilla floral aroma", ["doc6", "doc1"]),
        ("woody amber warm", ["doc1", "doc3", "doc5"]),
        ("fresh citrus mint", ["doc3", "doc8"]),
        ("musk jasmine floral", ["doc2", "doc6", "doc7"]),
        ("lavender sandalwood calm", ["doc9"])
    ]

    print(f"{'Query':35s} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10}")
    print("-" * 75)

    total_p, total_r, total_f = 0, 0, 0
    for query, rel in test_cases:
        p, r, f = eval_query_case(query, rel, docs, vocab, df)
        total_p += p
        total_r += r
        total_f += f
        print(f"{query:35s} | {p:10.2f} | {r:10.2f} | {f:10.2f}")

    n = len(test_cases)
    print("-" * 75)
    print(f"{'Rata-rata':35s} | {total_p/n:10.2f} | {total_r/n:10.2f} | {total_f/n:10.2f}")

# -------------------------------------------------
# Jalankan Program
# -------------------------------------------------
if __name__ == "__main__":
    main_menu()
