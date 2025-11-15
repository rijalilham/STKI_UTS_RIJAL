# app/main.py
"""
main.py — Interface utama (CLI)
Sebagai pintu masuk untuk menjalankan seluruh sistem IR (UTS STKI).
"""

import os
import subprocess
import sys

# Tentukan path ke folder src/
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(BASE_PATH, "src")

def clear_screen():
    # Membersihkan layar terminal (Windows & Mac/Linux)
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    while True:
        clear_screen()
        print("=======================================================")
        print("SISTEM TEMU KEMBALI INFORMASI — PARFUM HMNS")
        print("=========================================================")
        print("1. Jalankan Preprocessing Dokumen")
        print("2. Boolean Retrieval")
        print("3. Vector Space Model (TF-IDF)")
        print("4. Evaluasi Sistem (Manual)")
        print("5. Analisis Hasil Evaluasi (Otomatis)")
        print("0. Keluar")
        print("=========================================================")

        choice = input("Pilih menu [0-5]: ").strip()

        if choice == "1":
            run_script("preprocess.py")
        elif choice == "2":
            run_script("search.py", "2")
        elif choice == "3":
            run_script("search.py", "3")
        elif choice == "4":
            run_script("search.py", "4")
        elif choice == "5":
            run_script("search.py", "5")
        elif choice == "0":
            print("\nTerima kasih, sampai jumpa!")
            break
        else:
            print("\nPilihan tidak valid.")
            input("Tekan Enter untuk lanjut...")

def run_script(script_name, mode=None):
    """Menjalankan script Python dari folder src."""
    script_path = os.path.join(SRC_PATH, script_name)

    if not os.path.exists(script_path):
        print(f"File {script_name} tidak ditemukan di {SRC_PATH}")
        input("Tekan Enter untuk kembali ke menu...")
        return

    # Menjalankan perintah python
    cmd = ["python", script_path]
    if mode:
        cmd.append(mode)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nProses dibatalkan oleh pengguna.")
    input("\nTekan Enter untuk kembali ke menu utama...")

if __name__ == "__main__":
    main_menu()
