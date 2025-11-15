

import os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Nanti disesuaiin sendiri path directory nya yaa
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_PATH, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_PATH, "data", "processed")

stemmer = StemmerFactory().create_stemmer()
stop_factory = StopWordRemoverFactory()
stopword_list = set(stop_factory.get_stop_words())

def clean_text(text):
    """Menghapus karakter non-huruf dan ubah ke huruf kecil"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # hanya huruf dan spasi
    text = re.sub(r'\s+', ' ', text).strip()  # hilangkan spasi ganda
    return text

def tokenize(text):
    """Memecah teks menjadi token"""
    return text.split()

def remove_stopwords(tokens):
    """Hapus stopword umum Bahasa Indonesia"""
    return [t for t in tokens if t not in stopword_list and len(t) > 1]

def stemming(tokens):
    """Stem setiap token ke bentuk dasar"""
    return [stemmer.stem(t) for t in tokens]

def preprocess_text(text):
    """Pipeline lengkap preprocessing"""
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens

def process_folder(input_dir=RAW_DIR, output_dir=PROCESSED_DIR):
    """Proses semua file .txt di data/raw"""
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith('.txt'):
            in_path = os.path.join(input_dir, fname)
            with open(in_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            tokens = preprocess_text(raw_text)
            out_path = os.path.join(output_dir, fname)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(tokens))

            print(f" {fname} selesai diproses ({len(tokens)} token)")

if __name__ == "__main__":
    print("Mulai preprocessing korpus parfum ...")
    print(f"Input  : {RAW_DIR}")
    print(f"Output : {PROCESSED_DIR}")
    process_folder()
    print("Semua dokumen selesai diproses!")
