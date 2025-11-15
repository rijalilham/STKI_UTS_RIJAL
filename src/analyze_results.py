# src/analyze_results.py

import os
import math
from collections import Counter

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_PATH, "data", "processed")

# ------------------------------
# Fungsi VSM (dari langkah sebelumnya)
# ------------------------------
def load_documents(processed_dir=PROCESSED_DIR):
    docs = {}
    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith('.txt'):
            with open(os.path.join(processed_dir, fname), 'r', encoding='utf-8') as f:
                tokens = f.read().split()
                docs[fname.replace('.txt', '')] = tokens
    return docs

def build_vocabulary(docs):
    vocab = set()
    df = Counter()
    for tokens in docs.values():
        unique_terms = set(tokens)
        vocab.update(unique_terms)
        for t in unique_terms:
            df[t] += 1
    return sorted(list(vocab)), df

def compute_tfidf(docs, vocab, df):
    N = len(docs)
    tfidf = {}
    for doc_id, tokens in docs.items():
        tf = Counter(tokens)
        weights = {}
        for term in vocab:
            if term in tf:
                tf_val = tf[term]
                idf_val = math.log(N / (1 + df[term]))
                weights[term] = tf_val * idf_val
            else:
                weights[term] = 0
        tfidf[doc_id] = weights
    return tfidf

def build_query_vector(query, vocab, df, N):
    q_tokens = query.lower().split()
    q_tf = Counter(q_tokens)
    q_vec = {}
    for term in vocab:
        if term in q_tf:
            idf_val = math.log(N / (1 + df.get(term, 0)))
            q_vec[term] = q_tf[term] * idf_val
        else:
            q_vec[term] = 0
    return q_vec

def cosine_similarity(vec1, vec2):
    dot = sum(vec1[t] * vec2[t] for t in vec1)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def rank_documents(query, docs, vocab, df):
    N = len(docs)
    tfidf = compute_tfidf(docs, vocab, df)
    q_vec = build_query_vector(query, vocab, df, N)
    scores = {}
    for doc_id, vec in tfidf.items():
        scores[doc_id] = cosine_similarity(q_vec, vec)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ------------------------------
# Evaluasi Precision, Recall, F1
# ------------------------------
def evaluate(query, relevant_docs, docs, vocab, df):
    ranked_results = rank_documents(query, docs, vocab, df)
    retrieved_docs = [doc for doc, score in ranked_results if score > 0]

    TP = len([d for d in retrieved_docs if d in relevant_docs])
    FP = len([d for d in retrieved_docs if d not in relevant_docs])
    FN = len([d for d in relevant_docs if d not in retrieved_docs])

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# ------------------------------
# Main: menjalankan beberapa query
# ------------------------------
if __name__ == "__main__":
    docs = load_documents()
    vocab, df = build_vocabulary(docs)

    test_cases = [
        ("vanilla floral aroma", ["doc6", "doc1"]),
        ("woody amber warm", ["doc1", "doc3", "doc5"]),
        ("fresh citrus mint", ["doc3", "doc8"]),
        ("musk jasmine floral", ["doc2", "doc6", "doc7"]),
        ("lavender sandalwood calm", ["doc9"])
    ]

    print("Hasil Evaluasi Sistem Pencarian (TF-IDF Cosine Similarity)")
    print("-" * 75)
    print(f"{'Query':35s} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10}")
    print("-" * 75)

    total_p, total_r, total_f = 0, 0, 0

    for query, rel in test_cases:
        p, r, f = evaluate(query, rel, docs, vocab, df)
        total_p += p
        total_r += r
        total_f += f
        print(f"{query:35s} | {p:10.2f} | {r:10.2f} | {f:10.2f}")

    avg_p = total_p / len(test_cases)
    avg_r = total_r / len(test_cases)
    avg_f = total_f / len(test_cases)
    print("-" * 75)
    print(f"{'Rata-rata':35s} | {avg_p:10.2f} | {avg_r:10.2f} | {avg_f:10.2f}")
