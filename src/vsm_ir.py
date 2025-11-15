
import os
import math
from collections import Counter


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_PATH, "data", "processed")


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
                idf_val = math.log(N / (1 + df[term]))  # +1 untuk menghindari div 0
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

    # Urutkan berdasarkan skor tertinggi
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    print(" Membaca dokumen parfum ...")
    docs = load_documents()
    print(f"Total dokumen: {len(docs)}")

    vocab, df = build_vocabulary(docs)
    print(f"Total kosakata unik: {len(vocab)}")

    query = input("\nMasukkan query pencarian: ").lower()
    results = rank_documents(query, docs, vocab, df)

    print("\nHasil Ranking Dokumen (Cosine Similarity):")
    for doc, score in results:
        print(f"{doc}: {score:.4f}")
