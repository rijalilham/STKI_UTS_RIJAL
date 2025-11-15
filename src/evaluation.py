
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


def evaluate_system(query, relevant_docs):
    docs = load_documents()
    vocab, df = build_vocabulary(docs)
    ranked_results = rank_documents(query, docs, vocab, df)

    retrieved_docs = [doc for doc, score in ranked_results if score > 0]

    # hitung TP, FP, FN
    TP = len([d for d in retrieved_docs if d in relevant_docs])
    FP = len([d for d in retrieved_docs if d not in relevant_docs])
    FN = len([d for d in relevant_docs if d not in retrieved_docs])

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nQuery: {query}")
    print(f"Dokumen relevan (ground truth): {relevant_docs}")
    print(f"Dokumen ditemukan: {retrieved_docs}")
    print(f"\nTP={TP}, FP={FP}, FN={FN}")
    print(f"Precision = {precision:.2f}")
    print(f"Recall    = {recall:.2f}")
    print(f"F1-score  = {f1:.2f}")

if __name__ == "__main__":
    
    query = "woody amber warm"
    relevant_docs = ["doc6", "doc1", "doc8"]
    evaluate_system(query, relevant_docs)
