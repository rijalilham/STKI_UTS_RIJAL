
import os
from collections import defaultdict


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_PATH, "data", "processed")

def build_inverted_index(processed_dir=PROCESSED_DIR):
    inverted = defaultdict(set)
    doc_ids = []

    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith(".txt"):
            doc_id = fname.replace(".txt", "")
            doc_ids.append(doc_id)

            with open(os.path.join(processed_dir, fname), "r", encoding="utf-8") as f:
                tokens = f.read().split()

            for term in set(tokens):
                inverted[term].add(doc_id)

    inverted = {term: sorted(list(docs)) for term, docs in inverted.items()}
    return inverted, sorted(doc_ids)


def infix_to_postfix(tokens):
    precedence = {'not': 3, 'and': 2, 'or': 1}
    output = []
    stack = []

    for t in tokens:
        t = t.lower()
        if t == '(':
            stack.append(t)
        elif t == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack:
                stack.pop()
        elif t in precedence:
            while stack and stack[-1] in precedence and precedence[stack[-1]] >= precedence[t]:
                output.append(stack.pop())
            stack.append(t)
        else:
            output.append(t)
    while stack:
        output.append(stack.pop())
    return output


def eval_postfix(postfix, inverted_index, all_docs):
    stack = []
    all_set = set(all_docs)

    for t in postfix:
        t = t.lower()
        if t == 'and':
            b = set(stack.pop())
            a = set(stack.pop())
            stack.append(sorted(a & b))
        elif t == 'or':
            b = set(stack.pop())
            a = set(stack.pop())
            stack.append(sorted(a | b))
        elif t == 'not':
            a = set(stack.pop())
            stack.append(sorted(all_set - a))
        else:
            stack.append(inverted_index.get(t, []))

    return stack[-1] if stack else []


def eval_boolean_query(query, inverted_index, all_docs):
    tokens = []
    for raw in query.strip().split():
        # pisahkan tanda kurung jika menempel
        while raw.startswith('('):
            tokens.append('(')
            raw = raw[1:]
        closing = 0
        while raw.endswith(')'):
            raw = raw[:-1]
            closing += 1
        if raw:
            tokens.append(raw)
        tokens.extend([')'] * closing)

    postfix = infix_to_postfix(tokens)
    return eval_postfix(postfix, inverted_index, all_docs)


if __name__ == "__main__":
    print("Membangun Inverted Index ...")
    inverted_index, all_docs = build_inverted_index()
    print(f"Total dokumen: {len(all_docs)}")

    queries = [
        "vanilla and floral",
        "woody or citrus",
        "citrus not floral",
        "citrus and ( mint or lemon )"
    ]

    for q in queries:
        res = eval_boolean_query(q, inverted_index, all_docs)
        print(f"\nQuery: {q}")
        print("Dokumen cocok:", res)
