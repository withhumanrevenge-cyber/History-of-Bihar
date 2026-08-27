from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever


def dedupe(docs):
    seen, unique = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def hybrid_search(query, semantic_retriever, bm25_retriever, reranker, k=5):
    candidates = dedupe(semantic_retriever.invoke(query) + bm25_retriever.invoke(query))
    if not candidates:
        return []
    scores = reranker.predict([(query, d.page_content) for d in candidates]).tolist()
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k]]


def create_retriever(vectorstore, chunks, k=5, fetch_k=20):
    semantic = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = fetch_k
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(query):
        return hybrid_search(query, semantic, bm25, reranker, k)

    return retrieve   # returns a callable