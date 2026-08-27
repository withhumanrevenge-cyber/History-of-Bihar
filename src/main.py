from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval import create_retriever
from src.llm import get_llm
from src.chain import build_chain


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory="./.chroma_db", embedding_function=embeddings)


def load_chunks_from_chroma(vectorstore):
    raw = vectorstore.get()
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def build_rag():
    vectorstore = load_vectorstore()
    chunks = load_chunks_from_chroma(vectorstore)
    if not chunks:
        raise RuntimeError(
            "Chroma is empty. Run `python ingest.py` first to index the PDFs."
        )

    retriever = create_retriever(vectorstore, chunks, k=5, fetch_k=20)
    llm = get_llm()
    return build_chain(retriever, llm)


def main():
    rag = build_rag()
    print("Ask me about Legends of Bihar (type 'exit' to quit).\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = rag.invoke(question)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()
