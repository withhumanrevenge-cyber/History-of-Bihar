from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel


SYSTEM_PROMPT = """You are a helpful assistant answering questions about Bihar
using the provided context.

Rules:
- Answer ONLY from the context below. If the answer is not in the context,
  reply exactly: "I don't know based on the provided documents."
- Cite sources inline as [source: <filename>, page <n>] after each claim.
- Be concise. Do not invent facts.

Context:
{context}
"""

USER_PROMPT = "{question}"


def format_docs(docs) -> str:
    if not docs:
        return "(no context found)"
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[{i}] source={src} page={page}\n{d.page_content}")
    return "\n\n".join(blocks)


def build_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    retrieve_and_format = RunnableLambda(
        lambda q: format_docs(retriever(q))
    )

    return (
        RunnableParallel(
            {"context": retrieve_and_format, "question": RunnableLambda(lambda q: q)}
        )
        | prompt
        | llm
        | StrOutputParser()
    )
