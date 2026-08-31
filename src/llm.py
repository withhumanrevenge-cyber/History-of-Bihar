import os

from dotenv import load_dotenv

load_dotenv()

HF_ROUTER_URL = "https://router.huggingface.co/v1"
GROQ_API_URL = "https://api.groq.com/openai/v1"
# Default to a Llama model for generation so the app answers with the model family
# you requested unless an environment override is provided.
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
LLAMA_CHAT_MODEL = os.getenv("LLAMA_CHAT_MODEL", "llama-3.3-70b-instruct")
LOCAL_FALLBACK_MODEL = os.getenv("LOCAL_FALLBACK_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def _has_value(name: str) -> bool:
    value = os.getenv(name)
    return bool((value or "").strip())


def get_llm():
    """Select a cloud model only when a real API key is present.

    If no provider credential is configured, the app must use the local fallback
    model instead of making an unauthenticated OpenAI/LLAMA call.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

    if _has_value("GROQ_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=GROQ_CHAT_MODEL,
            base_url=GROQ_API_URL,
            api_key=GROQ_API_KEY,
            temperature=0.3,
            streaming=True,
            max_tokens=1200,
            top_p=0.9,
            timeout=30,
            max_retries=2,
        )

    if _has_value("HF_TOKEN") or _has_value("HUGGINGFACEHUB_API_TOKEN"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=HF_CHAT_MODEL,
            base_url=HF_ROUTER_URL,
            api_key=HF_TOKEN,
            temperature=0.3,
            streaming=True,
            max_tokens=1200,
            top_p=0.9,
            timeout=30,
            max_retries=2,
        )

    if _has_value("LLAMA_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLAMA_CHAT_MODEL,
            base_url="https://api.llama.ai/v1",
            api_key=LLAMA_API_KEY,
            temperature=0.3,
            streaming=True,
            max_tokens=1200,
            top_p=0.9,
            timeout=30,
            max_retries=2,
        )

    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipe = HuggingFacePipeline.from_model_id(
        model_id=LOCAL_FALLBACK_MODEL,
        task="chat-generation",
        pipeline_kwargs={"max_new_tokens": 400, "temperature": 0.3, "top_p": 0.9},
    )
    return ChatHuggingFace(llm=pipe)
