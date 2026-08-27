import os

from dotenv import load_dotenv

load_dotenv()

HF_ROUTER_URL = "https://router.huggingface.co/v1"
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")


def get_llm():
    """
    Uses HuggingFace's OpenAI-compatible Router with your HF_TOKEN.
    Falls back to a small local pipeline if the token is missing.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if hf_token:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=HF_CHAT_MODEL,
            base_url=HF_ROUTER_URL,
            api_key=hf_token,
            temperature=0.3,
            streaming=True,
            max_tokens=1200,
            top_p=0.9,
        )

    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipe = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 1200, "temperature": 0.3, "top_p": 0.9},
    )
    return ChatHuggingFace(llm=pipe)
