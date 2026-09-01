from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if OPENAI_API_KEY:
    print("OpenAI API Key loaded successfully!")
else:
    print("Error: OPENAI_API_KEY is empty")

if HF_TOKEN:
    print("HuggingFace Hub API Key loaded successfully!")
else:
    print("Error: HF_TOKEN is empty")

if GROQ_API_KEY:
    print("Groq API Key loaded successfully!")
else:
    print("Error: GROQ_API_KEY is empty")
    