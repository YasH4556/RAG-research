import os
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


client = Groq(
    api_key=GROQ_API_KEY
    )

MODEL = "openai/gpt-oss-20b"

messages = [{
    "role": "user",
    "content": "whats todays date?"
    }]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = embedding_model.encode()

response = client.chat.completions.create(
    model = MODEL,
    messages= messages
)

print(response.choices[0].message.content)