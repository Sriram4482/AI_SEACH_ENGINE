# test_openai_embed.py
from openai import OpenAI
import os

client = OpenAI()  # uses OPENAI_API_KEY env

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input="hello world"
)

print(len(resp.data[0].embedding))
