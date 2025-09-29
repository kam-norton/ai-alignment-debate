import openai, os

openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.base_url = "https://openrouter.ai/api/v1/"

import os
print("Key loaded:", os.getenv("OPENROUTER_API_KEY")[:10], "...")


print("Using key prefix:", openai.api_key[:6], "...")
print("Using base URL:", openai.base_url)

resp = openai.chat.completions.create(
    model="openai/gpt-4o-mini",  # or another model you know exists
    messages=[{"role": "user", "content": "Say hello world in one word"}],
)
print(resp)