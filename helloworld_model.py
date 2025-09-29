import openai, os

openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.base_url = "https://openrouter.ai/api/v1/"

resp = openai.models.list()
for m in resp.data:
    print(m.id)
