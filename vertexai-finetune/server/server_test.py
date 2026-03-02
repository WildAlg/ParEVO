from openai import OpenAI
from pathlib import Path
import os

with open(Path(os.path.dirname(__file__)) / "port.txt", 'r') as f:
    port = f.read().strip(' \n')

host = os.environ.get("HOST", "localhost")

client = OpenAI(
    base_url=f"http://{host}:{port}/v1",
    api_key="sk-1234"  # any dummy key works
)

response = client.chat.completions.create(
    model="gemini-2.5-pro-tuned",
    messages=[
        {"role": "user", "content": "Briefly explain what parlay lib is for?"}
    ]
)

print(response.choices[0].message.content)