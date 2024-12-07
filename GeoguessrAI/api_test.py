from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a poetic assistant"},
        {"role": "user", "content": "Why is the sky blue?"}
    ]
)

print(completion.choices[0].message.content)