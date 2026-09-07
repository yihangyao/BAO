from openai import OpenAI

client = OpenAI(
    base_url="http://p5-r23-n4:8000/v1",
    api_key="none"  # vLLM ignores it by default
)


resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Greetings!"}],
)
print(resp.choices[0].message.content)
