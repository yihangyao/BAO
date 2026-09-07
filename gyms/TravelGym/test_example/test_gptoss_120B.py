from openai import OpenAI

client = OpenAI(
    base_url="http://p1-r19-n1:8000/v1",
    api_key="none"  # vLLM ignores it by default
)


resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Explain reinforcement learning in one sentence."}],
)
print(resp.choices[0].message.content)
