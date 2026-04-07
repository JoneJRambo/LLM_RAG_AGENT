def demo01_ollama_utils():
    from openai import OpenAI

    client = OpenAI(base_url="https://ollama.example.com/v1", api_key="xx")

    stream = client.chat.completions.create(
        model="deepseek-r1:1.5b",
        stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the meaning of life?"},
        ],
    )
