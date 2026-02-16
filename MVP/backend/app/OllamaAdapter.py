import ollama
import os

class OllamaAdapter:

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL")
        self.base_url = os.getenv("OLLAMA_BASE_URL")

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": float(os.getenv("TEMPERATURE", 0.2)),
                "num_predict": int(os.getenv("MAX_TOKENS", 1024))
            }
        )
        return response["message"]["content"]
