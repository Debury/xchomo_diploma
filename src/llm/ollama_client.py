import os
import requests

class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    def generate_rag_answer(self, query, context_hits, temperature=0.7):
        context_str = "\n".join([f"- {h['metadata'].get('text_content', '')}" for h in context_hits])
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            })
            return resp.json().get("response", "No response.")
        except Exception as e:
            return f"LLM Error: {e}"

    def check_health(self):
        try:
            return requests.get(f"{self.base_url}/api/tags").status_code == 200
        except:
            return False