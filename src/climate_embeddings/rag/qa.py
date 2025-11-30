# climate_embeddings/rag/qa.py
from .vector_index import SimpleVectorIndex
from ..embeddings.text_models import TextEmbedder

class ClimateAssistant:
    def __init__(self, index_path: str, model_name="BAAI/bge-large-en-v1.5"):
        self.index = SimpleVectorIndex()
        self.index.load(index_path)
        self.embedder = TextEmbedder(model_name)

    def ask(self, question: str, llm_callback) -> str:
        """
        1. Embed Question
        2. Retrieve Chunks (based on metadata similarity)
        3. Formulate Context (Metadata + Numeric Stats)
        4. Send to LLM
        """
        # 1. Embed
        q_vec = self.embedder.embed_queries([question])[0]
        
        # 2. Retrieve
        results = self.index.search(q_vec, k=5)
        
        # 3. Context Construction
        context_str = "Here is the relevant climate data retrieved:\n\n"
        for i, res in enumerate(results):
            meta = res['metadata']
            stats = res['stats']
            # stats map: [mean, std, min, max, p10, p50, p90, trend]
            context_str += (
                f"Chunk {i+1}:\n"
                f" - Source: {meta.get('source')}\n"
                f" - Variable: {meta.get('variable')}\n"
                f" - Location: Lat [{meta.get('lat_min', '?')}, {meta.get('lat_max', '?')}], "
                f"Lon [{meta.get('lon_min', '?')}, {meta.get('lon_max', '?')}]\n"
                f" - Time: {meta.get('time_start', 'N/A')} to {meta.get('time_end', 'N/A')}\n"
                f" - Statistics: Mean={stats[0]:.2f}, Max={stats[3]:.2f}, Min={stats[2]:.2f}\n\n"
            )
            
        prompt = (
            f"You are a climate data assistant. Answer the user's question based ONLY on the data below.\n"
            f"Question: {question}\n\n"
            f"{context_str}"
        )
        
        # 4. LLM Call
        return llm_callback(prompt)

# Mock LLM Interface for demo
def mock_llm_call(prompt: str) -> str:
    print(f"\n--- PROMPT SENT TO LLM ---\n{prompt[:500]}...\n--------------------------\n")
    return "Based on the retrieved data, the mean temperature in the selected region is roughly X..."