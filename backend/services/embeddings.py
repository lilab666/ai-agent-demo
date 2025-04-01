# services/embeddings.py
from langchain.embeddings.base import Embeddings
from openai import OpenAI
from typing import List


class QwenEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v3", batch_size: int = 10):
        if not api_key:
            raise ValueError("API key is required")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = min(batch_size, 10)

    def embed_documents(self, texts: List[str], dimensions: int = 1024) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch, dimensions=dimensions,
                                                     encoding_format="float")
            embeddings.extend([data.embedding for data in response.data])
        return embeddings

    def embed_query(self, text: str, dimensions: int = 1024) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text], dimensions=dimensions,
                                                 encoding_format="float")
        return response.data[0].embedding
