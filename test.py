# To install: pip install tavily-python
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(os.getenv("TAVILY_API_KEY"))
response = client.search(
    query="病虫防害怎么治",
    search_depth="advanced",
    include_answer="advanced"
)
print(response['answer'])
