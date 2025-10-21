import os
import asyncio
from typing import Annotated
from pydantic import Field
from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv

from search_index_manager import SearchIndexManager


load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

embeddings_client = AsyncAzureOpenAI(
      azure_endpoint=AZURE_OPENAI_ENDPOINT,
      api_key=AZURE_OPENAI_API_KEY,
      api_version="2024-02-01"
)

azure_search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)

embed_dimensions = int(os.getenv('AZURE_AI_EMBED_DIMENSIONS'))

search_index_manager = SearchIndexManager(
    endpoint = AZURE_SEARCH_ENDPOINT,
    credential = azure_search_credential,
    index_name = AZURE_SEARCH_INDEX,
    dimensions = embed_dimensions,
    model = AZURE_OPENAI_EMBED_DEPLOYMENT,
    embeddings_client=embeddings_client
)

async def get_info(
    query: Annotated[str, Field(description="Get information from the RAG.")],
) -> str:
    """Get information from the RAG."""
    context = await search_index_manager.search(query)

    if context:
        return context
    else:
        return "No information found."

agent = AzureOpenAIChatClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
).create_agent(
    instructions="You are a helpful assistant that retrieves info using tools to answer user's question.  If you can't find any information, say 'Sorry, I don't know.'",
    tools=get_info
)

async def main():
    try:
        await search_index_manager.ensure_index_created(
            vector_index_dimensions=embed_dimensions if embed_dimensions else 100)

        result = await agent.run("Tell me about the Toronto Blue Jays")
        print(result.text)
    finally:
        await search_index_manager.close()
        await embeddings_client.close()


if __name__ == "__main__":
    asyncio.run(main())