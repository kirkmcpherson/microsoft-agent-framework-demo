import os
import asyncio
from typing import Any, Annotated
from pydantic import BaseModel, Field
from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI
from agent_framework import AgentRunUpdateEvent, WorkflowBuilder, AgentExecutor, WorkflowOutputEvent, WorkflowStatusEvent, WorkflowViz, AgentExecutorResponse, Case, Default
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv

from search_index_manager import SearchIndexManager


class CityInfo(BaseModel):
    """Information about a city."""
    name: str | None = None
    weather: str | None = None
    country: str | None = None

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


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    return f"The weather in {location} is snowy with a high of 34°F."

async def get_restaurants(
    query: Annotated[str, Field(description="City name.")],
) -> str:
    """Get restaurants from the RAG."""
    try:
        context = await search_index_manager.search(query)
        if context:
            return context
        else:
            return "No information found."
    except Exception as e:
        print(f"Error searching for restaurants: {str(e)}")
        return f"Error retrieving restaurant information: {str(e)}"

def get_country(expected_country: str):
    """Factory that returns a predicate matching a specific country."""

    def condition(message: Any) -> bool:
        if not isinstance(message, AgentExecutorResponse):
            return False

        try:
            parsed = CityInfo.model_validate_json(message.agent_run_response.text)
            return parsed.country == expected_country
        except Exception:
            print('invalid object')
            return False

    return condition

async def main() -> None:
    chat_client = AzureOpenAIChatClient(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
    )

    try:
        await search_index_manager.ensure_index_created(
            vector_index_dimensions=embed_dimensions if embed_dimensions else 100)

        city_info_agent = AgentExecutor(
            chat_client.create_agent(
                name="City Info",
                instructions="You are a helpful assistant that figures out the city from the information provided and also returns the weather",
                tools=get_weather,
                response_format=CityInfo,
            )
        ) 

        tourist_recommendations_agent = AgentExecutor(
            chat_client.create_agent(
                name="Tourist Recommendations",
                instructions=(
                    "You are an assistant who provides tourist recommendations based on a city. "
                    "Your input might be a JSON object that includes 'city'.  Your input might also be a JSON object that includes 'weather'."
                    "Base your response on 'city' and 'weather'.  If the weather is sunny and warm, recommend an outdoor place.  Else recommend an indoor place."
                    "Do not recommend the place you are already at."
                    "Return JSON with a single field response."
                ),
            )
        ) 

        restaurant_recommendations_agent = AgentExecutor(
                chat_client.create_agent(
                name="Restaurant Recommendations",
                instructions=(
                    "You are an assistant who provides restaurant recommendations based on a city. "
                    "Your input might be a JSON object that includes 'city'."
                    "Give a restaurant recommendation for the city you are currently in."
                    "Only provide restaurants from the RAG.  If you can't find any, so 'no recommendations'"
                    "Return JSON with a single field response."
                ),
                tools=get_restaurants
            )
        )

        hockey_agent = AgentExecutor(
            chat_client.create_agent(
                name="Hockey Recommendations",
                instructions=(
                    "You are an assistant who provides professional hockey information based on a city. "
                    "Your input might be a JSON object that includes 'city'."
                    "Give the user information about the hockey team and where they play."
                    "Return JSON with a single field response."
                ),
            )
        ) 

        workflow = (
            WorkflowBuilder()
            .set_start_executor(city_info_agent)
            .add_switch_case_edge_group(
               city_info_agent,
                [
                    Case(condition=get_country("Canada"), target=hockey_agent),
                    Case(condition=get_country("United States"), target=restaurant_recommendations_agent),
                    Default(target=tourist_recommendations_agent),
                ]
            )
            .build()
        )

        viz = WorkflowViz(workflow)
        doc_diagram = viz.save_svg("docs/workflow_architecture 3.svg")

        events = workflow.run_stream("You are at the Eiffel Tower.")

        last_executor_id: str | None = None

        async for event in events:
            if isinstance(event, AgentRunUpdateEvent):
                # Handle streaming updates from agents
                eid = event.executor_id
                if eid != last_executor_id:
                    if last_executor_id is not None:
                        print()
                    print(f"{eid}:", end=" ", flush=True)
                    last_executor_id = eid
                print(event.data, end="", flush=True)
            #elif isinstance(event, WorkflowStatusEvent):
            #    print("\n=== Status ===")
            #    print(event)


    finally:
        await search_index_manager.close()
        await embeddings_client.close()



if __name__ == "__main__":
    asyncio.run(main())