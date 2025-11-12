import os
import asyncio
from typing import Any, Annotated
from pydantic import BaseModel, Field
from agent_framework import AgentRunUpdateEvent, WorkflowBuilder, AgentExecutor, WorkflowOutputEvent, WorkflowStatusEvent, WorkflowViz
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv

class CityInfo(BaseModel):
    """Information about a city."""
    name: str | None = None
    weather: str | None = None

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    return f"The weather in {location} is sunny with a high of 20°C."

async def main() -> None:
    chat_client = AzureOpenAIChatClient(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
    )

    try:
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

        workflow = WorkflowBuilder().set_start_executor(city_info_agent).add_edge(city_info_agent, tourist_recommendations_agent).build()  

        viz = WorkflowViz(workflow)
        doc_diagram = viz.save_svg("docs/workflow_architecture.svg")

        events = workflow.run_stream("You are at the CN Tower.")

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
            elif isinstance(event, WorkflowStatusEvent):
                print("\n=== Status ===")
                print(event)


    finally:
        pass



if __name__ == "__main__":
    asyncio.run(main())