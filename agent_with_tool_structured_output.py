import os
import asyncio
from typing import Annotated
from pydantic import Field, BaseModel
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
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
    return f"The weather in {location} is cloudy with a high of 15°C."

agent = AzureOpenAIChatClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
).create_agent(
    instructions="You are a helpful assistant that figures out the city from the information provided and also returns the weather",
    tools=get_weather
)
async def main():
    result = await agent.run(
        "I'm at the CN Tower.",
        response_format=CityInfo    
    )
    
    if result.value:
        city_info = result.value
        print(f"City: {city_info.name}, Weather: {city_info.weather}")
    else:
        print("Nothing found")


if __name__ == "__main__":
    asyncio.run(main())