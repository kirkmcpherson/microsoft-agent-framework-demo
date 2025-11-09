import os
import asyncio
from typing import Annotated
from pydantic import Field
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import TextContent, DataContent, FunctionCallContent, FunctionResultContent
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

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
    instructions="You are a helpful assistant",
    tools=get_weather
)
async def main():
    #result = await agent.run("Tell me the weather in Los Angeles.")
    #print(f"Result type: {type(result)}")
    #print(result.text)

    #async for update in agent.run_stream("Tell me the weather is in Toronto."):
    #    print(f"Content count: {len(update.contents)}")    
    #    print(f"Update type: {type(update)}")
    #    if update.text:
    #        print(update.text)

    async for update in agent.run_stream("What is the weather like in Toronto?"):
        for content in update.contents:
            print(f"Content type: {type(content)}")
            if isinstance(content, TextContent):
                print(f"📝 Text: {content.text}")
            elif isinstance(content, FunctionCallContent):
                print(f"🔧 Function Call: {content.name}")
                print(f"   Call ID: {content.call_id}")
                print(f"   Arguments: {content.arguments}")
            elif isinstance(content, FunctionResultContent):
                print(f"🔧 Function Result: {content.result}")

if __name__ == "__main__":
    asyncio.run(main())