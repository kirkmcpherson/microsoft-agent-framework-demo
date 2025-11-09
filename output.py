import os
import asyncio
from pydantic import BaseModel
from agent_framework import AgentRunResponse
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

class PersonInfo(BaseModel):
    name: str | None = None
    age: int | None = None
    occupation: str | None = None

agent = AzureOpenAIChatClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
).create_agent(
    instructions="You are a helpful assistant that extracts person information from a given text."
)

async def main():
    query = "Please provide information about Bo Bichette who is a 26 year old baseball player."
    response = await agent.run(query,
        response_format=PersonInfo
    )

    if response.value:  
        person_info = response.value
        print(f"Name: {person_info.name}")
        print(f"Age: {person_info.age}")
        print(f"Occupation: {person_info.occupation}")
    else:
        print("No person information found.")

    # Get structured response from streaming agent using AgentRunResponse.from_agent_response_generator
    # This method collects all streaming updates and combines them into a single AgentRunResponse
    final_response = await AgentRunResponse.from_agent_response_generator(
        agent.run_stream(query, response_format=PersonInfo),
        output_format_type=PersonInfo,
    )

    if final_response.value:
        person_info = final_response.value
        print(f"Name: {person_info.name}")
        print(f"Age: {person_info.age}")
        print(f"Occupation: {person_info.occupation}")
    else:
        print("No person information found.")

if __name__ == "__main__":
    asyncio.run(main())