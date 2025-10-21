import os
import asyncio
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


async def main():
    # Initialize a chat agent with Azure OpenAI Responses
    agent = AzureOpenAIResponsesClient(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        # api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        # api_key=os.environ["AZURE_OPENAI_API_KEY"],  # Optional if using AzureCliCredential
        #credential=AzureCliCredential(), # Optional, if using api_key
    ).create_agent(
        name="HaikuBot",
        instructions="You are an upbeat assistant that writes beautiful poetry.",
    )

    print(await agent.run("Write a haiku about the Toronto Blue Jays."))


if __name__ == "__main__":
    asyncio.run(main())