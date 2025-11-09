import os
import asyncio
from typing import Annotated
from pydantic import Field
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import ai_function, ChatMessage, Role, TextContent, DataContent, FunctionCallContent, FunctionResultContent, FunctionApprovalRequestContent
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

@ai_function(approval_mode="always_require")
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
            elif isinstance(content, FunctionApprovalRequestContent):
                print(f"Function Approval Call: {content.function_call.name}")
                print(f"Function Approval Arguments: {content.function_call.arguments}")


    if update.user_input_requests:
        for user_input_needed in update.user_input_requests:
            print(f"Function: {user_input_needed.function_call.name}")
            print(f"Arguments: {user_input_needed.function_call.arguments}")

    user_approval = True

    approval_message = ChatMessage(role=Role.USER, contents=[user_input_needed.create_response(user_approval)])        

    #Uncomment if you don't want final to stream the end result
    #final_result = await agent.run([
    #    "What is the weather like in Toronto?",
    #    ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed]),
    #    approval_message
    #])

    #print(final_result)

    print("-"*50)

    async for update in agent.run_stream(["What is the weather like in Toronto?",
        ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed]),
        approval_message
    ]):
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