import os
from typing import Annotated
from pydantic import BaseModel, Field
from agent_framework import WorkflowBuilder, AgentExecutor
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

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

chat_client = AzureOpenAIChatClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
)

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

# Convert workflow to agent
workflow_agent = workflow.as_agent()

# Workaround: WorkflowAgent doesn't have as_mcp_server, so we wrap it as a tool
workflow_tool = workflow_agent.as_tool(
    name="get_tourist_recommendations",
    description="Get tourist recommendations for a location. Provide information about where you are."
)

# Create a wrapper agent that uses the workflow as a tool
wrapper_agent = chat_client.create_agent(
    name="tourist_guide",
    instructions="You are a helpful tourist guide. Use the get_tourist_recommendations tool to help users.",
    tools=workflow_tool
)

# Now expose the wrapper agent as an MCP server
server = wrapper_agent.as_mcp_server()
sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

async def handle_messages(request):
    await sse.handle_post_message(request.scope, request.receive, request._send)

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    print("Starting workflow MCP server on port 8001...")
    print("Workflow: City Info -> Tourist Recommendations")
    uvicorn.run(app, host="0.0.0.0", port=8001)
