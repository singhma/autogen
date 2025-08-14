from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from openai import DefaultHttpxClient
import asyncio
import httpx

load_dotenv()

api_base = "https://toolp1.dev.ai.dhp.cba/"  # litellm server
# Path to your certificate chain file
cert_chain_path = "certs/CBA-Group-Root-CA-G3.pem"
os.environ['SSL_CERT_FILE'] = cert_chain_path
# Create an HTTP client with the certificate chain
http_client = httpx.AsyncClient(verify=cert_chain_path)

model_client = OpenAIChatCompletionClient(
    model_info={
        "vision": True,
        "function_calling": True,
        "family": "claude-4-sonnet",
        "json_output": True,
        "structured_output": True,
    },
    model="bedrock-claude-4-sonnet",
    base_url=api_base,
    http_client=http_client)


#  Agent
agent = AssistantAgent(name="DigitalHuman",
                       model_client=model_client,)


async def main():
    result = await agent.run(task="Hi, How are you doing?")
    print(result.messages[-1].content)

asyncio.run(main())
