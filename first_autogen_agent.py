from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main() -> None:
    # connecting to model
    model_client = OpenAIChatCompletionClient(model='gpt-4o')

    agent_1 = AssistantAgent(name="myassistant", model_client= model_client)
    print(await agent_1.run(task="tell me something about you"))
    print(await agent_1.run(task="what is my last question"))
        
asyncio.run(main())