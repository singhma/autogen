from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

load_dotenv()

model_client = OpenAIChatCompletionClient(model='gpt-5-2025-08-07', )


#  Agent
agent = AssistantAgent(name="DigitalHuman",
                       model_client=model_client,)


async def main():
    result = await agent.run(task="Hi, How are you doing?")
    print(result.messages[-1].content)

asyncio.run(main())
