from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio

load_dotenv()
model_client = OpenAIChatCompletionClient(model='gpt-4o')

async def main():
    my_assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
    )
    result = await my_assistant.run(task="who are you")
    result = await my_assistant.run(task="what was the last question I asked")
    result = await my_assistant.run(task="What were the last two questions I asked")
    print(result)
        

if __name__ == "__main__":
    asyncio.run(main())