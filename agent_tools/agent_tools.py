from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import asyncio

load_dotenv()

model_client = OpenAIChatCompletionClient(model='gpt-5-2025-08-07', )

def reverse_string(text: str) -> str:
    '''
    reverse the given text

    input: str
    output: str

    The reverse string is returned
    '''
    return text[::-1]
    # return "test"


reverse_tool = FunctionTool(
    reverse_string, description="reverse the given text")


#  Agent
agent = AssistantAgent(name="DigitalHuman",
                       model_client=model_client,
                       tools=[reverse_tool],
                       #    reflect_on_tool_use=True,
                       system_message="You are a helpful asssitant that can reverse string using reverse_string tool. Give the result with summar",
                       )


async def main():
    result = await agent.run(task="reverse the string 'Hello world' ")
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
