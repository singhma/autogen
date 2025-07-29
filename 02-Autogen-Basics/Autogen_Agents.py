from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio

from io import BytesIO

import PIL
import requests
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image



load_dotenv()
model_client = OpenAIChatCompletionClient(model='gpt-4o')

async def main():
    my_assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
    )
    # result = await my_assistant.run(task="who are you")
    # result = await my_assistant.run(task="what was the last question I asked")
    # result = await my_assistant.run(task="What were the last two questions I asked")
    # print(result)

    # Create a multi-modal message with random image and text.
    pil_image = PIL.Image.open(BytesIO(requests.get("https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U").content))
    img = Image(pil_image)
    multi_modal_message = MultiModalMessage(content=["Can you describe the content of this image?", img], source="user")
    # Use asyncio.run(...) when running in a script.
    result = await my_assistant.run(task=multi_modal_message)
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())