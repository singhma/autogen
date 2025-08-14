from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_agentchat.ui import Console
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

# Agents when they work towards a common goal, then they are a team

#  Agent
writer = AssistantAgent(name="Writer",
                        model_client=model_client,
                        description='You are a great Writer',
                        system_message="You are a really helpful writer who writes in less then 30 words"
                        )


reviewer = AssistantAgent(name="Reviewer",
                          model_client=model_client,
                          description='You are a great reviewer',
                          system_message="You are a really helpful reviewer who reviews in less then 30 words"
                          )


editor = AssistantAgent(name="Editor",
                        model_client=model_client,
                        description='You are a great editor',
                        system_message="You are a really helpful editor who edits in less then 30 words"
                        )


# user_proxy = UserProxyAgent(
#     name="UserProxy",
#     description="you are a user proxy agent",
#     input_func=input,
# )

termination_condition = TextMentionTermination(text="Approve")

teams = RoundRobinGroupChat(
    participants=[writer, reviewer, editor],
    termination_condition=termination_condition,
    max_turns=3
)


async def run_team():
    task = "write 3 lines poem on sky"
    while True:
        stream = teams.run_stream(task=task)
        await Console(stream)

        feedback_from_user = input("please provide input")
        if feedback_from_user.lower().strip() == "exit":
            break
        task = feedback_from_user

    final_state = await teams.save_state()
    print(final_state)
if __name__ == "__main__":
    asyncio.run(run_team())
