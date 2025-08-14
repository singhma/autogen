from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult
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
dsa_solver = AssistantAgent(name="COMPLEX_DSA_SOLVER",
                            model_client=model_client,
                            description='A DSA solver',
                            system_message="you give code in python to solve complex DSA problem. Give under 100 words"
                            )

code_reviewer = AssistantAgent(name="CODE_REVIEWER",
                               model_client=model_client,
                               description='A code reviewer',
                               system_message="You review the code given by the complex_dsa_solver and make sure its optimised. Give under 10 words. If you feel the code is right, please say 'TERMINATE'"
                               )
code_editor = AssistantAgent(name="CODE_EDITOR",
                             model_client=model_client,
                             description='A Code Editor',
                             system_message="You make the code easy to understand and add comments whereever required. Give under 10 words"
                             )

my_termination = TextMentionTermination(
    text="TERMINATE") | MaxMessageTermination(max_messages=6)

team = RoundRobinGroupChat(
    participants=[dsa_solver, code_reviewer, code_editor],
    termination_condition=my_termination,
    max_turns=100
)


async def run_team():
    text = TextMessage(
        content='Write a simple code in python to add 2 numbers', source='user')
    # result = await team.run(task=text)
    async for message in team.run_stream(task=text):
        if isinstance(message, TaskResult):
            print(f"Stop Reason: {message.stop_reason}")
        else:
            print(message.source, message)

    state = await team.save_state()
    print(state)
if __name__ == "__main__":
    asyncio.run(run_team())
