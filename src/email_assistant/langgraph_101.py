from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, END, START
from dotenv import load_dotenv
import os

load_dotenv(".env")

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

# Azure OpenAI configuration (falls back to OpenAI if Azure env vars not set)
azure_endpoint = os.getenv("BRICK_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

if azure_endpoint and azure_api_key:
    llm = init_chat_model(
        "azure_openai:gpt-4o",
        temperature=0,
        azure_endpoint=azure_endpoint.split("/openai/deployments/")[0],
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
        api_key=azure_api_key
    )
else:
    # Fallback to regular OpenAI
    llm = init_chat_model("openai:gpt-4o", temperature=0)
model_with_tools = llm.bind_tools([write_email], tool_choice="any")

def call_llm(state: MessagesState) -> MessagesState:
    """Run LLM"""

    output = model_with_tools.invoke(state["messages"])
    return {"messages": [output]}

def run_tool(state: MessagesState) -> MessagesState:
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        observation = write_email.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["run_tool", "__end__"]:
    """Route to tool handler, or end if Done tool called"""
    
    # Get the last message
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is a tool call, check if it's a Done tool call
    if last_message.tool_calls:
        return "run_tool"
    # Otherwise, we stop (reply to the user)
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("run_tool", run_tool)
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue, {"run_tool": "run_tool", END: END})
workflow.add_edge("run_tool", END)

app = workflow.compile()