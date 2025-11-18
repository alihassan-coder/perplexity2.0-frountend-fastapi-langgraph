from typing import Annotated,TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
import os
from dotenv import load_dotenv
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage

from uuid import uuid4

load_dotenv()

try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
except Exception as e:
    print(f"Error: {e}")
    exit(1)


class State(TypedDict):
    message: Annotated[list, add_messages]
    search_urls: list
    summary: str




llm = ChatGroq(
    model="openai/gpt-oss-120b",  
    api_key=GROQ_API_KEY,
)

search_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=8
)

tools = {"tavily_search": search_tool}

memory = MemorySaver()


llm_with_tools = llm.bind_tools(tools=list(tools.values()))


async def model(state: State):
    messages = state["message"]
    summary = state.get("summary", "")

    existing_system = None
    for msg in messages:
        if isinstance(msg, SystemMessage):
            existing_system = msg
            break

    if existing_system is None:
        system_message = SystemMessage(content="""You are Perplexity 2.0, an advanced AI assistant that helps users find accurate, up-to-date information by searching the web when needed. 

Key characteristics:
- You are Perplexity 2.0, not ChatGPT or any other AI
- You provide helpful, accurate, and context-aware responses
- You always verify factual or real-world information by calling the tavily_search tool before answering, unless the question is purely about reasoning or personal preferences with no need for external data
- You can search the web using Tavily when you need current information
- You always identify yourself as Perplexity 2.0 when asked about your identity
- You are knowledgeable, friendly, and professional

Response style:
- Always give a formatted and clean response with clear headings and bullet points when helpful
- Responses should be detailed, but stay focused and avoid unnecessary repetition
- Use the running conversation summary you are given to stay consistent with the user's preferences and previous questions

At the end of EVERY answer, include a short section:

Summary:
- 2-4 bullet points with the key takeaways for the user

Sources:
- Markdown bullet list of the most relevant web pages you used (title and URL as a Markdown link)

Make sure the content of the Sources section matches the URLs returned by tavily_search and shown in the UI.""")
    else:
        system_message = existing_system

    conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    max_recent_messages = 8
    summary_trigger_messages = 12

    if len(conversation_messages) > summary_trigger_messages:
        older_messages = conversation_messages[:-max_recent_messages]
        recent_messages = conversation_messages[-max_recent_messages:]

        def serialize_messages(msgs):
            parts = []
            for m in msgs:
                if isinstance(m, HumanMessage):
                    role = "user"
                else:
                    role = "assistant"
                parts.append(f"{role}: {m.content}")
            text = "\n".join(parts)
            if len(text) > 4000:
                text = text[-4000:]
            return text

        conversation_segment = serialize_messages(older_messages)

        if conversation_segment.strip():
            summary_prompt = [
                SystemMessage(content="You are a concise assistant that maintains a running summary of a chat between a user and Perplexity 2.0."),
                HumanMessage(content=(
                    "Existing summary (can be empty):\n" + summary + "\n\n" +
                    "New conversation segment:\n" + conversation_segment + "\n\n" +
                    "Update the summary to include all important user preferences, facts, and unresolved questions. Keep it under 200 words."
                )),
            ]

            summary_result = await llm.ainvoke(summary_prompt)
            summary = summary_result.content

        recent_conversation = recent_messages
    else:
        recent_conversation = conversation_messages

    context_messages = [system_message]

    if summary:
        context_messages.append(SystemMessage(content=(
            "Conversation summary so far. Do not repeat this verbatim in your answer; use it only for context:\n" + summary
        )))

    if len(recent_conversation) > max_recent_messages:
        recent_conversation = recent_conversation[-max_recent_messages:]

    context_messages.extend(recent_conversation)

    result = await llm_with_tools.ainvoke(context_messages)
    return {"message": [result], "search_urls": state.get("search_urls", []), "summary": summary}


async def tool_router(state: State):
    last_message = state["message"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    else:
        return END


async def tool_node(state: State):
    last_message = state["message"][-1]
    tool_call = last_message.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]

    tool_instance = tools.get(tool_name)
    if not tool_instance:
        raise ValueError(f"Tool '{tool_name}' not found!")

    tool_result = await tool_instance.ainvoke(tool_args)

    # Extract URLs from Tavily search results
    urls = []
    if tool_name == "tavily_search" and isinstance(tool_result, list):
        for result in tool_result:
            if isinstance(result, dict) and "url" in result:
                content = result.get("content", "")
                urls.append({
                    "url": result["url"],
                    "title": result.get("title", "Untitled"),
                    "content": content[:200] + "..." if len(content) > 200 else content
                })

    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_id,
        name=tool_name,
    )

    if hasattr(tool_message, 'additional_kwargs'):
        tool_message.additional_kwargs['urls'] = urls
    else:
        pass

    return {"message": [tool_message], "search_urls": urls, "summary": state.get("summary", "")}


workflow = StateGraph(State)

workflow.add_node("model", model)
workflow.add_node("tool_router", tool_router)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "model")
workflow.add_conditional_edges("model", tool_router)
workflow.add_edge("tool_node", "model")

# Compile the workflow with memory checkpointing
graph_app = workflow.compile(checkpointer=memory)
