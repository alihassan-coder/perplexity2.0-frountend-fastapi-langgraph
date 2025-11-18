from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
import asyncio
import json
from typing import Optional
from agent import graph_app  


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


def serialize_chunk(chunk):
    """Serialize message chunk to text/event-stream format."""
    return f"data: {chunk}\n\n"


async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    config = {"configurable": {"thread_id": checkpoint_id or "1"}}
    collected_urls = []

    try:
        # Send initial status
        yield serialize_chunk('{"type": "status", "message": "Preparing Perplexity 2.0...", "stage": "initializing"}')
        
        async for event in graph_app.astream_events(
            {"message": [HumanMessage(content=message)]},
            config=config,
            version="v1",
        ):
            event_type = event["event"]
            
            # Handle different event types for better progress tracking
            if event_type == "on_chat_model_start":
                yield serialize_chunk('{"type": "status", "message": "Analyzing your question...", "stage": "thinking"}')
            
            elif event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:  # Only yield non-empty chunks
                    # Escape special characters in JSON
                    escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    yield serialize_chunk(f'{{"type": "content", "content": "{escaped_chunk}"}}')
            
            elif event_type == "on_tool_start":
                tool_name = event["name"]
                yield serialize_chunk(f'{{"type": "status", "message": "Searching the web for reliable sources...", "stage": "searching", "tool": "{tool_name}"}}')
            
            elif event_type == "on_tool_end":
                # Check if we have search URLs to send
                if "search_urls" in event["data"]["output"] and event["data"]["output"]["search_urls"]:
                    urls = event["data"]["output"]["search_urls"]
                    collected_urls.extend(urls)
                    yield serialize_chunk(json.dumps({"type": "urls", "urls": urls}))
                yield serialize_chunk('{"type": "status", "message": "Reviewing search results...", "stage": "processing"}')
            
            elif event_type == "on_chain_end":
                if event["name"] == "model":
                    yield serialize_chunk('{"type": "status", "message": "Crafting final answer...", "stage": "finalizing"}')
    
    except Exception as e:
        # Yield error message if something goes wrong
        error_msg = str(e).replace('\\', '\\\\').replace('"', '\\"')
        yield serialize_chunk(f'{{"type": "error", "message": "Error: {error_msg}"}}')
    
    # Signal completion
    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "message": "Perplexity 2.0 API is running",
        "version": "2.0.0",
        "features": ["streaming", "web_search", "agent_progress"]
    }


@app.get("/chat")
async def chat(
    message: str,
    checkpoint_id: Optional[str] = Query(None)
):
    """
    Get a single chat response from the agent (non-streaming).
    Example: /chat?message=hello
    """
    config = {"configurable": {"thread_id": checkpoint_id or "1"}}
    
    result = await graph_app.ainvoke(
        {"message": [HumanMessage(content=message)]},
        config=config
    )
    
    # Extract the final message content
    final_message = result["message"][-1]
    return {"response": final_message.content}


@app.get("/chat/stream")
async def chat_stream(
    message: str,
    checkpoint_id: Optional[str] = Query(None)
):
    """
    Stream chat responses from the agent.
    Example: /chat/stream?message=hello
    """
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )



