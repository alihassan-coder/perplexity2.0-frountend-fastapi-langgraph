#!/usr/bin/env python3
"""
Simple startup script for the Perplexity 2.0 API
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("🚀 Starting Perplexity 2.0 API...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🔄 Streaming endpoint: http://localhost:8000/chat/stream?message=hello")
    print("💬 Regular endpoint: http://localhost:8000/chat?message=hello")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
