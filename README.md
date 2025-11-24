# Perplexity 2.0 Backend

This is the backend service for the Perplexity 2.0 AI Assistant. It handles the core AI logic and API endpoints.

## 📂 Folder Structure

```
perplexity-backend/
├── agent.py          # Core AI agent logic and processing
├── main.py           # Main API entry point (FastAPI)
├── run.py            # Script to run the application
├── pyproject.toml    # Project dependencies and configuration
└── .env              # Environment variables configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Fast Python package installer and resolver)

### Installation

1.  **Install dependencies:**

    ```bash
    uv sync
    ```

2.  **Set up environment variables:**

    Ensure you have a `.env` file in the root directory with the necessary API keys and configuration.

### Usage

**Start the server:**

```bash
python run.py
```
*Or using uv:*
```bash
uv run run.py
```

The server will start, typically on `http://localhost:8000` (check console output for exact address).
