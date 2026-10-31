from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from backend.app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "agent:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8081")),
        reload=os.getenv("WORKAID_RELOAD", "0") == "1",
    )
