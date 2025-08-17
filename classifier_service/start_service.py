#!/usr/bin/env python3
"""
Startup script for the LLM Classifier Service
"""

import uvicorn
import os
from server import app

if __name__ == "__main__":
    # Get configuration from environment variables or use defaults
    host = os.getenv("CLASSIFIER_HOST", "0.0.0.0")
    port = int(os.getenv("CLASSIFIER_PORT", "8000"))
    reload = os.getenv("CLASSIFIER_RELOAD", "false").lower() == "true"
    
    print(f"Starting LLM Classifier Service on {host}:{port}")
    print(f"Reload mode: {reload}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

