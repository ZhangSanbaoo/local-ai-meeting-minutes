"""
Meeting AI - FastAPI Backend API

启动方式:
    uvicorn meeting_ai.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from .main import app

__all__ = ["app"]
