"""
Public ASGI entry for Hugging Face Spaces (``app_file: app.py``) and Docker CMD.

The FastAPI instance is defined in ``server.app`` (OpenEnv ``create_app`` plus
MACE routes). Import this module so HF metadata and the container use the same
``app`` object; use ``uvicorn app:app`` (matches ``openenv.yaml``).
"""

from server.app import app

__all__ = ["app"]
