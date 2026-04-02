"""
Server — Uses openenv's create_fastapi_app to serve the environment.

This auto-generates all required endpoints:
    POST /reset, POST /step, GET /state, /health, /ws, /docs
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openenv.core.env_server import create_fastapi_app
from server.environment import GeoPolicyEnv
from models.action import GeoAction
from models.observation import GeoObservation

# create_fastapi_app expects a callable that returns an Environment instance
app = create_fastapi_app(
    env=GeoPolicyEnv,
    action_cls=GeoAction,
    observation_cls=GeoObservation,
)


def main():
    """Entry point for `uv run server` and `python -m server.app`."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
