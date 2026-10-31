"""App factory. Run with: uvicorn sports_watcher.api.main:create_app --factory"""

from __future__ import annotations

from fastapi import FastAPI

from sports_watcher import __version__
from sports_watcher.api.jobs import JobManager
from sports_watcher.api.routes import games, health
from sports_watcher.config import Settings


def create_app(settings: Settings | None = None, jobs: JobManager | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Sports Watcher", version=__version__)
    app.state.settings = settings
    app.state.jobs = jobs or JobManager(settings)
    app.include_router(health.router)
    app.include_router(games.router)
    return app
