"""Request and response bodies."""

from __future__ import annotations

from pydantic import BaseModel

from sports_watcher.events.model import Event
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.watcher import WatcherStatus


class CreateGameRequest(BaseModel):
    source: SourceSpec
    league: str = "pba"
    broadcaster: str = "one_sports"
    home: str
    away: str
    home_color: str | None = None
    away_color: str | None = None


class CreateGameResponse(BaseModel):
    game_id: str


class GameListResponse(BaseModel):
    game_ids: list[str]


class GameStatusResponse(BaseModel):
    game_id: str
    home: str
    away: str
    league: str
    broadcaster: str
    status: WatcherStatus
    recent_events: list[Event]


class HealthResponse(BaseModel):
    status: str
    version: str
