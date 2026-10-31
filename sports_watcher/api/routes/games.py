"""Game jobs: create, inspect, backfill events, stream events, read firsts, stop."""

from __future__ import annotations

import asyncio
import contextlib
import queue

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from sports_watcher.api.jobs import GameJob, JobManager
from sports_watcher.api.schemas import (
    CreateGameRequest,
    CreateGameResponse,
    GameListResponse,
    GameStatusResponse,
)
from sports_watcher.events.firsts import Firsts
from sports_watcher.events.model import Event

router = APIRouter(prefix="/api/v1/games", tags=["games"])
RECENT_EVENTS = 20
WS_POLL_S = 0.25


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _job_or_404(jobs: JobManager, game_id: str) -> GameJob:
    try:
        return jobs.get(game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown game: {game_id}") from exc


@router.post("", status_code=201, response_model=CreateGameResponse)
def create_game(req: CreateGameRequest, request: Request) -> CreateGameResponse:
    try:
        game_id = _jobs(request).create(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateGameResponse(game_id=game_id)


@router.get("", response_model=GameListResponse)
def list_games(request: Request) -> GameListResponse:
    return GameListResponse(game_ids=_jobs(request).list_ids())


@router.get("/{game_id}", response_model=GameStatusResponse)
def game_status(game_id: str, request: Request) -> GameStatusResponse:
    job = _job_or_404(_jobs(request), game_id)
    events = job.watcher.events
    return GameStatusResponse(
        game_id=game_id,
        home=job.request.home,
        away=job.request.away,
        league=job.request.league,
        broadcaster=job.request.broadcaster,
        status=job.watcher.status(),
        recent_events=events[-RECENT_EVENTS:],
    )


@router.get("/{game_id}/events", response_model=list[Event])
def game_events(game_id: str, request: Request, since_seq: int = 0) -> list[Event]:
    return _job_or_404(_jobs(request), game_id).watcher.events_since(since_seq)


@router.get("/{game_id}/firsts", response_model=Firsts)
def game_firsts(game_id: str, request: Request) -> Firsts:
    return _job_or_404(_jobs(request), game_id).watcher.firsts()


@router.post("/{game_id}/stop", status_code=202)
def stop_game(game_id: str, request: Request) -> dict[str, str]:
    _job_or_404(_jobs(request), game_id).watcher.stop()
    return {"game_id": game_id, "stopping": "true"}


@router.websocket("/{game_id}/events")
async def stream_events(websocket: WebSocket, game_id: str, since_seq: int = 0) -> None:
    """Backfill from ``since_seq``, then stream live events. Closes once the game has
    ended and everything has been sent, or as soon as the client goes away (a receiver
    task watches for the disconnect so a quiet game never keeps a zombie subscription)."""
    jobs: JobManager = websocket.app.state.jobs
    try:
        job = jobs.get(game_id)
    except KeyError:
        await websocket.close(code=4404, reason=f"unknown game: {game_id}")
        return
    watcher = job.watcher
    await websocket.accept()
    live = watcher.broadcaster.subscribe()
    last_sent = since_seq
    receiver = asyncio.ensure_future(websocket.receive())
    try:
        for event in watcher.events_since(since_seq):
            await websocket.send_json(event.model_dump(mode="json"))
            last_sent = event.seq
        while True:
            getter = asyncio.ensure_future(asyncio.to_thread(live.get, True, WS_POLL_S))
            done, _pending = await asyncio.wait({receiver, getter}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                if _client_gone(receiver):
                    getter.cancel()
                    return
                receiver = asyncio.ensure_future(websocket.receive())  # ignore client chatter
            try:
                event = await getter
            except queue.Empty:
                if watcher.status().state in ("ended", "failed") and live.empty():
                    return
                continue
            if event.seq <= last_sent:
                continue
            await websocket.send_json(event.model_dump(mode="json"))
            last_sent = event.seq
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        receiver.cancel()
        watcher.broadcaster.unsubscribe(live)
        with contextlib.suppress(Exception):
            await websocket.close()


def _client_gone(receiver: asyncio.Future) -> bool:
    """True when the receiver finished with a disconnect message or an error."""
    if receiver.cancelled():
        return True
    exc = receiver.exception()
    if exc is not None:
        return True
    message = receiver.result()
    return isinstance(message, dict) and message.get("type") == "websocket.disconnect"
