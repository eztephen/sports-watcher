"""One GameWatcher per game, each on its own daemon thread. In-memory, like GodsEye."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from sports_watcher.api.schemas import CreateGameRequest
from sports_watcher.config import Settings, load_league
from sports_watcher.perception.scoreboard import RapidOcrTextReader, TextReader
from sports_watcher.watcher import GameWatcher


@dataclass
class GameJob:
    game_id: str
    request: CreateGameRequest
    watcher: GameWatcher
    thread: threading.Thread


class JobManager:
    def __init__(
        self,
        settings: Settings,
        text_reader_factory: Callable[[], TextReader] = RapidOcrTextReader,
        ocr_every_n_frames: int = 5,
    ) -> None:
        self._settings = settings
        self._text_reader_factory = text_reader_factory
        self._ocr_every_n_frames = ocr_every_n_frames
        self._jobs: dict[str, GameJob] = {}
        self._lock = threading.Lock()

    def create(self, req: CreateGameRequest) -> str:
        try:
            league = load_league(req.league, self._settings.configs_dir)
        except FileNotFoundError as exc:
            raise ValueError(f"unknown league: {req.league}") from exc
        if req.broadcaster not in league.broadcasters:
            raise ValueError(f"unknown broadcaster: {req.broadcaster}")
        game_id = uuid4().hex[:12]
        watcher = GameWatcher(
            req.source,
            league,
            req.broadcaster,
            game_id=game_id,
            home=req.home,
            away=req.away,
            text_reader=self._text_reader_factory(),
            ocr_every_n_frames=self._ocr_every_n_frames,
        )
        thread = watcher.start_in_thread()
        with self._lock:
            self._jobs[game_id] = GameJob(game_id, req, watcher, thread)
        return game_id

    def get(self, game_id: str) -> GameJob:
        with self._lock:
            return self._jobs[game_id]

    def stop(self, game_id: str) -> None:
        self.get(game_id).watcher.stop()

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._jobs)
