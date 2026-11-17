---
name: sports-watcher-context
description: "What sports-watcher is for, where the related repos and DB live, how to reach the GT sports database (VPN), and where the project stands"
metadata:
  type: project
---

sports-watcher (started 2026-10-31) is a Python computer-vision app that watches live
game video and emits a stream of game events (possession, pass, shot, score) to drive
and settle micro-betting markets on the GT/APS sportsbook. Modelled on GodsEye
(`~/Work_These_Out/AAFE/repositories/GodsEye/godseye-backend`, FastAPI + YOLO + 3D CNN).

**Why:** GT's live data (LSports XML over RabbitMQ, Don Best in-play over ActiveMQ)
carries score state only; props are graded only at FINAL by `AutoGrader`. There is no
per-action event stream. See [[user-profile-stephen]].

**How to apply:**
- Start from `docs/HANDOFF.md` in the repo, then the research notes in `docs/research/`
  and the specs and plan notes in `docs/superpowers/`.
- `aps` (not `gtservices`) is the maintained copy of the Java sportsbook engine.
- The GT `sports` Postgres DB sits on the GT internal network and needs Stephen's VPN
  on. Host and credentials are in `gtservices/goblin/src/properties/pooler.xml` under
  `~/Work_These_Out/GlobalNet/repositories`. Never copy them into this repo (it is public).
- Cheapest integration: publish LSports-shaped `<Feed>` XML to the in-play RabbitMQ
  queue; `local-tools/rabbitmq` under GlobalNet imitates LSports locally; sample
  feeds in `aps/feeds/*.xml`.
- Dev box is an M1 Pro Mac (no NVIDIA GPU); Python 3.12.9, ffmpeg, Docker present.
- Licensing: only Apache/MIT/BSD runtime deps (RF-DETR, SigLIP, RapidOCR); Ultralytics
  YOLO and boxmot are AGPL, a problem for a commercial betting product.
- Basketball first (Stephen's choice, 2026-10-31), PBA as the first league: One Sports
  streams every PBA game free on YouTube with no delay and keeps full recordings
  (downloads are for development only). NBA League Pass is Widevine DRM and its terms
  bar gambling use, so it is not a feed. Commentary audio (ASR) is a required evidence
  channel (M2, not started).
- Status 2026-11-17: M0 and M1 are on `main` (commit e7d50fc, pushed to
  github.com/eztephen/sports-watcher, 302 unit tests). Score changes, made shots, shot
  attempts, free throws and timeouts are solid on PBA file replays; possession, rebound
  and turnover are weak. Perception runs at about 2 frames a second on the Mac, so live
  vision needs a GPU. Open items that need Stephen: a GPU timing run on Colab, and shot
  and possession truth for the second game (`pba_gin_smb_q1`). Next: M2 commentary, M3
  hardening, spec 2 (markets and the APS adapter).
