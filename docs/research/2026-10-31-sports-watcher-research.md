# Sports Watcher: research notes

Date: 2026-10-31. Author: Stephen with Claude. Status: research only, no code yet.

## 1. Goal

Watch a live game feed and emit a running stream of what is happening: who has the
ball, passes, shots, scores, pitches, serves, tip-off, kickoff. Each event carries a
timestamp, the team, and where possible the player and pitch position. Micro-betting
markets ("first shot", "next goal", "first pitch strike", "who wins the tip-off") are
consumers of that stream: they are opened, priced and settled from it.

The app follows the GodsEye shape (Python, FastAPI, detect, track, classify, live
reader) but targets sports actions instead of violence.

## 2. What GT / APS already has

Source: `GlobalNet/repositories/{gtservices,aps}` and the `sports` Postgres database
on 192.168.181.20 (VPN). `aps` is the maintained copy of the Java sportsbook engine
("steam", inside `goblin`); build against it, not `gtservices`.

### 2.1 Live data today

Two provider feeds, both parsed by Java code in `goblin/src/com/goblinstudios/steam/parser/`:

| Provider | Transport | Messages | Code |
|---|---|---|---|
| LSports (current) | XML over RabbitMQ (AMQP), vhosts `StmInPlay` / `StmPreMatch` | `FIXTURE_METADATA` 1, `LIVESCORE` 2, `MARKET` 3, `KEEP_ALIVE` 31, `HEARTBEAT` 32, `SETTLEMENT` 35, `SNAPSHOT` 36 | `lsport/FeedParser.java` (8.4k lines), `lsport/Feed.java`, `feed.xsd` |
| Don Best (legacy) | XML over ActiveMQ + REST | `event_state`, `period_score`, `matchup_score`, `pitching_change`, `line` | `donbest/v2/inplay/ActiveMQParser.java` |

`externaleventmapping.sourceid` maps a GT event to the provider fixture:
`Source { DONBEST(1), LSPORT(2) }` in `ExternalEventMapping.java`. 673,848 LSports
mappings exist in the DB.

What the live feed carries is score state, not actions. `gamescore` has
`homescore, awayscore, period, timeleft, strikes, balls, outs, bases, fieldpos,
fieldside, down, firstdown, possession`. The LSports `Statistics` block is never read
by `FeedParser` (zero references). Nothing records "player X passed at 03:12".

### 2.2 Markets that already exist

The DB table `templateproposition` (174 rows) holds hand-made prop templates. Many
are exactly the "first X" questions this project is about:

| Sport | Template text |
|---|---|
| Football (NFL) | `{0} 1st Pass Will Be:`, `Yardage of {0} 1st Rush Attempt`, `What Will {0} Throw 1st?`, `Team With 1st Defensive QB Sack`, `1st OFFICIAL Play From Line of Scrimmage Will Be`, `Team to Score 1st`, `What Will Happen 1st? (TD vs FG)`, `Team To Commit 1st ACCEPTED Penalty`, `1st Scoring Play` |
| Baseball | `Which team will score first?`, `{0} First Pitch: Ball vs Strike`, `Will {0} Reach Base His First Plate Appearance?`, `What Will {0} Do First?`, `Team To Hit First Home Run` |
| Basketball | `Team to Score 1st`, `1st Team To Score 10 Points`, `Will {0} Make His 1st Free Throw Attempt?` |
| Soccer | `1st To Score`, `Total Corners` |
| Hockey | `Team To Score First` |
| Tennis | `Player to Win 1st Set` |

The LSports market catalog (`lsport/MarketType.java`, 490 entries) adds provider-run
micro-markets: `FIRST_TEAM_SCORE` 16, `NEXT_GOAL` 59, `FIRST_PLAYER_SCORE_BASKET` 2740,
`RACE_TO_20_PTS` 1050, `TEAM_FIRST_TIMEOUT` 845, `TEAM_FIRST_PENALTY` 933,
`FIRST_SCORE_METHOD` 453, `UNDER_OVER_SHOT_ON_TARGET` 132, `UNDER_OVER_CORNER_*`,
`UNDER_OVER_THROWIN_*`. There is no "first pass", "first shot attempt" or "first
possession" market yet. Convention for new ones: `FIRST_/NEXT_/LAST_<THING>[_HOME|_AWAY][_1H|_1P]`
plus a title in `propTitleMarketTypeMap`.

Prop volume by sport in `propositionbet` (644k rows): Football 55.9k, Soccer 49.2k,
Hockey 44.4k, Tennis 4.8k, Basketball 4.7k, Baseball 2.1k.

### 2.3 How props are settled today

`AutoGrader.java` grades only when the event status is FINAL, and
`PropositionBet.grade` is "restricted to feed events": the winner comes from the
provider's `SETTLEMENT` (type 35) message, matched by `PlayerName` + `Line`. There is
no path that settles a prop from an observed action mid-game.

### 2.4 Integration points for the watcher

1. Cheapest: emit LSports-shaped `<Feed>` XML and publish it to the in-play RabbitMQ
   queue. `FeedParser` then creates markets, updates scores and settles with no Java
   change. A local broker that imitates LSports exists:
   `local-tools/rabbitmq/docker-compose.yml` (RabbitMQ 3.13, ports 5672/15672,
   exchange `lsports.fanout.inplay`, queue `_2944_`). Real sample messages (NBA, NHL)
   are in `aps/feeds/*.xml`, including `suns_vs_pacers_settlement_PROPS.xml`.
2. Event identity: add `Source.WATCHER(3)` to `externaleventmapping`, or reuse the
   LSports `FixtureId` the frontends already know as `ls_id`.
3. New `MarketType` entries for first-pass / first-shot / first-possession.
4. Push to clients: Resin websocket servlet with `Sec-WebSocket-Protocol: gt-sports-*`,
   or REST `GET /sportsbook/score?eventid=`.

Caution: `goblin/src/properties/sportsbook.properties` and `pooler.xml` in both repos
contain live credentials. Do not copy them into this project.

## 3. Betting micro-markets (nVenue, bet365, others)

Full report with URLs: `appendix-b-micro-markets-report.md`.

### 3.1 What actually trades

Micro-markets that carry volume are "next X" on a discrete unit the official scorer
already records: next pitch, next plate appearance, next NFL play, next drive outcome,
next NBA possession result or next field-goal type, next tennis point or serve
outcome. "First X" is the same market applied to unit number one.

| Provider | Markets | Notes |
|---|---|---|
| nVenue | NFL per play / per drive (Sportradar UOF ids 1061-1065, 1448-1465, 1573-1574), MLB per pitch / per plate appearance (1378-1381, 1400-1403), NHL faceoff and 5-minute windows (1498-1505, 1569-1572) | Site is marketing only: "<50 ms prediction generation", "1M+ markets/day". No public API, market list or pricing. Inputs are official feeds, never video. |
| bet365 in-play | Baseball next pitch ball/strike/in play, at-bat hit/walk/K. Basketball Next Field Goal Method, Team to Score Next FG, First Basket, Race To. NFL drive markets, Next Play Type, Next Scoring Play. Soccer Next Goal, corners and cards in play, Race To, Time of First Corner, player shots (Opta). Tennis Point Betting, Next Game First Point. | Site blocks automated access; from help-page extracts. |
| Simplebet (DraftKings) | NFL Drive Result Exact (4-way) is the most popular football micro; NBA next made basket is 3/4 of volume; MLB pitch ball/strike/in play | $262m MLB handle 2022. |
| Genius BetVision | NFL Next Drive Outcome, Next TD Scorer in-stream; soccer "touch-to-bet" | Stream runs 4 to 6 s behind live. |
| Betr, Huddle, Kero, Swish | Possession result exact, next FG type, pitch result, contextual markets | Kero opens new markets every ~30 s from play-by-play. |

### 3.2 Regulatory direction

- 16 Oct 2026: the NBA told US sportsbooks to stop offering first basket, first 3,
  first rebound/assist/steal/block, "player misses first/next shot", and tip-off
  winner.
- MLB capped single-pitch bets at $200 and barred them from parlays (Nov 2025). Ohio
  is drafting a rule removing player-specific in-game micro props.
- Team-level and possession-level markets (first team to score, first corner, first
  throw-in, next possession result, drive result) are untouched.

GT's licensees are not US books, but the integrity signal is the same. The watcher
should lead with team-level and possession-level events and treat single-player
"first" props as optional outputs.

### 3.3 Who settles, and on what

Every book settles on the official stat feed. DraftKings house rules: bets "are
settled using the statistics provided by the Event's official statistical provider".
bet365 soccer props settle on Opta. The watcher is therefore not the settlement
authority on day one. Its value is:

1. Latency: an event the moment it is visible, before the provider's message.
2. Auto-grading GT's own hand-made props, which today wait for FINAL.
3. Markets where GT has no provider market (first pass, first shot attempt, first
   possession).
4. Redundancy and a second opinion on provider data.

Definitions the event model must honour:

- Opta shot on target includes keeper saves and last-line blocks; a shot off the
  woodwork is off target unless it goes in; a blocked shot is neither.
- LSports: First Match Corner (id 998) settles on the team awarded the corner; First
  Throw-in (id 1849) on the team awarded it; no data received means "Not Set".
  bet365 corner totals count only corners taken.
- DraftKings NBA: a possession ends on a score, after free throws, or when the other
  team gains possession; period end before a valid attempt voids.
- DraftKings MLB pitch result: hit by pitch and pitch-out count as a ball, a dropped
  third strike is a strike, a pitch-timer violation is not a pitch.
- DraftKings NFL next play: a sack is a pass attempt; a play nullified by an accepted
  penalty voids.

### 3.4 Event vocabularies to mirror

The watcher's event schema should map one-to-one onto these so GT can reconcile
against provider data later.

| Sport | Standard | Core types |
|---|---|---|
| Soccer | StatsBomb Open Data v4 | Pass 30, Shot 16, Carry 43, Ball Receipt 42, Duel 4, Interception 10, Clearance 9, Dribble 14; location on 120x80; pass end_location, recipient, outcome; shot outcome Goal 97, Saved 100, Blocked 96, Off Target 98, Post 99 |
| Soccer | Sportradar push | match_started, period_start, score_change, possible_goal, shot_on_target, shot_off_target, shot_saved, free_kick, goal_kick, throw_in, corner_kick, offside, cards; x/y 0-100 |
| NFL | Genius Match Events v2 | Snap, PassAttempt, CompletePass, IncompletePass, Run, Sack, Tackle, Fumble, Touchdown, FieldGoalAttempt, Punt, KickOff, Penalty, Possession, CoinToss, Down, YardsToEndzone |
| NBA | NBA PBP / Sportradar | EVENTMSGTYPE 1 FG made, 2 FG missed, 3 FT, 4 rebound, 5 turnover, 6 foul, 10 jump ball; Sportradar opentip, twopointmade/miss, threepointmade/miss, freethrowmade/miss, rebound, turnover, possession; coords 1128x600 |
| MLB | Statcast / Sportradar | pitch_type, release_speed, type B/S/X, description, events, launch_speed; outcome_id kKL, kKS, kF, bB, aS, aHR, oGO |
| NFL tracking | Next Gen Stats / Big Data Bowl | ball_snap, pass_forward, pass_arrived, pass_outcome_caught, handoff, tackle, first_contact at 10 Hz |

### 3.5 Candidate markets by video difficulty

Easy means one visible, unambiguous physical event. Medium needs player identity, an
official's decision, or sub-second timing. Hard depends on a scorer judgement or
off-camera information.

| Sport | Easy | Medium | Hard |
|---|---|---|---|
| Soccer | next/first team to score, next goal, throw-in taken, corner taken, race to X corners, goal in next N min, time of first corner | first/next goalscorer, first/next shot (intent), card, free kick, offside, corner awarded (referee off frame) | shot on target (trajectory to frame plus last-line-block rule) |
| Basketball | race to X points, first team to score, tip-off winner (visible, but banned for US books) | next possession result, next FG type (2 vs 3 is a foot-on-line call), shooter identity | player next shot made/missed |
| NFL | first team to score, next scoring play | next play pass/rush, drive result, TD scorer | yards to the yard, drive crosses yard line X |
| MLB | pitch count, runs, race to X runs | pitch result (umpire call, catcher hides the zone), at-bat result | pitch speed (needs radar) |
| Tennis | next point winner (scoreboard), next break, game to deuce | ace vs rally (receiver contact), stroke type | line calls |

## 4. Computer-vision stack

Full report with URLs: see the agent report saved under this session; the points
that decide the design are below.

### 4.1 Language

Python 3.12. Every model worth using ships as PyTorch, and TensorRT or ONNX Runtime
engines run at full speed from Python. C++ (DeepStream) only pays off past 3 to 4
streams per GPU and costs 5 to 10x the dev time. Rust has no CV research ecosystem and
immature TensorRT bindings. Go has nothing beyond OpenCV DNN.

Pattern: PyAV decode (NVDEC on Linux, VideoToolbox on the Mac), frames kept on the
GPU, one process per stream, asyncio stage queues, TensorRT FP16 engines on Linux
and ONNX Runtime CoreML / MPS on the Mac from the same ONNX graphs.

### 4.2 Models (licenses matter: this is a commercial betting product)

| Stage | Pick | License | Why |
|---|---|---|---|
| Player detection | RF-DETR (Roboflow) or D-FINE | Apache-2.0 | At or above YOLO26 accuracy. Ultralytics YOLO is AGPL, which pulls a closed network service into scope. GodsEye's YOLOv8 has the same problem. |
| Tracking | BoostTrack++ / OC-SORT (MIT), ARC-SORT (Apache) | MIT / Apache | Sports gains come from camera-motion compensation and tracking in pitch coordinates, not the association rule. `boxmot` is AGPL, avoid. |
| Ball | small RF-DETR / D-FINE head on full frame + WASB heatmap on a crop | Apache / MIT | Community soccer ball detection is around 60% without this. TrackNetV3 for tennis only. |
| Pose | RTMO (one-stage) or RTMPose on tracked crops | Apache-2.0 | RTMPose-m runs 90+ FPS on CPU. Sapiens is CC-BY-NC, not usable. |
| Action spotting | FOOTPASS TAAD/DST baselines (Apache) re-implemented; T-DEED is GPL, reference only | Apache | SoccerNet Ball Action Spotting 2024 winner reached 73.4 mAP@1 on 12 classes (pass, drive, cross, shot, throw-in, goal...). |
| Calibration | TVCalib (MIT) or own keypoint model; PnLCalib (GPL) as a separate process | MIT | Needed for "inside the box", pitch x/y, and tracking in pitch coordinates. |
| Team / player ID | SigLIP crops → KMeans (roboflow/sports, MIT); jersey OCR with a small CNN or PaddleOCR | MIT / Apache | Jersey numbers from broadcast top out around 85 to 93% tracklet accuracy. |
| VLM | Qwen3-VL (Apache) | Apache | Offline QA and hard-case labelling only. No source shows a VLM inside a 2 s live path. |

### 4.3 Nothing runs online out of the box

No SoccerNet winner was evaluated in streaming mode and none publish latency. Clip
heads need 2 to 4 s of video. Design that follows from this:

- Tier 1, provisional, causal, under 0.5 s: rules over tracking state. Possession
  change = ball within r of a new player's feet for k frames. Shot = ball speed spike
  toward goal. Kickoff = whistle, centered ball, two players.
- Tier 2, confirmed, 2 to 4 s: a windowed spotting head over per-player features.
  Emits `confirmed` or `retracted` for each provisional event.

Both tiers publish to a stream (Redis Stream or NATS) with stream PTS, wall clock,
team, jersey, pitch x/y and zone.

### 4.4 Pipelines to study or fork

| Repo | License | Gives us |
|---|---|---|
| roboflow/sports | MIT | Player/ball/pitch-keypoint datasets, ball tracker, team clustering, homography |
| SoccerNet/sn-gamestate + tracklab | GPL-3 / MIT | Full broadcast → minimap pipeline with swappable calibration |
| JeremieOchin/FOOTPASS | Apache-2.0 | Only Apache spotting baseline with team + jersey attribution |
| lRomul/ball-action-spotting | MIT | 2023 pass/drive winner, short frame stacks, easy to run causally |
| SkillCorner/opendata | MIT | 10 matches of commercial broadcast tracking output as a target format |

### 4.5 Data and feeds

- Labelled video: SoccerNet (NDA, 500 games, pass/shot labels), SportsMOT and
  MultiSports (CC BY-NC), MLB-YouTube (20 games), TenniSet (5 matches). Most sports
  video datasets are NC or NDA. Whether a betting product may train on them is a
  legal question to settle before v1.
- Live feeds: every consumer service (MLB.TV, League Pass, NFL+, ESPN) is DRM and 10
  to 30 s behind the venue over HLS. YouTube via `yt-dlp -g` works for private dev
  only. IPTV lists are fine for testing HLS plumbing and nothing else.
- For latency work, build a replay rig: `ffmpeg -re` looping SoccerNet or recorded
  games into mediamtx over SRT or RTSP. It is the only feed where we control
  glass-to-glass latency.

### 4.5.1 Basketball feed check (2026-10-31)

Checked at Stephen's request, from the Philippines.

**NBA TV / League Pass (nba.com/watch/nba-tv).** The NBA TV 24/7 player is locked
behind a subscription ("Unlock NBA TV to watch", Subscribe or Sign In) and no video
manifest loads without login. Pricing shown for the Philippines: League Pass
PHP 3,919 per season (1 device, with commercials), Premium PHP 5,039 (3 devices, no
commercials, offline downloads), Team Pass PHP 499 per month. The NBA's Philippines
page says no blackout restrictions. Prime Video also carries NBA games in the
Philippines from October 2025 under the league's global rights deal. Playback in a
browser uses Widevine DRM (forum reports; the NBA publishes no DRM page), so the
stream cannot be read by ffmpeg or PyAV. The only way to get frames is screen capture
of a licensed player.

NBA Terms of Use (nba.com/termsofuse) matter here:

- Basketball Content (video, audio, stats) is for "personal, noncommercial use" and
  may not be reproduced, transmitted or used for commercial purposes without written
  permission.
- NBA Statistics from the Services "may not be used in connection with any gambling
  activity (including legal gambling activity)" and may not be used in "any product
  or service that presents a live, near-live, or other real-time or archived
  play-by-play account or depiction of any NBA game".

Verdict: NBA League Pass is fine for a human to study games, but not a feed this
product can ingest, and the terms rule it out as a betting data source. Official NBA
data for betting comes through Sportradar or Genius, which is what LSports resells.

**PBA on One Sports (YouTube).** Since 24 October 2025 Cignal TV streams every PBA
game free on the One Sports YouTube channel, "with no delays" and "watchable
worldwide" (Inquirer, 27 Oct 2025). Confirmed today: "LIVE: TNT vs MAGNOLIA | PBA
SEASON 50 GOVERNORS' CUP QUARTERFINALS | OCT 28, 2026" (2:33:41) and "LIVE: RAIN OR
SHINE vs CONVERGE | OCT 30, 2026" (2:47:16) are on the channel as full recordings.
That gives us, for basketball:

- Live games several times a week with English and Filipino commentary.
- Full-game recordings for golden clips and replay-rig testing.
- A single-broadcaster production style, which helps the camera-state classifier.
- Ingest through `yt-dlp -g` to the HLS manifest, then PyAV. YouTube's terms bar
  downloading without permission, so this is a development source until Cignal or
  the PBA grants a feed. For a product feed, ask Cignal for the contribution stream
  (SRT or RTMP) that YouTube itself receives.

`yt-dlp` is not installed on this Mac yet.

**ESPN Philippines (espn.ph).** A news and scores site; it has no live player of its
own and loads no video manifest. Live NBA in the Philippines is sold as "ESPN on
Disney+" inside the Disney+ app under the league's 11-year global rights deal. The
published 2026-27 schedule (disney.ph/nba) is about 80 regular-season games, mostly
two per week on Thursday mornings PHT, plus Christmas Day, NBA Cup group play and
the All-Star Celebrity Game; one Conference Finals series live, the other series
and the Finals on delay. Disney+ playback is DRM-protected (Widevine in browsers,
PlayReady and FairPlay on other devices, HDCP enforced, browsers capped at 720p),
so, like League Pass, it cannot be read by ffmpeg or PyAV and its terms are for
personal viewing. ESPN5 on free TV ended in 2020. No subscription price is shown on
the NBA page; the plan price lives in the Disney+ app.

**Requirement (Stephen, 2026-10-31): the watcher must read whichever live feed we
are given.** Ingest is therefore a set of source adapters behind one interface
(frames with PTS and wall clock, plus demuxed audio): local file or directory,
HLS or DASH URL (including a YouTube live via yt-dlp), RTSP pull, RTMP, SRT or UDP
MPEG-TS push from a broadcaster's contribution encoder, a UVC HDMI capture card
(set-top box or venue switcher output), and screen or window capture on a Mac for
quick trials. DRM-protected apps (League Pass, Disney+) stay out: HDCP blocks HDMI
capture of them and their terms forbid it. Each adapter reports its own latency
class so events carry an honest "behind live" estimate.

**Decision for v1:** PBA (FIBA court dimensions, 24 s shot clock) is the first
league. The court template and roster loader are per-league config so NBA follows
when a lawful feed exists.

### 4.6 Which sport first

Soccer, main broadcast camera. It has the only open benchmark with pass/shot/cross
labels at scale, published winners with code, a working calibration stack and a full
reference pipeline. Kickoff is scripted, so first possession, first pass and first
touch are the easiest events to get right. Basketball is next: tip-off is equally
scripted, but there is no public broadcast event dataset and court calibration is
community grade. Tennis is mechanically simplest but has almost no labelled broadcast
data. Baseball and American football are hardest from broadcast (camera cuts,
occlusion at the snap and pitch).

Tension with GT's business: prop volume is NFL first, soccer second. The design
should keep the sport-specific part (event rules, calibration, spotting classes) in a
plug-in per sport so NFL and NBA follow without touching the core.

## 5. Dev machine

M1 Pro Mac, Python 3.12.9, ffmpeg, Docker, psql. No NVIDIA GPU. Dev runs on CPU /
MPS / CoreML with small models and recorded clips. Real-time inference needs a Linux
box with an RTX 4090-class GPU.

## 6. Open questions

1. Output target: a standalone event API and stream (GodsEye style) that GT consumes
   later, or LSports-shaped XML straight into the APS feed parser from day one?
2. First sport: soccer (easiest, recommended) or NFL (largest GT prop volume)?
3. Feed source: does GT or a partner have a broadcast or venue feed with low latency,
   or do we start from recorded games and a replay rig?
4. Where does the GPU runtime live (on-prem Linux box, cloud GPU)?
5. Dataset licensing for a commercial product (SoccerNet NDA, CC BY-NC sets).
