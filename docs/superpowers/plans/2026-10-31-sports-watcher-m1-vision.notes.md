# M1 execution notes (rulings, measurements and deferred items)

Copied from the execution ledger of `docs/superpowers/plans/2026-10-31-sports-watcher-m1-vision.md` on 2026-11-05.

## Rulings

- T7 -> T8,T9: confirmers — consistent. T8 extends Verdict to "retracted:<reason>"; sports/base.py Verdict and fuser Resolution are Literals. Ruling: Verdict becomes str (forms "pending" | "confirmed" | "retracted" | "retracted:<reason>"), EventEngine._terminal splits the reason — cost if wrong: a looser type alias.
- Task 6: Ruling: a release is the ball leaving the holder's grown box in any direction (the brief's "image y smaller than the previous frame" made a flat pass record no release, against its own test); the upward check lives in shot_path_passes_rim (fitted dy/dt at release must be negative) as spec 6.3 puts it on shot_attempt — cost if wrong: a bounce pass toward the rim is never a shot candidate, which is right.
- Task 6: Ruling: HOLD_S = 0.165 (five frame intervals at 30 Hz; the spec's 0.17 s rounded) so a sixth consecutive frame establishes the holder — cost if wrong: one frame of latency.
- Task 6: Ruling: the shot fit refits on every new ball observation from four points up to the 1.5 s window instead of once at exactly four (a 0.13 s baseline extrapolated over a second misses real rims); RimInteraction gains above_pts so a rim crossing is detected within 0.5 s of the ball being above the rim inside its span — cost if wrong: a draft a few frames later than the spec's "next four observations" wording.
- Task 6: Ruling: the lane formation is carried while the holder is None and the clock is stopped (the shooter has just released) so FreeThrowRule can match the release to the shooter — cost if wrong: a stale formation for one dead-ball stretch, cleared by any new holder or the clock running.
- Task 7: Ruling: Verdict and Resolution become str so a confirmer can answer "retracted:<reason>" and the engine records that reason (pre-flight row applied here, one task early, because team_holds_unless_attempt needs it) — cost if wrong: a looser type alias.
- Task 7: Ruling: the rebound confirmer is rebound_holds (team_holds plus a retraction when the shot's team scores after all) instead of plain team_holds: a made basket whose delta arrives after the next holder would otherwise confirm a phantom rebound — cost if wrong: one more confirmer.
- Task 7: Ruling: the brief's turnover test expected t_pts 0.77; possession_since_pts is the start of the hold (0.6), as spec 6.2 defines the holder's since — test corrected, code unchanged — cost if wrong: none.
- Task 8: Ruling: a confirmer may answer "confirmed:<confidence>" and the engine passes it to confirmed_copy (M0 already had the parameter): a vision-only shot_made confirmed by the scoreboard delta ends at 0.9, the two-signal confidence, and carries points from the attempt's zone; a 2-or-3 delta that disagrees with those points retracts it (points_disagree) and the delta path drafts the right one — cost if wrong: one more verdict form.
- Task 8: Ruling: retroactive attempts (and their shot_made) fire only once vision has been seen (GameState.has_vision); without weights the watcher must behave exactly as M0 (addendum 3.1), and three M0 tests pinned that — cost if wrong: none for vision runs.
- Task 8: Ruling: Release/ShotRecord carry no court position when the homography confidence is under 0.4, so shot_attempt.court_xy is None together with zone unknown (spec 8) — cost if wrong: none.
- Task 9: Ruling: a player's reach box is widened by 0.25 h on each side and raised by 0.1 h (hands up), not grown by 0.25 h above the head: the tip-off ball rising between the jumpers was otherwise inside both boxes and became a holder — cost if wrong: a ball held overhead for 0.17 s with arms fully extended is briefly unheld.
- Task 9: Ruling: a rising ball (image y falling by over 0.5 px per frame) never starts a holder candidate: in 2D a ball in flight passes over farther players' boxes (the free-throw ball crossed a lane player's box) — known limitation: a descending ball over a far player for 0.165 s can still start a candidate; revisit with the ball filter's height estimate — cost if wrong: a late holder on a two-handed overhead catch.
- Task 11: Ruling: build_perceiver imports cache.py and models.py inside the function (cache.py imports the Perceiver protocol from perceiver.py, so a module-level import would be a cycle) — cost if wrong: none.
- Task 12: Ruling: the observation-centric re-update after a gap re-seeds the Kalman state with the new box and the straight-line velocity from the last observation in one step instead of replaying virtual observations frame by frame — same effect on the next prediction, less code — cost if wrong: a slightly larger covariance after long gaps.
- Task 13: Ruling: the homography is fitted court -> image (RANSAC threshold and reprojection error in pixels) and inverted, because OpenCV measures the threshold in destination units and 8 metres rejects nothing — cost if wrong: none.
- Task 14: Ruling: the brief's sampling test observed 30 frames (t up to 0.967 s) and expected a second sample "at the one-second mark"; the test now runs 31 frames so t reaches 1.0 — test corrected, the one-per-second rule unchanged — cost if wrong: none.
- Task 15: Ruling: the keypoint model wrapper reshapes the keypoints output to (Q, K, 3) and takes the most confident query; the exact layout of rfdetr's keypoint ONNX output is verified against the exported court model in Task 17 — cost if wrong: one reshape to adjust.
- Task 15: Ruling: bench_perceiver.py takes no --resolution flag; the resolution is the detector manifest's, and a second size is benchmarked from a second models directory — cost if wrong: none.
- Task 16: Ruling: labels carry a 0.5 s margin inside every boundary (0.25 s for one-second segments) because they were read off 1 fps contact sheets; the camera-state scorer ignores frames outside segments, so boundary frames never count either way — cost if wrong: slightly fewer training and scoring frames.
- Task 16: Ruling: the timeout huddle and bench stretches are labelled closeup (the camera is on the bench, not the court) and the free-throw wide shots with a lower-third banner are main (the main camera with an overlay) — cost if wrong: a relabel of two segment kinds.
- Task 17: Ruling: the court keypoint template (configs/court_keypoints/basketball-court-detection-2.yaml) cannot be authored without the dataset's 33 keypoint names, which only the Roboflow download exposes (the Universe page and the RF-DETR keypoint tutorial list none); its test skips with that reason until the yaml exists — cost if wrong: the court model runs only after the template is written.
- Task 18: Ruling: the vision truth was drafted from bug-only crops at 2 fps (score changes exact to 0.5 s, matched by the scoreboard-only watcher within 0.7 s) and 2 and 4 fps play sheets; this feed resets the shot clock to 24 on offensive rebounds (seen with the Rain or Shine putback at 55.5 s), so the resets at 39.5, 75.0 and 167.5 s are read as offensive rebounds, not possession changes; the uncertain spots are listed under truth_uncertain in the clip's meta for Stephen's pass — cost if wrong: a few truth lines move by a second or change team.
- Task 18: Ruling: a golden clip whose scoreboard check ran but whose vision checks had no perceiver passes with a warning instead of skipping (a skip hid the scoreboard pass); it skips only when no check could run — cost if wrong: none.

## Deferred and waiting on weights

- Task 16: note: probe training (training/train_camera_probe.py) waits for the SigLIP export and the detector weights; labels are at data/labels/camera_state.jsonl (37 segments: closeup 16, main 15, graphics 4, replay 2); the first PBA clip is the hold-out and the golden camera truth.
- Task 17: deferred (needs weights): the integration test skips, the detector-size measurement (step 3) and the probe training have not run; all three wait for the Colab weights (Stephen's Roboflow key).
- Task 17: deferred (needs weights): the integration test skips, the detector-size measurement (step 3) and the probe training have not run; all three wait for the Colab weights (Stephen's Roboflow key).

## Measurements

- Scoreboard-only run of `data/golden/pba_ros_cvg_vision.mp4` (180 s, 1080p60, OCR every 15 frames): 1 min 37 s wall on the M1 Pro; the seven score changes it confirmed match the hand-read bug crops within 0.7 s.
- Detector size, per-stage milliseconds and ball recall: not measured yet (needs the Colab weights; `scripts/bench_perceiver.py`).
- Golden scores for `possession_start` and `shot_attempt`: not run yet (needs weights or a perception cache).

## Final review (fresh reviewer, 09c5c68..834fc37) and fix pass

Verdict before the fix pass: not ready, 2 Critical and 7 Important findings, all verified with probe scripts. All nine were fixed test-first in one pass; the suite went from 204 to 220 tests.

- Final: fixed ball filter never re-acquires after a jump or drop — test_ball_filter_reacquires_after_a_jump RED→GREEN
- Final: fixed shots and possessions deferred past the clock gate (phantom field-goal attempts after free throws, replay shots drafted at the restart) — test_shot_during_frozen_clock_is_not_deferred_to_the_restart, test_made_free_throw_is_neither_a_field_goal_nor_double_counted, test_possession_held_through_a_dead_ball_is_not_drafted_at_the_restart, test_possession_taken_just_before_the_restart_is_drafted RED→GREEN (ShotRecord carries clock_running and free_throw at record time; PossessionStartRule drafts only possessions that began within 2 s of the clock restarting)
- Final: fixed phantom defensive rebound after a made basket with a slow delta — test_made_basket_with_a_slow_delta_has_no_confirmed_rebound RED→GREEN (skip when the ball crossed the rim; confirm no earlier than 3 s after the shot)
- Final: fixed turnover confirmed against a team that scored unseen — test_turnover_is_retracted_when_the_losing_team_scores RED→GREEN (retract on a delta for the losing team; confirm no earlier than 3 s after the change)
- Final: fixed made free throw double counted when the inbounder picks the ball up before the delta — covered by test_made_free_throw_is_neither_a_field_goal_nor_double_counted (GameState.last_free_throw_pts; +1 deltas within 3 s of a lane-formation release belong to FreeThrowRule)
- Final: fixed holder flipping to an adjacent defender — test_holder_stays_while_the_ball_is_inside_their_box, test_adjacent_defender_does_not_take_possession RED→GREEN (the holder keeps the ball while it is inside their reach box)
- Final: fixed perception cache key and lookup — test_find_cache_matches_the_stem_and_header_exactly, test_partial_cache_with_an_inner_perceiver_is_rebuilt, test_complete_cache_is_reused_by_an_inner_perceiver, test_cache_only_miss_degrades_to_empty_vision, test_settings_version_changes_with_colours, test_model_set_rejects_a_manifest_whose_sha_does_not_match RED→GREEN (version covers colours, league, template and cadences; exact stem plus header check; completion marker; partial caches rebuilt; cache-only misses give empty vision; manifests verified by sha256 in build_perceiver)
- Final: fixed home/away swapping at a refit without operator colours — test_align_labels_to_previous_centroids RED→GREEN, test_refit_without_colours_keeps_the_mapping as regression
- Final: fixed API jobs never running vision — test_create_builds_a_perceiver_from_the_request_colours RED→GREEN
- Final: Ruling: turnover and rebound confirmations wait out the 3 s scoreboard delta window instead of the spec's 0.5 s hold (the spec's own latency allowance races its own hold; the reviewer verified both phantom confirmations) — cost if wrong: confirmed turnovers and rebounds arrive about 2.5 s later than the spec says.
- Final: Ruling: the synthetic miss goes long off the back of the rim (an image-x offset) instead of beside it in court depth, which the side camera rendered as a rim crossing — cost if wrong: none, test fixture only.
- Final: Ruling: declined-to-judge live-source throughput — M1 targets the offline cache loop; a live budget is M3 (replay rig) — cost if wrong: a slow live run, never a wrong event.
- Final: Ruling: declined-to-judge BallObs.floor_xy for an airborne ball — defined that way by the addendum and read by no rule — cost if wrong: none today.
- Final: Ruling: declined-to-judge the golden truth contents — drafted, with its uncertainties in the meta, for Stephen's pass — cost if wrong: truth lines move.
- Final: Ruling: declined-to-judge tracker IoU gating under fast pans — OC-SORT defaults until real footage says otherwise — cost if wrong: id switches on pans, visible once weights run.
- Final: Ruling: declined-to-judge the shot-clock-reset turnover against this feed's 24 s offensive-rebound reset — the rule needs the holder unknown and possession unchanged, so an offensive rebound with a visible holder does not fire it; revisit on the golden clip — cost if wrong: a false turnover on an unseen offensive rebound.
- Final: Ruling: declined-to-judge shot_missed firing for an air ball — a shot that never reaches the rim is a miss — cost if wrong: none.
- Final: Ruling: declined-to-judge weights, court template and training script correctness — cannot run without the Roboflow data; marked skipped — cost if wrong: found at the first Colab run.
- Final: minor (deferred): possession and shot state never reset at a period start (an overtime tip won by the team with the last regulation possession yields no tip_off_winner)
- Final: minor (deferred): Verdict = str loses type checking; small constructors would keep call sites honest
- Final: minor (deferred): the homography averages over 1 s and lags feet positions during pans (spec wording; revisit with footage)
- Final: minor (deferred): the lob test accepts a ttl retraction as well as same_team_holds
- Final: minor (deferred): chest-pass false positives are only tested on noise-free arcs; add a noisy-pass test
- Final: minor (deferred): download_models.py filters ".." but not absolute zip member names
- Final: minor (deferred): the perception cadence logic is duplicated in watcher._perception_cadence and offline.build_cache
- Final: minor (deferred): court template fractions of length and width do not transfer key or arc geometry between FIBA and NBA courts

## Weights follow-up (2026-11-06, branch feat/m1-weights)

- SigLIP vision tower exported locally (`python -m training.export_siglip`, torch 2.15, transformers 5.22): `models/siglip/vision.onnx`, 372 MB, input `pixel_values` [1, 3, 224, 224], output `pooler_output` [1, 768].
- Camera probe trained on the labels in `data/labels/camera_state.jsonl` at 2 fps: 610 training frames (main 366, closeup 202, graphics 36, replay 6) from `pba_ros_cvg_2700_3100.mp4` and `lal_gsw_060_150.mp4`, hold-out `pba_ros_cvg_2500_2700.mp4`. No detector yet, so the probe uses the embedding alone (feature weights zeroed). Held-out frame accuracy: plain fit 0.867 (main 1.0, closeup 0.908, graphics 0.654, replay 0.0); class-balanced fit 0.885 (main 1.0, closeup 0.923, graphics 0.731, replay 0.0). The balanced probe is `models/camera/probe.json`. Replay has six training frames and is never predicted: more replay footage is needed before the 0.9 camera-state floor can be expected; the clock gate keeps replay shots out of the event stream regardless.
- Ruling: the probe trainer runs without the detector and court models (embedding only, feature weights zeroed in the saved probe) so the camera-state part can proceed before the Colab weights land; retrain with features once `models/detector` exists — cost if wrong: a few points of camera-state accuracy until then.
- Ruling: training scripts run both by path and as modules (`python -m training.<script>`): run by path they could not import the `training` package — cost if wrong: none.
- Still waiting on the Colab weights: the court keypoint template, `tests/integration/test_onnx_perceiver.py`, the detector benchmark, the perception caches and the golden `possession_start` / `shot_attempt` scores.

## Court keypoint template (2026-11-09, branch feat/m1-court-template)

- Both Roboflow sets downloaded with Stephen's key into `data/datasets/` (gitignored): players v20 (1,398 images; train annotations ball 973, player 9,718, referee 2,656, rim 1,004 after the remap) and court v19 (1,460 images, 33 keypoints). The dataset helper now creates only the parent folder and asks for an overwrite: the Roboflow client silently skips a location that already exists.
- The 33 court keypoints are named by number only ("01" to "41" with gaps). Read off the annotated NBA frames (skeleton edges plus four rendered images), the numbering runs left to right: per half the baseline corners, the three-point line's baseline ends, the key's baseline corners, the free-throw line ends and midpoint, the hoop, the three-point arc's junctions and apex, and the NBA sideline hash marks 28 ft from the baseline; the centre line has its sideline ends and the centre.
- Ruling: the template names court features resolved from the league geometry (`sports/basketball/geometry.py::resolve_feature`) instead of fractions of length and width, so FIBA and NBA courts each get their own key, free-throw and three-point positions (the reviewer's deferred minor on fractions is closed by this); the NBA hash marks are kept as sideline marks at 8.53 m, which FIBA courts lack — cost if wrong: two keypoints the model may hallucinate on PBA footage, left to RANSAC.
- Ruling: the court frame's "near" sideline is the broadcast camera's side, as the dataset's frames and our PBA feed both have it; a feed shot from the other side would need the template mirrored — cost if wrong: a flipped homography on such a feed.

## Training on the Mac: trial and decision (2026-11-09)

- One epoch of RF-DETR Nano at 384 on the M1 Pro (MPS, batch 4, grad-accum 4, default loader workers) took 56 minutes including the checkpoint download and validation; after that single epoch the EMA weights scored mAP50 0.58 and mAP50-95 0.33 on the player validation split (precision 0.60, recall 0.56). Its ONNX export (input `input` 1x3x384x384, outputs `dets` 1x300x4 and `labels` 1x300x5) ran through the perceiver on live PBA frames: five player tracks, no rim, the camera probe at 0.61, and a false ball of radius 105 px, which led to the ball-size gate (a ball over 4 % of the frame height is rejected). The trial model stays under `tmp/detector_nano_trial/` and is not in `models/`.
- A 20-epoch run with eight loader workers made the machine unusable (load 18 on 10 cores, 15 of 16 GB in use, 5.5 GB compressed) and was stopped before its first epoch finished. Ruling: training moves to Colab (`training/colab_m1_weights.ipynb`, upload `tmp/training-colab.zip`); the Mac keeps inference only — cost if wrong: a Colab session per model instead of a night of local training.
- `scripts/download_models.py --zip` installs the Colab `models.zip` (the `models/` prefix stripped, `run/` folders skipped). The unit-test settings now point at an empty models directory so real weights never leak into the unit suite.

## Colab weights (2026-11-09, branch feat/m1-colab-weights)

- Small detector, 512, 30 epochs, batch 4 x accum 4 on a T4, 1 min 40 s per epoch: val mAP50:95
  0.632, mAP50 0.93, F1 0.93 (EMA, epoch 28 best). Batch 8 was never tried for the detector;
  the court keypoint model (40.7 M params, trained at 696 px) ran out of memory at batch 8 and
  at batch 4 (14 GB in the keypoint cross-attention) and runs at batch 2 x accum 8 with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 8.5 min per epoch, so it trains for 20
  epochs instead of 40 (OKS mAP50:95 0.83 after one epoch).
- Class-map bug: RF-DETR gives COCO category id k the logit column k - 1 and leaves the last
  column unused. The manifests were keyed by dataset id (1..4), so on PBA frames the detector
  reported ten "balls" (the players), one "player" (the referee), one "referee" (the rim) and no
  ball at all (column 0 dropped as background). `training.manifest.logit_classes` writes the map
  by column; the decode test and the installed manifest follow it. With the fix the integration
  suite passes (15 main-camera frames at 90 s: 6+ tracks, rim, detected ball).
- Providers on the M-series Mac: the RF-DETR graphs split into about a hundred CoreML
  partitions and run twice as slow there (Small 512: 300 ms CPU vs 650 ms CoreML per frame);
  SigLIP is twice as fast on CoreML (67 vs 123 ms). RF-DETR sessions default to the CPU
  provider, SigLIP keeps CoreML. Bench on pba_ros_cvg_2500_2700 from 70 s, 100 frames, every
  2nd: 442 ms per perception frame (detector 316, probe 106, team 11); that segment has no
  visible ball or rim, so the ball count there says nothing.
- Full-frame ball confidences on main-camera frames are 0.25 to 0.5 at 512 px; the 384 px
  ball crop is what carries the ball once acquired.

## First run on real perception (2026-11-11, branch feat/m1-real-perception)

- Court keypoint model, first full run (6 epochs, batch 2 x accum 8, about 460 optimizer steps,
  validation OKS mAP50:95 0.95): not usable. On its own validation images it places visible
  points within about 10 px (640 px images), but slot i and slot i + 17 land on the same spot
  with about 0.5 confidence each, whichever half of the court is in view, and the confidence
  barely separates visible from invisible points (0.5 against 0.1 to 0.5). RF-DETR resizes the
  pretrained 17-keypoint head to 33 by repeating rows, and 460 steps did not pull the copies
  apart; OKS only scores labelled points, so the validation number hides it. A prototype
  decoder that treats coincident slots as one point with two candidate names (the pairs are
  not mirror images, so geometry can pick) fitted 52 of 60 validation images at a median 0.4 m,
  but only 6 of 26 PBA main-camera frames and 18 of 30 frames of the NBA clip. The model is
  kept aside as `models/_court_nba_6ep`; the watcher runs without a homography until a better
  one exists. The prototype lives in `tmp/court_proto.py` and is not in the code base.
- Ruling: the next court run uses batch 2 x accum 2 (four times the steps for the same GPU
  time), 6 epochs. A second run with those settings reached OKS mAP50:95 0.94 after one epoch
  (0.59 with accum 8) before Colab's free quota cut it off — cost if wrong: one more hour of T4.
- Colab: a free account gave about 75 minutes of T4 time per day here. The notebook now clones
  the public repo instead of taking an upload, and `files.download` in the package cell did
  save `models.zip` to `~/Downloads` from the built-in browser pane.
- Camera probe retrained with the detector features (player count, median player height;
  homography inliers zero): held-out accuracy 0.780 (main 1.0, closeup 0.73, graphics 0.73,
  replay 0.25) against 0.885 for the embedding-only probe, so the embedding-only probe stays
  as `models/camera/probe.json` (the other is `probe.detector-features.json`). Golden camera
  accuracy on `pba_ros_cvg_q1`: 0.866, under the 0.9 floor.
- Perception on `pba_ros_cvg_vision` (5,400 frames, about 0.36 s per frame without the court model):
  the detector holds up: ball in 99 % of main-camera frames (detected or predicted), rim in
  93 %, a median of 10 players. Tracks are short: 431 player tracks, median 29 frames.
- Team assignment was the first failure: KMeans on SigLIP crop embeddings labelled 21,722
  player observations home and 6,678 away and agreed with the jersey colour on 51 % of 1,098
  sampled crops (PCA to 2 to 16 dimensions changed nothing). KMeans on the torso's median
  Lab colour gives 14,560 home and 13,840 away and clean contact sheets. Ruling: the perceiver
  clusters `jersey_features` (torso Lab) and no longer needs SigLIP for teams; SigLIP stays
  for the camera probe — cost if wrong: kits of similar colour will need a richer feature.
- Golden scores on `pba_ros_cvg_vision` after the team fix (tolerance 1 s): score_change 7/7;
  possession_start 2 true, 17 false, 4 missed; shot_attempt 1 true, 9 false, 8 missed;
  no rebounds or free throws. Causes seen so far:
  - The holder jumps to a defender when the ball is occluded: the ball filter's predicted
    position, and some detections, sit on the adjacent opponent for 0.2 to 2 s (51 to 53 s),
    long enough for HOLD_S and the possession confirmer.
  - Scrambles after a shot flip the holder several times a second (150 to 168 s).
  - Seven of the nine false shot attempts are the score-derived backfills (confidence 0.4):
    they carry the score-change time, 3 to 4 s after the release in the truth, and the three
    free throws come out as shot attempts because lane formation needs the homography.
  - Truth times for possession_start sit at the inbound or rebound; the rule dates a
    possession from the first confirmed hold, up to 2.7 s away (55.3 against 58.0).
- Next, in order: a court model from the corrected recipe; holder changes only on detected
  ball observations and a longer hold before a possession flips; backfilled attempts dated
  from the rim interaction or the last release instead of the score change; a pass over the
  truth file against the video (it was drafted from contact sheets).
- Holder rule: a predicted ball position (the filter's guess while the ball is hidden) no
  longer establishes a holder; only a detection does. On `pba_ros_cvg_vision` that takes
  possession_start from 2 true / 17 false to 1 true / 9 false (the true one lost is the
  150.9 s Converge start, which no longer drafts; a home start at 151.6 s comes first). A 0.6 s
  team-level hold before possession flips was tried and dropped: it duplicates the 0.5 s
  confirmer, and the false possessions that remain all last longer than that.
- What is left is perception, not rule logic. Seen on frames: a "detected" ball in the crowd
  at 75.5 s (holder runs of 45 frames on the wrong team follow); the dribbler at 153.5 s
  (Converge 3) carrying a home label, his torso window covered by a defender, on a track too
  short for the per-track vote to correct; tracks of about one second throughout. The truth
  file is also in doubt at 72 to 80 s: at 75.5 s the frame shows Rain or Shine running a
  break to the left with the shot clock at 24, where the truth has a Converge rebound and
  basket.
- Ordered next steps for the floors: (1) check the truth file against the video at 39 to 41,
  72 to 80 and 150 to 170 s; (2) reject ball detections outside the playing area or far from
  every player; (3) longer tracks (tracker association through occlusion), so team votes and
  holders stop resetting each second; (4) team feature from the upper torso only, with votes
  weighted by box overlap; (5) backfilled attempts dated from the rim interaction; (6) the
  court model for free throws, zones and the playing-area mask.

## Truth file check and the scoreboard lag (2026-11-11)

- The truth for `pba_ros_cvg_vision` was checked against the video (0.5 s contact sheets and
  0.1 s crops of the rim). All four "miss, offensive rebound, second shot" sequences in the
  first draft were one made shot each: Converge two released 36.4 s (through the net about
  37.8), Rain or Shine layup 53.6 s (rolls on the rim, drops 54.3), Converge two 72.75 s
  (drops 73.9), Converge three 165.0 s (drops about 166.5). The draft had read the shot
  clock's reset to 24 as an offensive rebound; on this feed it resets right after a made
  basket. Sixteen invented events removed, four shot_made added at the release time, and the
  four possession starts after those baskets moved to the first frame the inbounding team
  holds the ball (39.5, 54.8, 75.0, 168.0 s). The draft is kept in
  `tmp/pba_ros_cvg_vision.truth.before.jsonl` (untracked). Free throws and 150 to 160 s are
  still unchecked.
- One Sports' score bug trails the basket by 5 to 6 s: net at 37.8, 54.3, 73.9 and 166.5 s,
  score change read at 43.0, 59.0, 79.5 and 172.5 s (a "+2" badge shows about 1.5 s earlier,
  the digits change last).
- That lag is what breaks the shot rules on this feed. Vision drafts the attempts at 36.58 and
  165.05 s and a shot_made from the rim crossing for each, then the made event dies on the 4 s
  TTL and the open attempt falls out of the 3 s DELTA_WINDOW_S, so the late delta drafts a
  second, retroactive attempt and make at the score-change time (43.5, 173.25 s). The same
  happens at 53.32 s, where the shooter also carries the wrong team. The 72.75 s shot is not
  seen by vision at all.
- Scorer: an event zone of "unknown" (no homography) is treated as a different zone from the
  truth's "two" or "three", so a correctly timed attempt without a zone counts as a miss.
- Scores against the corrected truth, unchanged code: possession_start 2 true / 8 false /
  4 missed; shot_attempt 0 / 10 / 5 (two of the ten are the right shots, rejected on zone).
- Ruling: `score_lag_s` per broadcaster (default 3 s, One Sports 10 s) replaces the fixed 3 s
  delta window: an open attempt waits that long for the score change, a vision-only
  shot_made, a rebound and a turnover live a second longer, and a miss is not called before
  it is out — cost if wrong: on One Sports those confirmations arrive up to 10 s after the
  play. The scorer treats an event zone of "unknown" as no claim.
- Scores on `pba_ros_cvg_vision` after both (corrected truth): shot_attempt 2 true / 6 false /
  3 missed (was 0 / 10 / 5), shot_made 2 / 2 / 2 (was 0 / 4 / 4), shot_missed 0 / 0 / 1,
  possession_start 2 / 8 / 4, score_change 7 / 7. The six false attempts: three free throws
  backfilled as shots (lane formation needs the homography), the 53.32 s layup credited to
  the wrong team and its backfilled twin at 59.75 s, and the backfill at 80.25 s for the
  72.75 s shot vision never drafts.

## Possession on real perception: what the measurements said (2026-11-11, evening)

- The two perception steps planned next were measured before building and both dropped:
  - "Reject ball detections far from every player": every run of detections more than one
    player height from the nearest player on `pba_ros_cvg_vision` (8 runs) is a shot in
    flight. The filter would remove real shots and no false balls.
  - Team vote and torso window: holder-team accuracy against the truth possessions is 0.80
    with the cached labels and 0.69 to 0.81 for every variant tried (per-frame labels,
    votes over 0.5, 1.5, 3 s or the track's life, upper or middle torso window, occluded
    frames dropped, lightness weighted down). Lightness carries the signal; under motion
    blur the Converge blue reads L 145, a 138, b 108, nearer the white centroid
    (185, 132, 118) than the blue one (76, 148, 91) in every weighting. Colour alone will
    not label a track born during a fast break (152 to 155 s).
- Of the 226 detected-ball frames with a wrong-team holder, 99 have a player of the team in
  possession also within reach of the ball (the defender's box is nearer), and 127 have only
  wrong-team players in reach (63 of them the 152 to 155 s fast break).
- Ruling: among the players within reach of the ball, one of the team in possession wins
  over a nearer opponent; the ball changes sides once it is out of reach of every player of
  the team that had it — cost if wrong: a steal in close contact is dated when the stealer
  separates.
- Ruling: after a shot that crossed the rim, the next holder starts the possession of the
  team scored on, whatever label he carries, and that hold does not flip it back — cost if
  wrong: a steal of the inbound pass before anyone on the inbounding team holds the ball is
  credited to the inbounding team first.
- Scores on `pba_ros_cvg_vision` after both (true / false / missed): possession_start
  4 / 3 / 2 (was 2 / 8 / 4), shot_attempt 2 / 5 / 3 (was 2 / 6 / 3), shot_made 2 / 2 / 2,
  turnover 0 / 2 / 0. The 53.6 s layup is now credited to Rain or Shine but dated 52.45 s,
  1.15 s early: the ball is hidden during the drive and the release is dated from the last
  hold before it disappeared.
- What is left for possession_start: 76.67 s against 75.0 (the 72.75 s shot is never seen,
  so the made-basket rule has nothing to act on, and the ball is only predicted from 74.2 to
  76.0 s); a home start at 154.83 s from the blurred fast break and the away start at
  156.03 s that corrects it; the 150.0 s Converge start is not drafted.
- Next for possession: carry a team label across a track that dies and is reborn in place
  (976 dies at 152.50 s, 1006 is born at 152.57 s 23 px away), and announce a possession
  that changed while the clock was stopped when the clock restarts.
- Carrying a team label across a track that dies and is reborn in place was simulated on
  the cache before building (rebirth within 0.3 to 1 s at IoU 0.2 or 0.4, 170 to 223
  inheritances): holder-team accuracy 0.749 to 0.760 against 0.762 without it. In a crowd
  the nearest dead track is often another player. Dropped; no cache rebuild.
- Ruling: a possession the team already had before the clock restarted is announced at the
  restart when the clock stood still (the first tick is one step down from the value at the
  previous change) and a hold of that team began within 2 s of the restart; it is dated
  from the first such hold. A hold carried through the stoppage, or a restart after the
  clock ran on unseen, drafts nothing — cost if wrong: a replay that hides a stopped clock
  for a whole dead ball loses that start.
- The first version compared the tick with the last clock read and drafted a false start at
  68.93 s: OCR read the clock as 20.0 at 61.0 s, the bug was unread from 61 to 68 s, and the
  tick from 313 to 312 looked like a restart. The step is now measured from the clock value
  at the previous change.
- Scores on `pba_ros_cvg_vision` (true / false / missed): possession_start 4 / 4 / 2. The
  Converge start after the free throws is now drafted, at 151.53 s against 150.0 in the
  truth (1.53 s, outside the 1 s tolerance, so it counts as one false and one missed). The
  cache has the dribbler holding from 150.17 s; why the rule's first hold is 151.53 s is
  not traced yet (camera-state hysteresis after the close-up is the suspect), and 150 to
  160 s of the truth is still unchecked against the video.
- Possession errors left: 76.67 s against 75.0; 151.53 s against 150.0; the home start at
  154.83 s from the blurred fast break and the away start at 156.03 s that corrects it.
  Colour cannot fix the last pair; it needs a learned appearance model or a possession
  rule that distrusts a flip with no visible transfer.

## The late holder and the 150 to 160 s truth (2026-11-12)

- Truth check, 148.5 to 160 s against the video: a close-up until about 150.2 s, then the
  main camera with a Converge guard already dribbling in the backcourt (shot clock 24, game
  clock 4:49, first tick to 4:48 at 151.25 s). Converge keeps the ball to past 160 s; there
  is no turnover. The truth's away possession_start at 150.0 s stands (the inbound itself is
  behind the close-up). The watcher's home start at 154.83 s and away start at 156.03 s are
  both false.
- Why the restart hold was dated 151.53 s: the "rising ball is airborne" guard cleared the
  holder candidate on every up-bounce of a dribble, so a dribbler picked up without a
  holder needed five consecutive falling frames to register. Ruling: a rising ball still
  starts no candidate and makes no holder, but a candidate it was already with keeps
  waiting, and the hold is dated from its first frame — cost if wrong: a lob that falls and
  rises again inside a farther player's box can register him sooner than before.
- Two things that change exposed, both fixed:
  - A holder registered before the team assigner has fitted kept the team "unknown" for the
    whole hold (the first possession moved from 22.97 to 23.77 s). The holder now takes his
    track's team when it arrives mid-hold; only unknown is replaced, a label that flips
    mid-hold is not followed.
  - After the 36.6 s basket an early pick-up at 38.70 s took the hand-over and the
    inbounder, labelled away at 39.17 s, flipped it back. Ruling: for 3 s after the
    hand-over a holder's label does not change possession — cost if wrong: a steal within
    3 s of a made basket is announced late or not at all.
- Scores on `pba_ros_cvg_vision` (true / false / missed): possession_start 5 / 3 / 1
  (precision 0.62, recall 0.83; was 4 / 4 / 2), shot_attempt 2 / 5 / 3, shot_made 2 / 2 / 2,
  turnover 0 / 2 / 0. Starts now at 22.73, 38.70, 54.70, 150.67 and 168.43 s against 22.0,
  39.5, 54.8, 150.0 and 168.0 s in the truth.
- Possession errors left: 76.67 s against 75.0 (the ball is only predicted from 74.2 to
  76.0 s and the 72.75 s shot is never seen), and the 154.83 / 156.03 s pair from the
  mislabelled dribbler on the fast break.

## The fast-break blip (2026-11-12)

- The home start at 154.83 s is one hold of 0.37 s by a real Rain or Shine defender (track
  1012) whose box covers the ball while the Converge dribbler, mislabelled home, has it.
  Nobody is seen holding the ball for the next 0.8 s, then Converge again at 156.03 s. The
  `team_holds` confirmer passed it because possession, once flipped, stayed home for 0.5 s
  with no holder at all.
- A minimum hold of 0.5 s in the state before possession changes sides was tried again and
  dropped again: it removes the provisional event that the spec retracts
  (`test_steal_flipping_back_within_half_a_second_is_retracted`) and delays rebounds.
- Ruling: `team_holds` confirms only when a player is seen with the ball in hand 0.5 s or
  more into the possession; possession alone is not enough — cost if wrong: a team that
  gains the ball and shoots within half a second has its possession_start retracted when
  the other team rebounds.
- Ruling: when the ball returns to the team that had it before a possession nobody was seen
  to hold, the old possession goes on and no new possession_start is drafted — cost if
  wrong: a real steal and steal-back inside half a second shows as one possession.
- Scores on `pba_ros_cvg_vision` (true / false / missed): possession_start 5 / 1 / 1,
  precision 0.83 and recall 0.83, over the 0.8 floors for the first time. shot_attempt
  2 / 5 / 3 and shot_made 2 / 2 / 2 unchanged; turnover 0 / 1 / 0 (was 0 / 2 / 0).
- Left for possession: 76.67 s against 75.0. The detector loses the ball from 74.2 to
  76.0 s (the inbounder behind the baseline), and the one detection at 74.03 s is 580 px
  from the ball's last position under the rim. Not a rule problem.
- Loose end: the turnover rule still drafts a provisional turnover at 156.03 s when the
  ball returns after the blip; it expires on the TTL and is retracted.
- `tests/golden` after this change: 1 passed, 2 failed. `pba_ros_cvg_vision` now passes the
  possession_start floor and fails on shot_attempt (2 / 5 / 3); `pba_ros_cvg_q1` camera
  accuracy is still 0.866 against 0.9.

## Shots on real perception (2026-11-12)

- Truth check against the video (0.1 s crops): the three free throws were dated about 3 s
  late in the first draft, read back from the score change. Releases are at 7.3, 115.6 and
  128.9 s (the third is shown from a top-down rim camera). The Rain or Shine jumper drafted
  at 177.0 s is released at 175.6 s, bounces high off the rim at about 176.5 s, and Rain or
  Shine no. 4 takes the rebound at about 177.5 s. Nine truth rows moved; the previous file
  is `tmp/pba_ros_cvg_vision.truth.before2.jsonl` (untracked).
- One Sports' net-to-digits delay holds for free throws too: rim crossings at 8.73 and
  116.93 s, score changes at 14.0 and 122.25 s (5.3 s each); the third, about 130.1 to
  135.25 s.
- Why vision missed shots: the shooter's body hides the ball before the release, so the
  release is dated at the last hold before it vanished (71.68 s for the 72.75 s shot,
  174.82 s for the 175.6 s shot). The path fit only looks at the 1.5 s after the dated
  release and needs the ball rising, so those flights never qualified.
- Ruling: a detected ball that has been out of every player's reach and then enters the rim
  region is a shot for the last release, when that release is at most 3 s before the
  flight was first seen and the flight at most 2.5 s long. It is dated from the first
  detection of the flight; a gap in detections longer than 0.75 s restarts the flight —
  cost if wrong: a pass that crosses the rim region in the image (two rim widths) after a
  flight counts as an attempt until a same-team hold retracts it.
- Ruling: a +1 with no lane formation seen drafts free_throw_attempt and free_throw_made
  (confidence 0.4, note "retroactive") instead of a shot_attempt with zone "ft": one point
  is a free throw by rule — cost if wrong: none known; a scoreboard correction of +1 would
  also come out as a free throw, as it came out as a shot before.
- Ruling: a retroactive event is dated, best guess first, from the start of a flight seen
  to reach the rim inside the scoreboard lag, else from the score change minus the
  broadcaster's `score_delay_s` (new, None by default, 5.3 s for One Sports) minus 1 s of
  flight, else at the score change as before — cost if wrong: on a feed whose delay varies
  the estimate is off by that variation.
- Scores on `pba_ros_cvg_vision` against the corrected truth (true / false / missed):
  free_throw_attempt 3 / 0 / 0 and free_throw_made 3 / 0 / 0 (dated 7.63, 115.87, 128.95 s);
  shot_attempt 4 / 1 / 1 (precision 0.80, recall 0.80; was 2 / 5 / 3); shot_made 3 / 1 / 1;
  possession_start 5 / 1 / 1; score_change 7 / 0 / 0.
- Left for shots: the 53.6 s layup is still dated 52.45 s (one false, one missed for
  attempt and made). A synthetic play that reproduces it was not found, so nothing was
  changed. The 175.6 s miss and its rebound (drafted at 177.37 s, home, offensive) cannot
  confirm: the clip ends at 180 s, before the 10 s scoreboard lag is out.
- Found on the way, not fixed: `shot_path_passes_rim` can pass on a path whose last point
  is a catch far below the earlier points (the quadratic fit flips the sign of the linear
  term). Seen on a synthetic play only.
