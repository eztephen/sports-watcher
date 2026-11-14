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
- `tests/golden` checked that every confirmed shot_made has a score_change within a fixed
  3 s. On One Sports the score follows by 5 to 6 s, so both clips failed that check once the
  floors passed. The check now uses the broadcaster's `score_lag_s`. Result: both clip
  tests pass (`pba_ros_cvg_vision` with score_change 0.9, possession_start 0.8 and
  shot_attempt 0.8; `pba_ros_cvg_q1`), and only camera accuracy on `pba_ros_cvg_q1` fails
  (0.866 against 0.9). `tests/integration` passed in the run before.

## Court keypoint model, third run (2026-11-12)

- Colab's free quota was back. Run: 6 epochs, batch 2 x accum 2, about 9 to 12 minutes per
  epoch on a T4 (54 minutes with the export). Validation OKS mAP50:95 per epoch: 0.939,
  0.942, 0.944, 0.942, 0.943, 0.943. ONNX sha256 starts 6941c22b458b18a8. Kept aside as
  `models/_court_nba_6ep_accum2`; `models/court` stays absent, so the watcher runs as
  before.
- The duplicated slots are still there: on 26 PBA main-camera frames, 3 to 6 pairs (slot i
  and slot i + 17) sit on the same spot per frame, both at 0.5 to 0.65 confidence, and the
  production `CourtMapper` gets no homography on any of the 26. Four times the optimizer
  steps changed nothing, so the first diagnosis (too few steps to pull the copied head rows
  apart) was wrong.
- Corrected diagnosis: COCO keypoint training only supervises labelled points. A point of
  the half court that is out of view carries v = 0 and gets neither a location nor a
  visibility loss, so its slot keeps whatever its twin (the row it was copied from when
  RF-DETR resized the 17-keypoint head to 33) predicts. In a right-half view the right-half
  point is correct and the left-half twin repeats it; training longer cannot change that.
  What did improve: slots with neither twin in view now read 0.0 to 0.03.
- On a drawn frame (30 s) the right-half names are right where the markings are clear
  (centre, the throw-in mark, the arc's apex, lane corners, the far corner) and wrong where
  players cover them (the corner-three junction is placed 250 px off).
- The pair-aware decoder prototype (`tmp/court_proto.py`: a coincident pair is one point
  with two candidate names, geometry picks) on this model: a fit on 2 of 26 PBA frames at
  an 8 px tolerance and 15 of 26 at 24 px, never more than 7 inliers, and the overlay at
  30 s is visibly skewed (the baseline about 150 px off). Not usable for zones or the lane.
- Ruling: no more Colab runs of this recipe — cost if wrong: none, the result is the same
  twice. The open problem is PBA accuracy, with two parts: the twins (decoder, or training
  with the out-of-view points labelled invisible) and an NBA-trained model on a FIBA court
  with ten players standing on the markings.
- Options, cheapest first: (1) fine-tune on 50 to 100 PBA main-camera frames labelled with
  the same 33 points (the labelling is the cost); (2) a pair-aware decoder plus temporal
  smoothing over the pan, accepted only above a stricter inlier count; (3) a line-based
  court fit for the fixed main camera. Zones and lane formation wait on one of these.

## Court keypoints: decoder and fixed-camera spike (2026-11-12)

Measured on raw keypoints of the third-run model for the whole vision clip, one frame in
six (1800 samples, 1089 on the main camera; `tmp/court_dump.py` writes `tmp/court_kp.npz`).
No production code changed; `models/court` is absent again.

- Pair-aware decode (`tmp/court_spike.py`: coincident slots are one point with several
  candidate names, RANSAC picks; a previous fit within 1 s names the twins first). Fits at
  6 points minimum: 11 % of main frames at 8 px, 33 % at 12 px, 74 % at 24 px. The median
  fit has 6 points and no frame reaches 8. The fitted court is not stable: the projected
  court centre's second difference between samples 0.1 s apart has a median of 450 to 700
  px. The temporal prior changes the fit rate by a few points and does not steady it.
- Fixed camera centre with pan, tilt and focal length free per frame (`tmp/court_ptz.py`,
  3 unknowns instead of 8). The fitter recovers 40 of 40 synthetic views with twins, 3 px
  noise and a third wrong points (`tmp/court_ptz_syn.py`). On the clip, no camera centre
  explains the points: over a 27-position grid the inliers at 15 px stay between 1.4 and
  2.3 per frame out of about 9 point groups, and at the refined centre only 18 % of frames
  reach 4 inliers.
- Why, on one frame (90 s, left half in view, twins resolved by hand to the left names):
  12 named points, and a homography agrees with 5 to 7 of them. The lane's near corners,
  the spot under the hoop and the corner-three junction are 45 to 180 px from where the
  other points put them, with either league's dimensions. The points look plausible to
  the eye, and they are not mutually consistent to better than about a metre.
- Ruling: the twins are a side issue. The NBA-trained model is not accurate enough on the
  PBA floor for any decoder to recover a court, with or without time smoothing — cost if
  wrong: zones and lane formation stay off until a better model exists, which is the
  state today. Remaining options: fine-tune on 50 to 100 labelled PBA main-camera frames
  (the fixed-camera fit above is then a ready, tested way to make 4 good points enough),
  or a line-based fit that does not use this model.

## Court keypoints: what the twin slots are, and the training fix (2026-11-12)

- Read in rfdetr 1.12.0 (`models/heads/keypoints.py`, `models/lwdetr.py`,
  `models/weights.py`): the "findable" loss is a BCE against v > 0 over every keypoint, so
  out-of-view points are supervised. The third-run note's cause (no loss on out-of-view
  points) is wrong.
- The cause: loading the 17-keypoint checkpoint into a 33-keypoint model calls
  `reinitialize_keypoint_head`, which resizes the per-keypoint tables (`keypoint_pos_embed`
  and the query initializers) with `_resize_parameter_rows`, and that tiles rows. Slot
  i + 17 starts as an exact copy of slot i. Raw output on a PBA frame confirms it: twins
  give the same x, y and a findable logit of 0.0 to 0.6 (the BCE optimum for two
  indistinguishable slots with targets 1 and 0 is 0.5), while slots with neither twin in
  view give -4. The watcher's "0.5 to 0.65 confidence" was the sigmoid of those logits.
  The tables sit in the decoder's reduced learning-rate group, and 6 epochs do not pull
  the copies apart.
- Fix in `training/train_court.py`: `install_untie_hook` wraps the resize and redraws
  every row that copies an earlier one (mean and spread of the original rows), at
  construction and again when `train()` reloads the weights. The script refuses to export
  if the hook never fired. Not yet run on Colab — cost if wrong: one 55 minute run.
- `training/datasets.py::merge_keypoint_sets` adds a small labelled set of the target
  broadcast to the NBA set (keypoints reordered by name, images listed `repeat` times).
  The labelling pack for Stephen is `data/labels/court_pba` (52 main-camera frames with
  pre-labels, gitignored).
- Also seen: `KeypointTrainConfig.skip_best_epochs` defaults to 10, so a 6-epoch run never
  selects a best checkpoint by validation; which weights the export used was not checked.

## Court keypoint model, fourth run: the untie fix works (2026-11-13)

- Colab T4, NBA set only, 6 epochs, batch 2 x accum 2, `train_court.py` at e1c7607 (written
  into the Colab checkout, sha256 checked). The hook fired at construction and again when
  `train()` reloaded the weights: 16 rows redrawn in each of `decoder.keypoint_pos_embed`,
  `keypoint_query_initializer.queries` and `keypoint_query_initializer_enc.queries`.
  Validation OKS mAP50:95 per epoch: 0.938, 0.944, 0.946, 0.942, 0.942, 0.944 (the same as
  the tied model, so the NBA score never showed the defect). About 13 minutes per epoch.
  ONNX sha256 starts 444c461819d6097b; kept as `models/_court_nba_6ep_untied`, `models/court`
  absent.
- Twins: gone. On the 26 PBA frames the confident points read 0.97 to 1.00 (was 0.5 to
  0.65), out-of-view slots read 0.00, and coincident pairs number 0 on 25 frames and 1 on
  one. The frame at 90 s carries left-half names only.
- PBA accuracy: unchanged. Production `CourtMapper` gets a homography on 2 of 26 frames
  (6 inliers each). At 90 s, 11 points at confidence 1.0, and 5 to 6 agree with one
  homography: the free-throw line's midpoint sits on a player 114 px off the line through
  its two ends, and the arc's apex, the near lane corner and the sideline mark are 60 to
  700 px off. Confidence no longer separates right from wrong points.
- Ruling: the untie fix stays in the recipe; the PBA labels are still required — cost if
  wrong: none, nothing ships without a fit that passes.
- The labelling pack's pre-labels were rebuilt with this model (single names, 8 to 21
  points per frame); the earlier ones are in `data/labels/court_pba_prelabels_v1.json`.

## The 53.6 s layup and the stray turnover (2026-11-13)

- The layup's date: from 51.9 to 53.3 s the detector reports a still blob 250 px from the
  driver (confidence 0.3 to 0.65) while the real ball is hidden in his hands, so the state
  records a release at 52.45 s. The ball is detected again in his box from 53.33 s, rising
  8 to 20 px a frame, which the rising guard keeps from making a holder. The trajectory fit
  passes at 53.70 s over a path of blob points plus the real rise, and dates the shot at the
  release.
- Fix: `_apply_release_touch` notes when the detected ball is back inside the releaser's
  reach after having been seen out of it (`release_left_reach`, `release_touch_pts`), and a
  shot from the trajectory fit is dated at that touch when there is one. The synthetic drive
  only reaches the rim-entry path (the fit needs the footage's larger rim box), so the fit
  branch is tested by setting the touch on the state one frame before the fit records.
- That moved the shot after its rim entry (53.67 s, the rim area is large on a layup), and
  the attempt's confirmer wanted an entry after the shot time: the attempt was retracted
  when the scorer picked the ball up at 54.9 s while its shot_made confirmed. The confirmer
  now counts the entry from the release the shot was recorded for.
- The turnover drafted at 156.03 s was the ball going back after the 0.37 s blip at
  154.83 s. Ruling: a turnover needs the losing team seen with the ball in hand 0.5 s or
  more into its possession (the same `holding` test as `team_holds`) — cost if wrong: a
  strip within half a second of a catch drafts no turnover for the second team.
- Scores on `pba_ros_cvg_vision` (true / false / missed): shot_attempt 5 / 0 / 0 (was
  4 / 1 / 1), shot_made 4 / 0 / 0 (was 3 / 1 / 1), free throws 3 / 0 / 0, possession_start
  5 / 1 / 1, score_change 7 / 0 / 0; one turnover draft (154.83 s, retracted) instead of two.
  The layup is dated 53.70 s against 53.6. rebound and shot_missed 0 / 0 / 1 as before (the
  clip ends inside the scoreboard lag).
- Suites: 257 unit tests pass, ruff clean, `tests/integration` passes, `tests/golden` 7 of
  8 (camera accuracy on `pba_ros_cvg_q1` 0.866 against 0.9, unchanged).
- Seen, not touched: a confirmed `timeout` at 94.17 s (note camera_off_main) with no
  timeout in the truth, and a retracted one at 125.07 s. Whether a timeout happened at 94 s
  has not been checked against the video.

## The timeout at 94 s (2026-11-13)

- Checked on contact sheets (92 to 152 s, every 4 s): no timeout. A shooting foul stops the
  clock at 4:49, close-ups run to about 114 s, the main camera shows the two free throws
  (116 and 129 s, with a top-down replay between), close-ups again to the inbound at 150 s.
  The truth file is right.
- The rule drafted at 20 s off main with the clock stopped and confirmed at 30 s on the
  clock alone, although the main camera was back at 116 s. Ruling: the confirmer
  (`off_main_and_stopped_until`) also retracts when the camera returns to main before the
  30 s mark — cost if wrong: a real timeout during which the director cuts to a wide shot
  that reads as main is retracted.
- On the clip both timeout drafts (94.17 and 125.07 s) are now retracted with
  `camera_back_on_main`; every other score is unchanged. 258 unit tests, ruff clean,
  integration passes, golden 7 of 8 (camera accuracy on `pba_ros_cvg_q1`, 0.866).

## Second clip, third quarter (2026-11-13)

- New footage from the same stream: `pba_ros_cvg_q3` (01:35:00 to 01:38:30, Q3 7:39 to
  5:31, the teams have changed ends, 13 shots and two free throws) and `pba_ros_cvg_2700`
  (27:00 to 28:00, one whole minute inside a timeout). Truth drafted from contact sheets;
  no rule was tuned on either clip and neither has floors in `checks` yet.
- `pba_ros_cvg_2700`: the watcher emits one timeout at 0.0 s, provisional then confirmed,
  and nothing else.
- `pba_ros_cvg_q3`, first run (true / false / missed, tolerance 1 s): free_throw_attempt
  2 / 0 / 0, free_throw_made 2 / 0 / 0, shot_made 4 / 0 / 0, shot_attempt 5 / 2 / 6,
  score_change 5 / 1 / 1, possession_start 3 / 7 / 6, rebound 0 / 1 / 3, shot_missed
  0 / 0 / 4, timeout 0 / 1 / 0, turnover 0 / 1 / 0. Retraction rate 0.24.
- Every made basket and free throw is found with the right team and within 0.7 s. The
  misses are where it breaks. Checked on frames, the truth stands on each of these:
  - Team of the ball holder. At 41.9 to 47.7 s and 193.1 to 199.6 s Rain or Shine has the
    ball and the holder's track is labelled away (a white shirt in a crowd, on a track id
    a second or two old); at 200.8 s the Converge rebounder is labelled home. That gives
    the two shots credited to the wrong team (46.73, 197.50 s), a confirmed home turnover
    at 41.57 s that never happened, and most of the false possession starts. Overlay of
    the cached labels at 42.0 and 197.0 s: nine players right, the ball handler wrong, and
    referees labelled as players of either team. This is perception (team per track), the
    rules did what the labels said.
  - No holder, no shot. From 13.8 to 17.7 s nobody is the holder, so the three at 15.4 s
    has no release and no attempt; the same after made baskets, where the new possession
    is found 5 to 8 s late (117.0 in truth against 123.83, 124.5 against 129.57, 178.0
    against 186.23).
  - Sponsor banner over the score bug, 51 to 98 s: the clock is unreadable, the clock gate
    is shut, the layup and two putbacks at 52.4 to 54.6 s draft nothing. Expected with the
    gate as specified.
  - Timeout at 58.63 s, confirmed, false: a foul with a player down, close-ups for 32 s
    with the clock stopped at 6:44, so the camera does not return to main before the 30 s
    mark. The second stoppage (144.23 s, 23 s long) is drafted and retracted.
  - score_change 77 to 79: readable at 36.5 s, solid at 37.0 s, emitted at 38.75 s. A +2
    animation covers the digits from 34.5 s and the camera is on a close-up until 37.2 s.
    The other five are 0.25 to 0.75 s after the truth.
- Build note: two cache builders ran against the same file for an hour (a stalled waiter
  loop had started one). `CachedPerceiver._load` keeps the last record per frame and all
  of those came from the builder that finished; the completion marker reports 6301 frames.
- Not done: any fix. Candidates, in the order the clip suggests: a team label that a
  track keeps by vote over its life and across re-identification, referees out of the
  player class, a longer or camera-independent test for timeouts, the late score read.

## Team of the ball holder: two attempts, both backed out (2026-11-14)

- Looked at the four wrong-team moments on `pba_ros_cvg_q3` with the cached boxes drawn
  over the frames. Three causes, none of them a rule:
  - identity swaps in the tracker: track 331 follows a Converge player until 40 s and the
    Rain or Shine ball handler from 40.8 s, track 1370 does the same at 195.6 s. The team
    is a majority vote over the track's life, so the first player's team stays;
  - the holder given to the defender whose box overlaps the ball at the rim (46.6 s);
  - a false ball at a player's foot (193.5 s).
  831 player tracks in 210 s, median life 0.7 s.
- Attempt 1, a crop abstains when its torso window is covered by another box or its
  patches split between the two cluster colours, and new tracks are sampled five times in
  their first second. Labels recomputed over the cached tracks (same boxes and ids, so a
  fair measure): Q3 shot_attempt 5 / 2 / 6 to 6 / 2 / 5, possession_start 3 / 7 / 6 to
  4 / 6 / 5, but a second false turnover confirmed; vision clip one new false
  shot_attempt (87.12 s). The targets (46.73, 197.50, 41.57 s) did not move.
- Attempt 2, on top of it: the tracker refuses to match a detection whose crop reads as
  the other kit once a track's kit leads by two readings. Real rebuild of
  `pba_ros_cvg_vision`: possession_start 5 / 1 / 1 to 5 / 5 / 1, shot_attempt 5 / 0 / 0 to
  5 / 1 / 0, one false turnover confirmed. The new false starts (26.73, 29.2, 68.6,
  69.6 s) are a lone Converge dribbler whose crop reads as the white kit: the gate splits
  his track and the new track carries the misreading, where the old vote buried it.
- Why the crop misreads: the fitted "white" centre is Lab (176, 133, 122), close to the
  floor and to skin, so a torso window with floor in it looks like the white kit. One
  crop agrees with its track's settled team 88 % of the time on the vision clip and 90 %
  on Q3 (2541 and 3416 crops of tracks that live 4 s or more), and narrower torso windows
  score lower (84 to 89 %). A hard decision on one crop is wrong one time in ten.
- Both changes are backed out; the patch is kept in `tmp/kitgate_abstain.patch`. Measuring
  note: re-running the tracker over cached track boxes is not a valid test (it scores as
  badly with the gate off); only a rebuild from the detector is, about 25 s of compute per
  second of clip.
- What this points to: the kit reading needs a better feature than the torso's median
  colour before the tracker can lean on it (pixels near neither kit left out, or a small
  trained classifier), and the swaps themselves need appearance in the association cost
  as a soft term, not a gate. Neither is started.

## Kit read from kit-coloured pixels, kit gate in the tracker (2026-11-14)

- 300 player crops from the two clips labelled by eye (130 white, 118 blue, 34 unclear,
  11 non-players, 7 referees; `tmp/kitset`, not tracked). The median torso colour is
  right on 236 of the 248 clear crops (95.2 %) and votes on every unclear one. Its
  "white" centre is Lab (187, 131, 119), the floor; the shirts are at 230.
- `KitAssigner` (used when the operator gives both colours): a five-colour palette from
  torso pixels, the entries nearest the operator's colours are the kits, a crop votes for
  a kit that holds 70 % of its kit-coloured pixels and abstains otherwise. Held out (fit
  on one clip, scored on the other): 103 of 103 on Q3, 113 of 115 on the vision clip,
  abstaining on about 13 % of clear crops.
- A kit colour that fills the outer columns of the box is the court behind the player
  and is not counted: a Converge dribbler leaning over the white centre logo read as the
  white kit (150 s of the vision clip). With this the reading abstains more (188 of 248
  clear crops voted, 3 wrong).
- The tracker refuses a detection whose crop reads as the other kit once a track's kit
  leads by two readings. With the old reading this regressed the vision clip (previous
  section); with the new one it does not.
- Scores, true / false / missed (baseline is the committed code before this change):

  | | vision shot_attempt | vision possession_start | Q3 shot_attempt | Q3 possession_start |
  |---|---|---|---|---|
  | baseline | 5 / 0 / 0 | 5 / 1 / 1 | 5 / 2 / 6 | 3 / 7 / 6 |
  | new reading alone | 5 / 0 / 0 | 4 / 3 / 2 | 5 / 3 / 6 | 3 / 9 / 6 |
  | + kit gate | 5 / 0 / 0 | 4 / 3 / 2 | 6 / 1 / 5 | 4 / 5 / 5 |
  | + gate, no votes on close-ups | 4 / 2 / 1 | 3 / 4 / 3 | 6 / 1 / 5 | 4 / 5 / 5 |
  | + gate, court-paint check (committed) | 5 / 0 / 0 | 5 / 1 / 1 | 6 / 3 / 5 | 5 / 4 / 4 |

  Makes, free throws and score changes are the same in every row. "No votes on
  close-ups" was tried and dropped: the camera hint lags the cut, tracks go without votes
  at the start of a main-camera stretch.
- Costs of the committed row on Q3: two false shot attempts beside the real three at
  113.4 s (110.92 s away, 113.12 s no team), and a second false turnover confirmed
  (124.25 s, with 192.83 s). The 46.7 s shot is now Rain or Shine's and the false turnover
  at 41.57 s is gone; the 197.5 s shot is still credited to Converge. The palette and the
  thresholds were chosen on crops from these two clips, so Q3 is no longer an untouched
  clip for team labels.
- Suites: 268 unit tests, ruff clean, integration passes, golden 9 of 10 (camera accuracy
  on `pba_ros_cvg_q1`, 0.866 against 0.9, as before). `pba_ros_cvg_2700`: one timeout,
  confirmed.
- Tooling: `tmp/perceive_memo.py` runs `scripts/perceive.py` with every ONNX call
  memoised in `data/cache/onnx_memo/memo.sqlite` (key: model file and the input tensor's
  sha1). A Q3 rebuild drops from 55 minutes to under 3 once recorded; all four golden
  clips are recorded. Re-running the tracker over cached track boxes is not a valid
  shortcut; this is.

## Third clip, fourth quarter, unseen (2026-11-14)

- `pba_ros_cvg_q4`: the stream at 02:07:00 to 02:10:00, Q4 7:31 to 5:50, 105-105 to
  Converge 110-109, then a timeout from the dead ball at 101 s to the end. Truth (31
  events: 8 shots, 4 makes, 4 score changes, 7 possession starts, 3 rebounds, 1 timeout)
  drafted from sheets before the watcher ran on it. After the first run two rebound times
  were corrected on 0.25 s sheets (68.7 s, drafted 69.8; 82.8 s, drafted 83.2). No floors.
- Scores, true / false / missed, the same truth for both; the baseline is the code before
  the kit change (f683560), its cache rebuilt from the same recorded ONNX outputs:

  | | baseline | committed (26af625) |
  |---|---|---|
  | score_change | 4 / 0 / 0 | 4 / 0 / 0 |
  | shot_attempt | 6 / 3 / 2 | 7 / 4 / 1 |
  | shot_made | 4 / 0 / 0 | 3 / 1 / 1 |
  | possession_start | 2 / 5 / 5 | 3 / 4 / 4 |
  | rebound | 1 / 1 / 2 | 1 / 1 / 2 |
  | shot_missed | 0 / 0 / 4 | 0 / 0 / 4 |
  | timeout | 0 / 1 / 1 | 0 / 1 / 1 |
  | turnover | 0 / 1 / 0 | 0 / 1 / 0 |

  On an unseen clip the kit change is about neutral: it finds the blocked layup at
  98.5 s and dates the possession after the 35.1 s basket right, and it loses the make at
  35.1 s (hung on a duplicate attempt at 37.27 s) and adds a team-less duplicate attempt
  at 17.43 s. The Q3 gain did not carry over in size.
- What is wrong on this clip in both runs, checked on frames:
  - Duplicate attempts: a team-less twin of the real shot (2.02 s; 17.43 s with the new
    code), a second attempt on the same miss (68.13 s), an attempt after the dead ball
    (101.63 s).
  - The 92.3 s layup is under a sponsor banner (87 to 92.5 s): clock unreadable, gate
    shut, nothing drafted. Same cause as 51 to 98 s on Q3.
  - Timeout found but dated 123.77 s; the dead ball is at 101.0 s. The replays at 112 to
    126 s read as the main camera and restart the off-main timer.
  - After a made basket the watcher starts the possession when the other team picks the
    ball up (19.5, 50.5 s); the truth dates the inbound pass (21.0, 56.0 s). A convention
    to settle, not a perception error. The start after the first basket is found 6 to 10 s
    late (close-up until 10 s).
  - The Converge rebound at 82.8 s is found at 84.7 s (possession) and 86.37 s (rebound).
  - A false turnover confirmed at 95.5 s on a Rain or Shine fast break.
  - No shot_missed on any clip so far (0 of 9 across Q3 and Q4).
- Build note: the scoring run was started while the cache builder was still writing and
  truncated its file (`CachedPerceiver` rebuilds an incomplete cache). Rebuilt from the
  memo. `tmp/clip_run.py` should refuse to run without the completion marker.

## Shot rules: misses, twins, the make on the wrong attempt (2026-11-14)

- No `shot_missed` was drafted on any PBA clip. Cause, in `ShotOutcomeRule`: the miss
  waits out the scoreboard lag (`max(3, score_lag_s)` after the ball leaves the rim), and
  an open attempt was dropped at `max(10, score_lag_s + 1)` s. With One Sports' lag of
  10 s the attempt was gone at 11 s and the miss was due at 11 s or later. The unit tests
  ran at the default 3 s lag and never met it.
- Changes, each with a synthetic play that failed first:
  - an attempt stays open until its miss deadline has passed;
  - a miss needs the ball to have reached the rim after the release (an attempt that
    never did was retracted, there is no miss to call) and is confirmed at once: it is
    only called after the lag with no score change, and waiting for the next holder
    retracted it whenever the director cut to a close-up (138.6 s on Q3);
  - a rim crossing belongs to the latest open attempt before it, not to every open one;
  - a score change goes to the latest attempt of that team the ball reached the rim
    after (the release read off the ball dropping out of the net took the 35.1 s make on
    Q4 and dated it 37.27 s);
  - a shot by a player whose team is unknown is held back: a 2 or 3 inside the lag gives
    it the scoring team, otherwise it is drafted as unknown when the lag is out if the
    ball reached the rim — cost: such an attempt is emitted up to the lag later, and one
    whose ball never reached the rim is not emitted at all (it used to be drafted and
    retracted).
- Scores, true / false / missed, before (26af625) and after:

  | | Q4 before | Q4 after | Q3 before | Q3 after | vision |
  |---|---|---|---|---|---|
  | shot_attempt | 7 / 4 / 1 | 7 / 2 / 1 | 6 / 3 / 5 | 6 / 2 / 5 | 5 / 0 / 0, same |
  | shot_made | 3 / 1 / 1 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0, same |
  | shot_missed | 0 / 0 / 4 | 3 / 2 / 1 | 0 / 0 / 4 | 1 / 3 / 3 | 0 / 0 / 1, same |

  Every other type is unchanged on all three clips.
- The false misses follow attempts that were already false or wrong: 69.63 s (the second
  attempt at 68.13 s, possibly a real tip) and 103.13 s (the attempt after the dead ball)
  on Q4; 112.42 s (the false attempt at 110.92 s) and 199.05 s (right shot, wrong team)
  on Q3. The Q3 miss at 48.23 s is dated when the ball leaves the rim region, 1.3 s after
  the truth's 46.9 s. The misses still missing are the shots that are not found at all.
- Suites: 274 unit tests, ruff clean, integration passes, golden 10 of 11 (camera
  accuracy on `pba_ros_cvg_q1`, 0.866 against 0.9, as before).

## Possessions: the pick-up convention, one bug, two changes tried and dropped (2026-11-14)

- Convention (Stephen, 2026-11-14): after a made basket or free throw the next possession
  starts when the other team picks the ball up, not at the inbound pass. Nine starts on Q3
  and Q4 re-dated on 0.4 s frames (b28e153); the vision clip's were already at the pick-up.
  Under it, before any code change: possession_start 5 / 1 / 1 (vision), 4 / 5 / 5 (Q3),
  4 / 3 / 3 (Q4).
- Holder trace on the late and missing starts:
  - Q3, 101 s: a free throw drops with no release seen (close-up). `_scored_on` took the
    shot on record, Rain or Shine's from 55 s earlier, as the scorer and gave the ball to
    Converge when a Rain or Shine player picked it up. Possession stayed wrong for 20 s.
    Fixed: a rim crossing counts for a shot released at most 5.5 s before it
    (`RIM_SHOT_RELEASE_GAP_S + RIM_SHOT_FLIGHT_S`). The same bound is now in
    `ShotOutcomeRule._on_crossing`: with attempts kept open for the miss, the Q3 attempt at
    18.3 s collected the basket at 31 s and confirmed a `shot_made` with no score change
    near it (the golden invariant caught it).
  - Q4, 82.8 s: the ball is in Converge hands from the rebound, the nearest track id
    changes four times in two seconds and each change restarts the 0.165 s hold timer;
    credited at 84.7 s.
  - Q4, 4.8 s: every team is `unknown` until the kit palette is fitted at 200 samples,
    about 16 s into a clip; the holder is right from 10 s, credited at 15.97 s.
- Tried and dropped, each measured on real rebuilds:
  - The hold timer carried over between teammates. Q4 possession_start 4 / 3 / 3 to
    5 / 2 / 2, but Q3 4 / 5 / 5 to 4 / 9 / 5 with a third false turnover and the 46.7 s
    shot back on the wrong team, and the vision clip gained an attempt and a miss on a
    fouled drive (87 to 90 s), failing its floors. Holders found sooner only help when
    their team label is right.
  - The palette fitted at 60 samples. As accurate on the labelled crops (0 to 3 wrong in
    about 155 over six random fits) and it gave Q3 6 / 6 / 3, but it moves the 30 s refits
    and the vision clip's 150 s inbound flipped to the wrong team again (4 / 4 / 2).
  The 60-sample caches (version 329103b3f070 and similar) are unused.
- Kept, each with a test that failed first: the two crossing bounds above; releases by one
  team less than 0.5 s apart are one shot (the ball changing track ids on the way up; a
  second shot 0.8 s later is still a putback); an attempt retracted because the shooters
  held the ball again gets no miss and no make from a later shot at that rim.
- Scores after (true / false / missed): the vision clip and Q4 are event for event the
  same as before. Q3: shot_attempt 6 / 2 / 5 to 6 / 1 / 5 and shot_missed 1 / 3 / 3 to
  1 / 2 / 3 (the false attempt at 110.92 s and its miss are gone); possession_start
  4 / 5 / 5 to 4 / 7 / 5 (a home start now confirmed at 38.37 s, the right team 5.8 s
  late, where a draft at 33.5 s used to be retracted; a false away start at 123.8 s).
- Suites: 281 unit tests, ruff clean, integration passes, golden 10 of 11 (camera
  accuracy on `pba_ros_cvg_q1`, as before).
- Possessions did not improve in this round. What is left depends on the team label of
  the ball holder (false flips at 131, 133, 193 s on Q3), on a holder being found at all
  after a basket (116.0, 102.8 s on Q3), and on the cold start.
- Process: `tmp/clip_run.py` was once more started before a cache had its completion
  marker; see the Q4 section.

## The clock gate under a sponsor banner (2026-11-14)

- Four real shots on Q3 (52.4, 53.6, 54.6 s) and Q4 (92.3 s) drafted nothing: the gate
  was shut although the rule assumes a running clock for 5 s after the bug disappears.
- Trace: as the banner slides over the bug the OCR vote repeats the last clock value for
  about 1.5 s with confidence falling 0.8, 0.6, 0.4. The gate asked whether the clock had
  changed within 2 s of now, so it closed while still in state `read` (52.5 s on Q3,
  88.0 s on Q4) and the `assumed` seconds copied the closed gate.
- Fix in `update_clock_gate`: running means the clock had changed within 2 s of the last
  confident read; in `assumed` the same value holds. A clock read as standing still
  still closes the gate (test). The gate now stays open to 58.0 s on Q3 and 94.25 s on Q4.
- Effect, true / false / missed: Q3 shot_attempt 6 / 1 / 5 to 7 / 1 / 4 (the layup, at
  52.73 s against 52.4), shot_missed 1 / 2 / 3 to 1 / 3 / 3 (a miss at 54.23 s for that
  layup, real on the floor but the truth lists no misses for that sequence),
  possession_start 4 / 7 / 5 to 4 / 8 / 5 (a home start at 56.63 s during the foul). The
  two putbacks at 53.6 and 54.6 s and Q4's 92.3 s layup are still not found: the gate is
  open, vision records no release there. The vision clip's confirmed events and Q4 are
  unchanged.
- Suites: 283 unit tests, ruff clean, integration passes, golden 10 of 11 (camera
  accuracy on `pba_ros_cvg_q1`, as before).
- Court labels: four attempts to label `data/labels/court_pba` automatically failed (fit
  from the pre-labels, pan/tilt/zoom search, line-pair hypotheses with two scorings; a
  typical frame shows one long line across the court's width, the method needs two).
  `data/labels/court_pba/click_tool.html` and `tmp/court_clicks_to_coco.py` turn four to
  six clicked points per frame into all keypoints; waiting on the clicks.

## Timeouts: 45 s, dated from the dead ball, one per stoppage (2026-11-14)

- Time off the main camera with the clock stopped, per stoppage on the four clips: fouls
  and free throws 21 s and 25 s (vision clip), 23 s and 32 s (Q3, the 32 s one with a
  player down); timeouts 57 s and running at the clip's end (Q4) and the whole 60 s
  (`pba_ros_cvg_2700`).
- `TIMEOUT_CONFIRM_S` 30 to 45 (TTL 12 to 30 so the draft at 20 s lives that long): the
  Q3 stoppage at 58.63 s is retracted with `camera_back_on_main` — cost: a timeout that
  keeps the camera away for less than 45 s is missed, and a confirmation comes 15 s later.
- The timeout is dated at the clock's last change when that is before the camera left.
  On Q4 the replays at 113 to 123 s read as the main camera, the last absence began at
  123.77 s and that was the date; now 101.0 s, the truth's dead ball — cost: a timeout
  called later in a stoppage that began with a foul is dated at the foul.
- One timeout per stoppage of the clock: once a draft has stood to its 45 s mark, a later
  absence in the same stoppage drafts nothing. A retracted draft does not block the next.
- Scores: timeout 1 / 0 / 0 on Q4 (was 0 / 1 / 1) and on `pba_ros_cvg_2700`; none
  confirmed on Q3 (was 0 / 1 / 0) or the vision clip. No other event changed on any
  clip. The vision clip's two retracted drafts now share the date 91.75 s (one clock
  stoppage).
- Suites: 286 unit tests, ruff clean, integration passes, golden 10 of 11 (camera
  accuracy on `pba_ros_cvg_q1`, as before).

## Shots at the rim with no release: the last holder's shot, tips (2026-11-14)

- Trace of the rim shots still missed:
  - Q4, 92.3 s (layup): holder until 91.93 s, ball in the air from 92.2 s, in the rim
    region at 92.9 s. The hand-off to the air was never a release (the ball passed another
    player's box, or was a predicted position, on that frame), so the release on record
    was 4.75 s old and `_shot_from_rim_entry` refused it (limit 3 s).
  - Q3, 53.6 and 54.6 s (putbacks): the ball comes down into reach and goes back up
    twice with no holder; `rim_interaction.entered_pts` is set once per release.
  - Q3, 15.4 s (three): the same stale release, in the seconds before any team is known.
- Changes in `vision_state`, each with a synthetic play that failed first:
  - a flight into the rim region with no release on record for it is the last holder's
    shot when he held the ball at most 3 s before the flight began; dated at the start of
    the flight, zone unknown — cost if wrong: a loose ball knocked towards the rim within
    3 s of a hold is that team's attempt;
  - a tip: after the ball has reached the rim, has dropped at least 0.3 rim widths below
    it, was in a player's reach within the last second and is rising in the rim region
    again by at least 0.5 rim widths, a new shot by the player it came off; not after a
    crossing — cost if wrong: a defender's deflection up towards the rim is his team's
    attempt. A rebound carried down is none (test).
- Scores, true / false / missed, before (c311b4c) and after:

  | | Q4 before | Q4 after | Q3 before | Q3 after |
  |---|---|---|---|---|
  | shot_attempt | 7 / 2 / 1 | 8 / 2 / 0 | 7 / 1 / 4 | 8 / 2 / 3 |
  | shot_missed | 3 / 2 / 1 | 3 / 2 / 1 | 1 / 3 / 3 | 1 / 4 / 3 |
  | rebound | 1 / 1 / 2 | 1 / 1 / 2 | 0 / 1 / 3 | 1 / 1 / 2 |
  | possession_start | 4 / 3 / 3 | 4 / 3 / 3 | 4 / 8 / 5 | 4 / 7 / 5 |

  The vision clip's confirmed events are unchanged (one more retracted attempt, 88.8 s).
- Found: Q4's layup (92.2 s against 92.3) and Q3's first putback (53.73 s against 53.6).
  The second putback (54.63 s against 54.6) is drafted and retracted by TTL: it draws the
  foul and never reaches the rim. New and false by the truth: an attempt at 15.6 s with no
  team (the three at 15.4 s, taken before the palette is fitted) and a miss at 55.23 s for
  the first putback (real on the floor, the truth lists no misses for that scramble).
- Suites: 289 unit tests, ruff clean, integration passes, golden 10 of 11 (camera
  accuracy on `pba_ros_cvg_q1`, as before).
