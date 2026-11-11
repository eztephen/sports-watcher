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
