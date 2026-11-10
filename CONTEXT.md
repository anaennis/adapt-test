# CONTEXT: base-model question for tile-native work
Written Wed 4 Nov 2026 for a fresh session. Folder:
/scratch/users/anaennis/aion-tokenizer-diagnostic/<NEWFOLDER>/
(sits OUTSIDE cluster-v-field)

## The big question (Ana's framing)
Not led by one science case. Led by a data question: how to get the
most out of the whole sky rather than cutouts of galaxies.
 1) Build a tile-native model, or adapt an existing one?
 2) If adapt: what do we adapt (and is D1 the right base)?
 3) If build: can we even build? (Needs larger GPU access; parked.)
Current lean: ADAPT. Building only makes sense with bigger GPU access.
"D1" = DINOv3 (Meta general-purpose vision model). "AION-1" = the
astronomy-specific model. D1_native = DINOv3 on native pixels, no resize.

## How to work with Ana (standing rules)
- Plain language first ("what this means"), no unexplained shorthand.
  ONE step per message. Do not tie work to the poster (already printed).
- Never invent numbers. Flag measured vs estimated. If a paste is cut
  off, ask for the CSV path or a re-print.
- Steps are LOOK ONLY unless Ana approves a write. Verdict format:
  CONFIRMED (with evidence) / NOT FOUND / COULD NOT VERIFY.
- Every CC prompt ends: "Every line under 80 characters, at most 25
  lines per block, no box-drawing."
- Python on the login node is 2.7. For python use:
  ml python/3.12.1 and ~/.venvs/aion (from probe/probe.sbatch:10-11).
  There is NO env.sh in cluster-v-field; the old handoff was wrong.
- Git is 1.8.3.1 (no --show-current). git add with explicit names,
  never -A. master has no upstream: verify pushes with
  git ls-remote origin master. Remote: git@github.com:anaennis/
  tokenizer-tests.git. CC commits add a Co-Authored-By line.
- Heavy jobs (downloads, extraction) go in sbatch jobs, not the login
  node. CC cannot move itself to a compute node.
- Pre-registration discipline: write any analysis rule down BEFORE
  seeing results. Ana does not want more log entries in
  cluster-v-field today. For this new folder, ask Ana where rules go
  (suggest a short log file here). Look-only steps need no entry.
- Do not write into cluster-v-field. Read it only.
- Line 313 of the cluster-v-field log: differences smaller than the
  bootstrap interval are not interpreted.

## Where the data and code live (verified by CC today)
- Embeddings (D1_native, tidal galaxies): $SCRATCH/cluster-v-field/
  probe/features_t6/D1_native/shard_000..019.npz (+ .json), 159 MB,
  one row per galaxy x window x copy, with a Name column.
- Cutouts: 1000x1000 px, 4 bands (griz), 0.262"/px, centred on the
  galaxy. Manifest: $SCRATCH/cluster-v-field/cutouts_sga/
  manifest_pilot.csv (diam_arcmin = D26, in arcmin).
- Extraction: probe/extract_sga.py (D1 only). Windows are fixed centre
  crops of 96/192/384/768/960 px, fed at native pixels, no resize.
  Patch size 16; RoPE (so any size works). Ran on CPU only: array
  50170841, 20 tasks, 4 CPUs each, 49-82 min per 100 galaxies.
- Probe: probe/t7_probe.py. Data root is $SCRATCH/cluster-v-field/
  (t7_probe.py:7), not the live tree. Always loads all 2000 galaxies.
- RISK: t5cd.py, missing.py, offset_logit.py (build the catalogue
  baseline features) exist ONLY in $SCRATCH/cluster-v-field/probe/.
  Not in git, not in $HOME. $SCRATCH purges after 90 days without
  changes. A backup (copy + commit, no analysis) was proposed and is
  NOT yet done. Ana to approve.
- Live log: /scratch/users/anaennis/aion-tokenizer-diagnostic/
  cluster-v-field/AGENT_LOG.md (append only). Older "Sherlock log"
  is AGENT_LOG_sherlock.md, a different file; name which one when
  citing lines.

## Results so far (cluster-v-field tidal pilot, 2000 SGA galaxies,
## 264 positives, label frac >= 0.5; S3-S2 AUC gain, 95% CI)
 W96  +0.018 [-0.011,0.050]   W192 +0.087 [0.056,0.118]
 W384 +0.111 [0.080,0.142]    W768 +0.105 [0.076,0.136]
 W960 +0.089 [0.061,0.118]
S2 = catalogue baseline, S3 = D1 embedding. Gain about +0.08 to +0.11
from 192 px up. No ordering among 192-960. Single run, fixed seeds.
Paired ablation (commit 7abe432, pushed): removing colour or masking
the galaxy shrinks the gain in 5 of 6 comparisons but it stays above
zero. W960 colour-removal is not distinguishable from zero. Safe line:
"Take away colour, or the galaxy itself, and the advantage shrinks
but doesn't disappear."

## Earlier findings that matter here
- Adapter on D1 closed about half to two-thirds of the gap to AION-1
  on brightness and colour (r-z, mag_r). NO model met the accuracy
  bars. A fixed 8 px circle photometry baseline beat models on
  brightness in most bins (it is given the catalogue position).
- AION-1 is a centred specialist: 3-4x worse off-centre (model_test/
  L6, post hoc exploratory, DECaLS 436 / DES 825 objects).
- Cluster-richness window effect: about two-thirds reproduced by a
  cheap crowding/colour-count baseline. Rule: rule out cheap baselines
  before crediting a model.

## Settled today
1. AION-1 cannot take more than 96 px without code changes. The codec
   is convolutional, but the transformer uses a FIXED 24x24 position
   table built for 96 px (encoder_embeddings.tok_image.pos_emb
   [1,576,768]); codec CenterCrop(96) is hardcoded. The project's
   cluster pipeline shrinks windows by block-averaging to 96 px first.
   Whether bigger input would error or silently misalign: unknown.
   No AION-1 features exist for the tidal (SGA) galaxies.
2. Coverage test (is the window gain just "more of the galaxy fits"?)
   is PARKED. Small-galaxy subset leaves too few positives (D26<=192
   px: 519 galaxies, 23 positives). Design A (1x and 2x D26 frames on
   1616 galaxies, 169 positives) is fully specified in chat but not
   run; it gives a mechanism caveat, not a build/adapt answer.
   State the confound plainly in talks.
3. Job 50925527 (gccap-pilot-tp) belongs to a different project.

## Current direction: option B, and "is D1 the right base?"
Option B = repair what the model forgot: D1 and AION-1 both discard
brightness/colour. Adapt D1 to keep it, using free training signal
(Legacy Survey catalogue magnitudes, colours, sizes at catalogued
positions). Judge ONLY on held-out sky regions. Cheap competitor:
fixed-aperture photometry. (AstroBridge caveat: gains from labels fed
as training signal show storage, not understanding.)
Other options considered: A = frozen D1, per-patch readers across
tiles (the baseline to beat); C = label-free masked training on random
tiles (most speculative, most compute).

Criteria for any base model: (1) takes real pixels at any frame size;
(2) gives per-patch features (a map); (3) handles four bands; (4) has
seen astronomy; (5) runs on CPU. AION-1 fails (1).
Candidates seen in a SHORT search (not exhaustive, nothing verified):
DINOv3 (incl. a satellite-pretrained release; one remote-sensing study
found no advantage for satellite over web pretraining), AstroPT
(autoregressive on patches; Euclid version), Fama (masked autoencoder,
transfers across surveys). Not checked: weights availability, input
size limits, Legacy Survey compatibility. DINOspec (Lastufka et al.,
arXiv 2608.30503) also uses frozen DINOv3-Large for astronomy images.
Idea: a small bake-off of 2-3 frozen models on the existing
brightness/colour test, same galaxies, same baselines. Rule written
before running.

## NEXT STEP (look-only, not yet run)
Find out what D1 actually is and how four bands become three channels:
 1. Which DINOv3 checkpoint (name, size, params, web vs satellite)?
 2. How griz becomes the 3 input channels (extract_sga.py near :36,
    :68, :86, and the s1c function): stretch, normalise, mapping.
 3. Is D1_shrunk the same checkpoint as D1_native?
Then: check which candidate models have usable weights and accept
native pixels, and propose a bake-off for Ana to approve.

## Open items (lower priority)
- Backup of t5cd.py, missing.py, offset_logit.py (see RISK).
- 2 Nov readout tables were never logged: if cited, say so.
- D16, Gate 3c, and original D17 wording (Sherlock log) still open.
- Poster follow-ups (repo address, push figures/CSVs) not urgent.
