[Wed 04 Nov 2026 12:33 PST]
ENTRY 1: DECLARED, written before any new embedding is run
Question: is the adapter's gap share explained by model size alone?
Status: nothing has been run for this test. No ViT-L embeddings
exist yet.

Data (read only, from tnb0/): Gate 3b objects_main, 8 bricks x 400
objects, 96 px windows, 8 positions each. Preprocessing is
common_g3.prep (g, r, z; i dropped; s1 stretch; ImageNet normalise).

Existing embeddings, reused as saved:
 D1 frozen: tnb0/gate3a/emb_d1_main_BRICK_96.npy, one per brick
 AION-1: tnb0/gate3a/emb_aion_main_BRICK_96.npy, version (i)
 Adapter: the arm and step behind the gap shares in
 AGENT_LOG_sherlock.md (around :1391), cited by line at scoring.

New models, frozen, no adapter, embedded with the same code path as
D1 except for model name, revision, width, layer count and register
check:
 MAIN: facebook/dinov3-vitl16-pretrain-lvd1689m (web photos)
 SECOND: facebook/dinov3-vitl16-pretrain-sat493m (aerial photos)
Revision hashes are recorded at download time in a later entry.

Readout (unchanged, ridge): train on centred windows of bricks
3377m332, 0728m047, 0184p052, 0227m280. Score on centred windows of
held-out bricks 2118m002, 1954p237, 3103m010, 3444m405. Mean-patch
pooling (index 0). Targets mag_r and r-z. R2, pooled and per brick.

Gap share: (model - D1) / (AION-1 - D1), same code as
p3_score3b.py :78, :120. A cell is dropped if AION-1 - D1 < 0.05.
95% interval by bootstrap as at p3_score3b.py:69.

Reading rule, per target, for each new model:
 - interval wholly above the adapter's gap-share interval: size
   alone does at least as much as the adapter did.
 - interval wholly below: the adapter adds something size did not.
 - overlap: not distinguishable. Nothing more is read into it
   (line 313 rule of cluster-v-field/AGENT_LOG.md).
The adapter's interval is the one for the same target in
AGENT_LOG_sherlock.md.

Precondition: the copied embedding script, run with D1 on one brick,
must reproduce that brick's saved D1 file with max absolute
difference at most 0.001. If it does not, stop and read no ViT-L
result.

Caveats declared now: one run; two targets; 96 px only; centred
windows only. Every model is DINOv3, so training method as the cause
of lost brightness and colour is NOT tested. Web ViT-L vs D1 changes
size. Satellite vs web ViT-L changes the training pictures. The two
new models are each compared with the adapter; their difference from
each other is not tested.

[Wed 04 Nov 2026 12:36 PST]
ENTRY 2: DECLARED, written before the checkpoints are downloaded
CORRECTION to Entry 1: it cites the bootstrap as "p3_score3b.py:69".
That line is in gate3a/p3_score.py:69. In gate3b/p3_score3b.py the
same calculation is share_rows (:71) and gap (:78). No number or
rule changes. BRICK in the Entry 1 paths stands for the 8 brick
names (4 training, 4 held-out, as listed in Entry 1).

Action declared: download two checkpoints into the shared cache
$SCRATCH/hf_cache (HF_HOME). Nothing is embedded or scored.
 facebook/dinov3-vitl16-pretrain-lvd1689m
 facebook/dinov3-vitl16-pretrain-sat493m
Files: config*.json, *.safetensors, preprocessor_config.json.

Job: sbatch, partition kipac, 1 CPU, 4G, 1 hour, submitted from
adapt-test/, logs in adapt-test/logs/. Heavy I/O goes in a job,
not on the login node. The job checks the connection first and
stops unless the reply is HTTP 200. The HF token is read from
~/.cache/huggingface/token via HF_TOKEN_PATH and is never printed.

The job records each resolved revision hash, snapshot path, file
sizes and sha256 of the weight files. Those become the pinned
revisions for the Entry 1 models; a later entry copies them in.
The existing ViT-B snapshot (models--facebook--dinov3-vitb16-
pretrain-lvd1689m) must be unchanged: its listing is compared
before and after the download.

Precondition for any embedding: both downloads complete, sizes and
hashes recorded. Otherwise stop.

[Wed 04 Nov 2026 13:28 PST]
ENTRY 3: RECORD of the download, and two declarations made before
any embedding is run.

Download: job 51106236, COMPLETED, exit 0, 2 min 17 s, partition
kipac. The job printed VITB UNCHANGED: the ViT-B cache listing was
identical before and after.

Pinned files, under $SCRATCH/hf_cache/hub/ (snapshots/REV):
 MAIN: facebook/dinov3-vitl16-pretrain-lvd1689m
  REV ea8dc2863c51be0a264bab82070e3e8836b02d51
  model.safetensors sha256:
  dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179
 SECOND: facebook/dinov3-vitl16-pretrain-sat493m
  REV f692fa42da72c6797b67cd73494a168d1120d3ee
  model.safetensors sha256:
  4e6356d992c1301b5e7c275f465e47296c5c4ad17052e262b29fc21e82ccc698
Sizes in bytes, both models: model.safetensors 1212559808,
config.json 745, preprocessor_config.json 585. Each blob file name
equals its sha256. These are the revisions Entry 1 refers to. After
a scratch purge, a re-download is accepted only if it reproduces
these hashes.

Shape, from config.json (identical in both): 24 layers, width 1024,
16 heads, 4 register tokens, 3 channels, patch 16, rope_theta
100.0. D1 is 12 layers and width 768. The copied script needs width
768 -> 1024 and the 12-layer assumption at common4c.py:38 changed.

Declaration A (layers): wherever the D1 code uses layer indices
CAT_L = (4, 8, 12) of 12, ViT-L uses the same fractions of depth,
(8, 16, 24). This is confirmed against the code in a look-only read
before the first run. Any other layer rule needs its own entry
before anything runs.

Declaration B (normalisation): the Entry 1 rule is the primary run
for both models: the same front end as D1, with ImageNet mean and
std. The satellite checkpoint's own preprocessor_config.json has
different numbers (mean 0.43, 0.411, 0.296; std 0.213, 0.156,
0.143). So the satellite model is ALSO run once with those numbers
as a secondary run. Both satellite runs are reported with the
Entry 1 reading rule applied to each. Neither is dropped and the
better one is not selected.

[Wed 04 Nov 2026 13:48 PST]
ENTRY 4: DECLARED, written before the reproduction check is run.

CORRECTION to Entry 3: it calls common4c.py:38 "the 12-layer
assumption". That line is CAT_L = (4, 8, 12), which the frozen
path never uses. The frozen embeddings pool last_hidden_state,
after the final LayerNorm (common_g3.py:161, common_g2.py:10).
Declaration A therefore does not apply: no layer indices and no
rescaling for ViT-L. Only the width (768 at p2_embed.py:87) and
the model name needed changing.

Script: adapt-test/embed_vit.py, 172 lines, sha256
 e9758e5906929b26fafbe213dc7adbe31144d33c765f2867e88a8995650897e3
A patched copy of tnb0/gate3a/p2_embed.py (138 lines, sha256
 2a6211b3fed8a345c37d1a7b15fa51d8a50b8c783da27f013d8ae302401b8396)
made by 11 replacements, each matching exactly once. The diff was
read. Nothing in tnb0 was changed.

Edits: repo, revision, normalisation, output folder and brick
list come from VIT_REPO, VIT_REV, VIT_NORM (imagenet or
checkpoint), VIT_OUT and VIT_BRICKS. Width is read from the model
config. Outputs go only to VIT_OUT; the script refuses a folder
inside tnb0. The disk-space check uses VIT_OUT, not $HOME. Only
the main_<brick> 96 px blocks run, and only the frozen path (ENC
d1). Cutting, stretch, i-band drop, normalisation, pooling, the 8
positions and the built-in 20-window bit-identical re-embed check
are the old code, imported unchanged. NORM=checkpoint sets the
mean and std in stretches.py from the checkpoint's own
preprocessor_config.json. normalize() reads those module values
at call time (stretches.py:52-53), so the swap takes effect.

Known leftovers, no effect on results: the docstring still says
ViT-B and 768 and is copied into the run record as "declared";
two recorded free-space numbers (lines 169-170) are for $HOME;
blocks() still reads the desi CSVs, unused.

Environment for every embedding job: partition kipac, 4 CPUs, 4
threads, --mem=31G (kipac allows 8000 MB per CPU, so 32G would
force 5 CPUs), 4 h, HF_HUB_OFFLINE=1, ml python/3.12.1 and
~/.venvs/aion-cpu (torch 2.6.0+cpu, transformers 5.19.0, numpy
2.2.6). The saved D1 and AION-1 files were made on a Mac; its
package versions are not recorded.

Reproduction check (the Entry 1 precondition, made precise): run
embed_vit.py d1 with VIT_REPO facebook/dinov3-vitb16-pretrain-
lvd1689m, VIT_REV 5931719e67bbdb9737e363e781fb0c67687896bc,
VIT_NORM imagenet, VIT_BRICKS 3377m332, into adapt-test/emb_check/.
Compare with tnb0/gate3a/emb_d1_main_3377m332_96.npy and its ri
file (400 rows). PASS needs all of: mean-patch slot [:, :, 0, :]
max absolute difference at most 0.001; ri arrays equal; the
script's own re-embed check True. Slot [:, :, 1, :] is compared
and reported but does not decide, since Gate 3b scored only slot
0. This narrows "that brick's saved D1 file" in Entry 1 to slot 0.
The tolerance is not loosened after seeing the number. On FAIL,
stop; any change needs a new entry first.

After a PASS, in this order: (1) web ViT-L, brick 3377m332 only,
to measure time; (2) the other 7 bricks; (3) satellite ViT-L with
NORM imagenet (primary); (4) satellite ViT-L with NORM checkpoint
(secondary, Declaration B). Revisions are those in Entry 3. Each
run must print its MODEL, WIDTH and NORM lines, and WIDTH must be
1024 for ViT-L. Scoring is a separate step; its script gets its
own entry before it runs.

## 2026-11-04 repro_d1 fix
repro_d1 51115978 FAILED (1m26s): transformers import of DINOv3ViTModel raised
"Cannot access accelerator device when none is available" on CPU node (same bug
as embed3b 10-15). Fix: embed_vit.py:23 imports tnb0/gate3b/accel_shim before
transformers. No other change. Resubmitting repro_d1.sbatch unchanged.
Resubmit 51119624 FAILED (8 s) at the sbatch sha256 guard: the edit changed
embed_vit.py from the Entry 4 hash. Nothing ran. Edit reverted; embed_vit.py
is back to e9758e59...97e3. The shim fix needs a new entry (Entry 4: "any
change needs a new entry first"); pending Ana's choice.

ENTRY 5: DECLARED, written before the resubmit.
Change: repro_d1.sbatch only. "python embed_vit.py d1" is replaced by
PYTHONPATH=tnb0/gate3b and python -c "import accel_shim, runpy ...;
runpy.run_path('embed_vit.py', run_name='__main__')" with argv ['embed_vit.py','d1'].
accel_shim (tnb0/gate3b/accel_shim.py, unchanged, same fix as p2_embed3b.py:23)
is imported before transformers. embed_vit.py is unchanged, sha256 e9758e59...97e3,
still checked by the sbatch guard. Environment, inputs, PASS rule and the
post-PASS order are Entry 4's, unchanged. repro_d1.sbatch sha256 d2ee90e70a1a526b519eabc9aa2f72c7408faaca5d0813bb4d3ffec37331ba8e.
Note: 51122640 was submitted seconds before this entry was committed (git add of the gitignored sbatch failed and submission was not gated on it); the job was still PENDING when the entry was committed as 1ef4752.
51122640 FAILED (8 s) at the "emb_check exists" guard: hash check OK, but
emb_check/ was left by 51115978 (created 13:56:01 by embed_vit.py:40 before the
import crash). It was empty; removed with rmdir. No other change; resubmitting
repro_d1.sbatch as declared in Entry 5.
51124077 (resubmit per Entry 5) COMPLETED 3m05s on sh03-03n56, 2026-11-04 14:55-14:58.
MODEL dinov3-vitb16 5931719e..., WIDTH 768, NORM imagenet. 3200 windows, 133 s
(0.0416 s/window). same_shape True, same_ri True, reembed_ok True, all finite.
Slot 0 max abs diff 2.02e-4 (0 of 2457600 over 0.001); slot 1 max 2.00e-4 (reported only).
RESULT PASS under Entry 4's rule. Outputs in emb_check/. Next per Entry 4 order:
(1) web ViT-L timing run on 3377m332; not started.

ENTRY 6: DECLARED (Ana, 2026-11-05), written before any job exists.
Declaration (step 1 timing run), written before any job exists.
Runs Entry 4 step (1): web ViT-L, brick 3377m332 only, timing.
Interpretation of Declaration A: the D1 path (G.d1_forward,
common_g3.py:161-163) uses no layer indices; it takes
last_hidden_state. ViT-L therefore uses last_hidden_state
(layer 24 of 24), patch-mean after the 4 register tokens.
CAT_L (common4c.py:38) is not used by this path.
Choices not stated in Entry 4 (mine, made before the result):
 partition kipac (same as the ViT-B reproduction, for timing
 comparability); 4 cores, 31G; time limit 4h; NORM imagenet;
 output emb_vitl_web/; log logs/vitl_web_t1_<jobid>.log;
 revision ea8dc286... (Entry 3 main).
Expected: 3200 windows (inferred from ViT-B, not verified).
Measured: seconds per window, plus hostname. Comparable to
ViT-B only if node types are similar.
Done: run prints MODEL, WIDTH 1024, NORM lines; output folder
written; all values finite. Failed: any traceback, WIDTH not
1024, or emb_vitl_web/ already present.
Not claimed: any accuracy result. This step measures time only.
Implementation: vitl_web_t1.sbatch (gitignored), sha256
75540d1f628b193be873f393de5485e389a7755159e3fd89c461b69c116e00fb.
Same as repro_d1.sbatch (Entry 5 shim, embed_vit.py hash guard) with the
Entry 6 model/output; the PASS comparison is replaced by a check that the
output's last dim is 1024 and all values are finite (exit 4 otherwise).
Submitted as 51218471 on 2026-11-05, after commit 6d6ea0f; sbatch hash re-checked at submit.
51218471 COMPLETED 10m35s on sh04-11n16, 2026-11-05 09:52-10:03.
MODEL dinov3-vitl16 ea8dc286..., WIDTH 1024 LAYERS 24, NORM imagenet.
3200 windows (matches expectation), 579 s (0.1809 s/window); re-embed bit-identical True;
output shape (400, 8, 2, 1024), all finite. RESULT DONE under Entry 6.
Ratio vs ViT-B (0.0416 s/window) is 4.35x, but the node differs (sh04 vs sh03 for ViT-B),
so per Entry 6 this ratio is not a like-for-like comparison. Outputs in emb_vitl_web/.
No accuracy result claimed. Next step per Entry 4 order: not started.
ENTRY 7: CORRECTION and RESULT HEADER (Ana, 2026-11-05), appended.
Lines 247-253 are the RESULT of job 51218471 under Entry 6. They
were appended without a header and read as part of Entry 6, which
is a declaration. Read them as the result entry for Entry 6.
Correction: line 247 gives the elapsed time as 10m35s, taken from
date stamps in the job log (09:52:46 to 10:03:21). sacct Elapsed
is 00:10:45. The sacct value is authoritative. Lines 247-253 stand
unedited, per the append-only rule.
Node comparison (sacct and scontrol, read-only, 2026-11-05):
ViT-B job 51124077 ran on sh03-03n56 (CPU_SKU 7502, 32 CPUs,
256000 MB); ViT-L job 51218471 ran on sh04-11n16 (CPU_SKU 9754,
256 CPUs, 1536000 MB). Entry 6 makes the times comparable only if
node types are similar; they are not. The 4.35x ratio is not
claimed as a model-size effect.
ENTRY 8: DECLARED (Ana, 2026-11-05), written before any job exists.
Declaration (step 2): web ViT-L, the other 7 main bricks (96 px):
0728m047, 0184p052, 0227m280, 2118m002, 1954p237, 3103m010,
3444m405. Entry 4 step (2). Brick 3377m332 is the timing run in
emb_vitl_web/ (Entry 6) and is not re-embedded.
Same model, revision, and layer reading as Entry 6 (Declaration A
interpretation, last_hidden_state, NORM imagenet).
Choices not stated in Entry 4 (mine, made before the result):
partition kipac; 4 cores, 31G; time limit 4h; no node pinning
(whatever node the scheduler gives); VIT_BRICKS set to the 7
bricks above; output emb_vitl_web_b7/ (new folder, not
emb_vitl_web/); log logs/vitl_web_b7_<jobid>.log.
Expected: 22400 windows (7 x 3200, from object counts, not yet
verified by a run). Measured: seconds per window and hostname.
Timing is not comparable across node types; Entry 7 records that
the ViT-B and ViT-L timing nodes differ.
Done: run prints MODEL, WIDTH 1024, NORM lines; each of the 7
bricks has its file in emb_vitl_web_b7/; all values finite.
Failed: any traceback, WIDTH not 1024, any brick missing or not
finite, or emb_vitl_web_b7/ already present at submit.
Changed from the Entry 6 sbatch (new, untested code): the
done-check loops over all 7 bricks in emb_vitl_web_b7/ instead of
checking only main_3377m332_96; the output folder and guard name.
Not claimed: any accuracy result. This step produces embeddings
only.
Implementation: vitl_web_b7.sbatch (gitignored), sha256 <to be
filled after the file is saved>.
ENTRY 9: HASH FOR ENTRY 8 (Ana, 2026-11-05), appended. Entry 8
line 293 holds a placeholder for the sbatch sha256; it stands
unedited. vitl_web_b7.sbatch sha256 is
f46c8db1dc1ab590034946ba769f2348adeac11ec39a86c1dd5fb3ac03b5e043,
computed after the file was saved and before submission.
Changes from the Entry 6 sbatch not listed in Entry 8: line 28
(import json, sys became import os, sys, because the loop uses
os.path.exists); the old per-brick report lines (n_windows and
s_per_window read from p2_embed_d1.json) were removed, so seconds
per window now comes from the BLOCK lines embed_vit.py prints.
Declared here before submission.
Pre-submission checks (read-only, on copies in /tmp, login node):
bash -n and py_compile pass; the done-check run on the existing
3377m332 output gave RESULT DONE (exit 0), and on a missing folder
gave RESULT FAILED (exit 4). Not tested: the wrong-width path, the
non-finite path, and the loop over all 7 bricks. Not claimed: any
accuracy result.
ENTRY 10: RESULT FOR ENTRY 8 (vitl_web_b7 51225603), appended
2026-11-05. State COMPLETED, exit 0, elapsed 01:38:32 (sacct),
node sh03-03n51. Log logs/vitl_web_b7_51225603.log.
MODEL facebook/dinov3-vitl16-pretrain-lvd1689m rev ea8dc286...;
WIDTH 1024 LAYERS 24; NORM imagenet. All 7 bricks shape
(400, 8, 2, 1024), WIDTH_OUT 1024, finite True; total_windows
22400 (matches expected 7 x 3200); re-embed bit-identical True for
every brick. RESULT DONE.
s/window per BLOCK line: 0728m047 0.7353 (first brick, 2353 s,
likely warm-up/I/O; not investigated), 0184p052 0.1493, 0227m280
0.1445, 2118m002 0.1460, 1954p237 0.1597, 3103m010 0.1997,
3444m405 0.2879. Node differs from Entry 6 timing; not comparable.
Not claimed: any accuracy result.
ENTRY 11: RECORD AND CORRECTION (Ana, 2026-11-05), appended.
1. Job 51225603 (vitl_web_b7) was submitted by Ana under Entry 8
   (commit fd43597; hash in Entry 9). No submit entry was written at
   submission. Entry 10 is the first log mention of the job id. This
   entry records the gap. It does not alter Entry 8, 9 or 10.
2. Correction to Entry 10, lines 320-321: "likely warm-up/I/O" is a
   guess, not evidence. The cause of the slow first brick (0728m047,
   0.7353 s/window) is not known and was not investigated. Bricks
   3103m010 (0.1997) and 3444m405 (0.2879) were also slower than the
   0.14-0.16 bricks. Cause unknown for all three.
3. Clarification to Entry 10: "re-embed bit-identical True" is the
   per-block check in embed_vit.py:153-170 on a sample of windows per
   block (sample size per the docstring at line 18), not on all 3200
   windows.
4. Logging rule (Ana): look-only probes are not logged. Log state
   changes, declarations before runs, results, corrections, and
   decisions fixed before results are seen.
ENTRY 12: DECLARED (Ana, 2026-11-05), written before any job exists.
Step 3 of the Entry 4 order: satellite ViT-L, NORM imagenet (primary).

Model: facebook/dinov3-vitl16-pretrain-sat493m, revision
f692fa42da72c6797b67cd73494a168d1120d3ee (Entry 3, line 98). The
snapshot is in $SCRATCH/hf_cache with config.json, model.safetensors
and preprocessor_config.json present (look-only check, 2026-11-05).
There is no refs/ folder, so the full hash is required.
HF_HUB_OFFLINE=1 stays as set in embed_vit.py.

Data: all 8 main bricks, 96 px windows, 8 positions each, same
preprocessing and same embed_vit.py as Entries 6 and 8. Bricks, in
order: 0728m047, 0184p052, 0227m280, 2118m002, 1954p237, 3103m010,
3444m405, 3377m332. Which positions get scored is for the scoring
entry, not decided here.

Expected, from counts and not verified by a run: 25600 windows
(8 x 3200), shape (400, 8, 2, 1024) per brick, WIDTH 1024, LAYERS 24.

Run: partition kipac, 4 cores, 31G, -t 08:00:00, no node pinning,
NORM imagenet, VIT_OUT emb_vitl_sat/ (new, must not exist at submit),
log logs/vitl_sat_<id>.log. The time limit comes from the worst step 2
rate (0.7353 s/window x 25600 = about 5.2 h) plus margin. It is a
limit, not an estimate. Step 2 averaged 0.260 s/window (5831 s for
22400 windows), about 1.9 h for 25600 on a similar node. Timing on
other nodes may differ and is not compared across runs.

Implementation: vitl_sat.sbatch (gitignored), copied from
vitl_web_b7.sbatch. Planned differences, and only these: job name,
log path, header comment, -t, VIT_REPO, VIT_REV, VIT_BRICKS (8
bricks), VIT_OUT, the output-folder test, and the done-check (loops
over the 8 bricks, reads emb_vitl_sat/, expects total_windows 25600).
Any other difference is to be disclosed in the hash entry. The sbatch
sha256 goes in its own entry before submission.

Done: the run prints MODEL (sat493m, revision above), WIDTH 1024 and
NORM imagenet lines; each of the 8 bricks has its file in
emb_vitl_sat/; all values finite; the done-check prints RESULT DONE.
Failed: any traceback, WIDTH not 1024, any brick missing or not
finite, re-embed check false, or emb_vitl_sat/ present at submit.
Done-check paths still untested: wrong width and non-finite.

Retry trap: if the run writes emb_vitl_sat/ and then fails or times
out, the folder test blocks a resubmit. Log the failure, move the
folder aside as a separate step, then resubmit under a new entry.

Not claimed: any accuracy result, or any timing comparison between
the web and satellite runs. Step 4 (NORM checkpoint) gets its own
entry after the step 3 result is logged.
[Thu 05 Nov 2026 14:08 PST]
ENTRY 13: HASH FOR ENTRY 12 (Ana, 2026-11-05), appended.
vitl_sat.sbatch, 47 lines, sha256
ed6e22da8db2c8327363482a46a643c85be6b14092737fb130be37387020c802
vitl_web_b7.sbatch is unchanged (sha256 f46c8db1dc1ab590034946ba769f2
348adeac11ec39a86c1dd5fb3ac03b5e043, as in Entry 9).

Differences from vitl_web_b7.sbatch (diff checked 2026-11-05): lines
2, 6-8, 19-21, 23-24, 30-31. All fall within the Entry 12 planned
list: job name, -t 08:00:00, log path, header comment, folder test,
VIT_REPO, VIT_REV, VIT_BRICKS (8 bricks, 3377m332 last), VIT_OUT, and
the done-check folder and brick list. No other line changed.

One gap, and a correction to wording in Entry 12 line 373: the
done-check does not expect or compare total_windows to 25600. It only
prints it (lines 41-43). RESULT DONE depends on width 1024, finite
values and no missing brick. The window count is to be judged by
reading the printed total_windows against the expected 25600. A wrong
count would still print RESULT DONE. The re-embed check in
embed_vit.py (line 161) still stops the job if it fails. Entry 12 is
not edited; this entry corrects it.

Checks run: bash -n passed; py_compile of the done-check block
(extracted to /tmp) passed. Not run: the done-check on data, the loop
over all 8 bricks, the wrong-width path, the non-finite path.

Not claimed: any accuracy result. Submission, with its job id, goes in
the next entry.
[Thu 05 Nov 2026 14:12 PST]
ENTRY 14: SUBMITTED (Ana, 2026-11-05), appended.
Job 51247690 (vitl_sat), submitted under Entry 12, with the sbatch
hash in Entry 13 (sha256 ed6e22da...20c802). Before submit, the sbatch
sha256 was checked against the Entry 13 value and matched in full, and
emb_vitl_sat/ did not exist. Submitted by Claude Code at Ana's
instruction. State at submit: PENDING, partition kipac, no estimated
start yet. Log logs/vitl_sat_51247690.log. Node and start time are
not known at submit and go in the result entry.
Not claimed: any result.
[Thu 05 Nov 2026 14:18 PST]
ENTRY 15: RESULT, step 3 (satellite ViT-L, ImageNet norm), job 51247690
(Ana, 2026-11-06), appended.
Job 51247690 COMPLETED, exit 0:0. Start 2026-11-05 14:19:44, end 15:21:39,
elapsed 01:01:55, node sh03-03n26 (32 CPUs, EPYC 7502).
MODEL facebook/dinov3-vitl16-pretrain-sat493m at f692fa42da72..., NORM
imagenet, WIDTH 1024, LAYERS 24.
8 of 8 bricks, 3200 windows each; done-check printed total_windows 25600,
compared against 25600 by reading the log (the check does not compare).
Re-embed bit-identical True on all 8 (20 random draws per brick, with
replacement). All values finite. No traceback or warning lines.
Per-brick s/window 0.1408-0.1464. Node differs from step 2 (sh03-03n51),
same type (32 CPUs, EPYC 7502, SH3_CBASE); no ratio or cause claim made.
emb_vitl_sat/ holds 33 files: 4 per brick (b0, emb_d1, objpatch_n,
ri_d1) plus p2_embed_d1.json. The done-check reads only the 8 emb_d1
files (vitl_sat.sbatch lines 30, 34). emb_d1 size matches float32
(400, 8, 2, 1024); files not opened.
[Fri 06 Nov 2026 11:25 PST]
ENTRY 16: PLAN, step 4 (satellite ViT-L, checkpoint norm), before submit
(Ana, 2026-11-06), appended.
Purpose: embeddings only, no scoring. Same model, bricks, windows and
code as step 3 (job 51247690, Entry 15). Only change: NORM=checkpoint,
i.e. mean/std read from the checkpoint's preprocessor_config.json
instead of ImageNet values. This is the secondary comparison.
Model facebook/dinov3-vitl16-pretrain-sat493m, revision
f692fa42da72c6797b67cd73494a168d1120d3ee (VIT_REV, full hash).
Expected checkpoint values (config seen, not yet run):
mean [0.430, 0.411, 0.296], std [0.213, 0.156, 0.143].
Script: vitl_sat_ckpt.sbatch (copy of vitl_sat.sbatch). Output folder
emb_vitl_sat_ckpt/ (must not exist before submit). Log
logs/vitl_sat_ckpt_<jobid>.log. Same resources as step 3.
sbatch sha256: cf34de8eb469c8542b30b6d322fa3f0dfb8fef1ab98ca728211b51ae16efaeef
Rules, fixed before the run:
DONE = exit 0; MODEL line shows sat493m at f692fa42...; NORM line shows
checkpoint values above (not ImageNet); WIDTH 1024, LAYERS 24; 8 BLOCK
lines, 3200 windows each, re-embed bit-identical True; printed
total_windows read by Claude against 25600 (check does not compare);
33 files in emb_vitl_sat_ckpt/; no traceback.
FAILED = any traceback; NORM line shows ImageNet values; wrong or
non-finite width; file count not 33; any BLOCK line not True.
Known untested: hf_hub_download at embed_vit.py:53-54 must resolve
offline with the full hash. If it fails, expected early and cheap.
Not claimed: any comparison of step 3 vs step 4 embeddings. That needs
its own plan.
[Fri 06 Nov 2026 11:25 PST]
ENTRY 17: SUBMIT, step 4 (satellite ViT-L, checkpoint norm)
(Ana, 2026-11-06), appended.
Submitted job 51358795 on 2026-11-06 11:30:53 PST, partition kipac.
At submit: PENDING, reason (None), no estimated start.
Script vitl_sat_ckpt.sbatch, sha256 cf34de8eb469c8542b30b6d322fa3f0d
fef1ab98ca728211b51ae16efaeef, matched the hash in Entry 16 (checked
with sha256sum -c before submit). emb_vitl_sat_ckpt/ was absent.
Log expected at logs/vitl_sat_ckpt_51358795.log (not yet created).
Addition to Entry 16: the folder guard on line 19 of the sbatch was
also repointed to emb_vitl_sat_ckpt/. Entry 16 listed the other changes
but not this one. The hash covers the exact file.
Comment on line 8 still reads "Entry 12 ... (Entry 4 step 3)"; it is
a stale comment only, left unchanged to keep the hash.
Nothing from this run has been seen yet.
[Fri 06 Nov 2026 11:32 PST]
ENTRY 18 - RESULT, step 4 (satellite ViT-L, checkpoint norm), job 51358795
Job 51358795 COMPLETED, exit 0:0. Start 2026-11-06T11:32:22, end 2026-11-06T12:32:23,
elapsed 01:00:01, node sh03-03n27.
MODEL facebook/dinov3-vitl16-pretrain-sat493m at f692fa42da72...,
NORM checkpoint, mean [0.430, 0.411, 0.296], std [0.213, 0.156, 0.143],
WIDTH 1024, LAYERS 24.
8 of 8 bricks, 3200 windows each; total_windows 25600 read against
25600 by Claude (the done-check does not compare). Re-embed
bit-identical True on all 8 (20-window sample per brick). Output
arrays finite, shape (400, 8, 2, 1024). 33 files in
emb_vitl_sat_ckpt/. No traceback.
offline hf_hub_download with the full hash resolved from cache.
Per-brick s/window 0.1346-0.1425. No ratio or cause claim.
No comparison of step 3 and step 4 embeddings has been planned or run.
ENTRY 19 - PLAN: score step 3, step 4 and web ViT-L (slot 0)
Why: steps 3 and 4 gave embeddings only. Score them the same way
as the earlier models so build-or-adapt has numbers behind it and
the normalization question gets a paired answer.
Code: adapt-test/p3_score_new.py, copy of gate3b/p3_score3b.py.
Changes: fixed map at top points each new set at its folders;
new sets load slot 0 ([:, :, 0]) and are row-checked against
ri_d1; DESI section removed; NB=13 check only for repro_5000;
repro gate moved before scoring, using fixed reference shares;
gate result written to p3_repro_gate_new.json; interim saves go
to p3_results_new.json.partial, the real name only at the end;
4 docstring lines added. Probe, split, E1/E2/E3, bootstrap
(seed 20261036) unchanged. Output: adapt-test/score_new/.
Nothing in gate3b/ is written.
Feature sets: D1, A1_i, repro_5000, vitl_sat (step 3),
vitl_sat_ckpt (step 4), vitl_web (7+1 bricks, both halves
lvd1689m, NORM imagenet; extra, not part of the 3 vs 4 verdict).
REPRO GATE (runs first): repro_5000, E1 centred, ridge, pooled,
gap share: mag_r 0.5905631, r-z 0.4990124, tol 5e-4. On failure:
exit code 2, only p3_repro_gate_new.json written, report nothing.
New sets also get gap shares and comparisons vs D1 and A1_i
(descriptive; cells dropped if A1 - D1 < 0.05).
STEP 4 vs STEP 3: raw score, paired bootstrap, same objects,
via common4a.setA_rule (checked: THR 0.05, paired 95% CI,
diff = score(step 4) - score(step 3)).
Deciding cells: E1 centred pooled, 5 targets (size: galaxies).
"Step 4 better" on a cell: diff >= +0.05 and 95% CI excludes 0.
Mirror for step 3.
Verdict: step 4 better / step 3 better (cells pass one way only)
/ mixed (both ways) / no difference (no cell passes).
E2, E3, per-position, per-brick signs: reported, exploratory.
Not claimed: official checkpoint preprocessing (script reads only
mean/std); web vs satellite beyond these numbers.
Not judged here: the earlier accuracy bars.
sbatch: adapt-test/score_new.sbatch sha256
  2acec47a20cbd869f7db0c8622cf8a6acef4736ebf2ea96e59b83dc2083020b4
p3_score_new.py sha256
  b1c542809a4b179e4ac1e8060e4a9bc11fffbcbc85e55de21de0a05d32cda6b6
ENTRY 20 - SUBMIT: score_new
Job 51687140, submitted 2026-11-09 11:08:14 PST, partition normal,
3:00:00 limit. Script adapt-test/score_new.sbatch, sha256
2acec47a20cbd869f7db0c8622cf8a6acef4736ebf2ea96e59b83dc2083020b4
(checked OK against Entry 19 before submit). p3_score_new.py sha256
b1c542809a4b179e4ac1e8060e4a9bc11fffbcbc85e55de21de0a05d32cda6b6
(checked OK). Output folder was empty at submit.
Log: adapt-test/logs/score_new_51687140.log.
Plan: Entry 19. No results yet.
ENTRY 21 - CORRECTION to Entry 20: partition
Job 51687140 moved from normal to kipac while PENDING, 2026-11-09
11:17:43 PST (scontrol update partition=kipac). score_new.sbatch
unchanged (still says normal; hash 2acec47a... still valid).
No other job setting changed. No results yet.
ENTRY 22 - RESULT: score_new (job 51687140)
Job completed cleanly. Repro gate passed (mag_r within 1e-6,
r-z within 2e-7 of reference; tol 5e-4). All six feature sets
scored, E1-E3, all-sources and galaxy-only. Output in
adapt-test/score_new/. A .partial file remains beside the final
JSON; harmless.
STEP 4 vs STEP 3 (E1 centred pooled; diff, 95% CI):
 mag_r  +0.037 [+0.020,+0.058]  g-r  +0.029 [+0.004,+0.055]
 star/gal -0.006 [-0.020,+0.009]  r-z +0.016 [-0.012,+0.046]
 size   +0.016 [-0.008,+0.041]
Verdict under Entry 19 rule: NO DIFFERENCE (all five cells
neither; none reaches diff >= 0.05). mag_r and g-r are small
positive edges for step 4 with CIs excluding 0 but below the
threshold; not claimed as step 4 better. Per-brick signs are
over the 4 held-out bricks only.
GAP SHARE (E1 centred pooled ridge; star/gal, size, mag_r, g-r,
r-z):
 repro_5000    0.446 0.224 0.591 0.640 0.499
 step 3        -0.023 -0.062 -0.285 0.136 0.053
 step 4        -0.068 -0.017 -0.138 0.222 0.088
 web ViT-L     0.055 -0.074 -0.436 -0.030 -0.161
No cell of these 20 was dropped (den >= 0.05); none inf/nan.
Raw ridge R2/AUC for D1 mag_r 0.716, step 3 0.643, step 4 0.681,
web 0.605, repro_5000 0.866, A1_i 0.970.
Read: frozen ViT-L features (sat or web) do not beat frozen D1
on these targets; only the adapter-trained repro_5000 closes a
real share of the gap to AION-1. Step 3 was not materially
handicapped by ImageNet normalization.
Notes: a divide-by-zero RuntimeWarning at p3_score_new.py:88
came from a per-resample array; the cell is unidentified; no
stored value is inf/nan (0 of 2400 gap cells). Not claimed:
official checkpoint preprocessing; web vs satellite beyond these
numbers (web is 7+1 bricks, frozen, unadapted); earlier accuracy
bars not judged here.

[Tue 10 Nov 2026 11:30 PST]
ENTRY 23 - DECLARED: backup of purgeable results into this repo
Purpose: GitHub copy of generated results and embeddings that live
on purged $SCRATCH (or only outside git). Requested by Ana in chat.
No computation; no file is modified, re-run or re-dated. Copies are
plain cp (new files in the checkout); originals are left untouched.
Files are force-added (git add -f) by exact name; .gitignore is not
changed. Approx. 790 MB total, no file over 50 MB.
Group 1, results (about 50 MB incl. small npz):
 - tnb0/gate3a, tnb0/gate3b: ignored *.json, *.csv, *.md, *.txt
   (34 + 24 files), plus p3_preds.npz, p3_preds3b.npz (p5 inputs).
 - adapt-test/score_new/: p3_results_new.json, p3_repro_gate_new.json,
   p3_preds_new.npz (the .partial leftover is NOT added).
 - adapt-test/logs/ (10 job logs) and the ignored adapt-test/*.sbatch.
 - cluster-v-field/probe/: copies of $SCRATCH/cluster-v-field/probe/
   results_t7_* and t7full_* (46 files) and of
   $HOME/cluster-v-field/probe/results_round* (26 files).
Group 2, embeddings (about 740 MB):
 - adapt-test/emb_vitl_sat, emb_vitl_sat_ckpt, emb_vitl_web,
   emb_vitl_web_b7, emb_check (all files).
 - tnb0/gate3b/emb_repro_5000_*_96.npy (13 files).
Not included: downloaded inputs (cutouts, survey data, hf_cache), and
the Gate 3a/3b D1/AION/adapter embeddings other than repro_5000.
Status: DECLARED, not run.
