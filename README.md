# adapt-test

Is the D1 adapter's gap share explained by model size alone?

D1 is frozen DINOv3 ViT-B. AION-1 is the astronomy model. "Gap share"
is (model - D1) / (AION-1 - D1): how much of the D1-to-AION-1 gap a
model closes. The trained adapter (repro_5000) closes a real share.
This test asks whether a bigger frozen DINOv3 (ViT-L, no adapter)
does the same.

Moved out of aion-tokenizer-diagnostic on 2026-11-10. History before
that is in that repo (adapt-test/ path, up to commit c7b0c28).

## Setup (AGENT_LOG.md Entry 1)
- Data: Gate 3b objects_main from tnb0/, 8 bricks x 400 objects,
  96 px windows. Same preprocessing as D1 (common_g3.prep).
- Models, frozen, no adapter:
  - web: facebook/dinov3-vitl16-pretrain-lvd1689m
  - satellite: facebook/dinov3-vitl16-pretrain-sat493m
- Readout: ridge on centred windows. Train on 3377m332, 0728m047,
  0184p052, 0227m280. Score on held-out 2118m002, 1954p237,
  3103m010, 3444m405.
- A gap-share cell is dropped if AION-1 - D1 < 0.05.

## Result (Entry 22, score_new job 51687140)
Gap share, E1 centred pooled ridge:

    model        star/gal  size   mag_r  g-r    r-z
    repro_5000    0.446   0.224  0.591  0.640  0.499
    step 3 (sat) -0.023  -0.062 -0.285  0.136  0.053
    step 4 (sat) -0.068  -0.017 -0.138  0.222  0.088
    web ViT-L     0.055  -0.074 -0.436 -0.030 -0.161

Frozen ViT-L (web or satellite) does not beat frozen D1 on these
targets. Only the adapter-trained repro_5000 closes a real share of
the gap. Step 4 vs step 3: NO DIFFERENCE under the Entry 19 rule.
Step 3 = ImageNet normalisation, step 4 = checkpoint normalisation.

## Files
- AGENT_LOG.md: declarations and results, append only. Entries
  1-23. This is the authoritative record.
- CONTEXT.md: background on the base-model question (written 4 Nov).
  The text is repeated three times; the copies are identical.
- embed_vit.py: embeds windows with a DINOv3 model.
- p3_score_new.py: ridge scoring and gap shares.
- *.sbatch: one per job (repro_d1, dl_vitl, vitl_web_t1,
  vitl_web_b7, vitl_sat, vitl_sat_ckpt, score_new).
- emb_*/: embeddings (emb_check = reproduction check).
- score_new/: scoring outputs (JSON + preds npz).
- logs/: Slurm job logs.

## Rerunning
The sbatch files (not the .py scripts) use absolute paths under
$SCRATCH/aion-tokenizer-diagnostic/adapt-test/. Update them
before you resubmit. Inputs come from tnb0/ in that repo.
Environment: ml python/3.12.1, $HOME/.venvs/aion-cpu, with
AION_DATA=$SCRATCH/aion-tokenizer-diagnostic/data and
HF_HUB_OFFLINE=1.
