# adapt-test

## The question in one paragraph

We want a model that understands images of the sky. One option is to
take a general-purpose vision model (trained on everyday photos) and
adapt it to astronomy. Earlier work showed that adding a small trained
"adapter" to such a model makes it much better at astronomy tasks.
But a sceptic could ask: was it really the adapter, or would a bigger
general-purpose model have done just as well without one? This folder
tests that.

**Short answer: it was the adapter.** A bigger model with no adapter
did not help, and on some tasks it did worse.

## The models involved

- **D1**: DINOv3 ViT-B, a general-purpose vision model from Meta,
  used frozen (its weights are never changed). This is the baseline.
- **AION-1**: a model built specifically for astronomy. This is the
  target we would like to match.
- **repro_5000**: D1 plus a trained adapter (from earlier work).
- **ViT-L (new here)**: larger versions of DINOv3, also frozen, with
  no adapter. We tried two:
  - *web*: trained on everyday web photos
    (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
  - *satellite*: trained on satellite imagery
    (`facebook/dinov3-vitl16-pretrain-sat493m`)

## How we scored them: "gap share"

For each task we measure how well D1 does and how well AION-1 does.
The difference between them is the "gap". Gap share says how much of
that gap a new model closes:

    gap share = (new model - D1) / (AION-1 - D1)

- 0 means no better than D1.
- 1 means as good as AION-1.
- Negative means worse than D1.

## What we did

1. Took 3,200 galaxy and star images (8 sky patches, called
   "bricks", with 400 objects each; 96 x 96 pixel cutouts).
2. Ran each model over the images to get a numerical summary
   (an "embedding") of each object.
3. Fit a simple linear model (ridge regression) on 4 bricks to
   predict five properties from the embedding, then tested it on the
   other 4 bricks, which it never saw:
   - star vs galaxy
   - size
   - brightness (mag_r)
   - two colours (g-r and r-z)
4. Computed gap share for each model and property.

Tasks where D1 and AION-1 differ by less than 0.05 were to be
skipped, since the gap is too small to divide by. None were skipped.

## Result

Gap share (higher is better; 1.0 would match AION-1):

    model                 star/gal  size   mag_r  g-r    r-z
    D1 + adapter           0.446   0.224  0.591  0.640  0.499
    ViT-L satellite (v3)  -0.023  -0.062 -0.285  0.136  0.053
    ViT-L satellite (v4)  -0.068  -0.017 -0.138  0.222  0.088
    ViT-L web              0.055  -0.074 -0.436 -0.030 -0.161

What this means:

- The adapter closes roughly a quarter to two-thirds of the gap,
  depending on the task.
- The bigger frozen models close essentially none of it. On
  brightness they are clearly worse than the smaller D1.
- So the adapter's gain is not just "a bigger model would do this".

The two satellite rows differ only in how pixel values were scaled
before entering the model ("step 3" used the standard ImageNet
scaling; "step 4" used the scaling that came with the model). We set
a rule in advance (AGENT_LOG Entry 19) for calling a difference real;
under that rule there is **no difference** between them.

Source: AGENT_LOG.md Entry 22, Slurm job 51687140.

## Files

- `AGENT_LOG.md`: the full lab notebook. Every test was written down
  here before it was run, and every result after. Append only. This
  is the authoritative record (Entries 1-23).
- `CONTEXT.md`: background on the wider question (written 4 Nov
  2026, for a new working session).
- `embed_vit.py`: runs a DINOv3 model over the image cutouts.
- `p3_score_new.py`: fits the ridge models and computes gap share.
- `*.sbatch`: one Slurm job script per step (repro_d1, dl_vitl,
  vitl_web_t1, vitl_web_b7, vitl_sat, vitl_sat_ckpt, score_new).
- `emb_*/`: saved embeddings. `emb_check` is a check that we could
  reproduce the original D1 numbers before trying anything new.
- `score_new/`: scoring outputs (JSON plus predictions).
- `logs/`: Slurm job logs.

## Technical details

- Images: the Gate 3b `objects_main` set from `tnb0/` in the
  aion-tokenizer-diagnostic repo, prepared exactly as for D1
  (`common_g3.prep`).
- Training bricks: 3377m332, 0728m047, 0184p052, 0227m280.
  Test bricks: 2118m002, 1954p237, 3103m010, 3444m405.
- Ridge is fit on centred embeddings, pooled over all objects
  (called "E1 centred pooled" in the log).

## History

This folder was moved out of aion-tokenizer-diagnostic on
2026-11-10. Earlier history is in that repo, under the `adapt-test/`
path, up to commit c7b0c28.

## Rerunning

The sbatch files (not the .py scripts) contain absolute paths under
`$SCRATCH/aion-tokenizer-diagnostic/adapt-test/`. Update them to this
folder before resubmitting. Inputs come from `tnb0/` in the
aion-tokenizer-diagnostic repo.

Environment: `ml python/3.12.1`, then the venv `$HOME/.venvs/aion-cpu`,
with `AION_DATA=$SCRATCH/aion-tokenizer-diagnostic/data` and
`HF_HUB_OFFLINE=1`.
