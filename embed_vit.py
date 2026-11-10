"""TNB-0 Gate 3a PART D (NEW script). Usage: p2_embed.py aion | d1 | real_5000 | control_2000.
Blocks (resumable: a block whose output exists with the right shape is skipped):
 main_<brick> 96 px, 8 conditions (objects_main_<brick>.csv), all 8 bricks;
 desi_<brick> 96 px centered only, objects = union of objects_desi_<brick>_
   {strict,relaxed}.csv (Tractor row order), 5 gate3 DESI bricks; pilot
   1501p022 for d1 only;
 d1 only: main_<brick> 256 px (centered, x+48) for the 5 gate3 DESI bricks;
   desi_<brick> 256 px (centered, x+48) for those 5 + pilot.
Encoders: aion = AION-1 base as day3a (L6 top, 2 threads, image tokens only,
 576 encoder tokens, DES-G/R/I/Z, variants (i), (ii)) -> (N, C, 2, 768);
 d1 = frozen DINOv3 ViT-B/16, S1, mean-patch and oracle object-patch pooling
 (day3a) -> (N, C, 2, 768), plus objpatch counts (negative = fallback) and,
 at 96 px, B0 (common3a.b0, pilot DECaLS sigmas) -> (N, C, 24);
 real_5000 / control_2000 = gate2 adapter / shuffled-target control, LoRA
 unmerged, mean-patch final -> (N, C, 768). DINO: 4 threads.
Windows flattened (object, condition), batches of 10 consecutive windows.
Every window is a copy asserted unchanged (prep / tokenize batch snapshot).
Check per block (STOP): 20 windows re-embedded bit-identical (DINO: 20 random
windows in 2 batches; AION: 2 random whole batches recomputed as batches).
Adapter: base hash == checkpoint's. Disk floor 35 GiB before every save.
Output emb_<enc>_<block>_<size>.npy, p2_embed_<enc>.json."""
import csv, json, os, sys, time, zlib
ENC = sys.argv[1]; assert ENC in ("aion", "d1", "real_5000", "control_2000")
os.environ["HF_HUB_OFFLINE"] = "1"
if ENC != "aion":
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "4"
sys.path.insert(0, os.path.expanduser("~/aion-tokenizer-diagnostic/tnb0/gate3a"))
import common_g3 as G
from common_g3 import C3, CN, COND, C256, out
import numpy as np
# --- adapt-test edits: model, normalisation, outputs from env ---
OUT = os.environ["VIT_OUT"]; REPO = os.environ["VIT_REPO"]
REV = os.environ["VIT_REV"]; NORM = os.environ["VIT_NORM"]
BRICKS = [x for x in os.environ.get("VIT_BRICKS", "").split(",")
          if x]
assert ENC == "d1" and NORM in ("imagenet", "checkpoint")
TNB0 = os.path.realpath(os.path.expanduser(
    "~/aion-tokenizer-diagnostic/tnb0"))
assert not os.path.realpath(OUT).startswith(TNB0), "OUT in tnb0"
os.makedirs(OUT, exist_ok=True)


def load_vit():
    import torch
    from transformers import DINOv3ViTModel
    m = DINOv3ViTModel.from_pretrained(
        REPO, revision=REV, dtype=torch.float32).eval()
    assert m.embeddings.register_tokens.shape[1] == G.C4C.REG
    assert str(next(m.parameters()).device) == "cpu"
    ST = G.C4C.ST
    if NORM == "checkpoint":
        from huggingface_hub import hf_hub_download
        pc = json.load(open(hf_hub_download(
            REPO, "preprocessor_config.json", revision=REV)))
        ST.MEAN = np.array(pc["image_mean"], np.float32)
        ST.STD = np.array(pc["image_std"], np.float32)
    out(f"MODEL {REPO} {REV}")
    out(f"WIDTH {m.config.hidden_size} "
        f"LAYERS {m.config.num_hidden_layers}")
    out(f"NORM {NORM} mean {ST.MEAN.tolist()} "
        f"std {ST.STD.tolist()}")
    return m
if ENC == "aion":
    NS6 = C3.load_top(f"{C3.H}/model_test/L6/L6_build_embed.py", "t_first"); tokenize, embed = NS6["tokenize"], NS6["embed"]
else:
    import torch; torch.set_num_threads(4)
    if ENC == "d1": model = load_vit()
    else:
        model, ck = G.dino_adapted(ENC)
        if G.G2.base_hash(model) != ck["base_hash"]: out("STOP: base hash differs from checkpoint"); sys.exit(2)
WIDTH = model.config.hidden_size; RJ = f"{OUT}/p2_embed_{ENC}.json"; rep = json.load(open(RJ)) if os.path.exists(RJ) else {"declared": __doc__, "blocks": {}}
def save(): json.dump(rep, open(RJ, "w"), indent=1, default=float)
def stop(m): out(f"STOP: {m}"); save(); sys.exit(2)
def rows_of(p): return list(csv.DictReader(open(p)))


def blocks():
    L = []
    for b in G.TRAIN + G.HELD:
        L.append((f"main_{b}", b, 96, CN, [int(r["ri"]) for r in rows_of(f"{G.O}/objects_main_{b}.csv")]))
    for b in G.DESI_TRAIN + G.DESI_HELD + [G.PILOT]:
        if b == G.PILOT and ENC != "d1": continue
        ri = sorted({int(r["ri"]) for v in ("strict", "relaxed") for r in rows_of(f"{G.O}/objects_desi_{b}_{v}.csv")})
        L.append((f"desi_{b}", b, 96, ["centered"], ri))
    if ENC == "d1":
        for b in G.DESI_TRAIN + G.DESI_HELD:
            L.append((f"main_{b}", b, 256, list(C256), [int(r["ri"]) for r in rows_of(f"{G.O}/objects_main_{b}.csv")]))
        for b in G.DESI_TRAIN + G.DESI_HELD + [G.PILOT]:
            ri = sorted({int(r["ri"]) for v in ("strict", "relaxed") for r in rows_of(f"{G.O}/objects_desi_{b}_{v}.csv")})
            L.append((f"desi_{b}", b, 256, list(C256), ri))
    return L


BR = {}
def brick(b):
    if b not in BR: BR.clear(); BR[b] = G.Brick(b)
    return BR[b]


def window(B, i, size, c):
    dy, dx = (COND if size == 96 else C256)[c]; h = size // 2
    W = B.win(int(B.cy[i]) - h + dy, int(B.cx[i]) - h + dx, size)
    if W is None or not np.isfinite(W).all(): raise RuntimeError(f"invalid window {B.b} {i} {size} {c}")
    return W


def run_batch(Ws, masks):
    if ENC == "aion":
        x = np.stack(Ws).astype(np.float32); snap = x.copy(); e1, e2, tf = embed(tokenize(x))
        assert np.array_equal(x, snap), "AION batch mutated"; return np.stack([e1, e2], 1)
    X = np.stack([G.prep(w) for w in Ws])
    if ENC == "d1": m, o = G.d1_forward(model, X, np.stack(masks)); return np.stack([m, o], 1)
    return G.adapter_forward(model, X)


TOTAL_T, TOTAL_W = 0.0, 0
for tag, b, size, conds, ri in [
        x for x in blocks() if x[0].startswith("main_")
        and x[2] == 96 and (not BRICKS or x[1] in BRICKS)]:
    name = f"{tag}_{size}"; path = f"{OUT}/emb_{ENC}_{name}.npy"; N, C = len(ri), len(conds)
    shp = (N, C, 2, WIDTH) if ENC in ("aion", "d1") else (N, C, 768)
    if os.path.exists(path) and np.load(path, mmap_mode="r").shape == shp and name in rep["blocks"]:
        out(f"skip {name} (exists)"); continue
    B = brick(b); ell = B.ellipse() if ENC == "d1" else None
    flat = [(n, ci) for n in range(N) for ci in range(C)]
    def masks_for(pairs):
        if ENC != "d1": return None
        h = size // 2; L = []
        for n, ci in pairs:
            dy, dx = (COND if size == 96 else C256)[conds[ci]]; L.append(ell.patch_mask(ri[n], int(B.cy[ri[n]]) - h + dy, int(B.cx[ri[n]]) - h + dx, size)[0])
        return L
    E = np.empty(shp, np.float32); NP_ = np.empty((N, C), np.int16) if ENC == "d1" else None
    B0 = np.empty((N, C, 24)) if (ENC == "d1" and size == 96) else None; nnf = 0
    if ENC == "d1":
        h = size // 2
        for n, ci in flat:
            dy, dx = (COND if size == 96 else C256)[conds[ci]]; m, fb = ell.patch_mask(ri[n], int(B.cy[ri[n]]) - h + dy, int(B.cx[ri[n]]) - h + dx, size)
            NP_[n, ci] = -m.sum() if fb else m.sum()
    t0 = time.time()
    for k in range(0, len(flat), 10):
        pairs = flat[k:k + 10]; Ws = [window(B, ri[n], size, conds[ci]) for n, ci in pairs]
        if B0 is not None:
            for (n, ci), w in zip(pairs, Ws): B0[n, ci], _, nf = C3.b0(w, G.SIG4); nnf += nf
        R = run_batch(Ws, masks_for(pairs))
        for j, (n, ci) in enumerate(pairs): E[n, ci] = R[j]
        if (k // 10) % 40 == 0: out(f"  {ENC} {name}: {k + len(pairs)}/{len(flat)} windows, {time.time() - t0:.0f} s")
    dt = time.time() - t0
    rng = np.random.default_rng(G.SEED + zlib.crc32(f"{ENC}_{name}".encode()) % 1000)
    if ENC == "aion":
        nb = (len(flat) + 9) // 10; bs = sorted(int(x) for x in rng.choice(nb, min(2, nb), replace=False)); ok = True
        for bi in bs:
            pairs = flat[10 * bi:10 * bi + 10]; R = run_batch([window(B, ri[n], size, conds[ci]) for n, ci in pairs], None)
            ok &= all(np.array_equal(R[j], E[n, ci]) for j, (n, ci) in enumerate(pairs))
        chk = {"reembed_batches": bs}
    else:
        sel = [flat[int(x)] for x in rng.integers(0, len(flat), 20)]; ok = True
        for i in (0, 10):
            pairs = sel[i:i + 10]; R = run_batch([window(B, ri[n], size, conds[ci]) for n, ci in pairs], masks_for(pairs))
            ok &= all(np.array_equal(R[j], E[n, ci]) for j, (n, ci) in enumerate(pairs))
        chk = {}
    if not ok: stop(f"{ENC} {name}: re-embedded windows not bit-identical")
    nbytes = E.nbytes + (B0.nbytes if B0 is not None else 0)
    if __import__("shutil").disk_usage(OUT).free - nbytes < G.FLOOR: stop("disk floor")
    np.save(path, E); np.save(f"{OUT}/ri_{ENC}_{name}.npy", np.array(ri, np.int64))
    if NP_ is not None: np.save(f"{OUT}/objpatch_n_{name}.npy", NP_)
    if B0 is not None: np.save(f"{OUT}/b0_{name}.npy", B0)
    TOTAL_T += dt; TOTAL_W += len(flat)
    rep["blocks"][name] = {"n_objects": N, "conditions": conds, "n_windows": len(flat), "seconds": dt, "s_per_window": dt / max(len(flat), 1),
                           "reembed_bit_identical": bool(ok), **chk, "b0_nonfinite_features": nnf if B0 is not None else None, "free_GiB": G.free() / 2**30}
    save(); out(f"BLOCK {ENC} {name}: {len(flat)} windows, {dt:.0f} s ({dt / max(len(flat), 1):.4f} s/window); re-embed bit-identical {ok}; free {G.free() / 2**30:.1f} GiB")
rep["this_run_total_seconds"] = TOTAL_T; rep["this_run_total_windows"] = TOTAL_W; save()
out(f"P2 {ENC} DONE ({TOTAL_W} windows this run, {TOTAL_T / 60:.1f} min)")
