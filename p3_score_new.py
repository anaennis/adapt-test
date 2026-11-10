"""adapt-test p3_score_new.py: copy of gate3b/p3_score3b.py (Entry 19). Changes only:
new sets vitl_sat / vitl_sat_ckpt / vitl_web read from adapt-test folders (NEW map),
slot 0; repro gate first vs fixed refs; DESI skipped; step 4 - step 3 paired diffs;
output to adapt-test/score_new/. Original docstring follows.
TNB-0 Gate 3b scoring (NEW script; numpy; reads saved embeddings only; gate3a
code imported unedited). Design declared in AGENT_LOG_sherlock.md 2026-10-14
("Gate 3b embedding and scoring design"). Usage: p3_score3b.py
Feature sets: D1 (frozen DINOv3 mean-patch) and A1_i (AION-1) from the gate3a
embeddings; repro_5000 (gate2 adapter re-embedded on Sherlock); <arm>_<step>
for every 3b checkpoint embedded so far (p2_embed3b.py). Missing arms are
skipped, so this runs with repro alone.
Readouts exactly as gate3a p3_score.py: L6 Model ridge (alpha by inner
leave-one-group-out CV) + kNN k=10 secondary, fitted once per (feature set,
experiment, row set) on the four TRAIN bricks (groups = brick x L5 fold),
tested on the four held-out bricks, per brick and pooled; E1 / E2 / E3;
row sets 'all' and 'gal'; bootstrap common_g3.boot_strat (2,000, seed
20261036); gap share and drop rule as gate3a gap(); comparisons vs D1 and vs
A1_i by common4a.setA_rule.
STOP (exit 2), run first: repro_5000 pooled E1 centred ridge gap share for
mag_r and r-z must be within |diff| <= 0.0005 of the gate3a G_adapter value
(declared 2026-10-15; replaces the round-to-3-decimals test).
Decision quantities (final checkpoints, written as numbers; no verdict here):
 HYBRID 1 / A4 ceiling: B3 E1 centred share, mag_r and r-z, pooled and per
     held-out brick (D17 counts bricks: >= 0.5 / <= 0.7 in >= 3 of 4).
 A6: B3 vs D1, E1 ridge, pooled, 14 cells (mag_r, r-z x 7 off-centre
     conditions); worse = setA_rule verdict b_better (diff <= -0.05, CI
     excludes 0); holds if worse in <= 2.
 A7: share difference larger minus smaller arm (B3 - B1 steps; B2 - A1 data),
     same resamples; falls if the 95% CI upper bound < 0 on mag_r or r-z;
     holds if neither check falls.
Exploratory (no decision): 25% and 50% checkpoints (share vs step); DESI
redshift (strict labels if the gate3a A2 gate was met, as gate3a).
Output gate3b/p3_results3b.json, p3_preds3b.npz (suffix $P3_OUT_TAG if set)."""
import csv, glob, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/aion-tokenizer-diagnostic/tnb0/gate3a"))
import common_g3 as G
from common_g3 import C3B, C4, CN, out
Model = C3B.Model; score_rows, pct, t5_rows = C3B.score_rows, C3B.pct, C3B.t5_rows
O = os.path.expanduser("~/aion-tokenizer-diagnostic/tnb0/gate3b")  # read only (repro_5000)
A = os.path.dirname(os.path.abspath(__file__)); OUT = f"{A}/score_new"
# new feature sets: brick -> embedding folder (adapt-test); slot 0 (mean-patch) only
NEW = {"vitl_sat": {b: f"{A}/emb_vitl_sat" for b in G.TRAIN + G.HELD},
       "vitl_sat_ckpt": {b: f"{A}/emb_vitl_sat_ckpt" for b in G.TRAIN + G.HELD},
       "vitl_web": {b: f"{A}/emb_vitl_web" if b == "3377m332" else f"{A}/emb_vitl_web_b7" for b in G.TRAIN + G.HELD}}
REPRO_REF = {"mag_r": 0.5905631, "r-z": 0.4990124}
EXP = {"E1": ["centered"], "E2": ["centered", "x+12", "x+24", "x+48"], "E3": list(CN)}
RS = {"all": ["mag_r", "star_vs_galaxy", "g-r", "r-z"], "gal": ["log10_size"]}
PRIM = ["mag_r", "r-z"]; OFF = [c for c in CN if c != "centered"]
STEPS = {"A1": 20000, "B1": 5000, "B2": 20000, "B3": 80000}
res = {"declared": __doc__, "feature_sets": [], "main": {}, "gap": {}, "cmp": {}, "repro": {}, "decision": {}, "exploratory": {}, "redshift": {}, "n": {}}; PRED = {}
TAG = os.environ.get("P3_OUT_TAG", "")  # optional output suffix (interim runs); unset = original names
def save(final=False):  # interim saves -> .partial (overwritten); final=True only at the very end
    json.dump(res, open(f"{OUT}/p3_results_new{TAG}.json" + ("" if final else ".partial"), "w"), indent=1, default=float)
t0 = time.time(); el = lambda: f"{(time.time() - t0) / 60:.1f} min"
rows_of = lambda p: list(csv.DictReader(open(p)))

# 3b encoders present on disk (all blocks embedded)
NB = len(G.TRAIN + G.HELD) + len(G.DESI_TRAIN + G.DESI_HELD)
ENCS = ["repro_5000"] if len(glob.glob(f"{O}/emb_repro_5000_*_96.npy")) == NB else []
ENCS += list(NEW)  # no file-count check for the new sets; a missing file fails at load
FS = ["D1", "A1_i"] + ENCS; res["feature_sets"] = FS
if "repro_5000" not in ENCS: out("STOP: repro_5000 embeddings missing; run p2_embed3b.py repro first"); sys.exit(2)
CMP = [(x, y) for x in ENCS for y in ("D1", "A1_i")]
out(f"feature sets: {FS}")


def load_feats(tag, b):
    nm = f"{tag}_{b}_96"; ri = np.load(f"{G.O}/ri_d1_{nm}.npy"); F = {}
    F["D1"] = np.load(f"{G.O}/emb_d1_{nm}.npy")[:, :, 0].astype(np.float64)
    assert np.array_equal(np.load(f"{G.O}/ri_aion_{nm}.npy"), ri); F["A1_i"] = np.load(f"{G.O}/emb_aion_{nm}.npy")[:, :, 0].astype(np.float64)
    for e in ENCS:
        if e in NEW:
            d = NEW[e][b]; assert np.array_equal(np.load(f"{d}/ri_d1_{nm}.npy"), ri), (e, nm)
            F[e] = np.load(f"{d}/emb_d1_{nm}.npy")[:, :, 0].astype(np.float64); continue
        assert np.array_equal(np.load(f"{O}/ri_{e}_{nm}.npy"), ri), (e, nm); F[e] = np.load(f"{O}/emb_{e}_{nm}.npy").astype(np.float64)
    return ri, F


def boot_rows(kind, y, p, BI): return score_rows(kind, y[BI], p[BI])
def summ(kind, y, p, BI): return {"score": float(score_rows(kind, y, p)), "ci95": pct(boot_rows(kind, y, p, BI))}


def share_rows(kind, y, pv, pd, pa, BI):
    """point share, per-resample shares (same resamples), denominator; as gate3a gap()"""
    sp = {k: float(score_rows(kind, y, p)) for k, p in (("v", pv), ("d", pd), ("a", pa))}; den = sp["a"] - sp["d"]
    r = lambda p: boot_rows(kind, y, p, BI)
    return sp, den, (sp["v"] - sp["d"]) / den, (r(pv) - r(pd)) / (r(pa) - r(pd))


def gap(kind, y, pv, pd, pa, BI):
    sp, den, s, rows = share_rows(kind, y, pv, pd, pa, BI); d = {"s_D1": sp["d"], "s_A1": sp["a"], "s_X": sp["v"], "den": den}
    if den < 0.05: d["dropped"] = True; return d
    return d | {"dropped": False, "share": s, "ci95": pct(rows)}


# ---------------- main readouts ----------------
D = {}
for b in G.TRAIN + G.HELD:
    rows = rows_of(f"{G.O}/objects_main_{b}.csv"); ri, F = load_feats("main", b)
    assert np.array_equal(ri, [int(r["ri"]) for r in rows])
    D[b] = {"F": F, "y": {t: np.array([float(r[t]) for r in rows]) for t in G.TARGETS}, "fold": np.array([int(r["fold"]) for r in rows])}
res["n"]["main"] = {b: len(D[b]["fold"]) for b in D}
BOOT = {}

# ---------------- repro gate (STOP, runs before any new set is scored) ----------------
# repro_5000 E1 centred ridge pooled gap share, row set 'all', same Model / boot_strat / gap as below
REPRO_TOL = 0.0005; ci0 = CN.index("centered")
_nh = [len(D[b]["fold"]) for b in G.HELD]; _per, _pooled = G.boot_strat(_nh)
_g = np.concatenate([bi * 5 + D[b]["fold"] for bi, b in enumerate(G.TRAIN)])
_M = {fs: Model(np.vstack([D[b]["F"][fs][:, ci0] for b in G.TRAIN]), _g, [np.vstack([D[b]["F"][fs][:, ci0] for b in G.HELD])]) for fs in ("D1", "A1_i", "repro_5000")}
for t in PRIM:
    ytr = np.concatenate([D[b]["y"][t] for b in G.TRAIN]); yte = np.concatenate([D[b]["y"][t] for b in G.HELD])
    P_ = {fs: m.predict(ytr)[0][0].astype(np.float32).astype(np.float64) for fs, m in _M.items()}
    s = gap(G.KIND[t], yte, P_["repro_5000"], P_["D1"], P_["A1_i"], _pooled).get("share")
    d = None if s is None else abs(s - REPRO_REF[t]); ok = d is not None and d <= REPRO_TOL
    res["repro"][t] = {"reference_share": REPRO_REF[t], "share": s, "abs_diff": d, "within_tol": ok}
res["repro"]["tol"] = REPRO_TOL; res["repro"]["pass"] = all(res["repro"][t]["within_tol"] for t in PRIM)
json.dump(res["repro"], open(f"{OUT}/p3_repro_gate_new{TAG}.json", "w"), indent=1, default=float)
out(f"repro: {res['repro']}")
if not res["repro"]["pass"]: out(f"STOP: repro gate failed (repro_5000 share differs from {REPRO_REF} by more than {REPRO_TOL}); nothing from this run is to be reported"); sys.exit(2)
del _M, P_

for rs, tl in RS.items():
    valid = {b: (np.isfinite(D[b]["y"]["log10_size"]) if rs == "gal" else np.ones(len(D[b]["fold"]), bool)) for b in D}
    nh = [int(valid[b].sum()) for b in G.HELD]; per, pooled = G.boot_strat(nh); off = np.cumsum([0] + nh)
    res["n"][f"held_{rs}"] = dict(zip(G.HELD, nh)); res["n"][f"train_{rs}"] = {b: int(valid[b].sum()) for b in G.TRAIN}
    for fs in FS:
        for e, tc in EXP.items():
            Xtr = np.vstack([np.vstack([D[b]["F"][fs][valid[b], CN.index(c)] for b in G.TRAIN]) for c in tc])
            g = np.concatenate([bi * 5 + D[b]["fold"][valid[b]] for bi, b in enumerate(G.TRAIN)])
            mdl = Model(Xtr, np.tile(g, len(tc)), [np.vstack([D[b]["F"][fs][valid[b], ci] for b in G.HELD]) for ci in range(8)])
            for t in tl:
                ytr = np.concatenate([D[b]["y"][t][valid[b]] for b in G.TRAIN]); yte = np.concatenate([D[b]["y"][t][valid[b]] for b in G.HELD])
                a, k = mdl.predict(np.tile(ytr, len(tc)))
                for ci, c in enumerate(CN):
                    for pr, P in (("ridge", a[ci]), ("knn", k[ci])):
                        PRED[f"{t}|{fs}|{e}|{pr}|{c}"] = P.astype(np.float32)
                        R = res["main"].setdefault(f"{t}|{fs}|{e}|{pr}|{c}", {})
                        R["pooled"] = summ(G.KIND[t], yte, P, pooled)
                        for j, b in enumerate(G.HELD): R[b] = summ(G.KIND[t], yte[off[j]:off[j + 1]], P[off[j]:off[j + 1]], per[j])
            out(f"main {rs} {fs} {e} done ({el()})")
        save()
    for t in tl:
        yte = np.concatenate([D[b]["y"][t][valid[b]] for b in G.HELD]); kind = G.KIND[t]; BOOT[t] = (yte, pooled)
        views = [("pooled", yte, slice(None), pooled)] + [(b, yte[off[j]:off[j + 1]], slice(off[j], off[j + 1]), per[j]) for j, b in enumerate(G.HELD)]
        for e in EXP:
            for c in CN:
                P = lambda fs: PRED[f"{t}|{fs}|{e}|ridge|{c}"].astype(np.float64)
                for v, y, sl, BI in views:
                    for xa, xb in CMP: res["cmp"][f"{t}|{e}|{c}|{v}|{xa}-{xb}"] = C4.setA_rule(kind, y, P(xa)[sl], P(xb)[sl], BI)
                    for xv in ENCS: res["gap"][f"{t}|{e}|{c}|{v}|{xv}"] = gap(kind, y, P(xv)[sl], P("D1")[sl], P("A1_i")[sl], BI)
    save(); out(f"cmp/gap {rs} done ({el()})")

# ---------------- decision quantities (final checkpoints) ----------------
def pshare(t, e):
    """E1 centred ridge share, pooled and per held-out brick (D17 counts bricks)"""
    return {v: {k: res["gap"][f"{t}|E1|centered|{v}|{e}"].get(k) for k in ("share", "ci95", "dropped")} for v in ["pooled"] + G.HELD}


fin = {a: f"{a}_{STEPS[a]}" for a in STEPS if f"{a}_{STEPS[a]}" in ENCS}; Dd = res["decision"]; Dd["final_present"] = sorted(fin)
for a, e in fin.items(): Dd[f"share_{a}"] = {t: pshare(t, e) for t in PRIM}
if "B3" in fin:
    cells = {f"{t}|{c}": res["cmp"][f"{t}|E1|{c}|pooled|{fin['B3']}-D1"] for t in PRIM for c in OFF}
    worse = [k for k, v in cells.items() if v["verdict"] == "b_better"]
    Dd["A6"] = {"cells": {k: {"diff": v["diff"], "ci95": v["ci95"], "verdict": v["verdict"]} for k, v in cells.items()},
                "n_cells": len(cells), "worse": worse, "n_worse": len(worse), "holds": len(worse) <= 2}
A7 = {}
for nm, (big, small) in (("steps_B3_minus_B1", ("B3", "B1")), ("data_B2_minus_A1", ("B2", "A1"))):
    if big not in fin or small not in fin: continue
    o = {}
    for t in PRIM:
        yte, BI = BOOT[t]; P = lambda e: PRED[f"{t}|{e}|E1|ridge|centered"].astype(np.float64)
        _, dB, sB, rB = share_rows(G.KIND[t], yte, P(fin[big]), P("D1"), P("A1_i"), BI)
        _, dS, sS, rS = share_rows(G.KIND[t], yte, P(fin[small]), P("D1"), P("A1_i"), BI)
        if dB < 0.05: o[t] = {"dropped": True}; continue
        ci = pct(rB - rS); o[t] = {"dropped": False, "diff": sB - sS, "ci95": ci, "falls": ci[1] < 0}
    o["falls"] = any(o[t].get("falls", False) for t in PRIM); A7[nm] = o
if A7: Dd["A7"] = A7 | ({"holds": not any(v["falls"] for v in A7.values())} if len(A7) == 2 else {})
res["exploratory"]["share_vs_step"] = {e: {t: pshare(t, e) for t in PRIM} for e in ENCS if e != "repro_5000"}
save(); out(f"decision: {json.dumps({k: v for k, v in Dd.items() if k != 'A6'}, default=float)}" + (f"; A6 worse {Dd['A6']['n_worse']}/14" if "A6" in Dd else ""))

# ---------------- DESI redshift: skipped (new sets have main blocks only) ----------------

# ---------------- step 4 minus step 3 (vitl_sat_ckpt - vitl_sat), paired ----------------
# raw score per cell, same objects, same boot_strat resamples and score_rows (common4a.setA_rule);
# verdict a_better = step 4 better, b_better = step 3 better (|diff| >= 0.05 and CI excludes 0)
res["step4_minus_step3"] = {}; S4, S3 = "vitl_sat_ckpt", "vitl_sat"
for rs, tl in RS.items():
    valid = {b: (np.isfinite(D[b]["y"]["log10_size"]) if rs == "gal" else np.ones(len(D[b]["fold"]), bool)) for b in D}
    nh = [int(valid[b].sum()) for b in G.HELD]; per, pooled = G.boot_strat(nh); off = np.cumsum([0] + nh)
    for t in tl:
        yte = np.concatenate([D[b]["y"][t][valid[b]] for b in G.HELD]); kind = G.KIND[t]
        views = [("pooled", yte, slice(None), pooled)] + [(b, yte[off[j]:off[j + 1]], slice(off[j], off[j + 1]), per[j]) for j, b in enumerate(G.HELD)]
        for e in EXP:
            for c in CN:
                P = lambda fs: PRED[f"{t}|{fs}|{e}|ridge|{c}"].astype(np.float64)
                for v, y, sl, BI in views:
                    r = C4.setA_rule(kind, y, P(S4)[sl], P(S3)[sl], BI); r["sign"] = int(np.sign(r["diff"]))
                    res["step4_minus_step3"][f"{t}|{e}|{c}|{v}"] = r
save()
for t in RS["all"] + RS["gal"]:
    k = f"{t}|E1|centered"; r = res["step4_minus_step3"][f"{k}|pooled"]
    out(f"S4-S3 {k} pooled diff {r['diff']:+.4f} ci95 [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] {r['verdict']}; "
        f"per-brick sign {[res['step4_minus_step3'][f'{k}|{b}']['sign'] for b in G.HELD]}")
np.savez_compressed(f"{OUT}/p3_preds_new{TAG}.npz", **PRED); res["minutes"] = (time.time() - t0) / 60; save(final=True); out(f"P3 NEW DONE ({el()})")
