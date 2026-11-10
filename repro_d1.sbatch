#!/bin/bash
#SBATCH -J repro_d1
#SBATCH -p kipac
#SBATCH -c 4
#SBATCH --mem=31G
#SBATCH -t 04:00:00
#SBATCH -o logs/repro_d1_%j.log
# Entry 4 of adapt-test/AGENT_LOG.md. Submit from adapt-test/.
set -euo pipefail
ml purge; ml python/3.12.1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=$SCRATCH/hf_cache
source $HOME/.venvs/aion-cpu/bin/activate
A=/scratch/users/anaennis/aion-tokenizer-diagnostic/adapt-test
cd "$A"
hostname; date
E=e9758e5906929b26fafbe213dc7adbe31144d33c765f2867e88a8995650897e3
echo "$E  embed_vit.py" | sha256sum -c -
test ! -e emb_check || { echo "emb_check exists, stopping"; exit 1; }
export VIT_REPO=facebook/dinov3-vitb16-pretrain-lvd1689m
export VIT_REV=5931719e67bbdb9737e363e781fb0c67687896bc
export VIT_NORM=imagenet
export VIT_BRICKS=3377m332
export VIT_OUT="$A/emb_check"
# Entry 5: shim loaded here so embed_vit.py stays at the Entry 4 hash.
export PYTHONPATH=/scratch/users/anaennis/aion-tokenizer-diagnostic/tnb0/gate3b
python -c "import accel_shim, runpy, sys; sys.argv=['embed_vit.py','d1']; runpy.run_path('embed_vit.py', run_name='__main__')"
python - <<'PYEOF'
import json, os, sys
import numpy as np
T = os.path.expanduser(
    "~/aion-tokenizer-diagnostic/tnb0/gate3a/")
N = "emb_check/"
b = "main_3377m332_96"
new = np.load(N + "emb_d1_" + b + ".npy")
old = np.load(T + "emb_d1_" + b + ".npy")
rn = np.load(N + "ri_d1_" + b + ".npy")
ro = np.load(T + "ri_d1_" + b + ".npy")
rep = json.load(open(N + "p2_embed_d1.json"))
re_ok = bool(rep["blocks"][b]["reembed_bit_identical"])
print("shapes", new.shape, old.shape, new.dtype, old.dtype)
same_shape = new.shape == old.shape
same_ri = bool(np.array_equal(rn, ro))
print("same_shape", same_shape, "same_ri", same_ri,
      "reembed_ok", re_ok)
m0 = float("nan")
fin0 = False
if same_shape:
    for s in (0, 1):
        x = new[:, :, s, :].astype(np.float64)
        y = old[:, :, s, :].astype(np.float64)
        d = np.abs(x - y)
        print("slot", s, "max", np.nanmax(d), "mean", np.nanmean(d))
        print("slot", s, "n over 0.001:", int((d > 0.001).sum()),
              "of", d.size, "nonfinite new/old:",
              int((~np.isfinite(x)).sum()),
              int((~np.isfinite(y)).sum()))
        print("slot", s, "old abs mean", np.nanmean(np.abs(y)),
              "old abs max", np.nanmax(np.abs(y)))
        if s == 0:
            m0 = float(np.nanmax(d))
            fin0 = bool(np.isfinite(x).all())
            k = np.unravel_index(np.nanargmax(d), d.shape)
            print("slot 0 worst (object, position, dim):", k)
ok = bool(same_shape and same_ri and re_ok and fin0 and m0 <= 0.001)
print("RULE: slot0 max<=0.001, ri equal, reembed True, finite")
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 3)
PYEOF
date
