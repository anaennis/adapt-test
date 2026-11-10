#!/bin/bash
#SBATCH -J vitl_web_b7
#SBATCH -p kipac
#SBATCH -c 4
#SBATCH --mem=31G
#SBATCH -t 04:00:00
#SBATCH -o logs/vitl_web_b7_%j.log
# Entry 8 of adapt-test/AGENT_LOG.md (Entry 4 step 2). Submit from adapt-test/.
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
test ! -e emb_vitl_web_b7 || { echo "emb_vitl_web_b7 exists, stopping"; exit 1; }
export VIT_REPO=facebook/dinov3-vitl16-pretrain-lvd1689m
export VIT_REV=ea8dc2863c51be0a264bab82070e3e8836b02d51
export VIT_NORM=imagenet
export VIT_BRICKS=0728m047,0184p052,0227m280,2118m002,1954p237,3103m010,3444m405
export VIT_OUT="$A/emb_vitl_web_b7"
export PYTHONPATH=/scratch/users/anaennis/aion-tokenizer-diagnostic/tnb0/gate3b
python -c "import accel_shim, runpy, sys; sys.argv=['embed_vit.py','d1']; runpy.run_path('embed_vit.py', run_name='__main__')"
python - <<'PYEOF'
import os, sys
import numpy as np
N = "emb_vitl_web_b7/"
BR = ["0728m047", "0184p052", "0227m280", "2118m002", "1954p237", "3103m010", "3444m405"]
ok = True; total = 0
for k in BR:
    b = "main_" + k + "_96"; p = N + "emb_d1_" + b + ".npy"
    if not os.path.exists(p):
        print(b, "MISSING"); ok = False; continue
    x = np.load(p)
    w = x.shape[-1]
    fin = bool(np.isfinite(x).all())
    print(b, "shape", x.shape, "WIDTH_OUT", w, "finite", fin)
    total += x.shape[0] * x.shape[1]
    ok = ok and w == 1024 and fin
print("total_windows", total)
print("RESULT", "DONE" if ok else "FAILED")
sys.exit(0 if ok else 4)
PYEOF
date
