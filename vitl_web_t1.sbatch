#!/bin/bash
#SBATCH -J vitl_web_t1
#SBATCH -p kipac
#SBATCH -c 4
#SBATCH --mem=31G
#SBATCH -t 04:00:00
#SBATCH -o logs/vitl_web_t1_%j.log
# Entry 6 of adapt-test/AGENT_LOG.md (Entry 4 step 1). Submit from adapt-test/.
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
test ! -e emb_vitl_web || { echo "emb_vitl_web exists, stopping"; exit 1; }
export VIT_REPO=facebook/dinov3-vitl16-pretrain-lvd1689m
export VIT_REV=ea8dc2863c51be0a264bab82070e3e8836b02d51
export VIT_NORM=imagenet
export VIT_BRICKS=3377m332
export VIT_OUT="$A/emb_vitl_web"
export PYTHONPATH=/scratch/users/anaennis/aion-tokenizer-diagnostic/tnb0/gate3b
python -c "import accel_shim, runpy, sys; sys.argv=['embed_vit.py','d1']; runpy.run_path('embed_vit.py', run_name='__main__')"
python - <<'PYEOF'
import json, sys
import numpy as np
N = "emb_vitl_web/"
b = "main_3377m332_96"
x = np.load(N + "emb_d1_" + b + ".npy")
rep = json.load(open(N + "p2_embed_d1.json"))["blocks"][b]
w = x.shape[-1]
fin = bool(np.isfinite(x).all())
print("shape", x.shape, "WIDTH_OUT", w, "finite", fin)
print("n_windows", rep.get("n_windows"), "s_per_window", rep.get("s_per_window"))
ok = w == 1024 and fin
print("RESULT", "DONE" if ok else "FAILED")
sys.exit(0 if ok else 4)
PYEOF
date
