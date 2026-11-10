#!/bin/bash
#SBATCH -J dl_vitl
#SBATCH -p kipac
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o logs/dl_vitl_%j.log
# Entry 2 of adapt-test/AGENT_LOG.md. Submit from adapt-test/.
set -euo pipefail
ml purge; ml python/3.12.1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=$SCRATCH/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
unset HF_HUB_OFFLINE
source $HOME/.venvs/aion-cpu/bin/activate
hostname; date
code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' \
  https://huggingface.co || true)
echo "HTTP $code"
[ "$code" = "200" ] || { echo "no connection, stopping"; exit 1; }
cd /tmp
H=$SCRATCH/hf_cache/hub
VB=$H/models--facebook--dinov3-vitb16-pretrain-lvd1689m
test -d "$VB" || { echo "ViT-B cache missing"; exit 1; }
lst() { find "$VB" -printf '%p %y %s %T@ %l\n' | sort; }
B=$(mktemp); A=$(mktemp)
lst > "$B"
python -I - <<'PYEOF'
from huggingface_hub import HfApi, snapshot_download
P = ["config*.json", "*.safetensors", "preprocessor_config.json"]
for r in ("facebook/dinov3-vitl16-pretrain-lvd1689m",
          "facebook/dinov3-vitl16-pretrain-sat493m"):
    sha = HfApi().model_info(r).sha
    p = snapshot_download(r, revision=sha, allow_patterns=P)
    print("REPO", r)
    print("REV", sha)
    print("PATH", p)
PYEOF
lst > "$A"
if cmp -s "$B" "$A"; then
  echo "VITB UNCHANGED"
else
  echo "VITB CHANGED"; diff "$B" "$A" || true; exit 1
fi
for d in $H/models--facebook--dinov3-vitl16-*/snapshots/*/; do
  echo "== $d"; ls -lL "$d"; sha256sum "$d"*.safetensors
done
rm -f "$B" "$A"
date
