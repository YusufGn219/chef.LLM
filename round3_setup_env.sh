#!/bin/bash
# Chef.LLM - Konsolide EC2 ortam kurulumu (egitim sirasinda kesfedilen TUM duzeltmeler tek script'te)
set -e

echo "=== 1. NVMe instance store mount ==="
ROOT_DEV=$(findmnt -n -o SOURCE / | sed 's/p[0-9]*$//')
CANDIDATE=$(lsblk -dno NAME,TYPE | grep disk | grep -v "$(basename $ROOT_DEV)" | head -1 | awk '{print $1}')
if ! mountpoint -q /mnt/data; then
  sudo mkdir -p /mnt/data
  sudo mkfs -t xfs -f /dev/$CANDIDATE
  sudo mount /dev/$CANDIDATE /mnt/data
  sudo chown ec2-user:ec2-user /mnt/data
fi
mkdir -p /mnt/data/hf_cache /mnt/data/triton_cache /mnt/data/pip_cache
df -h | grep -E 'nvme|Filesystem'

echo "=== 2. Python 3.12 + devel + venv ==="
sudo dnf install -y python3.12 python3.12-pip python3.12-devel git gcc 2>&1 | tail -5
python3.12 -m venv /mnt/data/venv
source /mnt/data/venv/bin/activate

echo "=== 3. Temel paketler ==="
pip install -q torch peft accelerate huggingface_hub 'jinja2>=3.1' sentencepiece protobuf boto3 datasets tokenizers safetensors

echo "=== 4. trl (API'ye dikkat: max_length, processing_class kullan) ==="
pip install -q trl

echo "=== 5. transformers GitHub main (--no-deps, safetensors>=0.8.0 kisiti PyPI'de yok) ==="
pip install -q --no-deps 'git+https://github.com/huggingface/transformers.git'

echo "=== 6. Tokenizer duzeltmesi (extra_special_tokens bug) ==="
export HF_HOME=/mnt/data/hf_cache
python -c "
import json, os
from huggingface_hub import snapshot_download
MODEL_NAME = 'unsloth/gemma-4-12b-it'
LOCAL_DIR = '/mnt/data/gemma4_tokenizer_fixed'
snapshot_download(repo_id=MODEL_NAME, allow_patterns=['tokenizer*','special_tokens_map.json'], local_dir=LOCAL_DIR)
cfg_path = os.path.join(LOCAL_DIR, 'tokenizer_config.json')
with open(cfg_path, encoding='utf-8') as f:
    cfg = json.load(f)
if 'extra_special_tokens' in cfg and isinstance(cfg['extra_special_tokens'], list):
    cfg['extra_special_tokens'] = {}
with open(cfg_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
print('TOKENIZER OK:', type(tok))
"

echo "=== KURULUM TAMAMLANDI ==="
python -c "import torch, transformers, peft, trl; print('torch',torch.__version__, torch.cuda.is_available()); print('transformers',transformers.__version__); print('peft',peft.__version__); print('trl',trl.__version__)"
