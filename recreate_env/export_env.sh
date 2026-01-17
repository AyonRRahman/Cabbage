#!/usr/bin/env bash
set -e

ENV_NAME=yolo-trt
echo "📦 Snapshotting environment: $ENV_NAME"

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 1. Minimal conda env (Python + system deps)
python - <<'EOF'
import yaml

BAD = ("pytorch", "torch", "cuda", "cudatoolkit", "nvidia")
OUT_FILE = "env.yml"
ENV_NAME = "yolo-trt"

import subprocess, sys

# Try to get from_history export
try:
    result = subprocess.run(["conda", "env", "export", "--from-history"],
                            capture_output=True, text=True, check=True)
    data = yaml.safe_load(result.stdout)
except Exception:
    data = None

# Fallback minimal env if nothing exported
if data is None:
    data = {
        "name": ENV_NAME,
        "channels": ["conda-forge"],
        "dependencies": ["python=3.10", "pip"],
    }

deps = []
for d in data.get("dependencies", []):
    if isinstance(d, str):
        if not d.startswith(BAD):
            deps.append(d)
    else:
        deps.append(d)

data["dependencies"] = deps

with open(OUT_FILE, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
EOF

echo "✅ env.yml created"

# 2. Pip top-level packages (valid format)
pip list --not-required --format=freeze | grep -Ev 'torch|nvidia|cuda' > requirements.txt
echo "✅ requirements.txt created"

# 3. Torch + CUDA versions
python - <<'EOF' > torch_versions.txt
import torch, torchvision, torchaudio
print(f"TORCH_VERSION={torch.__version__.split('+')[0]}")
print(f"TORCHVISION_VERSION={torchvision.__version__.split('+')[0]}")
print(f"TORCHAUDIO_VERSION={torchaudio.__version__.split('+')[0]}")
print(f"CUDA_TAG=cu{torch.version.cuda.replace('.', '')}")
EOF
echo "✅ torch_versions.txt created"

echo "📦 Snapshot complete"
