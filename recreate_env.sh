#!/usr/bin/env bash
set -e

ENV_NAME=yolo-trt       # Desired environment name
OUT_ENV_NAME="$ENV_NAME"  # Start with the same name

source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if env with same name exists
if conda info --envs | grep -q "^$ENV_NAME"; then
    # Rename the NEW env by appending timestamp
    OUT_ENV_NAME="${ENV_NAME}_$(date +%Y%m%d%H%M%S)"
    echo "⚠️ Environment '$ENV_NAME' already exists. Creating new env as '$OUT_ENV_NAME'"
fi

echo "🐍 Creating environment: $OUT_ENV_NAME"
conda env create -f env.yml -n "$OUT_ENV_NAME"

conda activate "$OUT_ENV_NAME"

echo "📦 Installing pip packages"
pip install --upgrade pip
pip install -r requirements.txt

echo "🔥 Installing PyTorch + CUDA"

# Load torch versions
set -a
source torch_versions.txt
set +a

pip install \
  torch==${TORCH_VERSION} \
  torchvision==${TORCHVISION_VERSION} \
  torchaudio==${TORCHAUDIO_VERSION} \
  --index-url https://download.pytorch.org/whl/${CUDA_TAG}

echo "🧪 Verifying installation"
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
EOF

echo "✅ Environment '$OUT_ENV_NAME' created successfully"
