#!/usr/bin/env bash
# One-time (and safe to repeat) setup of a fresh Ubuntu 24.04 GPU server: system tools, Node 22, the CUDA runtime + cuDNN that onnxruntime-node's
# GPU build needs (CUDA 13 or 12, chosen by the driver), and this project. Run it as your normal user (it uses sudo where needed):
#   curl -fsSL https://raw.githubusercontent.com/rohitindustry09/Watermark-and-Bg-Remover/main/scripts/server-setup.sh | bash
# Every step checks whether it is already done, so running it again only fixes what is missing.
set -euo pipefail

REPO="https://github.com/rohitindustry09/Watermark-and-Bg-Remover.git"
DIR="${DIR:-$HOME/Watermark-and-Bg-Remover}"
say() { printf '\n== %s\n' "$*"; }

say "GPU and driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found: the NVIDIA driver is missing, so this is not a GPU machine (or the driver is not installed)."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "System tools"
# A duplicate NVIDIA package source (left by an earlier run of this script) makes apt refuse to read its sources; remove it and try again.
if ! sudo apt-get update -qq >/dev/null 2>&1; then
  # "dpkg -P" (purge) and not "-r": removing keeps the package's config file, which is the conflicting source line.
  sudo dpkg -P cuda-keyring >/dev/null 2>&1 || true
  if ! sudo apt-get update -qq; then
    echo "apt still cannot read its package sources. The NVIDIA entries are:"
    grep -rn "developer.download.nvidia.com" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null || true
    exit 1
  fi
fi
sudo apt-get install -y -qq git curl wget zip unzip build-essential >/dev/null
echo "git, curl, zip and unzip are installed."

say "Node.js 22"
export NVM_DIR="$HOME/.nvm"
# nvm's own script is not written for bash's strict mode (unset variables, non-zero returns), so it is loaded with the checks switched off.
load_nvm() {
  if [ -s "$NVM_DIR/nvm.sh" ]; then
    set +eu
    . "$NVM_DIR/nvm.sh"
    set -eu
  fi
}
load_nvm
if ! command -v node >/dev/null 2>&1 || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 22 ]; then
  if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
  fi
  load_nvm
  set +e
  nvm install 22
  nvm alias default 22 >/dev/null
  set -e
fi
echo "node $(node -v), npm $(npm -v)"

say "CUDA runtime and cuDNN 9 (needed for the GPU)"
# The newest onnxruntime-node is built for CUDA 13, which needs a driver that supports CUDA 13 (580 or newer). Some machines (Hostinger Texas: driver 570,
# CUDA 12.8) have an older driver: there the CUDA 13 build fails with "CUDA driver version is insufficient", so we use the CUDA 12 build (1.23.2) and the
# CUDA 12 cuDNN instead. The driver's CUDA version decides which one.
DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)"
DRIVER_CUDA="${DRIVER_CUDA:-12}"
if [ "$DRIVER_CUDA" -ge 13 ]; then CUDA_MAJOR=13; CUDA_PKG=cuda-libraries-13-0; else CUDA_MAJOR=12; CUDA_PKG=cuda-libraries-12-8; fi
echo "The driver supports CUDA $DRIVER_CUDA, so the CUDA $CUDA_MAJOR libraries are used."
# The checks capture the command's output first and search that: with "set -o pipefail", "ldconfig -p | grep -q x" reports failure even when x is found
# (grep -q exits at the first match and the left side gets SIGPIPE).
libs_present() {
  local libs
  libs="$(ldconfig -p 2>/dev/null || true)"
  grep -q "libcublasLt.so.$CUDA_MAJOR" <<<"$libs" && grep -q 'libcudnn.so.9' <<<"$libs"
}
# cuDNN 9 comes in one build per CUDA major; the wrong one loads but fails with CUDNN_STATUS_NOT_INITIALIZED.
cudnn_matches() {
  local installed
  installed="$(dpkg -l 2>/dev/null | awk '/^ii +libcudnn9-cuda-/ { print $2 }' || true)"
  grep -qx "libcudnn9-cuda-$CUDA_MAJOR" <<<"$installed"
}
cuda_visible() {
  local policy
  policy="$(apt-cache policy "$CUDA_PKG" 2>/dev/null || true)"
  grep -q 'Candidate: [0-9]' <<<"$policy"
}
if libs_present && cudnn_matches; then
  echo "Already installed."
else
  # Add NVIDIA's package source only if apt cannot already see the CUDA packages (the Hostinger image has it set up, and adding it twice makes apt fail).
  if ! cuda_visible; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq
  fi
  # libcudnn9-cuda-12 and -13 replace each other, so apt swaps the wrong one out.
  if libs_present; then
    sudo apt-get install -y "libcudnn9-cuda-$CUDA_MAJOR"
  else
    sudo apt-get install -y "$CUDA_PKG" "libcudnn9-cuda-$CUDA_MAJOR"
  fi
  sudo ldconfig
  if libs_present && cudnn_matches; then
    echo "Installed."
  else
    echo "The CUDA $CUDA_MAJOR runtime or cuDNN 9 is still missing after the install; the GPU cannot be used. Stopping."
    exit 1
  fi
fi

say "The project"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone "$REPO" "$DIR"
fi
cd "$DIR"
npm install --no-audit --no-fund
if [ "$CUDA_MAJOR" = 12 ]; then
  # The project's onnxruntime-node is the CUDA 13 build; swap in the last CUDA 12 build for this older driver (not saved to package.json).
  npm install --no-save --no-audit --no-fund onnxruntime-node@1.23.2
fi
echo "Project ready in $DIR (CUDA $CUDA_MAJOR)"

say "Done"
echo "Next: bash $DIR/scripts/server-run.sh help"
echo "(In a new SSH session run:  source ~/.bashrc   so that 'node' is found.)"
