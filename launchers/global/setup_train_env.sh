#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/content/post-Train-proj}"
LOCKFILE="${REPO_ROOT}/requirements-colab.lock.txt"
HF_CACHE_DIR="${HF_HOME:-/content/hf-cache}"
INSTALL_PLAN="/tmp/post-train-colab-install.txt"
CAUSAL_PACKAGE="causal_conv1d"

if [[ ! -f "${LOCKFILE}" ]]; then
  echo "Lockfile not found: ${LOCKFILE}" >&2
  exit 1
fi

export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

python --version
python -m pip --version

python - "${LOCKFILE}" "${INSTALL_PLAN}" "${CAUSAL_PACKAGE}" <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import json
import sys
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version


lockfile = Path(sys.argv[1])
install_plan = Path(sys.argv[2])
causal_package = canonicalize_name(sys.argv[3])

need_install = []
status_rows = []
installed_versions = {
    canonicalize_name(dist.metadata["Name"]): dist.version
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}

for raw in lockfile.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name, version = line.split("==", 1)
    canonical_name = canonicalize_name(name)
    installed = installed_versions.get(canonical_name)
    installed_base = None
    if installed is not None:
        try:
            installed_base = Version(installed).base_version
        except Exception:
            installed_base = installed.split("+", 1)[0]
    expected_base = Version(version).base_version
    matches = installed_base == expected_base
    status_rows.append(
        {
            "package": name,
            "expected": version,
            "installed": installed,
            "action": "install" if not matches else "keep",
        }
    )
    if not matches and canonical_name != causal_package:
        need_install.append(line)

install_plan.write_text("\n".join(need_install) + ("\n" if need_install else ""), encoding="utf-8")
print(json.dumps({"checked_packages": status_rows, "install_count": len(need_install)}, ensure_ascii=False, indent=2))
PY

if [[ -s "${INSTALL_PLAN}" ]]; then
  echo "Installing pinned packages from ${LOCKFILE}"
  python -m pip install --upgrade-strategy only-if-needed -r "${INSTALL_PLAN}"
else
  echo "Pinned Colab packages already match the lockfile. Skipping pip install."
fi

CAUSAL_VERSION="$(awk -F'==' '/^causal_conv1d==/ {print $2}' "${LOCKFILE}" | tail -n 1)"
if [[ -n "${CAUSAL_VERSION}" ]]; then
  echo "Checking optional accelerator ${CAUSAL_PACKAGE}==${CAUSAL_VERSION}"
  if python - "${CAUSAL_VERSION}" <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import sys
from packaging.utils import canonicalize_name
from packaging.version import Version

expected = Version(sys.argv[1]).base_version
try:
    installed = metadata.version(canonicalize_name("causal_conv1d"))
except metadata.PackageNotFoundError:
    raise SystemExit(1)

raise SystemExit(0 if Version(installed).base_version == expected else 1)
PY
  then
    echo "Optional accelerator ${CAUSAL_PACKAGE} already matches the lockfile."
  elif command -v nvcc >/dev/null 2>&1; then
    echo "Attempting to build optional accelerator ${CAUSAL_PACKAGE}==${CAUSAL_VERSION} with nvcc."
    if ! python -m pip install --no-build-isolation "${CAUSAL_PACKAGE}==${CAUSAL_VERSION}"; then
      echo
      echo "Warning: failed to build optional accelerator ${CAUSAL_PACKAGE} on Colab."
      echo "Continuing without it because flash-linear-attention provides Triton conv1d implementations."
    fi
  else
    echo "Skipping optional accelerator ${CAUSAL_PACKAGE}: nvcc is unavailable in this Colab runtime."
    echo "Continuing without it because flash-linear-attention provides Triton conv1d implementations."
  fi
fi

python - <<'PY'
import importlib
import json
import os
import platform
import sys

CHECKS = [
    ("torch", "__version__"),
    ("unsloth", "__version__"),
    ("transformers", "__version__"),
    ("trl", "__version__"),
    ("accelerate", "__version__"),
    ("peft", "__version__"),
    ("datasets", "__version__"),
    ("huggingface_hub", "__version__"),
    ("bitsandbytes", "__version__"),
    ("xformers", "__version__"),
    ("yaml", "__version__"),
    ("PIL", "__version__"),
]

report = {
    "python": sys.version,
    "platform": platform.platform(),
    "hf_home": os.environ.get("HF_HOME"),
}

for module_name, attr in CHECKS:
    module = importlib.import_module(module_name)
    report[module_name] = getattr(module, attr, "unknown")

try:
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
except Exception as exc:  # pragma: no cover
    report["torch_cuda_probe_error"] = repr(exc)

try:
    from unsloth import FastLanguageModel, is_bf16_supported
    report["unsloth_fast_language_import"] = FastLanguageModel is not None
    report["unsloth_bf16_supported"] = is_bf16_supported()
except Exception as exc:  # pragma: no cover
    report["unsloth_fast_language_import"] = False
    report["unsloth_fast_language_error"] = repr(exc)

try:
    import fla
    report["flash_linear_attention_import"] = True
    report["flash_linear_attention_version"] = getattr(fla, "__version__", "unknown")
except Exception as exc:  # pragma: no cover
    report["flash_linear_attention_import"] = False
    report["flash_linear_attention_error"] = repr(exc)

try:
    import causal_conv1d
    report["causal_conv1d_import"] = True
    report["causal_conv1d_version"] = getattr(causal_conv1d, "__version__", "unknown")
except Exception as exc:  # pragma: no cover
    report["causal_conv1d_import"] = False
    report["causal_conv1d_error"] = repr(exc)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    report["transformers_auto_tokenizer_import"] = AutoTokenizer is not None
    report["transformers_auto_causal_lm_import"] = AutoModelForCausalLM is not None
except Exception as exc:  # pragma: no cover
    report["transformers_auto_import_error"] = repr(exc)

print(json.dumps(report, ensure_ascii=False, indent=2))
PY
