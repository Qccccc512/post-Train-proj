#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests
from huggingface_hub import snapshot_download

from scripts.common.runtime_utils import ensure_dir, get_hf_token, load_yaml, repo_root, sanitize_name, write_json


DEFAULT_CONFIG_PATH = repo_root() / "configs" / "benchmark" / "default.yaml"
FRAMEWORK_LOCK_PATH = repo_root() / "benchmark" / "framework_versions.lock.json"


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    repo_url: str
    ref: str
    commit: str
    local_dir: Path
    workdir_subpath: str | None = None

    @property
    def workdir(self) -> Path:
        if self.workdir_subpath:
            return self.local_dir / self.workdir_subpath
        return self.local_dir


@dataclass(frozen=True)
class EndpointSpec:
    host: str
    port: int
    model_name: str

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def completions_url(self) -> str:
        return f"{self.root_url}/completions"


@dataclass(frozen=True)
class ResolvedModelSpec:
    base_model: str
    tokenizer: str
    adapter: str | None
    adapter_local_path: str | None
    served_model_name: str


def load_benchmark_config(path: str | Path | None = None) -> dict[str, Any]:
    import os
    env_path = os.environ.get("BENCHMARK_CONFIG")
    return load_yaml(path or env_path or DEFAULT_CONFIG_PATH)


def load_framework_specs() -> dict[str, FrameworkSpec]:
    payload = json.loads(FRAMEWORK_LOCK_PATH.read_text(encoding="utf-8"))
    specs: dict[str, FrameworkSpec] = {}
    for name, entry in payload.items():
        specs[name] = FrameworkSpec(
            name=name,
            repo_url=entry["repo_url"],
            ref=entry["ref"],
            commit=entry["commit"],
            local_dir=repo_root() / entry["local_dir"],
            workdir_subpath=entry.get("workdir_subpath"),
        )
    return specs


def get_framework_spec(name: str) -> FrameworkSpec:
    specs = load_framework_specs()
    try:
        return specs[name]
    except KeyError as exc:
        raise BenchmarkError(f"Unknown benchmark framework: {name}") from exc


def _auto_bootstrap_frameworks() -> None:
    """Automatically bootstrap frameworks if they are missing."""
    bootstrap_script = repo_root() / "scripts" / "bootstrap_benchmarks.py"
    if not bootstrap_script.exists():
        raise BenchmarkError(f"Bootstrap script not found: {bootstrap_script}")
    print("Benchmark frameworks not found. Running bootstrap...")
    try:
        import sys
        subprocess.run(
            [sys.executable, str(bootstrap_script)],
            check=True,
            cwd=str(repo_root()),
        )
    except subprocess.CalledProcessError as exc:
        raise BenchmarkError(
            f"Failed to bootstrap benchmark frameworks. "
            f"Try running manually: python scripts/bootstrap_benchmarks.py"
        ) from exc


def framework_workdir(name: str) -> Path:
    """Get the working directory for a benchmark framework.
    
    If the framework is missing, automatically runs bootstrap to download it.
    """
    spec = get_framework_spec(name)
    workdir = spec.workdir
    if not workdir.exists():
        # Auto-bootstrap if framework is missing
        _auto_bootstrap_frameworks()
        # Re-check after bootstrap
        if not workdir.exists():
            raise BenchmarkError(
                f"Benchmark framework '{name}' is still missing at {workdir} after bootstrap. "
                "Try running manually: python scripts/bootstrap_benchmarks.py --force"
            )
    return workdir


def is_probable_hf_repo_id(value: str) -> bool:
    if not value or value.startswith(("http://", "https://")):
        return False
    if value.startswith("/"):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def resolve_repo_or_path(
    value: str | None,
    *,
    cache_dir: str | Path,
    repo_type: str = "model",
) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if not is_probable_hf_repo_id(value):
        return value
    cache_dir = ensure_dir(cache_dir)
    destination = cache_dir / sanitize_name(value.replace("/", "--"))
    snapshot_download(
        repo_id=value,
        repo_type=repo_type,
        local_dir=str(destination),
        token=get_hf_token({"allow_keys_json_fallback": True}),
        resume_download=True,
    )
    return str(destination.resolve())


def resolve_model_spec(
    *,
    base_model: str,
    adapter: str | None,
    tokenizer: str | None,
    served_model_name: str | None,
    cache_dir: str | Path,
) -> ResolvedModelSpec:
    cache_dir = ensure_dir(cache_dir)
    base_model_resolved = resolve_repo_or_path(
        base_model,
        cache_dir=cache_dir / "base_models",
        repo_type="model",
    ) or base_model
    adapter_local = None
    if adapter:
        adapter_local = resolve_repo_or_path(
            adapter,
            cache_dir=cache_dir / "adapters",
            repo_type="model",
        )
    tokenizer_source = tokenizer or base_model_resolved
    served_name = served_model_name or base_model
    return ResolvedModelSpec(
        base_model=base_model_resolved,
        tokenizer=tokenizer_source,
        adapter=adapter,
        adapter_local_path=adapter_local,
        served_model_name=served_name,
    )


def resolve_concurrency(explicit: int | None) -> int:
    """Resolve the concurrency setting for benchmark runs.
    
    Since all evaluations (C-Eval, IFEval, BFCL v4) use local vLLM server,
    this simply returns the explicit value or defaults to 1.
    """
    if explicit is not None:
        return max(1, int(explicit))
    return 1


def parse_host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if not parsed.hostname or not parsed.port:
        raise BenchmarkError(f"Unable to parse host/port from server URL: {base_url}")
    return parsed.hostname, parsed.port


def pick_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def _tail_log_text(path: str | Path | None, *, max_lines: int = 80) -> str:
    if not path:
        return ""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    tail = lines[-max_lines:]
    return "\n".join(tail)


def wait_for_server(
    base_url: str,
    *,
    timeout_seconds: int = 300,
    process: subprocess.Popen[str] | None = None,
    log_path: str | Path | None = None,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    models_url = f"{base_url.rstrip('/')}/models"
    while time.time() < deadline:
        if process is not None:
            returncode = process.poll()
            if returncode is not None:
                log_tail = _tail_log_text(log_path)
                details = f"vLLM server exited early with code {returncode}."
                if log_path:
                    details += f" Log: {log_path}."
                if log_tail:
                    details += "\n\nLast server log lines:\n" + log_tail
                raise BenchmarkError(details)
        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # pragma: no cover
            last_error = repr(exc)
        time.sleep(2)
    log_tail = _tail_log_text(log_path)
    message = f"Timed out waiting for vLLM server at {models_url}: {last_error}"
    if log_path:
        message += f" Log: {log_path}."
    if log_tail:
        message += "\n\nLast server log lines:\n" + log_tail
    raise BenchmarkError(message)


def make_run_slug(base_model: str, adapter: str | None, label: str | None = None) -> str:
    if label:
        return sanitize_name(label)
    base_part = sanitize_name(base_model.split("/")[-1])
    adapter_part = "base-only"
    if adapter:
        adapter_part = sanitize_name(Path(adapter).name if Path(adapter).exists() else adapter.split("/")[-1])
    return sanitize_name(f"{base_part}_{adapter_part}")


def benchmark_output_dir(base_output_dir: str | Path, run_slug: str, suite_name: str) -> Path:
    base_output_dir = Path(base_output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = repo_root() / base_output_dir
    return ensure_dir(base_output_dir / "benchmarks" / run_slug / suite_name)


def runtime_dir() -> Path:
    return ensure_dir(repo_root() / "runs" / "_benchmark_runtime")


def get_conda_env_path(env_name: str) -> Path | None:
    """Get the path to a conda environment by name."""
    import subprocess
    try:
        result = subprocess.run(
            ["conda", "info", "--envs", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        import json as json_mod
        info = json_mod.loads(result.stdout)
        for env_path in info.get("envs", []):
            if Path(env_path).name == env_name:
                return Path(env_path)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def normalize_env_spec(env_dir_or_name: str | Path | None) -> str:
    if env_dir_or_name is None:
        return ""
    env_str = str(env_dir_or_name).strip()
    if env_str.lower() in {"", "none", "null"}:
        return ""
    return env_str


def runtime_python(env_dir_or_name: str | Path | None) -> Path:
    """Get the python executable for the benchmark environment.
    
    Args:
        env_dir_or_name: Either a conda environment name (e.g., 'post-train-benchmark')
            or a path to an environment directory.
    """
    env_str = normalize_env_spec(env_dir_or_name)
    # Check if it looks like a conda env name (no path separators)
    if "/" not in env_str and "\\" not in env_str and not Path(env_str).exists():
        conda_path = get_conda_env_path(env_str)
        if conda_path:
            python_bin = conda_path / "bin" / "python"
            if python_bin.exists():
                return python_bin
            raise BenchmarkError(
                f"Conda environment '{env_str}' exists but python not found at {python_bin}."
            )

    if not env_str:
        import shutil
        sys_python = shutil.which("python3") or shutil.which("python")
        if sys_python:
            return Path(sys_python)

    # Fall back to treating it as a path
    env_path = Path(env_str)
    if not env_path.is_absolute():
        env_path = repo_root() / env_path
    python_bin = env_path / "bin" / "python"
    if not python_bin.exists():
        raise BenchmarkError(
            f"Expected runtime python is missing at {python_bin}. "
            "Run `bash scripts/setup_benchmark_env.sh` first."
        )
    return python_bin


def write_command_log(path: Path, command: list[str], cwd: Path, env: dict[str, str] | None) -> None:
    ensure_dir(path.parent)
    payload = {
        "cwd": str(cwd),
        "command": command,
        "shell_command": " ".join(shlex.quote(part) for part in command),
        "env_overrides": env or {},
    }
    write_json(path, payload)


def run_logged_subprocess(
    command: list[str],
    *,
    cwd: str | Path,
    env_overrides: dict[str, str] | None,
    log_path: str | Path,
) -> None:
    cwd = Path(cwd)
    log_path = Path(log_path)
    write_command_log(log_path.with_suffix(".command.json"), command, cwd, env_overrides)
    env = build_vllm_process_env()
    if env_overrides:
        env.update(env_overrides)
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise BenchmarkError(
            f"Command failed with exit code {process.returncode}: {' '.join(command)}. "
            f"See log: {log_path}"
        )


def build_vllm_process_env() -> dict[str, str]:
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing}" if existing else conda_lib
    return env


def build_lm_eval_model_args(
    *,
    endpoint: EndpointSpec,
    tokenizer: str,
    num_concurrent: int,
    max_retries: int = 3,
) -> str:
    parts = {
        "model": endpoint.model_name,
        "base_url": endpoint.completions_url,
        "tokenizer": tokenizer,
        "tokenizer_backend": "huggingface",
        "tokenized_requests": False,
        "num_concurrent": num_concurrent,
        "max_retries": max_retries,
        "verify_certificate": False,
    }
    rendered: list[str] = []
    for key, value in parts.items():
        value_repr = str(value).lower() if isinstance(value, bool) else str(value)
        rendered.append(f"{key}={value_repr}")
    return ",".join(rendered)


def build_vllm_serve_command(
    *,
    vllm_env_dir_or_name: str | Path | None,
    model_spec: ResolvedModelSpec,
    host: str,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    dtype: str,
    max_lora_rank: int,
    repetition_penalty: float = 1.0,
) -> list[str]:
    env_str = normalize_env_spec(vllm_env_dir_or_name)
    vllm_bin: Path | None = None
    import shutil
    
    if not env_str:
        # if env_str is empty, fallback to system vllm
        sys_vllm = shutil.which("vllm")
        if sys_vllm:
            vllm_bin = Path(sys_vllm)
    elif "/" not in env_str and "\\" not in env_str:
        conda_path = get_conda_env_path(env_str)
        if conda_path:
            vllm_bin = conda_path / "bin" / "vllm"
    if vllm_bin is None or not vllm_bin.exists():
        # Fall back to treating it as a path if provided
        if env_str:
            vllm_env_dir = Path(env_str)
            if not vllm_env_dir.is_absolute():
                vllm_env_dir = repo_root() / vllm_env_dir
            vllm_bin = vllm_env_dir / "bin" / "vllm"
        else:
            import shutil
            vllm_bin = Path(shutil.which("vllm") or "vllm")

    if not vllm_bin.exists():
        raise BenchmarkError(
            f"vLLM executable was not found at {vllm_bin}. "
            "Run `bash scripts/setup_benchmark_env.sh` first."
        )
    command = [
        str(vllm_bin),
        "serve",
        model_spec.base_model,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        model_spec.served_model_name,
        "--dtype",
        dtype,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--trust-remote-code",
    ]
    if repetition_penalty > 1.0:
        # Keep server-side defaults aligned with per-request generation kwargs.
        # vLLM expects a JSON object for override-generation-config.
        command.extend(
            [
                "--override-generation-config",
                json.dumps({"repetition_penalty": repetition_penalty}),
            ]
        )
    reasoning_hint = " ".join(
        item for item in (model_spec.base_model, model_spec.served_model_name, model_spec.tokenizer) if item
    ).lower()
    if "qwen3" in reasoning_hint or "deepseek" in reasoning_hint:
        command.extend(["--reasoning-parser", "qwen3"])
    quantization_config_path = Path(model_spec.base_model) / "config.json"
    if quantization_config_path.exists():
        try:
            payload = json.loads(quantization_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        quant_cfg = payload.get("quantization_config")
        if isinstance(quant_cfg, dict):
            is_bitsandbytes = (
                quant_cfg.get("quant_method") == "bitsandbytes"
                or quant_cfg.get("load_in_4bit")
                or quant_cfg.get("load_in_8bit")
            )
            if is_bitsandbytes:
                command.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    if model_spec.adapter_local_path:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-modules",
                f"adapter={model_spec.adapter_local_path}",
            ]
        )
    return command


def _load_bfcl_data_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    if text[0] == "{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_bfcl_v4_run_ids(
    *,
    bfcl_root: str | Path,
    max_samples: int,
    selected_categories: list[str] | None = None,
) -> dict[str, list[str]]:
    # Memory backend types that need special handling
    MEMORY_BACKENDS = ["kv", "vector", "rec_sum"]
    # Web search variants that need special handling
    WEB_SEARCH_VARIANTS = ["base", "no_snippet"]

    data_dir = Path(bfcl_root) / "bfcl_eval" / "data"
    files = sorted(data_dir.glob("BFCL_v4_*.json"))
    category_to_ids: dict[str, list[str]] = {}
    for path in files:
        rows = _load_bfcl_data_file(path)
        if isinstance(rows, dict):
            for category, ids in rows.items():
                if selected_categories and category not in selected_categories:
                    continue
                category_to_ids.setdefault(category, []).extend([item for item in ids if item])
            continue

        category = path.stem.removeprefix("BFCL_v4_")

        # Special handling for memory tests: expand "memory" into memory_kv, memory_vector, memory_rec_sum
        # The BFCL framework expects these specific category names to properly configure backends
        if category == "memory":
            ids = [row["id"] for row in rows if isinstance(row, dict) and "id" in row]
            for backend in MEMORY_BACKENDS:
                backend_category = f"memory_{backend}"
                if selected_categories and backend_category not in selected_categories:
                    continue
                # Transform IDs: memory_0-xxx -> memory_{backend}_0-xxx
                backend_ids = [id_.replace("memory_", f"memory_{backend}_", 1) for id_ in ids]
                category_to_ids[backend_category] = backend_ids
            continue

        # Special handling for web_search tests: expand into web_search_base and web_search_no_snippet
        # The BFCL framework expects these specific category names for proper test setup
        if category == "web_search":
            ids = [row["id"] for row in rows if isinstance(row, dict) and "id" in row]
            for variant in WEB_SEARCH_VARIANTS:
                variant_category = f"web_search_{variant}"
                if selected_categories and variant_category not in selected_categories:
                    continue
                # Transform IDs: web_search_N -> web_search_{variant}_N
                variant_ids = [id_.replace("web_search_", f"web_search_{variant}_", 1) for id_ in ids]
                category_to_ids[variant_category] = variant_ids
            continue

        if selected_categories and category not in selected_categories:
            continue
        category_to_ids[category] = [row["id"] for row in rows if isinstance(row, dict) and "id" in row]
    run_ids: dict[str, list[str]] = {key: [] for key in category_to_ids}
    remaining = max(0, int(max_samples))
    categories = [key for key, values in category_to_ids.items() if values]
    while remaining > 0 and categories:
        next_round: list[str] = []
        for category in categories:
            if category_to_ids[category]:
                run_ids[category].append(category_to_ids[category].pop(0))
                remaining -= 1
                if category_to_ids[category] and remaining > 0:
                    next_round.append(category)
                if remaining <= 0:
                    break
        categories = next_round
    return {key: value for key, value in run_ids.items() if value}


def write_bfcl_project_env(project_root: str | Path, *, host: str, port: int) -> Path:
    project_root = ensure_dir(project_root)
    env_path = project_root / ".env"
    env_path.write_text(
        f"LOCAL_SERVER_ENDPOINT={host}\nLOCAL_SERVER_PORT={port}\n",
        encoding="utf-8",
    )
    return env_path


@contextlib.contextmanager
def managed_vllm_server(
    *,
    vllm_env_dir_or_name: str | Path,
    model_spec: ResolvedModelSpec,
    host: str,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    dtype: str,
    max_lora_rank: int,
    log_path: str | Path,
    repetition_penalty: float = 1.0,
) -> Iterator[EndpointSpec]:
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    command = build_vllm_serve_command(
        vllm_env_dir_or_name=vllm_env_dir_or_name,
        model_spec=model_spec,
        host=host,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        max_lora_rank=max_lora_rank,
        repetition_penalty=repetition_penalty,
    )
    write_command_log(log_path.with_suffix(".command.json"), command, repo_root(), None)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(repo_root()),
            env=build_vllm_process_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_server(
                f"http://{host}:{port}/v1",
                process=process,
                log_path=log_path,
            )
            yield EndpointSpec(host=host, port=port, model_name=model_spec.served_model_name)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()
                process.wait(timeout=30)
