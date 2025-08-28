#!/usr/bin/env python3
"""
Minimal Ray launcher for Isaac Lab that:
- lets you tweak only 'task' and 'num_samples' (and a few run knobs),
- pulls task-specific Ray settings (cfg_file, cfg_class, metric, mode) from per-task YAML,
- spawns Isaac Lab's ray/tuner.py with the correct Python (sys.executable).

Usage (inside container):
  /workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/train_ray.py \
      --task Isaac-Lift-Cube-Franka-OSC-v0 --num_samples 32

You can also override library (rsl_rl|skrl), run_mode, etc., if needed.
"""

import argparse, os, sys, subprocess, textwrap
from pathlib import Path

try:
    import yaml
except Exception as e:
    print("[ERROR] PyYAML not installed in this environment.", file=sys.stderr)
    raise

# -------- Paths --------
HERE = Path(__file__).resolve().parent
ISAACLAB_PATH = Path(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"))
TUNER_PATH = ISAACLAB_PATH / "scripts/reinforcement_learning/ray/tuner.py"

# Config files
RUN_YAML = HERE / "run_ray.yaml"                             # run-level knobs (you edit this)
TASK_CFG_DIR = HERE / "ray_task_configs"                     # per-task YAMLs

# -------- Utils --------
def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

def deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

# -------- Main --------
def main():
    p = argparse.ArgumentParser(description="Ray tuner launcher for Isaac Lab")
    p.add_argument("--task", type=str, help="Task registry name (e.g., Isaac-Lift-Cube-Franka-OSC-v0)")
    p.add_argument("--num_samples", type=int, help="Ray Tune samples", default=None)
    p.add_argument("--library", type=str, choices=["rsl_rl", "skrl"], default=None,
                   help="RL backend (default comes from run/task yaml)")
    p.add_argument("--run_mode", type=str, default=None, help="Ray tuner run mode (e.g., local)")
    p.add_argument("--num_workers_per_node", type=int, default=None)
    p.add_argument("--repeat_run_count", type=int, default=None)
    # Optional hard overrides (rare):
    p.add_argument("--cfg_file", type=str, default=None)
    p.add_argument("--cfg_class", type=str, default=None)
    p.add_argument("--metric", type=str, default=None)
    p.add_argument("--mode", type=str, default=None, choices=["max", "min"])
    args_cli = p.parse_args()

    # Load run yaml and task yaml
    run_cfg = load_yaml(RUN_YAML)
    task_name = args_cli.task or run_cfg.get("task")
    if not task_name:
        die("No task specified. Provide --task or set 'task' in ray_run.yaml")

    task_yaml = TASK_CFG_DIR / f"{task_name}.yaml"
    task_cfg = load_yaml(task_yaml)
    if not task_cfg:
        # Help the user see what's available
        available = sorted([p.stem for p in TASK_CFG_DIR.glob("*.yaml")])
        die(textwrap.dedent(
            f"""Task config not found: {task_yaml}
            Create it (see examples below) or choose one of:
            {available}
            """))

    # Resolve library & workflow script
    library = (args_cli.library or run_cfg.get("library") or task_cfg.get("library") or "rsl_rl")
    if library == "rsl_rl":
        workflow_script = ISAACLAB_PATH / "scripts/reinforcement_learning/rsl_rl/train.py"
    else:
        workflow_script = ISAACLAB_PATH / "scripts/reinforcement_learning/skrl/train.py"

    # Merge values with precedence: task_cfg < run_cfg < CLI overrides
    merged = {}
    deep_update(merged, task_cfg)
    deep_update(merged, run_cfg)

    # CLI overrides
    if args_cli.num_samples is not None: merged["num_samples"] = args_cli.num_samples
    if args_cli.run_mode    is not None: merged["run_mode"] = args_cli.run_mode
    if args_cli.num_workers_per_node is not None: merged["num_workers_per_node"] = args_cli.num_workers_per_node
    if args_cli.repeat_run_count     is not None: merged["repeat_run_count"] = args_cli.repeat_run_count
    if args_cli.cfg_file   is not None: merged["cfg_file"] = args_cli.cfg_file
    if args_cli.cfg_class  is not None: merged["cfg_class"] = args_cli.cfg_class
    if args_cli.metric     is not None: merged["metric"] = args_cli.metric
    if args_cli.mode       is not None: merged["mode"] = args_cli.mode

    # Fill defaults (same as your tmux script)
    merged.setdefault("run_mode", "local")
    merged.setdefault("num_workers_per_node", 1)
    merged.setdefault("repeat_run_count", 1)
    merged.setdefault("num_samples", 24)
    merged.setdefault("process_response_timeout", 120.0)
    merged.setdefault("max_lines_to_search_experiment_logs", 1000)
    merged.setdefault("max_log_extraction_errors", 2)

    # Required keys
    for k in ("cfg_file", "cfg_class", "metric", "mode"):
        if k not in merged:
            die(f"Missing required key '{k}' in {task_yaml} (or override via CLI).")

    # Build command for tuner.py using the *same Python* we're running under
    cmd = [
        sys.executable, str(TUNER_PATH),
        "--cfg_file", str(merged["cfg_file"]),
        "--cfg_class", str(merged["cfg_class"]),
        "--run_mode", str(merged["run_mode"]),
        "--workflow", str(workflow_script),
        "--num_workers_per_node", str(merged["num_workers_per_node"]),
        "--repeat_run_count", str(merged["repeat_run_count"]),
        "--num_samples", str(merged["num_samples"]),
        "--metric", str(merged["metric"]),
        "--mode", str(merged["mode"]),
        "--process_response_timeout", str(merged["process_response_timeout"]),
        "--max_lines_to_search_experiment_logs", str(merged["max_lines_to_search_experiment_logs"]),
        "--max_log_extraction_errors", str(merged["max_log_extraction_errors"]),
    ]

    print("[INFO] Launching Ray tuner with:")
    print("      ", " ".join(cmd))
    os.makedirs(ISAACLAB_PATH / "tmux" / "ray", exist_ok=True)  # harmless; keeps parity with your log layout
    # Inherit env (so CUDA, PYTORCH_JIT, etc., persist)
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
