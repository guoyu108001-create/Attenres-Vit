#!/usr/bin/env bash
# Shared helpers for experiment launch / logging / anti-overwrite.

set -euo pipefail

yaml_get() {
  local key="$1"
  local file="$2"
  python - "$key" "$file" <<'PY'
import sys, yaml
key, path = sys.argv[1], sys.argv[2]
data = yaml.safe_load(open(path)) or {}
val = data.get(key, "")
print("" if val is None else val)
PY
}

# Build run directory name: {date}_{slug}_s{seed}
build_run_name() {
  local slug="$1"
  local seed="$2"
  local run_id="${3:-$(date +%Y%m%d_%H%M%S)}"
  echo "${run_id}_${slug}_s${seed}"
}

# Block until every GPU in a comma list uses < GPU_FREE_MIB MiB (default 2000),
# polling every GPU_POLL_SEC seconds (default 5).
wait_gpus_free() {
  local gpus="$1"
  local limit="${GPU_FREE_MIB:-2000}"
  local poll="${GPU_POLL_SEC:-5}"
  local announced=0 g used busy
  while true; do
    busy=""
    for g in ${gpus//,/ }; do
      used="$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if (( used >= limit )); then
        busy="$g"
        break
      fi
    done
    [[ -z "$busy" ]] && return 0
    if (( announced == 0 )); then
      echo "[WAIT] GPU $busy busy (>= ${limit} MiB); polling every ${poll}s until GPUs $gpus are free"
      announced=1
    fi
    sleep "$poll"
  done
}

# Start a detached tmux session running a command from PROJECT_ROOT.
# PATH / LD_LIBRARY_PATH are passed explicitly: an already-running tmux server would
# otherwise give the session its own (possibly wrong) conda environment.
tmux_launch() {
  local session="$1"
  shift
  if tmux has-session -t "=$session" 2>/dev/null; then
    echo "[ERROR] tmux session '$session' already exists (tmux attach -t $session)"
    exit 1
  fi
  local env_cmd="export PATH=$(printf '%q' "$PATH");"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    env_cmd+=" export LD_LIBRARY_PATH=$(printf '%q' "$LD_LIBRARY_PATH");"
  fi
  tmux new-session -d -s "$session" -c "$PROJECT_ROOT" "$env_cmd $(printf '%q ' "$@")"
  echo "[INFO] tmux session '$session' started (tmux attach -t $session)"
}

# Refuse to clobber an existing run unless FORCE=1.
assert_output_safe() {
  local outdir="$1"
  if [[ -d "$outdir" && "${FORCE:-0}" != "1" ]]; then
    echo "[ERROR] Output directory already exists: $outdir"
    echo "        Set FORCE=1 to reuse, or RESUME=<checkpoint> to continue."
    exit 1
  fi
}

write_run_manifest() {
  local outdir="$1"
  local config="$2"
  local run_name="$3"
  local seed="$4"
  local gpus="$5"
  shift 5
  local extra_args=("$@")

  mkdir -p "$outdir"
  cp "$config" "$outdir/config.yaml"

  local git_commit="unknown"
  if git -C "$PROJECT_ROOT" rev-parse HEAD &>/dev/null; then
    git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  fi

  local git_dirty="clean"
  if ! git -C "$PROJECT_ROOT" diff --quiet 2>/dev/null; then
    git_dirty="dirty"
  fi

  python - "$outdir" "$run_name" "$seed" "$gpus" "$git_commit" "$git_dirty" "${extra_args[@]}" <<'PY'
import json, sys, datetime, socket, os
outdir, run_name, seed, gpus, git_commit, git_dirty = sys.argv[1:7]
extra = sys.argv[7:]
manifest = {
    "run_name": run_name,
    "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "hostname": socket.gethostname(),
    "seed": int(seed),
    "gpus": gpus,
    "git_commit": git_commit,
    "git_status": git_dirty,
    "extra_args": extra,
    "output_dir": os.path.abspath(outdir),
    "notes": "timm also writes args.yaml and summary.csv inside output_dir",
}
with open(os.path.join(outdir, "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[INFO] Manifest: {outdir}/run_manifest.json")
PY
}
