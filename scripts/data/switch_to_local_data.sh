#!/usr/bin/env bash
# Verify a local ImageNet copy against the NAS folders, then atomically repoint the training
# data symlinks (<data_dir>/train, <data_dir>/val) at it. Running jobs read the local copy from
# their next file open: timm keeps the symlinked paths, so no restart is needed.
# Nothing is switched if the file lists differ or a sampled file's content differs.
#
# Usage:
#   bash scripts/data/switch_to_local_data.sh <local_root> <nas_train_dir> <nas_val_dir> [data_dir]
#     local_root  directory holding train/ and val/ (e.g. /mnt/d/bench/imagenet)
#     data_dir    folder with the train/val symlinks used as data_dir (default <project>/data/imagenet)
#
# Example (147):
#   bash scripts/data/switch_to_local_data.sh /mnt/d/bench/imagenet \
#     /mnt/a100b/default/wangjin/data/imagenet/train /mnt/a100b/default/wujiannan/data/imagenet-1k/val
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$1"
NAS_TRAIN="$2"
NAS_VAL="$3"
DATA_DIR="${4:-$PROJECT_ROOT/data/imagenet}"
LISTS="$PROJECT_ROOT/logs/setup"
mkdir -p "$LISTS"

for split in train val; do
  nas="$NAS_TRAIN"
  [[ $split == val ]] && nas="$NAS_VAL"
  (cd "$nas" && find . -type f | sort) > "$LISTS/nas_${split}_list.txt"
  (cd "$LOCAL/$split" && find . -type f | sort) > "$LISTS/local_${split}_list.txt"
  echo "$split: NAS $(wc -l < "$LISTS/nas_${split}_list.txt") files / local $(wc -l < "$LISTS/local_${split}_list.txt") files"
  if ! cmp -s "$LISTS/nas_${split}_list.txt" "$LISTS/local_${split}_list.txt"; then
    echo "[ERROR] $split file lists differ; not switching"
    exit 1
  fi
  while read -r f; do
    if ! cmp -s "$nas/$f" "$LOCAL/$split/$f"; then
      echo "[ERROR] content differs: $split/$f; not switching"
      exit 1
    fi
  done < <(shuf -n 200 "$LISTS/local_${split}_list.txt")
done
echo "file lists identical, 400 sampled files identical"

cd "$DATA_DIR"
for split in train val; do
  ln -s "$LOCAL/$split" "$split.local"
  mv -T "$split.local" "$split"
done
echo "[$(date '+%F %T')] switched $DATA_DIR to $LOCAL"
ls -l "$DATA_DIR"
