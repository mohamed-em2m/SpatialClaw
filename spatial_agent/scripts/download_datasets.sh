#!/usr/bin/env bash
# download_datasets.sh — Download all SpatialAgent evaluation datasets from HuggingFace.
#
# Usage:
#   bash download_datasets.sh <HF_TOKEN>
#   bash download_datasets.sh hf_XXXXXXXXXXXXXXXXXXXX
#
# Datasets are downloaded into data/ relative to this script's parent directory
# (i.e. spatialagent/data/). Zip/tar archives are extracted automatically.
#
# Requires: conda env "spatialagent" with huggingface_hub installed.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <HUGGINGFACE_TOKEN>"
    exit 1
fi

HF_TOKEN="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"

mkdir -p "$DATA_DIR"
echo "=== Downloading datasets to: $DATA_DIR ==="

# Maximum parallel downloads (adjust based on bandwidth/disk IO)
MAX_PARALLEL=${MAX_PARALLEL:-4}

# ─── Dataset registry ────────────────────────────────────────────────────────
# Format: "HF_REPO_ID|TARGET_FOLDER_NAME"
# TARGET_FOLDER_NAME must match factory.py data_dir values exactly.
DATASETS=(
    "shijiezhou/VLM4D|vlm4d"
    "rbler/MMSI-Video-Bench|MMSI-Video-Bench"
    "dmarsili/Omni3D-Bench|Omni3D-Bench"
    "MLL-Lab/MindCube|MindCube"
    "RunsenXu/MMSI-Bench|MMSI-Bench"
    "nyu-visionx/VSI-Bench|VSI-Bench"
    "jasonzhango/SPAR-Bench|SPAR-Bench"
    "BLINK-Benchmark/BLINK|BLINK"
    "HarmlessSR07/OSI-Bench|OSI-Bench"
    "Journey9ni/vstibench|vstibench"
    "qizekun/OmniSpatial|OmniSpatial"
    "array/SAT|SAT-v2"
    "PaulineLi/QuantiPhy-validation|QuantiPhy"
    "shi-labs/physical-ai-bench-understanding|PAI-Bench"
    "turing-motors/STRIDE-QA-Bench|STRIDE-QA-Bench"
    "LongfeiLi/SpatialTree-Bench|SpatialTree-Bench"
    "TencentARC/DSR_Suite-Data|DSR-Bench"
    "yu2hi13/Dyn-Bench|Dyn-Bench-new"
    "hongxingli/SPBench|SPBench"
    "lmms-lab/Video-MME|Video-MME"
    "MME-Benchmarks/Video-MME-v2|Video-MME-v2"
    "chanhee-luke/RoboSpatial-Home|RoboSpatial-Home"
    "hrinnnn/PerceptionComp|PerceptionComp"
    "Oliver-Ma/Real-3DQA|Real-3DQA"
    "FlagEval/ERQA|ERQA"
    "Alibaba-DAMO-Academy/RynnBrain-Bench|RynnBrain-Bench"
)

# ─── Helper: download one dataset ────────────────────────────────────────────
download_one() {
    local repo="$1"
    local folder="$2"
    local target="$DATA_DIR/$folder"

    if [[ -d "$target" ]] && [[ "$(find "$target" -type f 2>/dev/null | head -1)" != "" ]]; then
        echo "[SKIP] $folder — already exists at $target"
        return 0
    fi

    echo "[DOWN] $folder ← $repo"
    conda run --no-banner -n spatialagent \
        hf download --repo-type dataset --token "$HF_TOKEN" \
        "$repo" --local-dir "$target" \
        2>&1 | tail -3

    if [[ $? -ne 0 ]]; then
        echo "[FAIL] $folder — download failed"
        return 1
    fi
    echo "[DONE] $folder downloaded"
}

# ─── Helper: extract all archives inside a directory ─────────────────────────
extract_archives() {
    local dir="$1"
    local name="$(basename "$dir")"

    # Extract .zip files
    find "$dir" -maxdepth 3 -name "*.zip" -print0 2>/dev/null | while IFS= read -r -d '' zipfile; do
        local parent="$(dirname "$zipfile")"
        echo "[UNZIP] $name: $(basename "$zipfile")"
        unzip -o -q "$zipfile" -d "$parent" && rm "$zipfile"
    done

    # Extract .tar.gz files
    find "$dir" -maxdepth 3 -name "*.tar.gz" -print0 2>/dev/null | while IFS= read -r -d '' tarfile; do
        local parent="$(dirname "$tarfile")"
        echo "[UNTAR] $name: $(basename "$tarfile")"
        tar xzf "$tarfile" -C "$parent" && rm "$tarfile"
    done

    # Extract .tar files
    find "$dir" -maxdepth 3 -name "*.tar" -print0 2>/dev/null | while IFS= read -r -d '' tarfile; do
        local parent="$(dirname "$tarfile")"
        echo "[UNTAR] $name: $(basename "$tarfile")"
        tar xf "$tarfile" -C "$parent" && rm "$tarfile"
    done
}

# ─── Helper: post-download fixups for datasets with non-standard layouts ─────
fixup_dataset() {
    local folder="$1"
    local target="$DATA_DIR/$folder"

    case "$folder" in
        SAT-v2)
            # Code expects data/{split}-*.parquet but HF has SAT_{split}.parquet at top level
            if [[ -f "$target/SAT_val.parquet" ]]; then
                echo "[FIX]  $folder: moving parquets into data/ subdirectory"
                mkdir -p "$target/data"
                for split in val test train static; do
                    local src="$target/SAT_${split}.parquet"
                    local dst="$target/data/${split}-00000-of-00001.parquet"
                    [[ -f "$src" ]] && mv "$src" "$dst"
                done
            fi
            ;;
    esac
}

# ─── Main: download all datasets with parallelism ────────────────────────────
echo ""
echo "Downloading ${#DATASETS[@]} datasets (max $MAX_PARALLEL parallel)..."
echo ""

running=0
for entry in "${DATASETS[@]}"; do
    repo="${entry%%|*}"
    folder="${entry##*|}"

    # Run download in background
    (
        download_one "$repo" "$folder"
        extract_archives "$DATA_DIR/$folder"
        fixup_dataset "$folder"
    ) &

    running=$((running + 1))
    if [[ $running -ge $MAX_PARALLEL ]]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done

# Wait for all remaining background jobs
wait

# ─── Final report ────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Dataset Download Report"
echo "========================================="

total=0
done_count=0
missing_count=0

for entry in "${DATASETS[@]}"; do
    folder="${entry##*|}"
    target="$DATA_DIR/$folder"
    total=$((total + 1))

    if [[ -d "$target" ]]; then
        fcount=$(find "$target" -type f 2>/dev/null | wc -l)
        zcount=$(find "$target" -maxdepth 3 \( -name "*.zip" -o -name "*.tar.gz" \) 2>/dev/null | wc -l)
        if [[ $zcount -gt 0 ]]; then
            echo "  WARN  $folder ($fcount files, $zcount archives not extracted)"
        else
            echo "  OK    $folder ($fcount files)"
        fi
        done_count=$((done_count + 1))
    else
        echo "  MISS  $folder"
        missing_count=$((missing_count + 1))
    fi
done

echo "========================================="
echo "  Total: $total | Downloaded: $done_count | Missing: $missing_count"
echo "========================================="
echo ""
echo "Note: 4DP-QA-Bench is not included (not publicly available on HuggingFace)."
echo "Done."
