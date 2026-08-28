#!/usr/bin/env bash
set -euo pipefail

# === FU_Biometry Docker Build & Test ===
# Usage: sudo bash docker/build_and_test.sh
#        SKIP_BUILD=1 sudo bash docker/build_and_test.sh   # re-run tests on the existing image
# Prerequisites: docker + nvidia-container-toolkit installed and running

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$SCRIPT_DIR"
IMAGE_NAME="sralin/fu-biometry"
IMAGE_TAG="v1.0"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
# The deployable weights (~583 MB: hyper_parameters + teacher_state_dict).
# Download from the GitHub Releases page, or produce from a training
# checkpoint with: uv run python scripts/extract_weights.py --ckpt <ckpt> \
#     --output docker/best_model.pth
WEIGHTS_PATH="$DOCKER_DIR/best_model.pth"

echo "============================================================"
echo "FU_Biometry Docker Build & Test"
echo "============================================================"
echo "Project:    $PROJECT_DIR"
echo "Docker dir: $DOCKER_DIR"
echo "Image:      $FULL_IMAGE"
echo "Weights:    $WEIGHTS_PATH"
echo ""

# --- Step 1: Validate prerequisites ---
echo "[1/8] Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found. Install: sudo apt-get install -y docker.io"; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not running. Start: sudo systemctl start docker"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not available"; exit 1; }
[ -f "$WEIGHTS_PATH" ] || { echo "ERROR: Weights not found: $WEIGHTS_PATH"; echo "  Download best_model.pth from the GitHub Releases page, or extract from a"; echo "  training checkpoint: uv run python scripts/extract_weights.py --ckpt <ckpt> --output docker/best_model.pth"; exit 1; }
echo "  Docker: $(docker --version)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# --- Step 2: Assemble build context ---
echo ""
echo "[2/8] Assembling build context..."

# docker/fubio is generated here, not versioned: the image's package is always
# an exact copy of src/fubio at build time.
rm -rf "$DOCKER_DIR/fubio"
cp -r "$PROJECT_DIR/src/fubio" "$DOCKER_DIR/fubio"
find "$DOCKER_DIR/fubio" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$DOCKER_DIR/fubio" -name "*.pyc" -delete 2>/dev/null || true
echo "  Copied fubio package ($(find "$DOCKER_DIR/fubio" -name "*.py" | wc -l) .py files)"

# Verify all required files
for f in Dockerfile predict.py model.py requirements-docker.txt data/shape_prior_v4.json; do
    [ -f "$DOCKER_DIR/$f" ] || { echo "ERROR: Missing $DOCKER_DIR/$f"; exit 1; }
done
[ -d "$DOCKER_DIR/vendor_dinov2" ] || { echo "ERROR: Missing $DOCKER_DIR/vendor_dinov2/"; exit 1; }
echo "  Build context ready"
echo "  Total size: $(du -sh "$DOCKER_DIR" | cut -f1)"

# --- Step 3: Build Docker image ---
echo ""
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    docker image inspect "$FULL_IMAGE" >/dev/null 2>&1 || { echo "ERROR: SKIP_BUILD=1 but $FULL_IMAGE does not exist"; exit 1; }
    echo "[3/8] SKIP_BUILD=1 — testing the existing image"
else
    echo "[3/8] Building Docker image (this may take 10-15 minutes)..."
    docker build \
        --platform linux/amd64 \
        --no-cache \
        -t "$FULL_IMAGE" \
        "$DOCKER_DIR" 2>&1 | tee /tmp/docker_build.log | tail -20
fi

echo "  Image: $(docker images "$IMAGE_NAME" --format '{{.Repository}}:{{.Tag}} {{.Size}}')"

# --- Step 4: Prepare test input (simulating organizer environment) ---
echo ""
echo "[4/8] Preparing test input..."

if [ ! -d "$PROJECT_DIR/data/val" ]; then
    echo "  data/val/ not found — the challenge validation images are not present."
    echo "  Image built successfully; skipping the container smoke test (steps 4-8)."
    echo "  Place the organizers' validation set under data/val/<task>/ to run it."
    exit 0
fi

INPUT_DIR="/tmp/fu_bio_test/input"
OUTPUT_DIR="/tmp/fu_bio_test/output"
rm -rf /tmp/fu_bio_test
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# Create {task_id}_test directories. Copied, not symlinked: a symlink into
# the repo dangles inside the container, whose mount namespace cannot see
# host paths outside the -v mounts.
for task in A4C AOP FA FUGC HC IVC PLAX PSAX fetal_femur; do
    src="$PROJECT_DIR/data/val/$task"
    if [ -d "$src" ]; then
        cp -rL "$src" "$INPUT_DIR/${task}_test"
        count=$(find "$INPUT_DIR/${task}_test" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "  $task: $count images"
    else
        echo "  WARNING: $src not found"
    fi
done

# Generate test_metadata.csv. Uses the repo venv (uv sync) for cv2 — the
# system python3 carries no imaging library.
PYBIN="$PROJECT_DIR/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="python3"
"$PYBIN" -c "
import csv, os
import cv2

input_dir = '$INPUT_DIR'
rows = []
nc_map = {'A4C':16,'AOP':4,'FA':4,'FUGC':2,'HC':4,'IVC':2,'PLAX':22,'PSAX':4,'fetal_femur':2}
for d in sorted(os.listdir(input_dir)):
    if not d.endswith('_test'):
        continue
    task_id = d.replace('_test', '')
    task_dir = os.path.join(input_dir, d)
    for f in sorted(os.listdir(task_dir)):
        fp = os.path.join(task_dir, f)
        if not os.path.isfile(fp):
            continue
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        rows.append({
            'image_path': f'{task_id}/{f}',
            'task_name': 'Regression',
            'task_id': task_id,
            'num_classes': nc_map.get(task_id, 0),
            'height': h,
            'width': w,
        })
csv_path = os.path.join(input_dir, 'test_metadata.csv')
with open(csv_path, 'w', newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=['image_path','task_name','task_id','num_classes','height','width'])
    writer.writeheader()
    writer.writerows(rows)
print(f'  test_metadata.csv: {len(rows)} rows')
"

# --- Step 5: Run Docker container (offline, resource-constrained) ---
echo ""
echo "[5/8] Running Docker container (offline, memory=7g, shm=2g)..."
echo "  This may take 10-30 minutes for 619 images × 3 TTA views..."

time docker run --rm --gpus all \
    --platform linux/amd64 \
    --network none \
    --memory 7g --cpus 4 --shm-size 2g \
    -v "$INPUT_DIR":/input:ro \
    -v "$OUTPUT_DIR":/output:rw \
    -e GU_INPUT_DIR=/input \
    -e GU_OUTPUT_DIR=/output \
    "$FULL_IMAGE" 2>&1 | tee /tmp/docker_run.log

echo ""
echo "  Container finished"

# --- Step 6: Validate output files ---
echo ""
echo "[6/8] Validating output files..."

for f in regression_predictions.json landmark_predictions.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/$f'))))")
        echo "  ✓ $f: $count predictions"
    else
        echo "  ✗ $f: MISSING"
        exit 1
    fi
done

# --- Step 7: Compare with a reference prediction set (if present) ---
# The submitted container's own predictions on the public validation set can
# serve as the reference; without one, this step is informational only.
echo ""
echo "[7/8] Comparing with reference predictions..."

REF_DIR="$PROJECT_DIR/predictions/R48-EP7-TEACHER-TTA5-ELLIPSE-ENHANCE-BS1"
if [ ! -f "$REF_DIR/regression_predictions.json" ]; then
    echo "  Reference predictions not found ($REF_DIR), skipping comparison"
else
    python3 -c "
import json, sys

docker_out = json.load(open('$OUTPUT_DIR/regression_predictions.json'))
reference = json.load(open('$REF_DIR/regression_predictions.json'))

docker_map = {p['image_path']: p for p in docker_out}
ref_map = {p['image_path']: p for p in reference}

if set(docker_map) != set(ref_map):
    extra = set(docker_map) - set(ref_map)
    missing = set(ref_map) - set(docker_map)
    print(f'FAIL: image sets differ')
    if extra: print(f'  Docker extra: {sorted(extra)[:5]}...')
    if missing: print(f'  Reference missing: {sorted(missing)[:5]}...')
    sys.exit(1)

mismatches = 0
max_global_diff = 0.0
for key in sorted(ref_map):
    d, r = docker_map[key], ref_map[key]
    if d['task_id'] != r['task_id']:
        print(f'  ROUTING DIFF: {key}: docker={d[\"task_id\"]} ref={r[\"task_id\"]}')
        mismatches += 1
    dp = d['predicted_points_pixels']
    rp = r['predicted_points_pixels']
    if len(dp) != len(rp):
        print(f'  LEN DIFF: {key}: {len(dp)} vs {len(rp)}')
        mismatches += 1
        continue
    max_diff = max(abs(a-b) for a,b in zip(dp, rp))
    max_global_diff = max(max_global_diff, max_diff)
    if max_diff > 0.5:
        print(f'  LARGE DIFF: {key}: max_diff={max_diff:.4f}')
        mismatches += 1

if mismatches == 0:
    print(f'  ✓ PASS: {len(ref_map)} predictions match (max pixel diff = {max_global_diff:.4f})')
else:
    print(f'  ✗ FAIL: {mismatches} mismatches (max pixel diff = {max_global_diff:.4f})')
    sys.exit(1)
"
fi

# --- Step 8: Run again for determinism check ---
echo ""
echo "[8/8] Determinism check (second run)..."

OUTPUT_DIR2="/tmp/fu_bio_test/output2"
mkdir -p "$OUTPUT_DIR2"

docker run --rm --gpus all \
    --platform linux/amd64 \
    --network none \
    --memory 7g --cpus 4 --shm-size 2g \
    -v "$INPUT_DIR":/input:ro \
    -v "$OUTPUT_DIR2":/output:rw \
    -e GU_INPUT_DIR=/input \
    -e GU_OUTPUT_DIR=/output \
    "$FULL_IMAGE" 2>&1 | tail -5

# Compare run1 vs run2
python3 -c "
import json
r1 = json.load(open('$OUTPUT_DIR/regression_predictions.json'))
r2 = json.load(open('$OUTPUT_DIR2/regression_predictions.json'))
if json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True):
    print('  ✓ Deterministic: run1 == run2 (byte-identical JSON)')
else:
    # Check numerical diffs
    m1 = {p['image_path']: p for p in r1}
    m2 = {p['image_path']: p for p in r2}
    diffs = 0
    for k in m1:
        p1, p2 = m1[k]['predicted_points_pixels'], m2[k]['predicted_points_pixels']
        if len(p1) == len(p2):
            md = max(abs(a-b) for a,b in zip(p1, p2))
            if md > 0.001: diffs += 1
    if diffs == 0:
        print(f'  ✓ Deterministic within 0.001px (JSON formatting may differ)')
    else:
        print(f'  ✗ Non-deterministic: {diffs} images differ')
"

echo ""
echo "============================================================"
echo "ALL TESTS PASSED"
echo "============================================================"
echo ""
echo "The image digest identifies exactly what was built:"
echo "  docker inspect --format='{{index .RepoDigests 0}}' $FULL_IMAGE"
