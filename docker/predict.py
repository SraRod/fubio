#!/usr/bin/env python3
"""Docker entry — provided by organizers, DO NOT MODIFY."""

import os
import shutil
import sys

INPUT_DIR  = os.environ["GU_INPUT_DIR"]
OUTPUT_DIR = os.environ["GU_OUTPUT_DIR"]
WORK_DIR   = "/work"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_images():
    """Map platform directories (e.g. A4C_test) to metadata paths (e.g. A4C)."""
    images_dir = os.path.join(WORK_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    for name in os.listdir(INPUT_DIR):
        source = os.path.join(INPUT_DIR, name)
        if not os.path.isdir(source):
            continue
        task_name = name[:-5] if name.endswith("_test") else name
        target = os.path.join(images_dir, task_name)
        if not os.path.lexists(target):
            os.symlink(source, target)

os.makedirs(os.path.join(WORK_DIR, "csv"), exist_ok=True)
prepare_images()

shutil.copy2(
    os.path.join(INPUT_DIR, "test_metadata.csv"),
    os.path.join(WORK_DIR, "csv", "test_metadata.csv"),
)

# Copy weights from /app/ to working directory
shutil.copy2("/app/best_model.pth", os.path.join(WORK_DIR, "best_model.pth"))
os.chdir(WORK_DIR)

# Import your Model
sys.path.insert(0, "/app")
from model import Model

model = Model()
model.predict(data_root=WORK_DIR, output_dir=OUTPUT_DIR, batch_size=8)

print("Done!", flush=True)
