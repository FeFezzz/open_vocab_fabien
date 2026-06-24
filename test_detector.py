"""
test_detector.py
----------------
Test script for OpenVocabularyDetector.


Make sure you have images in data/test_images/ before running.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Add project root to path so Python finds our modules ──────────────
PROJECT_ROOT = r"C:\Users\FeFe\open_vocab_nav"
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# ── Import our detector ───────────────────────────────────────────────
from perception.open_vocab_detector import OpenVocabularyDetector

# =============================================================================
# CONFIG — modify if needed
# =============================================================================

IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "test_images")

# Text queries for open-vocabulary search
TEXT_QUERIES = ["chair", "person", "bottle", "desk", "bag", "screen"]

# Camera intrinsics — placeholder values (replace with real ones later)
CAMERA_INTRINSICS = {
    "fx": 615.0,
    "fy": 615.0,
    "cx": 320.0,
    "cy": 240.0,
}

# =============================================================================
# LOAD DETECTOR
# =============================================================================

print("Loading detector...")
detector = OpenVocabularyDetector(yolo_model="yolov8n.pt")
print("Detector ready!\n")

# =============================================================================
# PROCESS EACH IMAGE
# =============================================================================

# Collect image paths
image_paths = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

if not image_paths:
    print(f"No images found in {IMAGE_FOLDER}")
    print("Please add some images and re-run.")
    sys.exit()

print(f"Found {len(image_paths)} image(s) to process.\n")

for image_path in image_paths:

    print("=" * 55)
    print(f"Processing : {os.path.basename(image_path)}")
    print("=" * 55)

    # ── Load image ────────────────────────────────────────────────────
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read image : {image_path}")
        continue

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Run detection ─────────────────────────────────────────────────
    detected_objects = detector.detect_objects(
        image_bgr,
        text_queries=TEXT_QUERIES,
        conf_threshold=0.25
    )

    if not detected_objects:
        print("No objects detected in this image.\n")
        continue

    # ── Print results in console ──────────────────────────────────────
    print(f"\nDetected {len(detected_objects)} object(s):\n")
    for i, obj in enumerate(detected_objects):
        print(f"  Object {i+1}:")
        print(f"    Label      : {obj['label']}")
        print(f"    Confidence : {obj['confidence']:.2f}")
        print(f"    BBox       : {[round(v) for v in obj['bbox']]}")
        print(f"    CLIP emb.  : shape {obj['clip_embedding'].shape}  "
              f"(first 3 values: {obj['clip_embedding'][:3].round(4)})")

        if obj['similarity']:
            # Show top 3 most similar text queries
            sorted_sim = sorted(
                obj['similarity'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            print(f"    Top CLIP similarities:")
            for query, score in sorted_sim:
                bar = "█" * int(score * 20)
                print(f"      '{query}' : {score:.4f}  {bar}")

        # project_to_3d test (no real depth image here — use zeros)
        depth_dummy = np.zeros(
            (image_bgr.shape[0], image_bgr.shape[1]), dtype=np.float32
        )
        centroid = detector.project_to_3d(
            obj["bbox"], depth_dummy, CAMERA_INTRINSICS
        )
        print(f"    Centroid 3D: {centroid} (zeros expected — no depth image)")
        print()

    # ── Visualise detections ──────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"{os.path.basename(image_path)} — {len(detected_objects)} detection(s)",
                 fontsize=13)
    ax.axis("off")

    colors = ["#378ADD", "#1D9E75", "#EF9F27", "#7F77DD",
              "#E24B4A", "#1D7A9E", "#9E1D75"]

    for i, obj in enumerate(detected_objects):
        x1, y1, x2, y2 = obj["bbox"]
        color = colors[i % len(colors)]

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Build label with top similarity if available
        label_text = f"{obj['label']} {obj['confidence']:.2f}"
        if obj["similarity"]:
            top_query, top_score = max(
                obj["similarity"].items(), key=lambda x: x[1]
            )
            label_text += f"\nCLIP: '{top_query}' {top_score:.2f}"

        # Draw label background
        ax.text(
            x1, y1 - 5, label_text,
            fontsize=8, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8)
        )

    plt.tight_layout()
    plt.show()
    print()

print("=" * 55)
print("Test complete!")
print("=" * 55)
