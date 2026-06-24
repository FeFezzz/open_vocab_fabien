"""
demo_detector.py
----------------
Student 1 — Perception Demo Script
Open-Vocabulary Object Detection with YOLO + CLIP

Demonstration script for Friday presentation.
Displays detection results with large readable visualisations
suitable for projector display.

Run with F5 in Spyder.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ── Project path setup ────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\FeFe\open_vocab_nav"
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from perception.open_vocab_detector import OpenVocabularyDetector

# =============================================================================
# CONFIG
# =============================================================================

IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "test_images")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "results", "visualizations")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Text queries for open-vocabulary search
# Adapt these to match the objects visible in your test images
TEXT_QUERIES = [
    "chair", "table", "couch", "sofa",
    "person", "car", "bicycle", "potted plant",
    "cup", "bowl", "sink", "tv"
]

# Camera intrinsics placeholder
CAMERA_INTRINSICS = {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0}

# Colour palette for bounding boxes
COLORS = [
    "#378ADD", "#1D9E75", "#EF9F27", "#7F77DD",
    "#E24B4A", "#1D7A9E", "#9E751D", "#75991D"
]

# =============================================================================
# LOAD DETECTOR
# =============================================================================

print("=" * 60)
print("  Open-Vocabulary Detector — Presentation Demo")
print("=" * 60)
print()
print("Loading models (YOLO + CLIP)...")
detector = OpenVocabularyDetector(yolo_model="yolov8n.pt")
print("Models ready!\n")

# =============================================================================
# COLLECT IMAGES
# =============================================================================

image_paths = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
])

if not image_paths:
    print(f"No images found in {IMAGE_FOLDER}")
    print("Please add images and re-run.")
    sys.exit()

print(f"Found {len(image_paths)} image(s).\n")

# =============================================================================
# PROCESS AND VISUALISE EACH IMAGE
# =============================================================================

for img_idx, image_path in enumerate(image_paths):

    filename = os.path.basename(image_path)
    print(f"Processing [{img_idx+1}/{len(image_paths)}] : {filename}")

    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  Could not read {filename}, skipping.")
        continue
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Run detection
    detected_objects = detector.detect_objects(
        image_bgr,
        text_queries=TEXT_QUERIES,
        conf_threshold=0.25
    )

    if not detected_objects:
        print(f"  No objects detected in {filename}.\n")
        continue

    print(f"  {len(detected_objects)} object(s) detected.")

    # ── Build annotated image ─────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("#1A1A1A")

    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1], wspace=0.04)

    # Left panel — annotated image
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image_rgb)
    ax_img.set_title(
        f"YOLO + CLIP Detection   |   {len(detected_objects)} object(s) found",
        fontsize=14, color="white", pad=12, fontweight="bold"
    )
    ax_img.axis("off")
    ax_img.set_facecolor("#1A1A1A")

    # Draw bounding boxes
    for i, obj in enumerate(detected_objects):
        x1, y1, x2, y2 = obj["bbox"]
        color = COLORS[i % len(COLORS)]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none"
        )
        ax_img.add_patch(rect)

        label_text = f"{obj['label']}  {obj['confidence']:.0%}"
        ax_img.text(
            x1 + 4, y1 + 4, label_text,
            fontsize=9, color="white", fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85)
        )

    # Right panel — CLIP similarity table
    ax_table = fig.add_subplot(gs[1])
    ax_table.set_facecolor("#1A1A1A")
    ax_table.axis("off")
    ax_table.set_title(
        "CLIP Open-Vocabulary Similarity",
        fontsize=13, color="white", pad=12, fontweight="bold"
    )

    y_pos = 0.97
    line_height = 0.97 / max(len(detected_objects), 1)

    for i, obj in enumerate(detected_objects):
        color = COLORS[i % len(COLORS)]

        # Object header
        ax_table.text(
            0.02, y_pos,
            f"  {obj['label'].upper()}  ({obj['confidence']:.0%})",
            transform=ax_table.transAxes,
            fontsize=10, color=color, fontweight="bold",
            va="top"
        )
        y_pos -= 0.04

        if obj["similarity"]:
            # Sort by score descending, show top 4
            sorted_sim = sorted(
                obj["similarity"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:4]

            for query, score in sorted_sim:
                # Bar chart inline
                bar_width = score * 0.55
                ax_table.barh(
                    y_pos - 0.01,
                    bar_width,
                    height=0.028,
                    left=0.28,
                    color=color,
                    alpha=0.7,
                    transform=ax_table.transAxes
                )
                ax_table.text(
                    0.03, y_pos,
                    f"{query}",
                    transform=ax_table.transAxes,
                    fontsize=8.5, color="#CCCCCC", va="top"
                )
                ax_table.text(
                    0.85, y_pos,
                    f"{score:.3f}",
                    transform=ax_table.transAxes,
                    fontsize=8.5, color="white", va="top", fontweight="bold"
                )
                y_pos -= 0.038

        y_pos -= 0.018

        # Stop if we run out of space
        if y_pos < 0.02:
            break

    plt.suptitle(
        f"Open-Vocabulary 3D Semantic Mapping   |   Student 1 — Perception",
        fontsize=11, color="#888888", y=0.01
    )

    plt.tight_layout()

    # Save to results/visualizations/
    save_path = os.path.join(
        OUTPUT_FOLDER, f"demo_{os.path.splitext(filename)[0]}.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#1A1A1A")
    print(f"  Saved : {save_path}")
    plt.show()
    print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("  Demo complete!")
print(f"  Visualisations saved in : {OUTPUT_FOLDER}")
print("=" * 60)
