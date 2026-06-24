"""
train_yolo11.py
---------------
Student 1 — YOLO11 Training Script on EnvoDat (mu-hall)

Windows-safe version with if __name__ == '__main__' guard.
"""

from ultralytics import YOLO
import os

# =============================================================================
# CONFIG — modify before running
# =============================================================================

DATA_YAML = r"C:\Users\FeFe\open_vocab_nav\data\envodat\envodata-mu-hall.yaml"
PROJECT_DIR = r"C:\Users\FeFe\open_vocab_nav\results\yolo11_training"

# Models to train
MODELS = [
    ("yolo11s.pt", "yolo11s_20ep"),
    #("yolo11n.pt", "yolo11n_10ep"),
    #("yolo11s.pt", "yolo11s_10ep"),
    #("yolo11m.pt", "yolo11m_10ep"),
    #("yolo11n.pt", "yolo11n_50ep"),
    #("yolo11s.pt", "yolo11s_50ep"),
    #("yolo11n.pt", "yolo11n_full"),
    #("yolo11s.pt", "yolo11s_full"),
    #("yolo11n.pt", "yolo11n_test"),    # quick test
    # ("yolo11s.pt", "yolo11s_envodat"),  # uncomment for full run
]

EPOCHS     = 20     # 20 for quick test, 100 for full training, 10 global running
IMG_SIZE   = 640
BATCH_SIZE = 16
DEVICE     = 0      # 0 = GPU, 'cpu' = CPU
WORKERS    = 0      # 0 avoids Windows multiprocessing issues

# =============================================================================
# TRAINING — protected by main guard (required on Windows)
# =============================================================================

def main():
    for weights, run_name in MODELS:

        print(f"\n{'='*55}")
        print(f"  Training : {run_name}")
        print(f"  Weights  : {weights}")
        print(f"  Epochs   : {EPOCHS}")
        print(f"{'='*55}\n")

        model = YOLO(weights)

        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=run_name,
            save_period=10,
            patience=30,
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            verbose=True,
        )

        print(f"\n  Training complete : {run_name}")

        # Validation
        print(f"\n  Validation on val split...")
        val_metrics = model.val(split='val')
        print(f"  mAP50    : {val_metrics.box.map50:.4f}")
        print(f"  mAP50-95 : {val_metrics.box.map:.4f}")
        print(f"  Precision: {val_metrics.box.mp:.4f}")
        print(f"  Recall   : {val_metrics.box.mr:.4f}")

        # Test
        print(f"\n  Evaluation on test split...")
        test_metrics = model.val(split='test')
        print(f"  mAP50    : {test_metrics.box.map50:.4f}")
        print(f"  mAP50-95 : {test_metrics.box.map:.4f}")
        print(f"  Precision: {test_metrics.box.mp:.4f}")
        print(f"  Recall   : {test_metrics.box.mr:.4f}")

    print(f"\n{'='*55}")
    print("  All trainings complete!")
    print(f"  Results in : {PROJECT_DIR}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()