"""
benchmark_spyder.py
-------------------
Version adaptée pour Spyder 6 / Python 3.12.
Configurez les paramètres dans la section CONFIG ci-dessous,
puis appuyez sur F5 (Run) ou Run > Run File.
"""

# =============================================================================
# CONFIG — modifiez ces valeurs selon votre machine
# =============================================================================

 
IMAGE_FOLDER = "C:/Users/FeFe/open_vocab_nav/data/test_images" # dossier avec vos images
MAX_IMAGES   = 50          # nombre d'images à tester (None = toutes)
CONF         = 0.25        # seuil de confiance (0.0 → 1.0)
IMG_SIZE     = 640         # taille d'entrée du modèle (pixels)
OUTPUT_DIR   = "results"   # dossier de sortie pour CSV et graphiques

# Modèles à comparer — commentez/décommentez selon vos besoins
MODELS_TO_RUN = [
    "yolov8n",        # YOLOv8 Nano     — le plus rapide, le moins précis
    "yolov8s",        # YOLOv8 Small    — bon équilibre
    "yolov8m",        # YOLOv8 Medium   — plus précis
    # "yolov8l",      # YOLOv8 Large    — décommentez si GPU puissant
    # "yolov8x",      # YOLOv8 XLarge   — le plus précis YOLOv8
    "yolov9c",        # YOLOv9
    "yolov8s-worldv2",  # YOLO World S  — open-vocabulary
    # "yolov8m-worldv2", # YOLO World M — décommentez si GPU puissant
    "yolo11n",    
    "yolo11s"
]

# Texte libre pour YOLO World (les objets qu'il doit chercher)
YOLO_WORLD_CLASSES = [
    "person", "chair", "table", "door", "window",
    "computer", "bookshelf", "fire extinguisher",
]

# =============================================================================
# IMPORTS
# =============================================================================

import os
import time
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# GPU monitoring (optionnel — pas obligatoire)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
    print("pynvml OK — mesure GPU activée")
except Exception:
    NVML_OK = False
    GPU_HANDLE = None
    print("pynvml absent — GPU watts/VRAM ne seront pas mesurés")
    print("  → pip install pynvml  pour activer")

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_gpu_stats():
    """Lit VRAM utilisée (MB) et puissance GPU (W). Retourne 0 si indisponible."""
    if not NVML_OK:
        return 0.0, 0.0
    mem   = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
    power = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0  # mW → W
    return mem.used / 1024**2, power


def collect_images(folder: str, max_images: int | None) -> list[Path]:
    """Récupère tous les fichiers image d'un dossier."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(Path(folder).rglob("*"))
    paths = [p for p in paths if p.suffix.lower() in exts]
    if max_images:
        paths = paths[:max_images]
    print(f"\nImages trouvées : {len(paths)} dans '{folder}'")
    return paths


def warmup(model: YOLO, n: int = 3):
    """Quelques inférences factices pour stabiliser les horloges GPU."""
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for _ in range(n):
        model.predict(dummy, verbose=False)


# =============================================================================
# BENCHMARK D'UN MODÈLE
# =============================================================================

def run_one_model(model_name: str, image_paths: list[Path]) -> dict:
    """
    Charge le modèle, fait l'inférence sur toutes les images,
    retourne un dictionnaire avec les métriques agrégées.
    """
    print(f"\n{'─'*50}")
    print(f"  Modèle : {model_name}")
    print(f"{'─'*50}")

    # Chargement
    model = YOLO(model_name + ".pt" if not model_name.endswith(".pt") else model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # YOLO World : définir les classes textuelles
    is_world = "world" in model_name.lower()
    if is_world:
        model.set_classes(YOLO_WORLD_CLASSES)
        print(f"  Classes YOLO World : {YOLO_WORLD_CLASSES}")

    warmup(model)

    latencies, vrams, powers, detections = [], [], [], []

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Mesure GPU avant inférence
        vram_before, _ = get_gpu_stats()

        # Inférence
        t0 = time.perf_counter()
        preds = model.predict(img, conf=CONF, imgsz=IMG_SIZE, verbose=False)
        t1 = time.perf_counter()

        # Mesure GPU après inférence
        vram_after, power_w = get_gpu_stats()

        latency_ms = (t1 - t0) * 1000.0
        num_det = len(preds[0].boxes) if preds and preds[0].boxes is not None else 0

        latencies.append(latency_ms)
        vrams.append(vram_after)
        powers.append(power_w)
        detections.append(num_det)

        # Progression
        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] "
                  f"lat={latency_ms:.1f}ms  "
                  f"det={num_det}  "
                  f"vram={vram_after:.0f}MB")

    # Libération mémoire GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Résumé
    result = {
        "model":          model_name,
        "fps":            round(1000.0 / np.mean(latencies), 1),
        "lat_mean_ms":    round(float(np.mean(latencies)), 2),
        "lat_std_ms":     round(float(np.std(latencies)), 2),
        "vram_mb":        round(float(np.mean(vrams)), 0),
        "power_w":        round(float(np.mean(powers)), 1),
        "detections_avg": round(float(np.mean(detections)), 1),
        "n_images":       len(latencies),
    }

    print(f"\n  → FPS={result['fps']}  "
          f"Latence={result['lat_mean_ms']}ms  "
          f"VRAM={result['vram_mb']}MB  "
          f"Puissance={result['power_w']}W")

    return result


# =============================================================================
# GRAPHIQUES
# =============================================================================

COLORS = {
    "yolov8n":          "#378ADD",
    "yolov8s":          "#185FA5",
    "yolov8m":          "#0C447C",
    "yolov8l":          "#042C53",
    "yolov8x":          "#021A33",
    "yolov8s-worldv2":  "#7F77DD",
    "yolov8m-worldv2":  "#534AB7",
}

def _bar_color(name):
    for key, col in COLORS.items():
        if key in name:
            return col
    return "#888780"


def make_charts(df: pd.DataFrame, out_dir: str):
    """Génère un dashboard 2x3 et des graphiques individuels."""
    os.makedirs(out_dir, exist_ok=True)
    df_sorted = df.sort_values("fps", ascending=True).reset_index(drop=True)
    colors = [_bar_color(m) for m in df_sorted["model"]]

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("YOLO Comparison — Performance & Energy",
                 fontsize=14, fontweight="bold")

    # 1. FPS
    ax = axes[0, 0]
    bars = ax.barh(df_sorted["model"], df_sorted["fps"], color=colors, height=0.55)
    ax.set_xlabel("FPS (frames/second)")
    ax.set_title("Inference Speed")
    for b, v in zip(bars, df_sorted["fps"]):
        ax.text(v + 0.5, b.get_y() + b.get_height()/2,
                f"{v:.1f}", va="center", fontsize=9)

    # 2. Latence ± écart-type
    ax = axes[0, 1]
    bars = ax.barh(df_sorted["model"], df_sorted["lat_mean_ms"],
                   xerr=df_sorted["lat_std_ms"],
                   color=colors, height=0.55, capsize=4)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_title("Latency ± std dev")
    for b, v in zip(bars, df_sorted["lat_mean_ms"]):
        ax.text(v + 0.3, b.get_y() + b.get_height()/2,
                f"{v:.1f} ms", va="center", fontsize=9)

    # 3. VRAM
    ax = axes[0, 2]
    bars = ax.barh(df_sorted["model"], df_sorted["vram_mb"], color=colors, height=0.55)
    ax.set_xlabel("VRAM used (MB)")
    ax.set_title("GPU Memory")
    for b, v in zip(bars, df_sorted["vram_mb"]):
        ax.text(v + 5, b.get_y() + b.get_height()/2,
                f"{v:.0f} MB", va="center", fontsize=9)

    # 4. Puissance
    ax = axes[1, 0]
    bars = ax.barh(df_sorted["model"], df_sorted["power_w"], color=colors, height=0.55)
    ax.set_xlabel("Average GPU power (W)")
    ax.set_title("Energy Consumption")
    for b, v in zip(bars, df_sorted["power_w"]):
        ax.text(v + 0.2, b.get_y() + b.get_height()/2,
                f"{v:.1f} W", va="center", fontsize=9)

    # 5. Scatter FPS vs Puissance
    ax = axes[1, 1]
    for _, row in df_sorted.iterrows():
        c = _bar_color(row["model"])
        ax.scatter(row["power_w"], row["fps"], color=c, s=120, zorder=3)
        ax.annotate(row["model"], xy=(row["power_w"], row["fps"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=8, color=c)
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("FPS")
    ax.set_title("Energy Efficiency (top-left = best)")

    # 6. Scatter FPS vs VRAM
    ax = axes[1, 2]
    for _, row in df_sorted.iterrows():
        c = _bar_color(row["model"])
        ax.scatter(row["vram_mb"], row["fps"], color=c, s=120, zorder=3)
        ax.annotate(row["model"], xy=(row["vram_mb"], row["fps"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=8, color=c)
    ax.set_xlabel("VRAM (MB)")
    ax.set_ylabel("FPS")
    ax.set_title("Speed vs Memory (top-left = best)")

    plt.tight_layout()
    path = os.path.join(out_dir, "dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGraphique sauvegardé → {path}")


# =============================================================================
# PROGRAMME PRINCIPAL — s'exécute quand vous appuyez sur F5 dans Spyder
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("  YOLO Benchmark — Spyder 6 / Python 3.12")
    print("=" * 55)
    print(f"CUDA disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Modèles testés  : {MODELS_TO_RUN}")

    # 1. Charger les images
    images = collect_images(IMAGE_FOLDER, MAX_IMAGES)
    if not images:
        raise FileNotFoundError(
            f"Aucune image trouvée dans : {IMAGE_FOLDER}\n"
            "Vérifiez le chemin dans la section CONFIG en haut du script."
        )

    # 2. Benchmarker chaque modèle
    all_results = []
    for model_name in MODELS_TO_RUN:
        result = run_one_model(model_name, images)
        all_results.append(result)

    # 3. Construire le tableau
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 65)
    print(df[["model", "fps", "lat_mean_ms", "vram_mb", "power_w"]].to_string(index=False))
    print("=" * 65)

    # 4. Sauvegarder le CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "yolo_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats CSV → {csv_path}")

    # 5. Générer les graphiques (s'affichent dans Spyder Plots)
    make_charts(df, OUTPUT_DIR)

    print("\nBenchmark terminé !")
