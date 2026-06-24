# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:58:04 2026

@author: FeFe
"""

import urllib.request
import os

os.makedirs(r"C:\Users\FeFe\open_vocab_nav\data\test_images", exist_ok=True)

# 20 images variées depuis COCO val2017 — objets du quotidien
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "http://images.cocodataset.org/val2017/000000174482.jpg",
    "http://images.cocodataset.org/val2017/000000403385.jpg",
    "http://images.cocodataset.org/val2017/000000006818.jpg",
    "http://images.cocodataset.org/val2017/000000480985.jpg",
    "http://images.cocodataset.org/val2017/000000458054.jpg",
    "http://images.cocodataset.org/val2017/000000331352.jpg",
    "http://images.cocodataset.org/val2017/000000579321.jpg",
    "http://images.cocodataset.org/val2017/000000226111.jpg",
    "http://images.cocodataset.org/val2017/000000153299.jpg",
    "http://images.cocodataset.org/val2017/000000010977.jpg",
    "http://images.cocodataset.org/val2017/000000119516.jpg",
    "http://images.cocodataset.org/val2017/000000418281.jpg",
    "http://images.cocodataset.org/val2017/000000541671.jpg",
    "http://images.cocodataset.org/val2017/000000092091.jpg",
    "http://images.cocodataset.org/val2017/000000314515.jpg",
]

folder = r"C:\Users\FeFe\open_vocab_nav\data\test_images"

for i, url in enumerate(urls):
    filename = f"coco_{i+1:02d}.jpg"
    path = os.path.join(folder, filename)
if not os.path.exists(path):
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved : {path}")
    except Exception as e:
        print(f"  Skipped {filename} : {e}")
else:
    print(f"  Already exists : {filename}")

print(f"\nDone! {len(urls)} images ready in {folder}")