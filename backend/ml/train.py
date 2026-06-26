import os
import json
import torch
import kagglehub
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_LOC =  kagglehub.dataset_download("emmarex/plantdisease") + r"\PlantVillage"
EXCLUDE_FOLDERS = {"PlantVillage"}

class_names = sorted([fol for fol in os.listdir(DATA_LOC)
                      if os.path.isdir(os.path.join(DATA_LOC, fol)) and fol not in EXCLUDE_FOLDERS])

class_to_idx = {name: idx for idx, name in enumerate(class_names)}

samples = []
for cn in class_names:
    class_dir = os.path.join(DATA_LOC, cn)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((os.path.join(class_dir, fname), class_to_idx[cn]))

print(f"Found {len(samples)} images across {len(class_names)} classes.")

