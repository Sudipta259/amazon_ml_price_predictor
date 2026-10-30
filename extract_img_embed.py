import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running vision extraction on: {device}")

# 1. Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 2. PyTorch Dataset with missing-image fallback
class CatalogImageDataset(Dataset):
    def __init__(self, sample_ids, img_dir="data/images", transform=None):
        self.sample_ids = sample_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = str(self.sample_ids[idx])
        path = os.path.join(self.img_dir, f"{sid}.jpg")
        
        # Load image; if missing or corrupted, return a blank black image
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        else:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            img = self.transform(img)
        return img

# 3. Load ResNet-34 and strip the classification head
print("Loading pretrained ResNet-34...")
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = nn.Identity()  # Outputs 512-dim pooled vector directly
model = model.to(device)
model.eval()

# 4. Feature Extraction Function
def extract_embeddings(sample_ids, batch_size=128):
    dataset = CatalogImageDataset(sample_ids, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True if device == "cuda" else False
    )
    
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting visual features"):
            batch = batch.to(device, non_blocking=True)
            feats = model(batch)
            embeddings.append(feats.cpu().numpy())
            
    return np.vstack(embeddings)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("Loading IDs from processed tabular sets...")
    train_df = pd.read_parquet("data/processed/train_tabular.parquet")
    test_df = pd.read_parquet("data/processed/test_tabular.parquet")

    # Extract Train
    print(f"Extracting {len(train_df)} train image embeddings...")
    train_img_emb = extract_embeddings(train_df["sample_id"].values)
    np.save("data/processed/train_img_emb.npy", train_img_emb)
    print(f"Saved: data/processed/train_img_emb.npy {train_img_emb.shape}")

    # Extract Test
    print(f"Extracting {len(test_df)} test image embeddings...")
    test_img_emb = extract_embeddings(test_df["sample_id"].values)
    np.save("data/processed/test_img_emb.npy", test_img_emb)
    print(f"Saved: data/processed/test_img_emb.npy {test_img_emb.shape}")