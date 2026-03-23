import torch
from PIL import Image
import cv2
import numpy as np
import sys
import os
import torch.nn.functional as F

from patchcore.feature_extractor import ResNetFeatureExtractor
from patchcore.anomaly_scoring import compute_anomaly_score

from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model once
extractor = ResNetFeatureExtractor().to(device)
extractor.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


def load_memory_bank(category):

    path = f"models/{category}_memory.pt"
    bank = torch.load(path, map_location=device)

    return bank


def extract_patches(feature_map):

    B, C, H, W = feature_map.shape
    patches = feature_map.permute(0,2,3,1)
    patches = patches.reshape(-1, C)

    return patches



def run_patchcore(category, image_path):

    # 1. Load memory bank
    bank = load_memory_bank(category)

    # 2. Load image
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 3. Extract features
    with torch.no_grad():
        feat2, feat3 = extractor(img_tensor)

    # 4. Align feature maps
    feat3 = F.interpolate(
        feat3,
        size=feat2.shape[2:],
        mode="bilinear",
        align_corners=False
    )

    # 5. Extract patches
    patches2 = extract_patches(feat2)
    patches3 = extract_patches(feat3)

    patches = torch.cat([patches2, patches3], dim=1)

    patches = F.normalize(patches, p=2, dim=1)
    bank = F.normalize(bank, p=2, dim = 1)

    # ✅ 6. NEW scoring (IMPORTANT)
    min_distances = compute_anomaly_score(patches, bank)

    # ✅ 7. Fix spatial mapping (IMPORTANT)
    B, C, H, W = feat2.shape
    # directly reshape (since only H*W patches exist)
    patch_map = min_distances.reshape(H, W).detach().cpu().numpy()

    # 8. Normalize safely
    if patch_map.max() > patch_map.min():
        patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min())
    else:
        patch_map = np.zeros_like(patch_map)

    # 9. Smooth
    patch_map = cv2.GaussianBlur(patch_map, (5, 5), 0)

    # 10. Resize
    heatmap = cv2.resize(patch_map, (256, 256))

    # 11. Return final score
    k = int(0.02 * min_distances.shape[0])

    if k < 1:
        k = 1

    topk_vals, _ = torch.topk(min_distances, k)
    final_score = topk_vals.mean().item()
    return heatmap, final_score