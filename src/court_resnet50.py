import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torchvision.models as models


class ResNetKeypointRegressor(nn.Module):
    def __init__(self, num_keypoints=14, backbone="resnet50", pretrained=False, dropout=0.25):
        super().__init__()
        self.num_kp = num_keypoints
        if backbone == "resnet50":
            # do not download weights by default (pretrained=False)
            self.backbone = models.resnet50(pretrained=pretrained)
            feat_dim = self.backbone.fc.in_features
        else:
            self.backbone = models.resnet50(pretrained=pretrained)
            feat_dim = self.backbone.fc.in_features
        # remove final fc
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_keypoints * 2),
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.head(f)
        out = torch.sigmoid(out)
        return out


def load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    # handle common checkpoint wrappers
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip 'module.' if present
    new_sd = {}
    try:
        for k, v in sd.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[nk] = v
    except Exception:
        # assume it's already a state_dict-like
        new_sd = sd

    # Try strict loading, fallback to non-strict
    try:
        model.load_state_dict(new_sd)
        return True
    except Exception as e:
        try:
            model.load_state_dict(new_sd, strict=False)
            print("Loaded checkpoint with strict=False (some keys missing or unexpected).")
            return True
        except Exception as e2:
            print("Failed to load state_dict:", e)
            print(e2)
            return False

def preprocess_image(img_instance, input_size):
    # Load image
    if isinstance(img_instance, str):
        img = Image.open(img_instance).convert("RGB")
        orig_w, orig_h = img.size
    elif isinstance(img_instance, np.ndarray):
        # OpenCV frame: convert BGR -> RGB, then to PIL Image
        img = Image.fromarray(cv2.cvtColor(img_instance, cv2.COLOR_BGR2RGB))
        orig_h, orig_w = img_instance.shape[:2]
    elif isinstance(img_instance, Image.Image):
        img = img_instance.convert("RGB")
        orig_w, orig_h = img.size
    else:
        raise TypeError("img_instance must be str, np.ndarray or PIL.Image.Image")

    tf = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img_t = tf(img)
    return img_t, (orig_w, orig_h)



def predict_keypoints(model, img_tensor, orig_size, device):
    model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        out = model(x)
        out_np = out.cpu().numpy()[0]
        # expected length = num_keypoints * 2
        coords = []
        w, h = orig_size
        for k in range(len(out_np) // 2):
            nx = float(np.clip(out_np[2 * k], 0.0, 1.0))
            ny = float(np.clip(out_np[2 * k + 1], 0.0, 1.0))
            px = nx * w
            py = ny * h
            coords.append((float(px), float(py)))
    return coords
