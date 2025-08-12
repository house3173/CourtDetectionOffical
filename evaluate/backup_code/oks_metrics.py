import json
import math
import numpy as np
import os
from typing import List, Optional, Tuple

def oks_per_instance(
    gt_kps: List[List[float]],
    pred_kps: List[List[float]],
    area: float,
    # sigmas: Optional[np.ndarray] = np.array([0.082, 0.083, 0.074, 0.075, 0.076, 0.078, 0.066, 0.060, 0.074, 0.077, 0.072, 0.081, 0.057, 0.045]),
    # sigmas: Optional[np.ndarray] = np.array([0.075, 0.075, 0.070, 0.070, 0.075, 0.075, 0.060, 0.060, 0.075, 0.075, 0.065, 0.065, 0.057, 0.045]),
    sigmas: Optional[np.ndarray] = np.array([0.075, 0.075, 0.060, 0.060, 0.0725, 0.0725, 0.0575, 0.0575, 0.07, 0.07, 0.055, 0.055, 0.055, 0.04]),
    # sigmas: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5
) -> float:
    """
    Compute OKS for a single instance (ground-truth vs prediction).

    Args:
        gt_kps: list of K keypoints: [ [x,y] ] or [ [x,y,v] ]   (v: visibility 0/1/2)
        pred_kps: list of K keypoints: [ [x,y] ] or [ [x,y,score] ]
        area: object area in pixels (if 0 or None, will fallback to 1.0)
        sigmas: np.array of length K (per-keypoint sigma). If None, use default small constant.
        confidence_threshold: if prediction score < threshold -> treat that keypoint as missing (KS=0)

    Returns:
        oks (float) in [0,1]
    """
    gt = np.asarray(gt_kps, dtype=float)
    dt = np.asarray(pred_kps, dtype=float)

    K = max(gt.shape[0], dt.shape[0])
    # ensure shapes
    if gt.shape[0] != K:
        raise ValueError("gt_kps and pred_kps must have same number of keypoints (or be compatible).")

    # Extract xy and visibility
    if gt.shape[1] >= 3:
        gt_xy = gt[:, :2]
        vis = (gt[:, 2] > 0).astype(np.bool_)
    else:
        gt_xy = gt[:, :2]
        vis = np.ones((K,), dtype=bool)

    # Predicted xy and scores
    if dt.shape[1] >= 3:
        dt_xy = dt[:, :2]
        scores = dt[:, 2]
    else:
        dt_xy = dt[:, :2]
        scores = np.ones((K,), dtype=float)

    # handle sigmas
    if sigmas is None:
        # conservative default: use 0.07 for all keypoints (you should tune for your dataset)
        sigmas = np.full((K,), 0.07, dtype=float)
    else:
        sigmas = np.asarray(sigmas, dtype=float)
        if sigmas.shape[0] != K:
            raise ValueError("sigmas length must match number of keypoints")

    # object scale: COCO uses s = sqrt(area). If area==0 -> fallback to 1.0
    if area <= 0:
        x_min = gt_xy[2][0]
        x_max = gt_xy[3][0]
        y_min = gt_xy[0][1]
        y_max = gt_xy[3][1]
        area = max(1.0, (x_max - x_min) * (y_max - y_min))
        s = math.sqrt(0.53 * area)
    else:
        s = math.sqrt(0.53 * area)

    # per-keypoint constant used in formula: COCO uses k_i = 2 * sigma_i
    # k = 2.0 * sigmas
    k = 1.0 * sigmas

    oks_sum = 0.0
    for i in range(K):
        if not vis[i]:
            continue
        d_i = math.sqrt((np.square(dt_xy[i, 0] - gt_xy[i, 0]) + np.square(dt_xy[i, 1] - gt_xy[i, 1])))
        if d_i > 0:
            weight_keypoint = d_i ** 2 / (2 * s ** 2 * k[i] ** 2)
            oks_sum += np.exp(-weight_keypoint)

    return oks_sum / float(vis.sum()) if vis.sum() > 0 else 0.0


def calculate_dataset_oks(
    image_folder_path: str,
    ground_truth_path: str,
    predicted_path: str,
    sigmas: Optional[np.ndarray] = np.array([0.075, 0.075, 0.060, 0.060, 0.0725, 0.0725, 0.0575, 0.0575, 0.07, 0.07, 0.055, 0.055, 0.055, 0.04]),
    confidence_threshold: float = 0.5,
    skip_images_without_prediction: bool = False
) -> Tuple[dict, float]:
    """
    Compute OKS per-image (matching by 'id' field) and dataset mean OKS.

    Args:
        image_folder_path: folder containing images (used to compute bbox area if needed)
        ground_truth_path: path to GT json: [ {"id": "name", "kps": [[x,y],...]} , ... ]
        predicted_path: path to predictions json: [ {"id": "name", "kps": [[x,y,score],...]} , ... ]
        sigmas: per-keypoint sigma array (length K). If None, defaults to 0.05 for all.
        confidence_threshold: threshold to treat predicted keypoint as present
        skip_images_without_prediction: if True, skip GT images that have no prediction; else treat as oks=0

    Returns:
        (oks_per_image_dict, mean_oks)
        oks_per_image_dict: { img_id: oks_value }
        mean_oks: average oks across considered images
    """
    with open(ground_truth_path, 'r') as f:
        gt_list = json.load(f)
    with open(predicted_path, 'r') as f:
        pred_list = json.load(f)

    gt_dict = {item['id']: item for item in gt_list}
    pred_dict = {item['id']: item for item in pred_list}

    oks_results = {}
    oks_values = []

    for img_id, gt_kps in gt_dict.items():
        if img_id not in pred_dict:
            if skip_images_without_prediction:
                continue
            else:
                # no prediction => oks = 0
                oks_results[img_id] = 0.0
                oks_values.append(0.0)
                continue

        gt_kps = gt_dict[img_id]["kps"]
        pred_kps = pred_dict[img_id]["kps"]
        area = gt_dict[img_id]["area"]

        oks = oks_per_instance(gt_kps, pred_kps, area, sigmas=sigmas, confidence_threshold=confidence_threshold)
        oks_results[img_id] = oks
        oks_values.append(oks)

    mean_oks = float(np.mean(oks_values)) if len(oks_values) > 0 else 0.0
    return oks_results, mean_oks

import cv2
# ---- Thư mục ảnh ----
image_folder_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\images_out"  # thư mục chứa ảnh gốc
image_out_folder_path = "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\images_out_oks_2"  # thư mục lưu ảnh đã vẽ
os.makedirs(image_out_folder_path, exist_ok=True)

# ---- Đọc dữ liệu ----
with open("C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\ground_truth.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)
with open("C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\predicted.json", "r", encoding="utf-8") as f:
    pred_data = json.load(f)

gt_dict = {item["id"]: item for item in gt_data}
pred_dict = {item["id"]: item for item in pred_data}

# ---- Tính OKS và vẽ keypoints ----
results = []
for img_id in sorted(gt_dict.keys()):
    if img_id not in pred_dict:
        print(f"[WARN] Không tìm thấy dự đoán cho ảnh {img_id}")
        continue

    gt_kps = gt_dict[img_id]["kps"]
    pred_kps = pred_dict[img_id]["kps"]
    area = gt_dict[img_id]["area"]

    oks = oks_per_instance(gt_kps, pred_kps, area)
    results.append({"id": img_id, "OKS": oks})

    # Đọc ảnh
    img_path = os.path.join(image_folder_path, img_id + ".png")
    if not os.path.exists(img_path):
        img_path = os.path.join(image_folder_path, img_id)  # đề phòng đã có đuôi
        if not os.path.exists(img_path):
            print(f"[WARN] Không tìm thấy file ảnh {img_id}")
            continue

    img = cv2.imread(img_path)

    # Vẽ keypoints ground_truth màu xanh dương
    for (x, y) in gt_kps:
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)  # BGR: xanh dương

    # Ghi giá trị OKS ở góc trái trên màu đỏ
        # Tạo text
    text = f"OKS: {oks:.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    thickness = 4

    # Lấy kích thước text
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Tọa độ góc trái trên và phải dưới của background
    bg_x1, bg_y1 = 10, 10
    bg_x2, bg_y2 = bg_x1 + text_w + 10, bg_y1 + text_h + baseline + 10

    # Vẽ nền trắng
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)

    # Vẽ text màu đen
    cv2.putText(img, text, (bg_x1 + 5, bg_y1 + text_h + 2),
                font, scale, (0, 0, 0), thickness)

    # Lưu ảnh
    out_path = os.path.join(image_out_folder_path, img_id + ".jpg")
    cv2.imwrite(out_path, img)

print("\nAverage OKS:", np.mean([r["OKS"] for r in results]))

# Example usage:
oks_dict, mean_oks = calculate_dataset_oks(
    "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\images_out",
    "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\ground_truth.json",
    "C:\\Users\\Admin\\Downloads\\Badminton\\CourtDetectionOffical\\evaluate\\data_example\\predicted.json",
)
print("mean OKS:", mean_oks)